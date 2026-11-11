# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab tiled-camera renderer that rasterizes on Tenstorrent Tensix cores (tt-raster).

Structure follows ``isaaclab_newton.renderers.NewtonWarpRenderer``: the scene is the Newton model of the
simulation, body transforms come from the scene data provider every frame, and the camera outputs are
written into the buffers the camera sensor binds in :meth:`TtRasterRenderer.set_outputs`.

Data path (v1): poses and camera state are read to the host, ttraster rasterizes (Tensix or CPU), and the
images are copied into the output buffers (host to device when those live on a GPU). Geometry is static
apart from rigid-body motion: deforming meshes (``geometry_offsets``) are not updated.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.scene_data import REQUIRES_STAGE_AND_MODEL, SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab_newton.physics import NewtonBackendCfg

import ttraster
from ttraster import newton_bridge as bridge

from .ttraster_renderer_cfg import TtRasterRendererCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

logger = logging.getLogger(__name__)

# Isaac Lab output kind -> (ttraster output name, channels written to the buffer).
_NATIVE = {
    str(RenderBufferKind.RGBA): ("rgb", 4),
    str(RenderBufferKind.RGB): ("rgb", 3),
    str(RenderBufferKind.ALBEDO): ("albedo", 4),
    str(RenderBufferKind.DISTANCE_TO_CAMERA): ("depth", 1),
    str(RenderBufferKind.DEPTH): ("forward_depth", 1),
    str(RenderBufferKind.DISTANCE_TO_IMAGE_PLANE): ("forward_depth", 1),
    str(RenderBufferKind.NORMALS): ("normal", 3),
    # Segmentation is derived from the per-pixel shape index (Newton's global shape index, 0xFFFFFFFF where nothing is hit).
    str(RenderBufferKind.SEMANTIC_SEGMENTATION): ("shape", 1),
    str(RenderBufferKind.INSTANCE_SEGMENTATION): ("shape", 1),
}
_SEG_KINDS = frozenset({str(RenderBufferKind.SEMANTIC_SEGMENTATION), str(RenderBufferKind.INSTANCE_SEGMENTATION)})
_DEPTH_KINDS = frozenset(
    {str(RenderBufferKind.DISTANCE_TO_CAMERA), str(RenderBufferKind.DEPTH), str(RenderBufferKind.DISTANCE_TO_IMAGE_PLANE)}
)


def _pack_background(color) -> int:
    """RGBA8 with R in the low byte; Newton's default clear colour (93 % grey, opaque) when no colour is given."""
    if color is None:
        return 0xFFEEEEEE
    r, g, b = (max(0, min(255, round(float(c) * 255))) for c in color[:3])
    return (0xFF << 24) | (b << 16) | (g << 8) | r


def _host(array) -> np.ndarray:
    """Host view or copy of a warp array, or of the ``.warp`` of a ProxyArray."""
    array = getattr(array, "warp", array)
    return array.numpy()


class RenderData:
    """Per-camera state: output bindings, the camera model and the latest poses."""

    def __init__(self, spec: CameraRenderSpec, cfg: TtRasterRendererCfg, seg_mapper=None):
        self.cfg = cfg
        self._seg_mapper = seg_mapper   # shared, scene-static lookup builder (None until a segmentation output is requested)
        self.seg_dests: dict[str, tuple[Any, Any]] = {}   # segmentation kind -> (destination view, mapping)
        c = spec.cfg
        if getattr(c, "isp_cfg", None) is not None:
            raise NotImplementedError("ttraster does not support an ISP pipeline")
        spawn = getattr(c, "spawn", None)
        if spawn is not None and getattr(spawn, "distortion", None) is not None:
            raise NotImplementedError("ttraster renders an ideal pinhole camera; lens distortion is not supported")
        self.width, self.height = int(c.width), int(c.height)
        clipping = getattr(spawn, "clipping_range", None) if spawn is not None else None
        self.near_clip = float(clipping[0]) if clipping is not None else cfg.near_distance
        self.far_clip = float(clipping[1]) if clipping is not None else cfg.max_distance
        self.background = _pack_background(getattr(c, "background_color", None))
        self.num_envs = int(spec.num_instances)
        self.camera: ttraster.Camera | None = None
        self.camera_poses: np.ndarray | None = None
        self.outputs: dict[str, Any] = {}          # Isaac Lab kind -> ProxyArray
        self._host_bufs: dict[str, np.ndarray] = {}  # ttraster output name -> persistent host array
        self.native: tuple[str, ...] = ()
        # Asynchronous rendering (cfg.async_rendering): two sets of host images (a frame in flight owns one while the previous
        # frame is published from the other), the frame in flight as (ticket, set index, capture metadata), the frame counter,
        # the metadata snapshot of the capture about to be submitted (prepare_capture) and of the one just published.
        self._sets: list[dict[str, np.ndarray]] = [{}, {}]
        self._inflight: tuple[int, int, dict | None] | None = None
        self._frames = 0
        self._started = False
        self.capture: dict | None = None
        self.ready_capture: dict | None = None

    def set_outputs(self, output_data: dict) -> None:
        unsupported = [k for k in output_data if str(k) not in _NATIVE]
        if unsupported:
            raise NotImplementedError(
                f"ttraster cannot produce {sorted(map(str, unsupported))}; supported: {sorted(_NATIVE)}"
            )
        self.outputs = {str(k): v for k, v in output_data.items()}
        self.native = tuple(dict.fromkeys(_NATIVE[k][0] for k in self.outputs))
        # Segmentation family: each requested output is bound to a view of the camera's buffer (colorized RGBA packed as uint32, or
        # int32 ids) and the lookup of the shared mapper, as the Newton Warp renderer does.
        import warp as wp

        self.seg_dests = {}
        for kind, proxy in self.outputs.items():
            if kind not in _SEG_KINDS:
                continue
            if self._seg_mapper is None:
                raise RuntimeError(f"output '{kind}' needs a segmentation mapper, but none was created (is it in the camera's data_types?)")
            colorize = bool(self.cfg.colorize_semantic_segmentation if kind == str(RenderBufferKind.SEMANTIC_SEGMENTATION)
                            else self.cfg.colorize_instance_segmentation)
            warp_array = getattr(proxy, "warp", proxy)
            view = wp.array(ptr=warp_array.ptr, dtype=wp.uint32 if colorize else wp.int32, shape=(self.num_envs, 1, self.height, self.width),
                            device=warp_array.device, copy=False)
            self.seg_dests[kind] = (view, self._seg_mapper.get_mapping(kind, colorize))
        self._sets = [{}, {}]
        self._started = False

    def update(self, positions, orientations, intrinsics) -> None:
        self.camera_poses = bridge.camera_poses_opengl(_host(positions), _host(orientations))
        self.set_intrinsics(intrinsics)

    def set_intrinsics(self, intrinsics) -> None:
        self.camera = bridge.camera_from_intrinsics(
            self.width, self.height, _host(intrinsics), self.near_clip, self.far_clip
        )
        self.camera.background = self.background

    def publish(self, images: dict[str, np.ndarray], depth_behavior: str) -> None:
        """Copy the rendered images into the bound output buffers."""
        for kind, (view, mapping) in self.seg_dests.items():
            import warp as wp

            index = np.ascontiguousarray(images["shape"], dtype=np.uint32).reshape(self.num_envs, 1, self.height, self.width)
            mapping.convert_shape_index_to_output(wp.array(index, dtype=wp.uint32, device=view.device), view)
        for kind, proxy in self.outputs.items():
            if kind in _SEG_KINDS:
                continue
            native, channels = _NATIVE[kind]
            img = images[native]
            if channels != img.shape[-1]:
                img = np.ascontiguousarray(img[..., :channels])
            if kind in _DEPTH_KINDS and depth_behavior == "max":
                img = np.where(img <= 0.0, np.float32(self.far_clip), img)
            dest = getattr(proxy, "warp", proxy)
            if tuple(dest.shape) != img.shape:
                img = img.reshape(tuple(dest.shape))  # e.g. a (N, 1, H, W) view of the same bytes
            dest.assign(img)


class TtRasterRenderer(BaseRenderer):
    """Tensix-backed tiled camera renderer."""

    RenderData = RenderData

    def __init__(self, cfg: TtRasterRendererCfg, *, sim: SimulationContext):
        super().__init__(cfg, sim=sim)
        self.newton_cfg = NewtonBackendCfg(physics_cfg=sim.cfg.physics, device=sim.device)
        # Same needs as the Newton Warp renderer: the Newton model, not the USD stage.
        requires_stage, requires_model = REQUIRES_STAGE_AND_MODEL["newton_warp"]
        sim.requires_usd_stage |= requires_stage
        sim.requires_newton_model |= requires_model
        self._scene: bridge.NewtonScene | None = None
        self.backend = None
        self._warned_geometry = False
        self.last_stats: dict | None = None
        self._async_cameras = 0
        self._seg_mapper = None   # shared, scene-static segmentation lookup builder, created when a camera asks for segmentation

    def initialize(self) -> None:
        self.backend = self._sim.get_or_create_backend(self.newton_cfg)
        self._scene_data_provider = self._sim.get_scene_data_provider()
        self._transform_mapping = self._scene_data_provider.create_mapping(list(self.backend.model.body_label))
        cfg = self.cfg
        self._scene = bridge.NewtonScene(
            self.backend.model,
            backend=cfg.backend,
            n_chips=cfg.num_chips,
            sphere_rings=cfg.sphere_rings,
            sphere_segments=cfg.sphere_segments,
            capsule_rings=cfg.capsule_rings,
            capsule_segments=cfg.capsule_segments,
            round_segments=cfg.round_segments,
            ground_half_extent=cfg.ground_half_extent,
        )
        if self._scene.skipped:
            warnings.warn(f"ttraster: {len(self._scene.skipped)} shape(s) of unsupported type are not rendered", stacklevel=2)

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        return self.cfg.supported_output_types()

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        pass  # the scene comes from the Newton model

    def create_render_data(self, spec: CameraRenderSpec) -> RenderData:
        wanted = [str(t) for t in getattr(spec.cfg, "data_types", ())]
        unsupported = [t for t in wanted if t not in _NATIVE]
        if unsupported:
            raise NotImplementedError(
                f"ttraster cannot produce {unsupported}; supported: {sorted(_NATIVE)} "
                "(HDR colour is not implemented)"
            )
        if self._scene is not None and spec.num_instances != self._scene.num_worlds:
            raise ValueError(f"camera has {spec.num_instances} instances, the Newton model has {self._scene.num_worlds} worlds")
        if self.cfg.async_rendering and self._async_cameras >= 1:
            raise NotImplementedError("async_rendering supports one camera per renderer (two frames in flight at most)")
        has_semantic = str(RenderBufferKind.SEMANTIC_SEGMENTATION) in wanted
        has_instance = str(RenderBufferKind.INSTANCE_SEGMENTATION) in wanted
        if (has_semantic or has_instance) and self._seg_mapper is None:
            # Isaac Lab's own mapper (shape index to semantic / instance ids from the USD stage's labels), shared with the Newton Warp renderer.
            from isaaclab_newton.renderers.segmentation import NewtonSegmentationMapper

            self._seg_mapper = NewtonSegmentationMapper(self.backend.model, self._sim.stage, self.cfg, self._sim.get_clone_plan())
        if has_semantic:
            self._seg_mapper.build_mapping(str(RenderBufferKind.SEMANTIC_SEGMENTATION), bool(self.cfg.colorize_semantic_segmentation))
        if has_instance:
            self._seg_mapper.build_mapping(str(RenderBufferKind.INSTANCE_SEGMENTATION), bool(self.cfg.colorize_instance_segmentation))
        render_data = RenderData(spec, self.cfg, seg_mapper=self._seg_mapper)
        self._async_cameras += 1 if self.cfg.async_rendering else 0
        return render_data

    def set_outputs(self, render_data: RenderData, output_data: dict[str, ProxyArray]) -> None:
        self._drain(render_data)
        render_data.set_outputs(output_data)

    def update_transforms(self) -> None:
        pass  # body transforms are read when rendering

    def update_geometries(self) -> None:
        pass  # deforming geometry is not supported; rigid bodies move through update_transforms

    def update_camera(self, render_data: RenderData, positions: ProxyArray, orientations: ProxyArray, intrinsics: ProxyArray):
        render_data.update(positions, orientations, intrinsics)

    def update_camera_intrinsics(self, render_data: RenderData, intrinsics, parameters) -> None:
        render_data.set_intrinsics(intrinsics)

    def _sync_scene(self) -> None:
        backend, provider = self.backend, self._scene_data_provider
        poses = SceneDataFormat.Transform()
        if provider.get_transforms(poses, mapping=self._transform_mapping, count=backend.model.body_count):
            backend.state_0.body_q = poses.transforms
        if getattr(backend, "geometry_offsets", None) and not self._warned_geometry:
            warnings.warn("ttraster does not update deforming geometry; meshes keep their initial shape", stacklevel=2)
            self._warned_geometry = True
        self._scene.update(backend.state_0.body_q)

    def render(self, render_data: RenderData) -> None:
        if render_data.camera is None or render_data.camera_poses is None:
            raise RuntimeError("update_camera() must be called before render()")
        self._sync_scene()
        if self.cfg.async_rendering:
            self._render_async(render_data)
            return
        images, stats = self._scene.render(
            render_data.camera, render_data.camera_poses, render_data.native, out=render_data._host_bufs
        )
        render_data._host_bufs.update(images)
        self.last_stats = stats
        render_data.publish(images, self.cfg.depth_clipping_behavior)

    def prepare_capture(self, render_data: RenderData, camera_data: CameraData, frame: ProxyArray) -> None:
        """Snapshot the metadata of the capture about to be submitted; it is delivered with its image one step later."""
        if not self.cfg.async_rendering:
            return
        import warp as wp
        from isaaclab.utils.warp import ProxyArray as _ProxyArray

        render_data.capture = {
            name: _ProxyArray(wp.clone(value.warp))
            for name, value in (
                ("pos_w", camera_data.pos_w),
                ("quat_w_world", camera_data.quat_w_world),
                ("intrinsic_matrices", camera_data.intrinsic_matrices),
                ("frame", frame),
            )
        }

    def reset(self, render_data: RenderData, env_ids=None) -> None:
        """Drop the frame in flight (an observation from before the reset); the next capture waits for a fresh image."""
        self._drain(render_data)
        render_data._started = False
        render_data.ready_capture = None

    def _render_async(self, rd: RenderData) -> None:
        """Start this step's frame and publish the previous one (see ``TtRasterRendererCfg.async_rendering``)."""
        slot = rd._frames % 2
        rd._frames += 1
        capture, rd.capture = rd.capture, None
        ticket = self._scene.submit(rd.camera, rd.camera_poses, rd.native, out=rd._sets[slot])
        if rd._inflight is None and not rd._started:
            # The first capture (or the first after a reset) has no predecessor: wait for it, so that the buffers hold a valid image.
            images, stats = self._scene.collect(ticket)
            rd._sets[slot] = images
            rd._started = True
            self.last_stats = stats
            rd.publish(images, self.cfg.depth_clipping_behavior)
            rd.ready_capture = capture
            return
        # Record the new frame first: if collecting the previous one fails, the new one is still owned and gets drained.
        previous, rd._inflight = rd._inflight, (ticket, slot, capture)
        if previous is not None:
            images, stats = self._scene.collect(previous[0])
            rd._sets[previous[1]] = images
            self.last_stats = stats
            rd.publish(images, self.cfg.depth_clipping_behavior)
            rd.ready_capture = previous[2]
        # else: the second capture; the buffers still hold the first image, which is the previous one

    def _drain(self, rd: RenderData) -> None:
        """Wait for the frame in flight of this camera and drop it."""
        if rd._inflight is not None and self._scene is not None:
            ticket, slot, _ = rd._inflight
            rd._inflight = None
            try:
                rd._sets[slot], _ = self._scene.collect(ticket)
            except Exception:  # a failed frame is dropped with the rest
                logger.exception("ttraster: dropping a frame in flight")

    def read_output(self, render_data: RenderData, camera_data: CameraData | None = None) -> None:
        """The images were written into the bound buffers by render(); with async_rendering the metadata of the image that
        was just delivered goes into ``camera_data.info[output]["capture"]`` (the pose, calibration and frame indices it was
        rendered with, as Isaac Lab defines for delayed captures)."""
        if camera_data is not None and camera_data.info is not None:
            for kind, (_view, mapping) in render_data.seg_dests.items():
                camera_data.info[kind] = mapping.info   # idToLabels / idToSemantics
        capture, render_data.ready_capture = render_data.ready_capture, None
        if capture is not None and camera_data is not None and camera_data.info is not None:
            for name in camera_data.info:
                camera_data.info[name] = {**(camera_data.info[name] or {}), "capture": capture}

    def cleanup(self, render_data: RenderData | None) -> None:
        if render_data is not None:
            self._drain(render_data)
            render_data.outputs = {}
            render_data._host_bufs = {}
            render_data._sets = [{}, {}]
            self._async_cameras = max(0, self._async_cameras - (1 if self.cfg.async_rendering else 0))

    def close(self) -> None:
        if self._scene is not None:
            self._scene.close()
        self._scene = None
        self.backend = None
