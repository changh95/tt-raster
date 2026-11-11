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
}
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

    def __init__(self, spec: CameraRenderSpec, cfg: TtRasterRendererCfg):
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

    def set_outputs(self, output_data: dict) -> None:
        unsupported = [k for k in output_data if str(k) not in _NATIVE]
        if unsupported:
            raise NotImplementedError(
                f"ttraster cannot produce {sorted(map(str, unsupported))}; supported: {sorted(_NATIVE)}"
            )
        self.outputs = {str(k): v for k, v in output_data.items()}
        self.native = tuple(dict.fromkeys(_NATIVE[k][0] for k in self.outputs))

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
        for kind, proxy in self.outputs.items():
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
                "(segmentation and HDR colour are not implemented)"
            )
        if self._scene is not None and spec.num_instances != self._scene.num_worlds:
            raise ValueError(f"camera has {spec.num_instances} instances, the Newton model has {self._scene.num_worlds} worlds")
        return RenderData(spec, self.cfg)

    def set_outputs(self, render_data: RenderData, output_data: dict[str, ProxyArray]) -> None:
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
        images, stats = self._scene.render(
            render_data.camera, render_data.camera_poses, render_data.native, out=render_data._host_bufs
        )
        render_data._host_bufs.update(images)
        self.last_stats = stats
        render_data.publish(images, self.cfg.depth_clipping_behavior)

    def read_output(self, render_data: RenderData, camera_data: CameraData) -> None:
        pass  # the images were written into the bound buffers by render()

    def cleanup(self, render_data: RenderData | None) -> None:
        if render_data is not None:
            render_data.outputs = {}
            render_data._host_bufs = {}

    def close(self) -> None:
        if self._scene is not None:
            self._scene.close()
        self._scene = None
        self.backend = None
