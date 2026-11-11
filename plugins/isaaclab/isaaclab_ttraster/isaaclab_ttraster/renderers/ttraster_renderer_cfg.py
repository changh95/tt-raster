# SPDX-License-Identifier: Apache-2.0
"""Configuration of the tt-raster renderer."""

from __future__ import annotations

from typing import Literal

import warp as wp

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils import configclass


@configclass
class TtRasterRendererCfg(RendererCfg):
    """Tiled-camera rendering by a tile-based software rasterizer on Tensix cores.

    Reads the scene from the Newton model of the simulation (as the Newton Warp renderer does) and draws
    its triangles on the chip(s); see the `isaaclab_ttraster` README for what is and is not supported.
    """

    class_type: type | str = "{DIR}.ttraster_renderer:TtRasterRenderer"
    renderer_type: str = "ttraster"
    cloning_contexts: tuple[type | str, ...] = ("isaaclab_newton.cloner:NewtonReplicateContext",)

    backend: Literal["tensix", "reference"] = "tensix"
    """``"tensix"`` rasterizes on the Tenstorrent device(s); ``"reference"`` is the CPU implementation of the
    same pipeline (bit-identical image, no device needed)."""

    num_chips: int = 1
    """Number of chips for the ``"tensix"`` backend (frames are split into bands across them)."""

    per_pixel_lighting: bool = False
    """Light ``rgb`` per pixel with a fragment shader on interpolated normals instead of at the vertices. Against Newton's image the
    shading of interior pixels improves from 49.4 to 50.8 dB PSNR and the whole image (dominated by silhouette pixels) not at all
    (38.05 to 38.13 dB), while a 1,024-environment frame takes about three times as long on the device (docs/PERF.md). Off by default."""

    max_distance: float = 1000.0
    """Far clip [m] used when the camera spawn config carries no clipping range."""

    near_distance: float = 0.01
    """Near clip [m] used when the camera spawn config carries no clipping range."""

    async_rendering: bool = False
    """Trade one capture of camera latency for pipelined rendering (the pattern of ``OVRTXRendererCfg.async_rendering``).
    Defaults to False (synchronous).

    ``render()`` starts the frame of the current step and returns the previous capture, so the chip renders while the
    physics step and the policy run and a step costs the longer of the two instead of their sum. The first capture and the
    first capture after a reset wait for a fresh image; the second repeats the first. Live pose and calibration fields stay
    current; ``camera.data.info[output_name]["capture"]`` carries the image's matching ``pos_w``, ``quat_w_world``,
    ``intrinsic_matrices`` and per-environment ``frame`` indices. One camera per renderer."""

    semantic_filter: str | list[str] = "*:*"
    """Semantic filter predicate for ``semantic_segmentation`` and ``instance_segmentation`` (``"*:*"``: all types and labels)."""

    colorize_semantic_segmentation: bool = True
    """``True``: ``semantic_segmentation`` is an (N, H, W, 4) uint8 RGBA image and ``data.info["semantic_segmentation"]["idToLabels"]``
    maps colours to labels; ``False``: an (N, H, W, 1) int32 id image."""

    colorize_instance_segmentation: bool = True
    """The same for ``instance_segmentation``."""

    semantic_segmentation_mapping: dict = {}
    """Optional label to colour mapping used for the colorized ``semantic_segmentation`` (as in the Newton Warp renderer)."""

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """``"max"`` replaces the depth of pixels that hit nothing with the far clip; ``"zero"`` and ``"none"`` leave 0."""

    ground_half_extent: float = 200.0
    """Half size [m] of the square that stands in for an infinite ground plane."""

    sphere_rings: int = 16
    sphere_segments: int = 32
    capsule_rings: int = 8
    capsule_segments: int = 24
    round_segments: int = 32
    """Tessellation of spheres/ellipsoids, capsules and cylinders/cones (more triangles: closer silhouettes,
    slower frames)."""

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        def segmentation_spec(colorize: bool) -> RenderBufferSpec:
            return RenderBufferSpec(4, wp.uint8) if colorize else RenderBufferSpec(1, wp.int32)

        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
            RenderBufferKind.SEMANTIC_SEGMENTATION: segmentation_spec(self.colorize_semantic_segmentation),
            RenderBufferKind.INSTANCE_SEGMENTATION: segmentation_spec(self.colorize_instance_segmentation),
        }
