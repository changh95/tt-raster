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

    max_distance: float = 1000.0
    """Far clip [m] used when the camera spawn config carries no clipping range."""

    near_distance: float = 0.01
    """Near clip [m] used when the camera spawn config carries no clipping range."""

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
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, wp.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.DISTANCE_TO_IMAGE_PLANE: RenderBufferSpec(1, wp.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, wp.float32),
        }
