# SPDX-License-Identifier: Apache-2.0
"""Renderer classes of isaaclab_ttraster (imported lazily, because they need Isaac Lab)."""

__all__ = ["TtRasterRenderer", "TtRasterRendererCfg"]


def __getattr__(name):
    if name == "TtRasterRendererCfg":
        from .ttraster_renderer_cfg import TtRasterRendererCfg

        return TtRasterRendererCfg
    if name == "TtRasterRenderer":
        from .ttraster_renderer import TtRasterRenderer

        return TtRasterRenderer
    raise AttributeError(name)
