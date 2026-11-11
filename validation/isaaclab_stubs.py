# SPDX-License-Identifier: Apache-2.0
"""Minimal stand-ins for the parts of Isaac Lab that integrations/isaaclab_ttraster imports.

Isaac Lab itself needs PyTorch, CUDA and USD, which this host does not have, so the renderer is exercised
against these stubs. They copy the *shape* of the contract from the Isaac Lab sources (method names and
parameters of BaseRenderer, the render-spec and config fields, the scene-data provider calls the Newton
Warp renderer makes); `check_against_isaaclab_source` compares them with the real base class when the
sources are available. A pass proves the renderer follows that contract, not that it runs inside a
live Isaac Lab.
"""
from __future__ import annotations

import abc
import ast
import dataclasses
import enum
import os
import sys
import types


class RenderBufferKind(str, enum.Enum):
    RGB = "rgb"
    RGBA = "rgba"
    RGB_HDR = "rgb_hdr"
    ALBEDO = "albedo"
    DEPTH = "depth"
    DISTANCE_TO_IMAGE_PLANE = "distance_to_image_plane"
    DISTANCE_TO_CAMERA = "distance_to_camera"
    NORMALS = "normals"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"

    def __str__(self):  # StrEnum behaviour
        return self.value


@dataclasses.dataclass(frozen=True)
class RenderBufferSpec:
    channels: int
    dtype: type


class BaseRenderer(abc.ABC):
    def __init__(self, cfg, *, sim):
        self.cfg = cfg
        self._sim = sim

    def initialize(self):
        return

    @abc.abstractmethod
    def supported_output_types(self): ...

    @abc.abstractmethod
    def prepare_stage(self, stage, num_envs): ...

    @abc.abstractmethod
    def create_render_data(self, spec): ...

    @abc.abstractmethod
    def set_outputs(self, render_data, output_data): ...

    def prepare_capture(self, render_data, camera_data, frame):
        pass

    @abc.abstractmethod
    def update_transforms(self): ...

    @abc.abstractmethod
    def update_geometries(self): ...

    @abc.abstractmethod
    def update_camera(self, render_data, positions, orientations, intrinsics): ...

    def update_camera_intrinsics(self, render_data, intrinsics, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, render_data): ...

    def render_batch(self, render_data):
        for d in render_data:
            self.render(d)

    @abc.abstractmethod
    def read_output(self, render_data, camera_data): ...

    @abc.abstractmethod
    def cleanup(self, render_data): ...

    def reset(self, render_data, env_ids=None):
        pass

    def close(self):
        return


@dataclasses.dataclass
class RendererCfg:
    class_type: object = None
    renderer_type: str = "default"
    cloning_contexts: tuple = ()

    def supported_output_types(self):
        return None


@dataclasses.dataclass(frozen=True)
class CameraRenderSpec:
    cfg: object
    device: str
    num_instances: int
    camera_prim_paths: tuple
    view_count: int


class SceneDataFormat:
    class Transform:
        transforms = None


REQUIRES_STAGE_AND_MODEL = {"newton_warp": (False, True)}


class SimulationContext:
    pass


@dataclasses.dataclass
class NewtonBackendCfg:
    physics_cfg: object = None
    device: object = None


def install():
    """Register the stub modules in sys.modules (idempotent)."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    mod("isaaclab")
    mod("isaaclab.renderers", BaseRenderer=BaseRenderer, RenderBufferKind=RenderBufferKind, RenderBufferSpec=RenderBufferSpec)
    mod("isaaclab.renderers.camera_render_spec", CameraRenderSpec=CameraRenderSpec)
    mod("isaaclab.renderers.renderer_cfg", RendererCfg=RendererCfg)
    mod("isaaclab.utils", configclass=dataclasses.dataclass)
    mod("isaaclab.scene_data", REQUIRES_STAGE_AND_MODEL=REQUIRES_STAGE_AND_MODEL, SceneDataFormat=SceneDataFormat)
    mod("isaaclab.sim", SimulationContext=SimulationContext)
    mod("isaaclab_newton")
    mod("isaaclab_newton.physics", NewtonBackendCfg=NewtonBackendCfg)


def _methods(source_path, class_name):
    """{method: (is_abstract, [parameter names])} of a class, parsed from a Python source file."""
    tree = ast.parse(open(source_path).read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out = {}
            for f in node.body:
                is_property = isinstance(f, ast.FunctionDef) and any(
                    isinstance(d, ast.Name) and d.id == "property" for d in f.decorator_list)
                if isinstance(f, ast.FunctionDef) and not f.name.startswith("_") and not is_property:
                    abstract = any(isinstance(d, ast.Name) and d.id == "abstractmethod" for d in f.decorator_list)
                    out[f.name] = (abstract, [a.arg for a in f.args.args + f.args.kwonlyargs if a.arg != "self"])
            return out
    raise LookupError(class_name)


def check_against_isaaclab_source(renderer_class, source_root=None):
    """Compare BaseRenderer in the Isaac Lab sources with the renderer class. Returns (problems, checked).

    Every public method of the real BaseRenderer must exist on the renderer with the same parameter names,
    and every abstract one must be overridden by it. `checked` is False when the sources are not present.
    """
    root = source_root or os.environ.get("ISAACLAB_SRC") or os.path.expanduser("~/experiments/refs/IsaacLab/source")
    path = os.path.join(root, "isaaclab", "isaaclab", "renderers", "base_renderer.py")
    if not os.path.exists(path):
        return [], False
    import inspect

    problems = []
    for name, (abstract, params) in _methods(path, "BaseRenderer").items():
        member = getattr(renderer_class, name, None)
        if member is None or not callable(member):
            problems.append(f"missing method {name}")
            continue
        mine = [p for p in inspect.signature(member).parameters if p != "self"]
        if mine != params:
            problems.append(f"{name}: parameters {mine} differ from Isaac Lab's {params}")
        if abstract and getattr(member, "__isabstractmethod__", False):
            problems.append(f"{name} is abstract and not overridden")
    return problems, True
