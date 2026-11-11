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


class ProxyArray:
    """isaaclab.utils.warp.ProxyArray: a warp array with a torch view (no torch here)."""

    def __init__(self, warp_array):
        self.warp = warp_array
        self.torch = None


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


def configclass(cls):
    """isaaclab.utils.configclass: a dataclass whose mutable defaults (dict, list) are copied per instance."""
    import copy

    for name, value in list(cls.__dict__.items()):
        if not name.startswith("__") and isinstance(value, (dict, list, set)):
            setattr(cls, name, dataclasses.field(default_factory=lambda v=value: copy.deepcopy(v)))
    return dataclasses.dataclass(cls)


class FakeSegmentationMapping:
    """Stand-in for isaaclab_newton.renderers.segmentation.NewtonSegmentationMapping: the same call, a simple lookup.

    Real ids come from the USD stage's semantic labels; here shape i gets id 2 + i % 3 (a 3-class "semantics"), a miss BACKGROUND_ID 0 and
    synthetic hits UNLABELLED_ID 1, which is what the real kernels write for those cases.
    """

    def __init__(self, kind, colorize, shape_count):
        self.kind, self.colorize, self.shape_count = kind, colorize, shape_count
        self.info = {"idToLabels": {str(2 + i): {"class": f"class{i}"} for i in range(3)}}

    def ids(self, shape_index):
        import numpy as np

        idx = shape_index.astype(np.int64)
        out = np.where(idx < self.shape_count, 2 + idx % 3, 1)
        return np.where(idx == 0xFFFFFFFF, 0, out).astype(np.int32)

    def convert_shape_index_to_output(self, shape_index, out_view):
        ids = self.ids(shape_index.numpy())
        if self.colorize:
            ids = (0xFF000000 | (ids.astype("uint32") * 0x00101010)).astype("uint32")   # a packed RGBA colour per id
        out_view.assign(ids.astype(out_view.numpy().dtype))


class FakeSegmentationMapper:
    def __init__(self, model, stage, cfg, plan):
        self.shape_count = len(getattr(model, "shape_label", []) or [])
        self.maps = {}

    def build_mapping(self, kind, colorize):
        self.maps[(kind, colorize)] = FakeSegmentationMapping(kind, colorize, self.shape_count)

    def get_mapping(self, kind, colorize):
        return self.maps[(kind, colorize)]


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
    mod("isaaclab.utils", configclass=configclass)
    mod("isaaclab.utils.warp", ProxyArray=ProxyArray)
    mod("isaaclab.scene_data", REQUIRES_STAGE_AND_MODEL=REQUIRES_STAGE_AND_MODEL, SceneDataFormat=SceneDataFormat)
    mod("isaaclab.sim", SimulationContext=SimulationContext)
    mod("isaaclab_newton")
    mod("isaaclab_newton.physics", NewtonBackendCfg=NewtonBackendCfg)
    mod("isaaclab_newton.renderers")
    mod("isaaclab_newton.renderers.segmentation", NewtonSegmentationMapper=FakeSegmentationMapper)


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
