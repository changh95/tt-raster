# SPDX-License-Identifier: Apache-2.0
"""3D Gaussian Splatting on tt-raster: scenes (numpy arrays or a trained model's PLY), cameras and a renderer.

    import ttraster_gs as gs
    scene = gs.Scene.load_ply("point_cloud.ply")          # or gs.Scene.synthetic(100_000), or gs.Scene(xyz=..., ...)
    cam = gs.Camera.look_at(eye=(0, 1, 4), target=(0, 0, 0), up=(0, 1, 0), fov_x=60, width=1280, height=720)
    image = gs.Renderer("cpu").render(scene, cam)           # uint8 [720, 1280, 4]; "reference", "tensix" and "remote" go through the core

The library is found through TTRASTER_GS_LIB or build/plugins/gaussian/libttraster_gaussian.so. Conventions are those of gs/gaussians.h:
world to camera row-major, the camera looks along +z with x right and y down, opacities in [0, 1], scales linear, unit quaternions (w, x, y, z).
"""
import ctypes as C
import os

import numpy as np

_lib = None


def _load():
    global _lib
    if _lib is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.environ.get("TTRASTER_GS_LIB") or os.path.join(here, "..", "..", "..", "..", "build", "plugins", "gaussian", "libttraster_gaussian.so")
        lib = C.CDLL(path)
        lib.ttr_gs_last_error.restype = C.c_char_p
        lib.ttr_gs_scene_create.restype = C.c_void_p
        lib.ttr_gs_scene_create.argtypes = [C.c_uint32, C.c_uint64] + [C.c_void_p] * 5
        lib.ttr_gs_scene_load_ply.restype = C.c_void_p
        lib.ttr_gs_scene_load_ply.argtypes = [C.c_char_p]
        lib.ttr_gs_scene_synthetic.restype = C.c_void_p
        lib.ttr_gs_scene_synthetic.argtypes = [C.c_uint64, C.c_uint32, C.c_uint32]
        lib.ttr_gs_scene_save_ply.argtypes = [C.c_void_p, C.c_char_p]
        lib.ttr_gs_scene_size.restype = C.c_uint64
        lib.ttr_gs_scene_size.argtypes = [C.c_void_p]
        lib.ttr_gs_scene_sh_degree.restype = C.c_uint32
        lib.ttr_gs_scene_sh_degree.argtypes = [C.c_void_p]
        lib.ttr_gs_scene_get.argtypes = [C.c_void_p] + [C.c_void_p] * 5
        lib.ttr_gs_scene_destroy.argtypes = [C.c_void_p]
        lib.ttr_gs_look_at.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_float, C.c_uint32, C.c_uint32]
        lib.ttr_gs_renderer_create.restype = C.c_void_p
        lib.ttr_gs_renderer_create.argtypes = [C.c_char_p, C.c_uint32]
        lib.ttr_gs_renderer_destroy.argtypes = [C.c_void_p]
        lib.ttr_gs_render.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        _lib = lib
    return _lib


def _error():
    return _load().ttr_gs_last_error().decode()


class _CameraStruct(C.Structure):
    _fields_ = [("view", C.c_float * 16), ("fx", C.c_float), ("fy", C.c_float), ("cx", C.c_float), ("cy", C.c_float),
                ("width", C.c_uint32), ("height", C.c_uint32), ("znear", C.c_float)]


class _StatsStruct(C.Structure):
    _fields_ = [("gaussians", C.c_uint64), ("splats", C.c_uint64), ("project_ms", C.c_double), ("render_ms", C.c_double)]


def _f32(a, shape_tail):
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.shape[1:] != shape_tail:
        raise ValueError(f"expected shape [n, {', '.join(map(str, shape_tail))}], got {a.shape}")
    return a


class Scene:
    """A set of Gaussians. `sh` is [n, (degree + 1)^2, 3]; degree 0 can be given as [n, 3] or [n, 1, 3]."""

    def __init__(self, xyz=None, scale=None, rot=None, opacity=None, sh=None, *, _handle=None):
        lib = _load()
        if _handle is not None:
            self._h = _handle
            return
        xyz = _f32(xyz, (3,))
        n = xyz.shape[0]
        sh = np.ascontiguousarray(sh, dtype=np.float32)
        if sh.ndim == 2:
            sh = sh[:, None, :]
        k = sh.shape[1]
        degree = {1: 0, 4: 1, 9: 2, 16: 3}.get(k)
        if degree is None or sh.shape != (n, k, 3):
            raise ValueError("sh must be [n, K, 3] with K = 1, 4, 9 or 16")
        scale, rot = _f32(scale, (3,)), _f32(rot, (4,))
        opacity = np.ascontiguousarray(opacity, dtype=np.float32).reshape(-1)
        if not (scale.shape[0] == rot.shape[0] == opacity.shape[0] == n):
            raise ValueError("arrays must have one row per Gaussian")
        self._h = lib.ttr_gs_scene_create(degree, n, xyz.ctypes.data, scale.ctypes.data, rot.ctypes.data, opacity.ctypes.data, sh.ctypes.data)
        if not self._h:
            raise ValueError(_error())

    @classmethod
    def load_ply(cls, path):
        h = _load().ttr_gs_scene_load_ply(os.fsencode(path))
        if not h:
            raise ValueError(_error())
        return cls(_handle=h)

    @classmethod
    def synthetic(cls, n, seed=1, sh_degree=3):
        h = _load().ttr_gs_scene_synthetic(n, seed, sh_degree)
        if not h:
            raise ValueError(_error())
        return cls(_handle=h)

    def save_ply(self, path):
        if _load().ttr_gs_scene_save_ply(self._h, os.fsencode(path)) != 0:
            raise ValueError(_error())

    def __len__(self):
        return int(_load().ttr_gs_scene_size(self._h))

    @property
    def sh_degree(self):
        return int(_load().ttr_gs_scene_sh_degree(self._h))

    def arrays(self):
        """(xyz, scale, rot, opacity, sh) as numpy copies."""
        n, k = len(self), (self.sh_degree + 1) ** 2
        xyz, scale, rot = np.empty((n, 3), np.float32), np.empty((n, 3), np.float32), np.empty((n, 4), np.float32)
        opacity, sh = np.empty(n, np.float32), np.empty((n, k, 3), np.float32)
        _load().ttr_gs_scene_get(self._h, xyz.ctypes.data, scale.ctypes.data, rot.ctypes.data, opacity.ctypes.data, sh.ctypes.data)
        return xyz, scale, rot, opacity, sh

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h and _lib is not None:
            _lib.ttr_gs_scene_destroy(h)


class Camera:
    def __init__(self, struct):
        self._s = struct

    @classmethod
    def look_at(cls, eye, target, up, fov_x, width, height):
        s = _CameraStruct()
        f = lambda v: (C.c_float * 3)(*[float(x) for x in v])
        if _load().ttr_gs_look_at(C.byref(s), f(eye), f(target), f(up), fov_x, width, height) != 0:
            raise ValueError(_error())
        return cls(s)

    @property
    def size(self):
        return int(self._s.width), int(self._s.height)

    @property
    def view(self):
        return np.array(self._s.view, dtype=np.float32).reshape(4, 4)


class Renderer:
    """backend: "cpu" (float reference), "reference" (the core's CPU rasterizer), "tensix" (chips of the tray, or a device server when
    TTR_SERVER is set), "remote" (a device server)."""

    def __init__(self, backend="cpu", chips=1):
        self._h = _load().ttr_gs_renderer_create(backend.encode(), chips)
        if not self._h:
            raise RuntimeError(_error())
        self.last_stats = None

    def render(self, scene, camera, background=(0.0, 0.0, 0.0)):
        w, h = camera.size
        out = np.empty((h, w, 4), np.uint8)
        bg = (C.c_float * 3)(*[float(x) for x in background])
        st = _StatsStruct()
        if _load().ttr_gs_render(self._h, scene._h, C.byref(camera._s), bg, out.ctypes.data, C.byref(st)) != 0:
            raise RuntimeError(_error())
        self.last_stats = {"gaussians": st.gaussians, "splats": st.splats, "project_ms": st.project_ms, "render_ms": st.render_ms}
        return out

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h and _lib is not None:
            _lib.ttr_gs_renderer_destroy(h)
