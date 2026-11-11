# SPDX-License-Identifier: Apache-2.0
"""ctypes binding of libttraster (include/ttraster/ttraster.h).

    import ttraster
    scene = ttraster.Scene(num_worlds, backend="reference")      # or "tensix"
    box = scene.add_box(0.3, 0.2, 0.1)
    scene.add_instance(box, world=0, shape_id=1, albedo=(0.8, 0.2, 0.2), pose=(x, y, z, qx, qy, qz, qw))
    out = scene.render(camera, camera_poses, outputs=("depth", "shape", "rgb"))
"""
import ctypes as C
import os
from dataclasses import dataclass

import numpy as np


class _Camera(C.Structure):
    _fields_ = [("width", C.c_uint32), ("height", C.c_uint32), ("focal_length", C.c_float),
                ("horizontal_aperture", C.c_float), ("vertical_aperture", C.c_float),
                ("near_clip", C.c_float), ("far_clip", C.c_float), ("background", C.c_uint32)]


class _Images(C.Structure):
    _fields_ = [("depth", C.c_void_p), ("forward_depth", C.c_void_p), ("shape", C.c_void_p),
                ("albedo", C.c_void_p), ("rgb", C.c_void_p), ("normal", C.c_void_p)]


class _Stats(C.Structure):
    _fields_ = [("vertex_ms", C.c_double), ("raster_ms", C.c_double), ("assemble_ms", C.c_double),
                ("build_ms", C.c_double), ("total_ms", C.c_double),
                ("triangles", C.c_uint64), ("passes", C.c_uint32)]


@dataclass
class Camera:
    width: int = 96
    height: int = 96
    focal_length: float = 24.0
    horizontal_aperture: float = 20.955
    vertical_aperture: float = 20.955
    near_clip: float = 0.1
    far_clip: float = 20.0
    background: int = 0xFF000000  # RGBA8, R in the low byte: colour of the rgb image where nothing is hit


_CHANNELS = {"depth": (np.float32, 1), "forward_depth": (np.float32, 1), "shape": (np.uint32, 1),
             "albedo": (np.uint8, 4), "rgb": (np.uint8, 4), "normal": (np.float32, 3)}


def _load():
    path = os.environ.get("TTRASTER_LIB")
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, "libttraster.so"),
                     os.path.join(here, "..", "..", "build", "src", "libttraster.so")):
            if os.path.exists(cand):
                path = cand
                break
    if not path:
        raise OSError("libttraster.so not found; set TTRASTER_LIB")
    lib = C.CDLL(path)
    lib.ttr_scene_create.restype = C.c_void_p
    lib.ttr_scene_create.argtypes = [C.c_uint32, C.c_char_p, C.c_uint32]
    lib.ttr_scene_destroy.argtypes = [C.c_void_p]
    lib.ttr_last_error.restype = C.c_char_p
    lib.ttr_backend_report.restype = C.c_char_p
    lib.ttr_backend_report.argtypes = [C.c_void_p]
    for name, args in (("ttr_add_box", [C.c_float] * 3), ("ttr_add_sphere", [C.c_float, C.c_int, C.c_int]),
                       ("ttr_add_capsule", [C.c_float, C.c_float, C.c_int, C.c_int]), ("ttr_add_ground", [C.c_float])):
        f = getattr(lib, name)
        f.restype = C.c_int32
        f.argtypes = [C.c_void_p] + args
    lib.ttr_add_mesh.restype = C.c_int32
    lib.ttr_add_mesh.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32]
    lib.ttr_add_instance.restype = C.c_int32
    lib.ttr_add_instance.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_void_p, C.c_void_p]
    lib.ttr_set_instance_poses.argtypes = [C.c_void_p, C.c_void_p]
    lib.ttr_set_lighting.argtypes = [C.c_void_p, C.c_int]
    lib.ttr_set_lighting.restype = None
    lib.ttr_render.argtypes = [C.c_void_p, C.POINTER(_Camera), C.c_void_p, C.c_void_p, C.POINTER(_Images), C.POINTER(_Stats)]
    lib.ttr_submit.argtypes = [C.c_void_p, C.POINTER(_Camera), C.c_void_p, C.c_void_p, C.POINTER(_Images)]
    lib.ttr_collect.argtypes = [C.c_void_p, C.c_int, C.POINTER(_Stats)]
    return lib


_lib = None


def _l():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


def _check(rc):
    if rc < 0:
        raise RuntimeError(_l().ttr_last_error().decode())
    return rc


def _f32(a, n):
    a = np.ascontiguousarray(a, dtype=np.float32)
    assert a.size == n, (a.size, n)
    return a


class Scene:
    def __init__(self, num_worlds, backend="reference", n_chips=1):
        self.num_worlds = num_worlds
        self._h = _l().ttr_scene_create(num_worlds, backend.encode(), n_chips)
        if not self._h:
            raise RuntimeError(_l().ttr_last_error().decode())
        self.num_instances = 0
        self._pending = {}   # ticket -> output arrays kept alive while a frame is in flight

    def close(self):
        if self._h:
            _l().ttr_scene_destroy(self._h)
            self._h = None

    __del__ = close

    def add_box(self, hx, hy, hz):
        return _check(_l().ttr_add_box(self._h, hx, hy, hz))

    def add_sphere(self, radius, rings=16, segments=32):
        return _check(_l().ttr_add_sphere(self._h, radius, rings, segments))

    def add_capsule(self, radius, half_height, rings=8, segments=24):
        return _check(_l().ttr_add_capsule(self._h, radius, half_height, rings, segments))

    def add_ground(self, half_extent=50.0):
        return _check(_l().ttr_add_ground(self._h, half_extent))

    def add_mesh(self, positions, normals, indices):
        p = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        n = np.ascontiguousarray(normals, dtype=np.float32).reshape(-1, 3)
        i = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1, 3)
        return _check(_l().ttr_add_mesh(self._h, p.ctypes.data, n.ctypes.data, len(p), i.ctypes.data, len(i)))

    def add_instance(self, mesh, world, shape_id, albedo, pose):
        a, p = _f32(albedo, 3), _f32(pose, 7)
        r = _check(_l().ttr_add_instance(self._h, mesh, world, shape_id, a.ctypes.data, p.ctypes.data))
        self.num_instances += 1
        return r

    def set_instance_poses(self, poses):
        p = _f32(poses, 7 * self.num_instances)
        _check(_l().ttr_set_instance_poses(self._h, p.ctypes.data))

    def set_lighting(self, per_pixel):
        """Light the ``rgb`` output per pixel (True: a fragment shader on interpolated normals; False: at the vertices, the default)."""
        _l().ttr_set_lighting(self._h, 1 if per_pixel else 0)

    def backend_report(self):
        return _l().ttr_backend_report(self._h).decode()

    def _prepare(self, camera, camera_poses, outputs, light_direction, out):
        """The ctypes arguments of a render call: (camera struct, poses, light, image pointers, out dict)."""
        W, H = self.num_worlds, camera.height
        out = {} if out is None else dict(out)
        ptrs = _Images()
        for name in outputs:
            dt, ch = _CHANNELS[name]
            shape = (W, H, camera.width, ch)
            if name in out:
                a = out[name]
                if a.shape != shape or a.dtype != np.dtype(dt) or not a.flags["C_CONTIGUOUS"]:
                    raise ValueError(f"output '{name}' must be a C-contiguous {np.dtype(dt).name} array of shape {shape}")
            else:
                out[name] = np.zeros(shape, dtype=dt)
            setattr(ptrs, name, out[name].ctypes.data)
        cam = _Camera(camera.width, camera.height, camera.focal_length, camera.horizontal_aperture,
                      camera.vertical_aperture, camera.near_clip, camera.far_clip, camera.background)
        poses = _f32(camera_poses, 7 * W)
        light = None if light_direction is None else _f32(light_direction, 3)
        return cam, poses, light, ptrs, out

    @staticmethod
    def _stats(st):
        return dict(vertex_ms=st.vertex_ms, raster_ms=st.raster_ms, assemble_ms=st.assemble_ms, build_ms=st.build_ms,
                    total_ms=st.total_ms, triangles=st.triangles, passes=st.passes)

    def render(self, camera, camera_poses, outputs=("depth",), light_direction=None, out=None):
        """Returns ({name: array [worlds, H, W, C]}, stats dict).

        `out` may hold preallocated C-contiguous arrays (name -> array of the right shape and dtype) that
        are filled in place; a render loop should pass the same ones every frame, since fresh multi-MB
        arrays cost page faults on every frame. Outputs missing from `out` are allocated."""
        cam, poses, light, ptrs, out = self._prepare(camera, camera_poses, outputs, light_direction, out)
        st = _Stats()
        _check(_l().ttr_render(self._h, C.byref(cam), poses.ctypes.data, None if light is None else light.ctypes.data,
                               C.byref(ptrs), C.byref(st)))
        return out, self._stats(st)

    def submit(self, camera, camera_poses, outputs=("depth",), light_direction=None, out=None):
        """Start a frame on another thread and return a ticket for :meth:`collect`.

        The instance poses are snapshotted now (set them with :meth:`set_instance_poses` before the call); the
        caller can move them on and submit the next frame while this one renders. At most two frames are in
        flight, and each needs its own output arrays (`out`), which must not be touched until collect()."""
        cam, poses, light, ptrs, out = self._prepare(camera, camera_poses, outputs, light_direction, out)
        ticket = _check(_l().ttr_submit(self._h, C.byref(cam), poses.ctypes.data,
                                        None if light is None else light.ctypes.data, C.byref(ptrs)))
        # The C side copied the camera, the poses and the light; only the output arrays must stay alive.
        self._pending[ticket] = out
        return ticket

    def collect(self, ticket):
        """Wait for a submitted frame. Returns ({name: array}, stats dict) like :meth:`render`."""
        st = _Stats()
        out = self._pending.pop(ticket, None)
        _check(_l().ttr_collect(self._h, ticket, C.byref(st)))
        return out, self._stats(st)
