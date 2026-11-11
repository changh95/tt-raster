# SPDX-License-Identifier: Apache-2.0
"""Build a ttraster scene from a Newton model and keep it in sync with a Newton state.

This is the data path Isaac Lab's Newton backend gives a renderer: a finalized ``newton.Model``
(shapes, per-world layout) and a ``body_q`` array of body transforms that changes every step.
`NewtonScene` mirrors the visible shapes of the model as ttraster instances, uses Newton's global
shape index as the shape id (so the "shape" output is directly comparable with Newton's
``shape_index_image``) and turns ``body_q`` into instance poses.

Supported shape types: plane, sphere, capsule, box, ellipsoid, cylinder, cone and triangle meshes.
Other types are skipped with a warning (``strict=True`` raises instead). Newton conventions used
(``shape_scale``): box half extents; sphere (radius); capsule and cylinder (radius, half height) along
the local z axis; cone (radius, half height), apex at +z; ellipsoid (rx, ry, rz); mesh per-axis scale.

Poses are (px, py, pz, qx, qy, qz, qw), the layout of Newton's ``body_q``.
"""
from __future__ import annotations

import warnings

import numpy as np

from . import Camera, Scene

# newton.GeoType values (stable across the versions this was written against; checked by the tests).
PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH, CONE, CONVEX_MESH = 1, 3, 4, 5, 6, 7, 8, 9, 10
_VISIBLE = 1  # newton.ShapeFlags.VISIBLE

# Camera orientation conversion used by Isaac Lab's Newton renderer: its cameras are given in the "world"
# convention (+x forward, +z up) and Newton wants OpenGL (-z forward, +y up). xyzw.
WORLD_TO_OPENGL = (0.5, -0.5, -0.5, 0.5)


def quat_mul(a, b):
    """Hamilton product of quaternion arrays in xyzw layout, broadcasting over leading dimensions."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack(
        [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_rotate(q, v):
    """Rotate vectors v by unit quaternions q (xyzw), broadcasting over leading dimensions."""
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def compose(parent, child):
    """World pose of `child` (a pose given in the frame of `parent`); both are (..., 7) xyzw poses."""
    pos = parent[..., :3] + quat_rotate(parent[..., 3:7], child[..., :3])
    quat = quat_mul(parent[..., 3:7], child[..., 3:7])
    return np.concatenate([pos, quat], axis=-1)


def _as_numpy(a, dtype=np.float32):
    """numpy, warp (``.numpy()``) or torch (``.cpu().numpy()``) array to a numpy array."""
    if hasattr(a, "numpy"):
        a = a.numpy()
    elif hasattr(a, "cpu"):
        a = a.cpu().numpy()
    return np.asarray(a, dtype=dtype)


# ---- mesh generators (counter-clockwise seen from outside, normals pointing out) -------------------


def _grid_indices(rows, cols, wrap=True):
    """Triangles over a (rows x cols) vertex grid, `cols` wrapping around; rows run from bottom to top."""
    idx = []
    for r in range(rows - 1):
        for c in range(cols):
            c1 = (c + 1) % cols if wrap else c + 1
            a, b = r * cols + c, r * cols + c1
            d, e = (r + 1) * cols + c, (r + 1) * cols + c1
            idx += [a, b, e, a, e, d]
    return np.array(idx, dtype=np.uint32).reshape(-1, 3)


def unit_sphere_mesh(rings=16, segments=32):
    """UV sphere of radius 1 with poles on +-z. Returns (positions, normals, triangles)."""
    theta = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)  # latitude from -z to +z
    phi = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    pos = np.stack([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)], axis=-1).reshape(-1, 3)
    return pos.astype(np.float32), pos.astype(np.float32), _grid_indices(rings + 1, segments)


def ellipsoid_mesh(rx, ry, rz, rings=16, segments=32):
    pos, _, tri = unit_sphere_mesh(rings, segments)
    scale = np.array([rx, ry, rz], dtype=np.float32)
    nrm = pos / scale
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pos * scale, nrm.astype(np.float32), tri


def cylinder_mesh(radius, half_height, segments=32):
    """Closed cylinder along z. Returns (positions, normals, triangles)."""
    phi = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    side_pos = np.concatenate([np.stack([radius * c, radius * s, np.full(segments, -half_height)], 1),
                               np.stack([radius * c, radius * s, np.full(segments, half_height)], 1)])
    side_nrm = np.concatenate([np.stack([c, s, np.zeros(segments)], 1)] * 2)
    tri = [_grid_indices(2, segments)]
    pos, nrm, base = [side_pos], [side_nrm], 2 * segments
    for sign in (-1.0, 1.0):  # bottom cap faces -z, top cap faces +z
        centre = base + segments
        ring = np.stack([radius * c, radius * s, np.full(segments, sign * half_height)], 1)
        pos += [ring, np.array([[0, 0, sign * half_height]])]
        nrm += [np.tile([0, 0, sign], (segments + 1, 1))]
        t = []
        for k in range(segments):
            a, b = base + k, base + (k + 1) % segments
            t += [centre, a, b] if sign > 0 else [centre, b, a]
        tri.append(np.array(t, dtype=np.uint32).reshape(-1, 3))
        base += segments + 1
    return (np.concatenate(pos).astype(np.float32), np.concatenate(nrm).astype(np.float32),
            np.concatenate(tri).astype(np.uint32))


def cone_mesh(radius, half_height, segments=32):
    """Closed cone along z: base disc at -half_height, apex at +half_height."""
    phi = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    h = 2 * half_height
    slant = np.hypot(radius, h)
    nr, nz = h / slant, radius / slant  # outward side normal (radial part, z part)
    base = np.stack([radius * c, radius * s, np.full(segments, -half_height)], 1)
    apex = np.tile([0, 0, half_height], (segments, 1))
    side_pos = np.concatenate([base, apex])
    side_nrm = np.concatenate([np.stack([nr * c, nr * s, np.full(segments, nz)], 1)] * 2)
    tri = [_grid_indices(2, segments)]
    cap_centre = 2 * segments + segments
    cap = np.concatenate([base, [[0, 0, -half_height]]])
    cap_nrm = np.tile([0, 0, -1.0], (segments + 1, 1))
    t = []
    for k in range(segments):
        t += [cap_centre, 2 * segments + (k + 1) % segments, 2 * segments + k]
    tri.append(np.array(t, dtype=np.uint32).reshape(-1, 3))
    return (np.concatenate([side_pos, cap]).astype(np.float32), np.concatenate([side_nrm, cap_nrm]).astype(np.float32),
            np.concatenate(tri).astype(np.uint32))


def flat_shaded(pos, tri):
    """Unweld a mesh so every triangle carries its own geometric normal (what Newton shades a mesh
    without vertex normals with). Returns (positions, normals, triangles)."""
    p = pos[tri].reshape(-1, 3)  # three private vertices per triangle
    a, b, c = (pos[tri[:, k]].astype(np.float64) for k in range(3))
    fn = np.cross(b - a, c - a)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    return p.astype(np.float32), np.repeat(fn, 3, axis=0).astype(np.float32), np.arange(len(p), dtype=np.uint32).reshape(-1, 3)


def smooth_normals(pos, tri):
    """Area-weighted vertex normals of an indexed triangle mesh."""
    n = np.zeros_like(pos, dtype=np.float64)
    p = pos.astype(np.float64)
    fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(norm, 1e-20)).astype(np.float32)


# ---- the scene ---------------------------------------------------------------------------------------


class NewtonScene:
    """A ttraster scene that mirrors the visible shapes of a Newton model.

    Args:
        model: finalized ``newton.Model`` (any object with the same ``shape_*`` / ``world_count`` arrays).
        backend, n_chips: forwarded to :class:`ttraster.Scene`.
        sphere_rings, sphere_segments, capsule_rings, capsule_segments, round_segments: tessellation of
            the analytic shapes. 8 x 24 for capsules was the setting used in the Newton comparison.
        ground_half_extent: half size [m] of the square standing in for an infinite plane.
        strict: raise on an unsupported shape type instead of skipping it.
    """

    def __init__(self, model, backend="reference", n_chips=1, *, sphere_rings=16, sphere_segments=32,
                 capsule_rings=8, capsule_segments=24, round_segments=32, ground_half_extent=200.0, strict=False):
        self.model = model
        self.num_worlds = int(model.world_count)
        self.scene = Scene(self.num_worlds, backend=backend, n_chips=n_chips)
        self.skipped: list[tuple[int, int]] = []  # (shape index, geo type) that were not mirrored

        shape_type = _as_numpy(model.shape_type, np.int64).reshape(-1)
        scale = _as_numpy(model.shape_scale).reshape(-1, 3)
        xform = _as_numpy(model.shape_transform).reshape(-1, 7)
        body = _as_numpy(model.shape_body, np.int64).reshape(-1)
        world = _as_numpy(model.shape_world, np.int64).reshape(-1)
        color = _as_numpy(model.shape_color).reshape(-1, 3)
        flags = _as_numpy(model.shape_flags, np.int64).reshape(-1)
        source = list(model.shape_source)
        n_shapes = len(shape_type)

        mesh_of = {}  # cache key -> ttraster mesh id

        def mesh_id(i):
            t, s = int(shape_type[i]), scale[i]
            if t == PLANE:
                half = float(max(s[0], s[1])) if max(s[0], s[1]) > 0 else ground_half_extent
                return mesh_of.setdefault(("plane", half), self.scene.add_ground(half))
            if t == SPHERE:
                return mesh_of.setdefault(("sphere", float(s[0])), self.scene.add_sphere(float(s[0]), sphere_rings, sphere_segments))
            if t == CAPSULE:
                key = ("capsule", float(s[0]), float(s[1]))
                return mesh_of.setdefault(key, self.scene.add_capsule(float(s[0]), float(s[1]), capsule_rings, capsule_segments))
            if t == BOX:
                return mesh_of.setdefault(("box", *map(float, s)), self.scene.add_box(*map(float, s)))
            if t == ELLIPSOID:
                key = ("ellipsoid", *map(float, s))
                if key not in mesh_of:
                    mesh_of[key] = self.scene.add_mesh(*ellipsoid_mesh(*map(float, s), sphere_rings, sphere_segments))
                return mesh_of[key]
            if t == CYLINDER:
                key = ("cylinder", float(s[0]), float(s[1]))
                if key not in mesh_of:
                    mesh_of[key] = self.scene.add_mesh(*cylinder_mesh(float(s[0]), float(s[1]), round_segments))
                return mesh_of[key]
            if t == CONE:
                key = ("cone", float(s[0]), float(s[1]))
                if key not in mesh_of:
                    mesh_of[key] = self.scene.add_mesh(*cone_mesh(float(s[0]), float(s[1]), round_segments))
                return mesh_of[key]
            if t in (MESH, CONVEX_MESH):
                src = source[i]
                key = ("mesh", id(src), *map(float, s))
                if key not in mesh_of:
                    pos = _as_numpy(src.vertices).reshape(-1, 3) * s
                    tri = _as_numpy(src.indices, np.uint32).reshape(-1, 3)
                    if getattr(src, "normals", None) is not None:
                        nrm = _as_numpy(src.normals).reshape(-1, 3) / s  # normals follow the inverse scale
                        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
                        mesh_of[key] = self.scene.add_mesh(pos, nrm, tri)
                    else:
                        mesh_of[key] = self.scene.add_mesh(*flat_shaded(pos, tri))
                return mesh_of[key]
            return -1

        mesh_ids = {}
        for i in range(n_shapes):
            if not (flags[i] & _VISIBLE):
                continue
            mid = mesh_id(i)
            if mid < 0:
                if strict:
                    raise NotImplementedError(f"shape {i}: Newton geometry type {int(shape_type[i])} is not supported")
                warnings.warn(f"ttraster: shape {i} (Newton geometry type {int(shape_type[i])}) is not rendered", stacklevel=2)
                self.skipped.append((i, int(shape_type[i])))
                continue
            mesh_ids[i] = mid

        # One instance per (shape, world); shapes with world -1 are global and appear in every world.
        inst_shape, inst_world = [], []
        for w in range(self.num_worlds):
            for i in mesh_ids:
                if world[i] == w or world[i] < 0:
                    inst_shape.append(i)
                    inst_world.append(w)
        self._inst_shape = np.array(inst_shape, dtype=np.int64)
        self._inst_body = body[self._inst_shape] if len(inst_shape) else np.zeros(0, dtype=np.int64)
        self._inst_local = xform[self._inst_shape] if len(inst_shape) else np.zeros((0, 7), dtype=np.float32)
        self.num_instances = len(inst_shape)

        body_q0 = _as_numpy(model.body_q).reshape(-1, 7) if getattr(model, "body_q", None) is not None else None
        poses0 = self._poses(body_q0)
        for k in range(self.num_instances):
            i = int(self._inst_shape[k])
            self.scene.add_instance(mesh_ids[i], inst_world[k], i, color[i], poses0[k])
        self._poses_buf = poses0

    def _poses(self, body_q):
        """World poses of all instances from body transforms (None: bodies at the origin)."""
        local = self._inst_local
        if self.num_instances == 0:
            return local.copy()
        out = local.copy()
        dynamic = self._inst_body >= 0
        if dynamic.any():
            if body_q is None:
                raise ValueError("the model has shapes attached to bodies but no body_q was given")
            out[dynamic] = compose(body_q[self._inst_body[dynamic]], local[dynamic])
        return np.ascontiguousarray(out, dtype=np.float32)

    def update(self, body_q):
        """Set instance poses from ``body_q`` (array of shape (bodies, 7); numpy, warp or torch)."""
        self._poses_buf = self._poses(_as_numpy(body_q).reshape(-1, 7))
        self.scene.set_instance_poses(self._poses_buf)

    def render(self, camera, camera_poses, outputs=("depth",), light_direction=None, out=None):
        return self.scene.render(camera, camera_poses, outputs, light_direction, out)

    def backend_report(self):
        return self.scene.backend_report()

    def close(self):
        self.scene.close()


# ---- Isaac Lab camera conventions --------------------------------------------------------------------


def camera_from_intrinsics(width, height, intrinsics, near_clip, far_clip):
    """A :class:`ttraster.Camera` from a 3x3 pixel intrinsics matrix (one matrix shared by all cameras).

    The pinhole is centred: a principal point away from the image centre is rejected, because the
    renderer has no lens shift.
    """
    k = _as_numpy(intrinsics).reshape(-1, 3, 3)
    if not np.allclose(k, k[0], atol=1e-4):
        raise ValueError("ttraster needs identical intrinsics for all cameras")
    k = k[0]
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    if abs(cx - width / 2) > 1e-2 * width or abs(cy - height / 2) > 1e-2 * height:
        raise ValueError(f"principal point ({cx}, {cy}) is not the image centre ({width / 2}, {height / 2}); lens shift is not supported")
    # fx = focal_length / horizontal_aperture * width, so taking focal_length = fx and aperture = width is exact.
    return Camera(width=width, height=height, focal_length=fx, horizontal_aperture=float(width),
                  vertical_aperture=float(height) * fx / fy, near_clip=near_clip, far_clip=far_clip)


def camera_poses_opengl(positions, orientations):
    """(N, 7) camera-to-world poses in the renderer's OpenGL convention from Isaac Lab's world convention."""
    pos = _as_numpy(positions).reshape(-1, 3)
    quat = _as_numpy(orientations).reshape(-1, 4)
    conv = np.broadcast_to(np.array(WORLD_TO_OPENGL, dtype=np.float32), quat.shape)
    return np.ascontiguousarray(np.concatenate([pos, quat_mul(quat, conv)], axis=-1), dtype=np.float32)
