# SPDX-License-Identifier: Apache-2.0
"""One abstract test scene, used by both renderers so that they draw exactly the same thing.

A cartpole-like scene in z-up world coordinates, repeated in W worlds with different poses:
a ground plane, a rail (box), a cart (box) that slides along the rail, and a pole (capsule)
that tilts about the y axis at the cart's centre. Camera: the Isaac Lab cartpole camera
(position (-5, 0, 2), world convention, 24 mm focal length, 20.955 mm aperture, clip 0.1 to 20 m).
"""
import math

import numpy as np

GROUND_COLOR = (0.125, 0.125, 0.15)
RAIL = dict(half=(2.0, 0.03, 0.03), center=(0.0, 0.0, 1.0), color=(0.55, 0.55, 0.6))
CART = dict(half=(0.30, 0.18, 0.12), color=(0.85, 0.25, 0.20))
POLE = dict(radius=0.05, half_height=0.5, color=(0.25, 0.65, 0.30))  # capsule along local z
CAMERA = dict(position=(-5.0, 0.0, 2.0), focal_length=24.0, horizontal_aperture=20.955, near=0.1, far=20.0)


def world_params(w):
    """Cart position along the rail and pole tilt for world w (deterministic, spread over the range)."""
    x = 0.9 * math.sin(0.7 * w + 0.2)
    theta = 0.55 * math.sin(1.3 * w + 0.4)
    return x, theta


def quat_mul(a, b):  # xyzw
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def quat_y(theta):  # rotation about y, xyzw
    return (0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2))


def rot_to_quat(R):  # xyzw from a 3x3 rotation matrix
    R = np.asarray(R, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = [0.0] * 4
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


# Camera-to-world rotation for the "world" convention: the camera looks along +x, up is +z
# (camera axes: +x right = world -y, +y up = world +z, -z forward = world +x).
CAMERA_QUAT = rot_to_quat([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
