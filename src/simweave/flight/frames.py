"""Rotation and angle helpers.

Conventions used throughout :mod:`simweave.flight`:

* World frame: right-handed, **Z up** (x east/forward, y north/left, z up).
* Body frame: x forward, y left, z up.
* Euler angles are ZYX (yaw about world z, then pitch about y, then roll about
  x), so ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` maps body vectors to world.
  Positive roll banks to the right (right side down); positive pitch is nose
  down; positive yaw is counter-clockwise seen from above.
"""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle):
    """Wrap an angle (or array of angles) in radians to ``[-pi, pi)``."""
    return (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation matrix for ZYX Euler angles (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def euler_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """``(roll, pitch, yaw)`` from a body-to-world rotation matrix.

    Pitch is returned in ``[-pi/2, pi/2]``. At the gimbal-lock singularity
    (pitch = +/-90 degrees) yaw is set to zero and roll absorbs the rotation.
    """
    sp = -float(R[2, 0])
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    if abs(sp) > 1.0 - 1e-12:
        return math.atan2(-R[1, 2], R[1, 1]), pitch, 0.0
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Unit quaternion ``[w, x, y, z]`` for ZYX Euler angles."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quaternion_to_euler(q) -> tuple[float, float, float]:
    """ZYX Euler angles ``(roll, pitch, yaw)`` from a quaternion ``[w, x, y, z]``."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=float) / np.linalg.norm(q))
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return euler_from_rotation(R)
