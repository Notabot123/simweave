"""Flight dynamics: quadrotor physics, control allocation and wind models.

Importing this package needs only numpy. Conventions (Z-up world, body x
forward / y left / z up, ZYX Euler angles) are described in
:mod:`simweave.flight.frames`.
"""

from simweave.flight.frames import (
    euler_from_rotation,
    euler_to_quaternion,
    quaternion_to_euler,
    rotation_matrix,
    wrap_angle,
)
from simweave.flight.mixer import Mixer
from simweave.flight.params import QuadrotorParams
from simweave.flight.quadrotor import Quadrotor
from simweave.flight.wind import ConstantWind, GustWind, TurbulentWind, Wind

__all__ = [
    "Quadrotor",
    "QuadrotorParams",
    "Mixer",
    "Wind",
    "ConstantWind",
    "GustWind",
    "TurbulentWind",
    "rotation_matrix",
    "euler_from_rotation",
    "euler_to_quaternion",
    "quaternion_to_euler",
    "wrap_angle",
]
