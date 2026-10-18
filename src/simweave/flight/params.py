"""Physical parameters of a quadrotor."""

from __future__ import annotations

import math
from dataclasses import dataclass

from simweave.units.si import SIUnit


def _num(x) -> float:
    return float(x.value) if isinstance(x, SIUnit) else float(x)


@dataclass
class QuadrotorParams:
    """Rigid-body and propulsion parameters (SI units; unit objects accepted).

    The defaults describe a roughly 1.2 kg sport quadrotor with a
    thrust-to-weight ratio of about 2.4. They are plausible, not a model of
    any particular airframe.

    Attributes
    ----------
    mass:
        Total mass, kg.
    inertia:
        Principal moments ``(Ixx, Iyy, Izz)``, kg m^2 (body axes, diagonal).
    arm_length:
        Distance from the centre of mass to each rotor, m. Rotors sit at
        45 degrees to the body axes ("X" layout).
    max_thrust:
        Maximum thrust of one rotor, N.
    torque_ratio:
        Rotor reaction torque per unit thrust (yaw), m.
    motor_tau:
        First-order rotor response time constant, s. Must be positive.
    drag_lin, drag_quad:
        Linear (N s/m) and quadratic (N s^2/m^2) drag coefficients on the
        velocity relative to the air.
    gravity:
        Gravitational acceleration, m/s^2.
    air_density:
        Air density, kg/m^3 (used by the energy model).
    rotor_radius:
        Propeller radius, m (used by the energy model).
    figure_of_merit:
        Hover efficiency of the rotors (ideal power / actual power).
    """

    mass: float = 1.2
    inertia: tuple[float, float, float] = (0.012, 0.012, 0.022)
    arm_length: float = 0.17
    max_thrust: float = 7.0
    torque_ratio: float = 0.016
    motor_tau: float = 0.03
    drag_lin: float = 0.1
    drag_quad: float = 0.0
    gravity: float = 9.81
    air_density: float = 1.225
    rotor_radius: float = 0.12
    figure_of_merit: float = 0.65

    def __post_init__(self) -> None:
        self.mass = _num(self.mass)
        self.inertia = tuple(_num(v) for v in self.inertia)
        for name in (
            "arm_length", "max_thrust", "torque_ratio", "motor_tau", "drag_lin",
            "drag_quad", "gravity", "air_density", "rotor_radius",
            "figure_of_merit",
        ):
            setattr(self, name, _num(getattr(self, name)))
        if self.mass <= 0:
            raise ValueError("mass must be positive.")
        if len(self.inertia) != 3 or any(v <= 0 for v in self.inertia):
            raise ValueError("inertia must be three positive moments.")
        for name in ("arm_length", "max_thrust", "torque_ratio", "motor_tau",
                     "gravity", "air_density", "rotor_radius"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.drag_lin < 0 or self.drag_quad < 0:
            raise ValueError("drag coefficients must be non-negative.")
        if not 0.0 < self.figure_of_merit <= 1.0:
            raise ValueError("figure_of_merit must be in (0, 1].")
        if self.thrust_to_weight <= 1.0:
            raise ValueError("Rotors cannot lift the vehicle (thrust-to-weight <= 1).")

    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    @property
    def hover_thrust(self) -> float:
        """Thrust per rotor needed to hover, N."""
        return self.weight / 4.0

    @property
    def thrust_to_weight(self) -> float:
        return 4.0 * self.max_thrust / self.weight

    @property
    def disk_area(self) -> float:
        return math.pi * self.rotor_radius**2
