from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.units.si import Mass, Distance, Acceleration, Damping, Angle, AngularVelocity

class SimplePendulum(DynamicSystem):
    """Simple pendulum with optional viscous damping and external torque.

    State x = [theta, theta_dot]
    Equation:
        theta'' + (c / (m l^2)) theta' + (g/l) sin(theta) = tau / (m l^2)
    """

    STATE_UNITS = {
        "theta": Angle,
        "theta_dot": AngularVelocity,        
    }

    def __init__(
        self,
        length: float | Distance,
        mass: float | Mass = 1.0,
        gravity: float | Acceleration = 9.81,
        damping: float | Damping = 0.0,
        x0: tuple[float, float] = (0.0, 0.0),
    ):
        if length <= 0 or mass <= 0:
            raise ValueError("length and mass must be positive")
        self.length = self._val(length)
        self.mass = self._val(mass)
        self.gravity = self._val(gravity)
        self.damping = self._val(damping)
        self._x0 = np.asarray([self._val(v) for v in x0], dtype=float)
        self._inertia = self.mass * self.length**2

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, str]:
        return ("theta", "theta_dot")

    def derivatives(
        self, t: float, state: np.ndarray, inputs: float | int | None = None
    ) -> np.ndarray:
        theta, theta_dot = state
        torque = 0.0 if inputs is None else float(inputs)
        theta_ddot = (
            torque
            - self.damping * theta_dot
            - self.mass * self.gravity * self.length * np.sin(theta)
        ) / self._inertia
        return np.array([theta_dot, theta_ddot], dtype=float)
