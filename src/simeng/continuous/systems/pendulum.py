from __future__ import annotations

import numpy as np

from simeng.continuous.solver import DynamicSystem


class SimplePendulum(DynamicSystem):
    """Simple pendulum with optional viscous damping and external torque.

    State x = [theta, theta_dot]
    Equation:
        theta'' + (c / (m l^2)) theta' + (g/l) sin(theta) = tau / (m l^2)
    """

    def __init__(self, length: float, mass: float = 1.0, gravity: float = 9.81,
                 damping: float = 0.0, x0: tuple[float, float] = (0.0, 0.0)):
        if length <= 0 or mass <= 0:
            raise ValueError("length and mass must be positive")
        self.length = float(length)
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.damping = float(damping)
        self._x0 = np.asarray(x0, dtype=float)
        self._inertia = self.mass * self.length ** 2

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, str]:
        return ("theta", "theta_dot")

    def derivatives(self, t: float, state: np.ndarray,
                    inputs: float | int | None = None) -> np.ndarray:
        theta, theta_dot = state
        torque = 0.0 if inputs is None else float(inputs)
        theta_ddot = (
            torque
            - self.damping * theta_dot
            - self.mass * self.gravity * self.length * np.sin(theta)
        ) / self._inertia
        return np.array([theta_dot, theta_ddot], dtype=float)
