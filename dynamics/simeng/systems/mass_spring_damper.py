from __future__ import annotations

import numpy as np

from simeng.solver import DynamicSystem


class MassSpringDamper(DynamicSystem):
    """Single degree-of-freedom mass-spring-damper.

    State x = [displacement, velocity]
    Equation: m x¨ + c x˙ + k x = F(t)
    """

    def __init__(self, mass: float, damping: float, stiffness: float, x0: tuple[float, float] = (0.0, 0.0)):
        if mass <= 0:
            raise ValueError('mass must be positive')
        self.mass = float(mass)
        self.damping = float(damping)
        self.stiffness = float(stiffness)
        self._x0 = np.asarray(x0, dtype=float)

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, str]:
        return ('x', 'x_dot')

    def derivatives(self, t: float, state: np.ndarray, inputs: float | int | None = None) -> np.ndarray:
        displacement, velocity = state
        force = 0.0 if inputs is None else float(inputs)
        acceleration = (force - self.damping * velocity - self.stiffness * displacement) / self.mass
        return np.array([velocity, acceleration], dtype=float)
