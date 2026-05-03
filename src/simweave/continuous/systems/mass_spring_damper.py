from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.units.si import Mass, SpringStiffness, Damping, Distance, Velocity

class MassSpringDamper(DynamicSystem):
    """Single degree-of-freedom mass-spring-damper.

    State x = [displacement, velocity]
    Equation: m x'' + c x' + k x = F(t)
    """

    STATE_UNITS = {
        "x": Distance,
        "x_dot": Velocity,        
    }

    def __init__(
        self,
        mass: float | Mass,
        damping: float | Damping,
        stiffness: float | SpringStiffness,
        x0: tuple[float, float] = (0.0, 0.0),
    ):
        if mass <= 0:
            raise ValueError("mass must be positive")
        self.mass = self._val(mass)
        self.damping = self._val(damping)
        self.stiffness = self._val(stiffness)
        self._x0 = np.asarray([self._val(v) for v in x0], dtype=float)

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, str]:
        return ("x", "x_dot")

    def derivatives(
        self, t: float, state: np.ndarray, inputs: float | int | None = None
    ) -> np.ndarray:
        displacement, velocity = state
        force = 0.0 if inputs is None else float(inputs)
        acceleration = (
            force - self.damping * velocity - self.stiffness * displacement
        ) / self.mass
        return np.array([velocity, acceleration], dtype=float)
