from __future__ import annotations

import numpy as np

from simeng.continuous.solver import DynamicSystem


class QuarterCarModel(DynamicSystem):
    """Two degree-of-freedom quarter-car suspension model.

    State vector x = [z_s, z_s_dot, z_u, z_u_dot]
    where:
        z_s: sprung mass displacement
        z_u: unsprung mass displacement
    """

    def __init__(
        self,
        sprung_mass: float,
        unsprung_mass: float,
        suspension_stiffness: float,
        damping: float,
        tyre_stiffness: float,
        x0: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ):
        if sprung_mass <= 0 or unsprung_mass <= 0:
            raise ValueError("Masses must be positive")
        self.m_s = float(sprung_mass)
        self.m_u = float(unsprung_mass)
        self.k_s = float(suspension_stiffness)
        self.c_s = float(damping)
        self.k_t = float(tyre_stiffness)
        self._x0 = np.asarray(x0, dtype=float)

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, str, str, str]:
        return ("z_s", "z_s_dot", "z_u", "z_u_dot")

    def derivatives(
        self, t: float, state: np.ndarray, inputs: float | int | None = None
    ) -> np.ndarray:
        z_s, z_s_dot, z_u, z_u_dot = state
        z_r = 0.0 if inputs is None else float(inputs)

        suspension_deflection = z_s - z_u
        suspension_velocity = z_s_dot - z_u_dot

        z_s_ddot = (
            -self.k_s * suspension_deflection - self.c_s * suspension_velocity
        ) / self.m_s
        z_u_ddot = (
            self.k_s * suspension_deflection
            + self.c_s * suspension_velocity
            - self.k_t * (z_u - z_r)
        ) / self.m_u

        return np.array([z_s_dot, z_s_ddot, z_u_dot, z_u_ddot], dtype=float)
