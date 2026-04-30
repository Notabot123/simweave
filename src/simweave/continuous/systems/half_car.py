from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem


class HalfCarModel(DynamicSystem):
    """Half-car suspension model with pitch dynamics.

    State vector:
    x = [z_s, z_s_dot, theta, theta_dot, z_uf, z_uf_dot, z_ur, z_ur_dot]
    """

    def __init__(
        self,
        sprung_mass: float,
        pitch_inertia: float,
        unsprung_mass_front: float,
        unsprung_mass_rear: float,
        k_sf: float,
        k_sr: float,
        c_sf: float,
        c_sr: float,
        k_tf: float,
        k_tr: float,
        a: float,  # CG → front axle
        b: float,  # CG → rear axle
        x0: tuple[float, ...] = (0.0,) * 8,
    ):
        if sprung_mass <= 0 or pitch_inertia <= 0:
            raise ValueError("Mass and inertia must be positive")

        self.m_s = float(sprung_mass)
        self.I_y = float(pitch_inertia)

        self.m_uf = float(unsprung_mass_front)
        self.m_ur = float(unsprung_mass_rear)

        self.k_sf = float(k_sf)
        self.k_sr = float(k_sr)

        self.c_sf = float(c_sf)
        self.c_sr = float(c_sr)

        self.k_tf = float(k_tf)
        self.k_tr = float(k_tr)

        self.a = float(a)
        self.b = float(b)

        self._x0 = np.asarray(x0, dtype=float)

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, ...]:
        return (
            "z_s",
            "z_s_dot",
            "theta",
            "theta_dot",
            "z_uf",
            "z_uf_dot",
            "z_ur",
            "z_ur_dot",
        )

    def derivatives(self, t: float, state: np.ndarray, inputs=None) -> np.ndarray:
        (
            z_s,
            z_s_dot,
            theta,
            theta_dot,
            z_uf,
            z_uf_dot,
            z_ur,
            z_ur_dot,
        ) = state

        # road inputs (front, rear)
        if inputs is None:
            z_rf = 0.0
            z_rr = 0.0
        else:
            z_rf, z_rr = inputs

        # suspension deflections
        delta_f = z_s + self.a * theta - z_uf
        delta_r = z_s - self.b * theta - z_ur

        # velocities
        delta_f_dot = z_s_dot + self.a * theta_dot - z_uf_dot
        delta_r_dot = z_s_dot - self.b * theta_dot - z_ur_dot

        # suspension forces
        F_sf = -self.k_sf * delta_f - self.c_sf * delta_f_dot
        F_sr = -self.k_sr * delta_r - self.c_sr * delta_r_dot

        # body dynamics
        z_s_ddot = (F_sf + F_sr) / self.m_s
        theta_ddot = (self.a * F_sf - self.b * F_sr) / self.I_y

        # wheel dynamics
        z_uf_ddot = (
            -F_sf - self.k_tf * (z_uf - z_rf)
        ) / self.m_uf

        z_ur_ddot = (
            -F_sr - self.k_tr * (z_ur - z_rr)
        ) / self.m_ur

        return np.array(
            [
                z_s_dot,
                z_s_ddot,
                theta_dot,
                theta_ddot,
                z_uf_dot,
                z_uf_ddot,
                z_ur_dot,
                z_ur_ddot,
            ],
            dtype=float,
        )