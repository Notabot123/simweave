from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem


class RollCarModel(DynamicSystem):
    """Roll-only vehicle model (left-right dynamics)."""

    def __init__(
        self,
        sprung_mass: float,
        roll_inertia: float,
        unsprung_mass_left: float,
        unsprung_mass_right: float,
        k_sl: float,
        k_sr: float,
        c_sl: float,
        c_sr: float,
        k_tl: float,
        k_tr: float,
        track_width: float,
        x0: tuple[float, ...] = (0.0,) * 8,
    ):
        self.m_s = float(sprung_mass)
        self.I_phi = float(roll_inertia)

        self.m_ul = float(unsprung_mass_left)
        self.m_ur = float(unsprung_mass_right)

        self.k_sl = float(k_sl)
        self.k_sr = float(k_sr)

        self.c_sl = float(c_sl)
        self.c_sr = float(c_sr)

        self.k_tl = float(k_tl)
        self.k_tr = float(k_tr)

        self.w = float(track_width)

        self._x0 = np.asarray(x0, dtype=float)

    def initial_state(self):
        return self._x0.copy()

    def state_labels(self):
        return (
            "z_s", "z_s_dot",
            "phi", "phi_dot",
            "z_ul", "z_ul_dot",
            "z_ur", "z_ur_dot"
        )

    def derivatives(self, t, state, inputs=None):
        z_s, z_s_dot, phi, phi_dot, z_ul, z_ul_dot, z_ur, z_ur_dot = state

        if inputs is None:
            z_rl = z_rr = 0.0
        else:
            z_rl, z_rr = inputs

        # geometry
        half_w = self.w / 2

        # deflections
        delta_l = z_s + half_w * phi - z_ul
        delta_r = z_s - half_w * phi - z_ur

        # velocities
        delta_l_dot = z_s_dot + half_w * phi_dot - z_ul_dot
        delta_r_dot = z_s_dot - half_w * phi_dot - z_ur_dot

        # forces
        F_sl = -self.k_sl * delta_l - self.c_sl * delta_l_dot
        F_sr = -self.k_sr * delta_r - self.c_sr * delta_r_dot

        # body
        z_s_ddot = (F_sl + F_sr) / self.m_s
        phi_ddot = (half_w * (F_sl - F_sr)) / self.I_phi

        # wheels
        z_ul_ddot = (-F_sl - self.k_tl * (z_ul - z_rl)) / self.m_ul
        z_ur_ddot = (-F_sr - self.k_tr * (z_ur - z_rr)) / self.m_ur

        return np.array([
            z_s_dot, z_s_ddot,
            phi_dot, phi_ddot,
            z_ul_dot, z_ul_ddot,
            z_ur_dot, z_ur_ddot
        ])