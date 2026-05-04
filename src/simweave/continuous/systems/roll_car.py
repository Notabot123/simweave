from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.units.si import Mass, Inertia, SpringStiffness, Damping, Distance, Velocity, Angle, AngularVelocity

class RollCarModel(DynamicSystem):
    """Roll-only vehicle model (left-right dynamics)."""

    STATE_UNITS = {
        "z_s": Distance,
        "z_s_dot": Velocity,
        "phi": Angle,
        "phi_dot": AngularVelocity,
        "z_ul": Distance,
        "z_ul_dot": Velocity,
        "z_ur": Distance,
        "z_ur_dot": Velocity,
    }

    def __init__(
        self,
        sprung_mass: float | Mass,
        roll_inertia: float | Inertia,
        unsprung_mass_left: float | Mass,
        unsprung_mass_right: float | Mass,
        k_sl: float | SpringStiffness,
        k_sr: float | SpringStiffness,
        c_sl: float | Damping,
        c_sr: float | Damping,
        k_tl: float | SpringStiffness,
        k_tr: float | SpringStiffness,
        track_width: float | Distance,
        x0: tuple[float, ...] = (0.0,) * 8,
        controller = None,
    ):
        self.m_s = self._val(sprung_mass)
        self.I_phi = self._val(roll_inertia)

        self.m_ul = self._val(unsprung_mass_left)
        self.m_ur = self._val(unsprung_mass_right)

        self.k_sl = self._val(k_sl)
        self.k_sr = self._val(k_sr)

        self.c_sl = self._val(c_sl)
        self.c_sr = self._val(c_sr)

        self.k_tl = self._val(k_tl)
        self.k_tr = self._val(k_tr)

        self.w = self._val(track_width)

        self._x0 = np.asarray([self._val(v) for v in x0], dtype=float)
        self.controller = controller

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

        # optional controller forces e.g. skyhook
        if self.controller is not None:
            F_sl += self.controller.force(z_s_dot, z_ul_dot)
            F_sr += self.controller.force(z_s_dot, z_ur_dot)

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