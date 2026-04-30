from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem

class FullCarModel(DynamicSystem):
    """Full car model with pitch and roll dynamics."""

    def __init__(
        self,
        sprung_mass: float,
        pitch_inertia: float,
        roll_inertia: float,
        unsprung_mass: float,
        k_s: float,
        c_s: float,
        k_t: float,
        a: float,       # CG → front axle
        b: float,       # CG → rear axle
        track_width: float,
        x0: tuple[float, ...] = (0.0,) * 14,
    ):
        self.m_s = float(sprung_mass)
        self.I_y = float(pitch_inertia)
        self.I_phi = float(roll_inertia)

        self.m_u = float(unsprung_mass)

        self.k_s = float(k_s)
        self.c_s = float(c_s)
        self.k_t = float(k_t)

        self.a = float(a)
        self.b = float(b)
        self.w = float(track_width)

        self._x0 = np.asarray(x0, dtype=float)

    def initial_state(self):
        return self._x0.copy()

    def state_labels(self):
        return (
            "z_s", "z_s_dot",
            "theta", "theta_dot",
            "phi", "phi_dot",
            "z_ufl", "z_ufl_dot",
            "z_ufr", "z_ufr_dot",
            "z_url", "z_url_dot",
            "z_urr", "z_urr_dot",
        )

    def derivatives(self, t, state, inputs=None):
        (
            z_s, z_s_dot,
            theta, theta_dot,
            phi, phi_dot,
            z_ufl, z_ufl_dot,
            z_ufr, z_ufr_dot,
            z_url, z_url_dot,
            z_urr, z_urr_dot,
        ) = state

        if inputs is None:
            z_rf = z_rr = z_rl = z_rrr = 0.0
        else:
            z_rf, z_rr, z_rl, z_rrr = inputs  # FL, FR, RL, RR

        half_w = self.w / 2

        # --- BODY DISPLACEMENTS AT WHEELS ---
        z_s_fl = z_s + self.a * theta + half_w * phi
        z_s_fr = z_s + self.a * theta - half_w * phi
        z_s_rl = z_s - self.b * theta + half_w * phi
        z_s_rr = z_s - self.b * theta - half_w * phi

        # --- VELOCITIES ---
        z_s_fl_dot = z_s_dot + self.a * theta_dot + half_w * phi_dot
        z_s_fr_dot = z_s_dot + self.a * theta_dot - half_w * phi_dot
        z_s_rl_dot = z_s_dot - self.b * theta_dot + half_w * phi_dot
        z_s_rr_dot = z_s_dot - self.b * theta_dot - half_w * phi_dot

        # --- DEFLECTIONS ---
        d_fl = z_s_fl - z_ufl
        d_fr = z_s_fr - z_ufr
        d_rl = z_s_rl - z_url
        d_rr = z_s_rr - z_urr

        d_fl_dot = z_s_fl_dot - z_ufl_dot
        d_fr_dot = z_s_fr_dot - z_ufr_dot
        d_rl_dot = z_s_rl_dot - z_url_dot
        d_rr_dot = z_s_rr_dot - z_urr_dot

        # --- SUSPENSION FORCES ---
        F_fl = -self.k_s * d_fl - self.c_s * d_fl_dot
        F_fr = -self.k_s * d_fr - self.c_s * d_fr_dot
        F_rl = -self.k_s * d_rl - self.c_s * d_rl_dot
        F_rr = -self.k_s * d_rr - self.c_s * d_rr_dot

        # --- BODY DYNAMICS ---
        z_s_ddot = (F_fl + F_fr + F_rl + F_rr) / self.m_s

        theta_ddot = (
            self.a * (F_fl + F_fr)
            - self.b * (F_rl + F_rr)
        ) / self.I_y

        phi_ddot = (
            half_w * (F_fl - F_fr + F_rl - F_rr)
        ) / self.I_phi

        # --- WHEEL DYNAMICS ---
        z_ufl_ddot = (-F_fl - self.k_t * (z_ufl - z_rf)) / self.m_u
        z_ufr_ddot = (-F_fr - self.k_t * (z_ufr - z_rr)) / self.m_u
        z_url_ddot = (-F_rl - self.k_t * (z_url - z_rl)) / self.m_u
        z_urr_ddot = (-F_rr - self.k_t * (z_urr - z_rrr)) / self.m_u

        return np.array([
            z_s_dot, z_s_ddot,
            theta_dot, theta_ddot,
            phi_dot, phi_ddot,
            z_ufl_dot, z_ufl_ddot,
            z_ufr_dot, z_ufr_ddot,
            z_url_dot, z_url_ddot,
            z_urr_dot, z_urr_ddot,
        ])