from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.units.si import Mass, Inertia, SpringStiffness, Damping, Distance, Velocity, Angle, AngularVelocity

class FullCarModel(DynamicSystem):
    """Full car model with pitch and roll dynamics."""

    STATE_UNITS = {
        "z_s": Distance,
        "z_s_dot": Velocity,
        "theta": Angle,
        "theta_dot": AngularVelocity,
        "phi": Angle,
        "phi_dot": AngularVelocity,
        "z_ufl": Distance,
        "z_ufl_dot": Velocity,
        "z_ufr": Distance,
        "z_ufr_dot": Velocity,
        "z_url": Distance,
        "z_url_dot": Velocity,
        "z_urr": Distance,
        "z_urr_dot": Velocity,
    }
       

    def __init__(
        self,
        sprung_mass: float | Mass,
        pitch_inertia: float | Inertia,
        roll_inertia: float | Inertia,
        unsprung_mass: float | Mass,
        k_s: float | SpringStiffness,
        c_s: float | Damping,
        k_t: float | SpringStiffness,
        a: float | Distance,       # CG → front axle
        b: float | Distance,       # CG → rear axle
        track_width: float | Distance,
        x0: tuple[float, ...] = (0.0,) * 14,
        controller = None,
    ):
        self.m_s = self._val(sprung_mass)
        self.I_y = self._val(pitch_inertia)
        self.I_phi = self._val(roll_inertia)

        self.m_u = self._val(unsprung_mass)

        self.k_s = self._val(k_s)
        self.c_s = self._val(c_s)
        self.k_t = self._val(k_t)

        self.a = self._val(a)
        self.b = self._val(b)
        self.w = self._val(track_width)

        self._x0 = np.asarray([self._val(v) for v in x0], dtype=float)
        self.controller = controller

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
            z_rfl, z_rfr, z_rrl, z_rrr = 0.0
        else:
            z_rfl, z_rfr, z_rrl, z_rrr = inputs  # FL, FR, RL, RR

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

        # optionally include controller force e.g. skyhook
        if self.controller is not None:
            F_fl += self.controller.force(z_s_dot, z_ufl_dot)
            F_fr += self.controller.force(z_s_dot, z_ufr_dot)
            F_rl += self.controller.force(z_s_dot, z_url_dot)
            F_rr += self.controller.force(z_s_dot, z_urr_dot)

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
        z_ufl_ddot = (-F_fl - self.k_t * (z_ufl - z_rfl)) / self.m_u
        z_ufr_ddot = (-F_fr - self.k_t * (z_ufr - z_rfr)) / self.m_u
        z_url_ddot = (-F_rl - self.k_t * (z_url - z_rrl)) / self.m_u
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