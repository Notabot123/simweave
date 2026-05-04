from __future__ import annotations

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.units.si import Mass, Inertia, SpringStiffness, Damping, Distance, Velocity, Angle, AngularVelocity


class HalfCarModel(DynamicSystem):
    """Half-car suspension model with pitch dynamics.

    State vector:
    x = [z_s, z_s_dot, theta, theta_dot, z_uf, z_uf_dot, z_ur, z_ur_dot]
    """
    STATE_UNITS = {
        "z_s": Distance,
        "z_s_dot": Velocity,
        "theta": Angle,
        "theta_dot": AngularVelocity,
        "z_uf": Distance,
        "z_uf_dot": Velocity,
        "z_ur": Distance,
        "z_ur_dot": Velocity,
    }

    def __init__(
        self,
        sprung_mass: float | Mass,
        pitch_inertia: float | Inertia,
        unsprung_mass_front: float | Mass,
        unsprung_mass_rear: float | Mass,
        k_sf: float | SpringStiffness,
        k_sr: float | SpringStiffness,
        c_sf: float | Damping,
        c_sr: float | Damping,
        k_tf: float | SpringStiffness,
        k_tr: float | SpringStiffness,
        a: float | Distance,  # CG → front axle
        b: float | Distance,  # CG → rear axle
        x0: tuple[float, ...] = (0.0,) * 8,
        controller = None
    ):
        if self._val(sprung_mass) <= 0 or self._val(pitch_inertia) <= 0:
            raise ValueError("Mass and inertia must be positive")

        self.m_s = self._val(sprung_mass)
        self.I_y = self._val(pitch_inertia)

        self.m_uf = self._val(unsprung_mass_front)
        self.m_ur = self._val(unsprung_mass_rear)

        self.k_sf = self._val(k_sf)
        self.k_sr = self._val(k_sr)

        self.c_sf = self._val(c_sf)
        self.c_sr = self._val(c_sr)

        self.k_tf = self._val(k_tf)
        self.k_tr = self._val(k_tr)

        self.a = self._val(a)
        self.b = self._val(b)

        self._x0 = np.asarray([self._val(v) for v in x0], dtype=float)
        self.controller = controller

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

        # suspension forces (passive  only)
        F_front = -self.k_sf * delta_f - self.c_sf * delta_f_dot
        F_rear = -self.k_sr * delta_r - self.c_sr * delta_r_dot

        # optionally supplement with controller forces e.g. skyhook
        if self.controller is not None:
            F_front += self.controller.force(z_s_dot, z_uf_dot)
            F_rear += self.controller.force(z_s_dot, z_ur_dot)

        # body dynamics
        z_s_ddot = (F_front + F_rear) / self.m_s
        theta_ddot = (self.a * F_front - self.b * F_rear) / self.I_y

        # wheel dynamics
        z_uf_ddot = (
            -F_front - self.k_tf * (z_uf - z_rf)
        ) / self.m_uf

        z_ur_ddot = (
            -F_rear - self.k_tr * (z_ur - z_rr)
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