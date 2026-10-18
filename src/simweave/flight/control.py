"""Cascaded position / attitude controller for a quadrotor.

Loops, outermost first::

    position P            -> velocity setpoint (speed limited)
    velocity PI           -> desired acceleration (+ reference feed-forward)
    thrust vector         -> collective thrust and desired attitude (tilt limited)
    attitude (SO(3) error)-> body-rate setpoint
    rate PI (x inertia)   -> body torques
    mixer                 -> four rotor thrusts (attitude torques first)

The controller is a sampled-data :class:`~simweave.continuous.solver.Controller`:
pass it to ``simulate(..., controller=ctrl, control_dt=...)``. The vehicle
model it uses (mass, inertia, limits) comes from ``params``; if the real
vehicle differs (a payload, say) the integral terms absorb the mismatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from simweave.continuous.control.pid import PIDController
from simweave.flight.frames import rotation_matrix
from simweave.flight.mixer import Mixer
from simweave.flight.params import QuadrotorParams
from simweave.flight.trajectory import RefPoint


@dataclass
class ControllerGains:
    """Gains and limits for :class:`CascadedController`.

    Rate-loop gains act on angular acceleration (rad/s^2 per rad/s), so they
    do not need rescaling when the inertia changes.
    """

    kp_pos_xy: float = 1.5
    kp_pos_z: float = 2.0
    kp_vel_xy: float = 3.5
    ki_vel_xy: float = 0.8
    kp_vel_z: float = 5.0
    ki_vel_z: float = 1.5
    kp_att: float = 8.0
    kp_yaw: float = 3.0
    kp_rate_rp: float = 20.0
    kp_rate_yaw: float = 10.0
    ki_rate: float = 2.0
    max_tilt: float = math.radians(35.0)
    max_speed_xy: float = 8.0
    max_speed_up: float = 3.0
    max_speed_down: float = 2.0
    max_accel_xy: float = 6.0
    max_accel_up: float = 6.0
    max_accel_down: float = 5.0
    max_angular_accel: float = 300.0


class CascadedController:
    """Tracks a reference ``ref(t) -> RefPoint`` (a trajectory or a mission).

    Parameters
    ----------
    reference:
        Callable returning the desired :class:`RefPoint` at time ``t``.
    params:
        The controller's model of the vehicle.
    gains:
        :class:`ControllerGains`; defaults suit the default vehicle.
    disarm_height:
        When both the reference and the vehicle are at or below this height
        the motors are switched off (as a real flight controller does after
        landing) and the integrators are reset.
    """

    def __init__(
        self,
        reference: Callable[[float], RefPoint],
        params: QuadrotorParams | None = None,
        gains: ControllerGains | None = None,
        disarm_height: float = 0.03,
    ) -> None:
        self.reference = reference
        self.params = params if params is not None else QuadrotorParams()
        self.gains = gains if gains is not None else ControllerGains()
        self.mixer = Mixer(self.params)
        self.disarm_height = float(disarm_height)
        self.armed = False
        g = self.gains
        a_xy, a_up, a_dn = g.max_accel_xy, g.max_accel_up, g.max_accel_down
        self._vel_pid = [
            PIDController(g.kp_vel_xy, g.ki_vel_xy, output_limits=(-a_xy, a_xy)),
            PIDController(g.kp_vel_xy, g.ki_vel_xy, output_limits=(-a_xy, a_xy)),
            PIDController(g.kp_vel_z, g.ki_vel_z, output_limits=(-a_dn, a_up)),
        ]
        lim = g.max_angular_accel
        self._rate_pid = [
            PIDController(g.kp_rate_rp, g.ki_rate, output_limits=(-lim, lim)),
            PIDController(g.kp_rate_rp, g.ki_rate, output_limits=(-lim, lim)),
            PIDController(g.kp_rate_yaw, g.ki_rate, output_limits=(-lim, lim)),
        ]
        self._inertia = np.array(self.params.inertia)

    def reset(self) -> None:
        for pid in self._vel_pid + self._rate_pid:
            pid.reset()
        self.armed = False

    # ------------------------------------------------------------------
    def update(self, t: float, state: np.ndarray, dt: float) -> np.ndarray:
        g = self.gains
        p = self.params
        ref = self.reference(t)
        pos = state[0:3]
        vel = state[3:6]
        omega = state[9:12]

        # --- disarm on the ground ------------------------------------
        if ref.pos[2] <= self.disarm_height and pos[2] <= self.disarm_height + 0.03:
            if self.armed:
                self.reset()
            return np.zeros(4)
        self.armed = True

        # --- position loop -> velocity setpoint ----------------------
        v_cmd = np.asarray(ref.vel, dtype=float).copy()
        err = ref.pos - pos
        v_cmd[0] += g.kp_pos_xy * err[0]
        v_cmd[1] += g.kp_pos_xy * err[1]
        v_cmd[2] += g.kp_pos_z * err[2]
        speed_xy = math.hypot(v_cmd[0], v_cmd[1])
        if speed_xy > g.max_speed_xy:
            v_cmd[0:2] *= g.max_speed_xy / speed_xy
        v_cmd[2] = min(max(v_cmd[2], -g.max_speed_down), g.max_speed_up)

        # --- velocity loop -> acceleration ---------------------------
        a_cmd = np.empty(3)
        for i in range(3):
            pid = self._vel_pid[i]
            pid.setpoint = float(v_cmd[i])
            a_cmd[i] = ref.acc[i] + pid(float(vel[i]), dt)

        # --- thrust vector, tilt limit -------------------------------
        force = p.mass * (a_cmd + np.array([0.0, 0.0, p.gravity]))
        force[2] = max(force[2], 0.1 * p.weight)
        f_xy = math.hypot(force[0], force[1])
        f_xy_max = force[2] * math.tan(g.max_tilt)
        if f_xy > f_xy_max:
            force[0:2] *= f_xy_max / f_xy
        z_d = force / np.linalg.norm(force)

        # --- desired attitude from thrust direction and heading ------
        x_c = np.array([math.cos(ref.yaw), math.sin(ref.yaw), 0.0])
        y_d = np.cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        x_d = np.cross(y_d, z_d)
        R_d = np.column_stack([x_d, y_d, z_d])
        R = rotation_matrix(state[6], state[7], state[8])

        collective = max(float(force @ R[:, 2]), 0.0)

        # --- attitude loop -> body-rate setpoint ----------------------
        m = 0.5 * (R_d.T @ R - R.T @ R_d)
        e_R = np.array([m[2, 1], m[0, 2], m[1, 0]])
        omega_cmd = -np.array([g.kp_att, g.kp_att, g.kp_yaw]) * e_R
        if ref.yaw_rate:
            omega_cmd += ref.yaw_rate * R[2, :]  # world yaw rate in body axes

        # --- rate loop -> torque --------------------------------------
        alpha = np.empty(3)
        for i in range(3):
            pid = self._rate_pid[i]
            pid.setpoint = float(omega_cmd[i])
            alpha[i] = pid(float(omega[i]), dt)
        inertia_omega = self._inertia * omega
        torque = self._inertia * alpha + np.cross(omega, inertia_omega)

        return self.mixer.allocate(collective, torque)
