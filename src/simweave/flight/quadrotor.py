"""Quadrotor rigid-body dynamics with first-order rotor response.

State vector (16)::

    x y z   vx vy vz   roll pitch yaw   p q r   f1 f2 f3 f4

position and velocity in the world frame (Z up), ZYX Euler angles, body rates
``p q r`` in the body frame, and the *actual* thrust of each rotor, which
follows the commanded thrust with time constant ``motor_tau``. The input is
the four commanded rotor thrusts (N); see :class:`~simweave.flight.mixer.Mixer`
for turning a collective thrust and body torques into rotor commands.

Modelling notes: rotor gyroscopic and blade-flapping effects are ignored,
inertia is diagonal, and drag acts on the velocity relative to the air. Euler
kinematics are singular at pitch = +/-90 degrees; keep missions tilt-limited.
"""

from __future__ import annotations

import math

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.flight.params import QuadrotorParams
from simweave.units.si import Angle, AngularVelocity, Distance, Force, Velocity

_LABELS = (
    "x", "y", "z", "vx", "vy", "vz",
    "roll", "pitch", "yaw", "p", "q", "r",
    "f1", "f2", "f3", "f4",
)


class Quadrotor(DynamicSystem):
    """Quadrotor in "X" layout. See the module docstring for the state layout.

    Parameters
    ----------
    params:
        Physical parameters; defaults to :class:`QuadrotorParams`.
    position, yaw:
        Initial position (m) and heading (rad).
    hover:
        If true, start with all rotors already producing hover thrust;
        otherwise the rotors start at zero thrust (vehicle on the ground).
    wind:
        Optional :class:`~simweave.flight.wind.Wind` callable.
    motor_eff:
        Per-rotor thrust effectiveness (1.0 = healthy). These are exposed as
        the scalar attributes ``motor_eff_1`` .. ``motor_eff_4`` so a
        :class:`~simweave.faults.FaultInjector` can degrade one rotor.

    Notes
    -----
    The commanded rotor thrusts are supplied through ``simulate(...,
    controller=...)`` or ``inputs=``; with no input the rotors are commanded
    to zero thrust.
    """

    STATE_UNITS = {
        "x": Distance, "y": Distance, "z": Distance,
        "vx": Velocity, "vy": Velocity, "vz": Velocity,
        "roll": Angle, "pitch": Angle, "yaw": Angle,
        "p": AngularVelocity, "q": AngularVelocity, "r": AngularVelocity,
        "f1": Force, "f2": Force, "f3": Force, "f4": Force,
    }

    def __init__(
        self,
        params: QuadrotorParams | None = None,
        position=(0.0, 0.0, 0.0),
        yaw: float = 0.0,
        hover: bool = False,
        wind=None,
        motor_eff=(1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.params = params if params is not None else QuadrotorParams()
        p = self.params
        # Plain attributes (not read through ``params``) so fault injection
        # and parameter sweeps can perturb them on the instance.
        self.mass = p.mass
        self.Ixx, self.Iyy, self.Izz = p.inertia
        self.arm_length = p.arm_length
        self.max_thrust = p.max_thrust
        self.torque_ratio = p.torque_ratio
        self.motor_tau = p.motor_tau
        self.drag_lin = p.drag_lin
        self.drag_quad = p.drag_quad
        self.gravity = p.gravity
        self.wind = wind
        eff = tuple(float(e) for e in motor_eff)
        if len(eff) != 4:
            raise ValueError("motor_eff needs four values.")
        self.motor_eff_1, self.motor_eff_2, self.motor_eff_3, self.motor_eff_4 = eff

        pos = np.asarray(position, dtype=float)
        if pos.shape != (3,):
            raise ValueError("position must have three components.")
        self._x0 = self.hover_state(pos, yaw) if hover else self._ground_state(pos, yaw)

    # ------------------------------------------------------------------
    # state construction
    # ------------------------------------------------------------------
    def _ground_state(self, position, yaw: float) -> np.ndarray:
        x0 = np.zeros(16)
        x0[0:3] = position
        x0[8] = yaw
        return x0

    def hover_state(self, position=(0.0, 0.0, 0.0), yaw: float = 0.0) -> np.ndarray:
        """State at rest with every rotor already producing hover thrust."""
        x0 = self._ground_state(np.asarray(position, dtype=float), yaw)
        x0[12:16] = self.mass * self.gravity / 4.0
        return x0

    def initial_state(self) -> np.ndarray:
        return self._x0.copy()

    def state_labels(self) -> tuple[str, ...]:
        return _LABELS

    def validate_state(self, state: np.ndarray) -> None:
        super().validate_state(state)
        if state.size != 16:
            raise ValueError("Quadrotor state must have 16 elements.")

    @property
    def hover_thrust(self) -> float:
        """Thrust per rotor that balances the current weight, N."""
        return self.mass * self.gravity / 4.0

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def derivatives(self, t: float, state: np.ndarray, inputs=None) -> np.ndarray:
        (x, y, z, vx, vy, vz, phi, theta, psi, p, q, r,
         f1, f2, f3, f4) = state.tolist()

        if inputs is None:
            c1 = c2 = c3 = c4 = 0.0
        else:
            u = np.asarray(inputs, dtype=float).ravel()
            if u.size != 4:
                raise ValueError("Quadrotor input must be four rotor thrusts.")
            fm = self.max_thrust
            c1, c2, c3, c4 = (min(max(v, 0.0), fm) for v in u.tolist())

        # thrust actually delivered
        e1 = f1 * self.motor_eff_1
        e2 = f2 * self.motor_eff_2
        e3 = f3 * self.motor_eff_3
        e4 = f4 * self.motor_eff_4
        thrust = e1 + e2 + e3 + e4
        d = self.arm_length / math.sqrt(2.0)
        k = self.torque_ratio
        tau_x = d * (-e1 + e2 + e3 - e4)
        tau_y = d * (-e1 - e2 + e3 + e4)
        tau_z = k * (-e1 + e2 - e3 + e4)

        cphi, sphi = math.cos(phi), math.sin(phi)
        ctheta, stheta = math.cos(theta), math.sin(theta)
        cpsi, spsi = math.cos(psi), math.sin(psi)

        # body z axis in the world frame (third column of the rotation matrix)
        bx = cpsi * stheta * cphi + spsi * sphi
        by = spsi * stheta * cphi - cpsi * sphi
        bz = ctheta * cphi

        # drag on air-relative velocity
        if self.wind is not None:
            wx, wy, wz = self.wind(t, (x, y, z))
        else:
            wx = wy = wz = 0.0
        rx, ry, rz = vx - wx, vy - wy, vz - wz
        coeff = self.drag_lin
        if self.drag_quad:
            coeff += self.drag_quad * math.sqrt(rx * rx + ry * ry + rz * rz)
        m = self.mass
        ax = (thrust * bx - coeff * rx) / m
        ay = (thrust * by - coeff * ry) / m
        az = (thrust * bz - coeff * rz) / m - self.gravity

        # Euler kinematics (ZYX)
        cth = ctheta if abs(ctheta) > 1e-6 else math.copysign(1e-6, ctheta)
        s_term = q * sphi + r * cphi
        phi_dot = p + s_term * stheta / cth
        theta_dot = q * cphi - r * sphi
        psi_dot = s_term / cth

        # rigid-body rotation
        p_dot = (tau_x + (self.Iyy - self.Izz) * q * r) / self.Ixx
        q_dot = (tau_y + (self.Izz - self.Ixx) * p * r) / self.Iyy
        r_dot = (tau_z + (self.Ixx - self.Iyy) * p * q) / self.Izz

        # first-order rotor response
        inv_tau = 1.0 / self.motor_tau
        return np.array(
            [
                vx, vy, vz, ax, ay, az,
                phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot,
                (c1 - f1) * inv_tau, (c2 - f2) * inv_tau,
                (c3 - f3) * inv_tau, (c4 - f4) * inv_tau,
            ]
        )

    def post_step(self, state: np.ndarray) -> np.ndarray:
        """Ground contact: the floor at ``z = 0`` stops downward motion, and a
        vehicle that cannot lift off stays upright and still."""
        if state[2] > 1e-9:
            return state
        s = state.copy()
        s[2] = 0.0
        if s[5] < 0.0:
            s[5] = 0.0
        thrust = (
            s[12] * self.motor_eff_1 + s[13] * self.motor_eff_2
            + s[14] * self.motor_eff_3 + s[15] * self.motor_eff_4
        )
        if thrust < self.mass * self.gravity:
            s[3] = s[4] = 0.0
            s[6] = s[7] = 0.0
            s[9:12] = 0.0
        return s

    # ------------------------------------------------------------------
    # energy
    # ------------------------------------------------------------------
    def power(self, thrusts) -> np.ndarray:
        """Electrical rotor power (W) for rotor thrusts, by momentum theory:
        ``P = f^1.5 / (sqrt(2 rho A) * figure_of_merit)`` summed over rotors.

        Accepts ``(..., 4)`` and returns the total over the last axis.
        """
        p = self.params
        f = np.maximum(np.asarray(thrusts, dtype=float), 0.0)
        per_rotor = f**1.5 / (math.sqrt(2.0 * p.air_density * p.disk_area) * p.figure_of_merit)
        return per_rotor.sum(axis=-1)
