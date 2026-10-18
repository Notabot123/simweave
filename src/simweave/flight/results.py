"""Flight results, metrics and the :func:`fly` convenience runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from simweave.continuous.solver import SimulationResult, simulate
from simweave.flight.control import CascadedController, ControllerGains
from simweave.flight.quadrotor import Quadrotor
from simweave.flight.trajectory import Mission, RefPoint

_COLUMNS = (
    "time", "x", "y", "z", "roll", "pitch", "yaw",
    "vx", "vy", "vz", "p", "q", "r", "f1", "f2", "f3", "f4",
)


@dataclass
class FlightResult:
    """A simulated flight plus the reference it was flying.

    ``reference`` is the callable the controller tracked (a
    :class:`~simweave.flight.trajectory.Mission`, usually).
    """

    result: SimulationResult
    reference: Callable[[float], RefPoint]
    vehicle: Quadrotor

    # --- raw channels ---------------------------------------------------
    @property
    def time(self) -> np.ndarray:
        return self.result.time

    @property
    def position(self) -> np.ndarray:
        return self.result.state[:, 0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.result.state[:, 3:6]

    @property
    def euler(self) -> np.ndarray:
        """``(roll, pitch, yaw)`` in radians, one row per sample."""
        return self.result.state[:, 6:9]

    @property
    def rates(self) -> np.ndarray:
        return self.result.state[:, 9:12]

    @property
    def thrusts(self) -> np.ndarray:
        """Actual rotor thrusts (N), one column per rotor."""
        return self.result.state[:, 12:16]

    @property
    def commands(self) -> np.ndarray:
        """Commanded rotor thrusts (N) held during each step."""
        return self.result.inputs

    @property
    def reference_position(self) -> np.ndarray:
        return np.array([self.reference(t).pos for t in self.time])

    # --- export ---------------------------------------------------------
    def to_dict(self) -> dict[str, np.ndarray]:
        """Columns ``time, x, y, z, roll, pitch, yaw`` first (angles in
        radians, Z up), then velocity, body rates and rotor thrusts."""
        s = self.result.state
        data = {"time": self.time}
        for i, name in enumerate(_COLUMNS[1:]):
            data[name] = s[:, i]
        return data

    def to_frame(self):
        """Same columns as :meth:`to_dict` as a pandas DataFrame."""
        import pandas as pd  # optional dependency

        return pd.DataFrame(self.to_dict())

    def to_viewer(self) -> dict[str, np.ndarray]:
        """Columns for the Y-up 3D viewer (model faces +Z, Euler order YXZ).

        Viewer ``(X, Y, Z) = (y, z, x)`` of this package's Z-up frame, which
        keeps the frame right-handed. The three angles keep their values:
        roll about the forward axis, pitch (positive nose down) about the
        left axis, yaw about the up axis. Angles are radians.
        """
        s = self.result.state
        return {
            "time": self.time,
            "x": s[:, 1],
            "y": s[:, 2],
            "z": s[:, 0],
            "roll": s[:, 6],
            "pitch": s[:, 7],
            "yaw": s[:, 8],
        }

    # --- metrics ----------------------------------------------------------
    def tracking_error(self) -> np.ndarray:
        """Distance (m) between the vehicle and the reference at each sample."""
        return np.linalg.norm(self.position - self.reference_position, axis=1)

    def tilt(self) -> np.ndarray:
        """Angle (rad) between the body z axis and vertical at each sample."""
        roll, pitch = self.euler[:, 0], self.euler[:, 1]
        return np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1.0, 1.0))

    def power(self) -> np.ndarray:
        """Electrical rotor power (W) at each sample."""
        return self.vehicle.power(self.thrusts)

    def energy_wh(self) -> float:
        """Rotor energy used over the flight, Wh."""
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        return float(trapezoid(self.power(), self.time) / 3600.0)

    def saturation_fraction(self) -> float:
        """Fraction of airborne control steps with any rotor command at a limit."""
        cmd = self.commands
        armed = np.any(cmd > 0.0, axis=1)
        if not armed.any():
            return 0.0
        fmax = self.vehicle.max_thrust
        hit = np.any((cmd[armed] >= fmax - 1e-9) | (cmd[armed] <= 1e-9), axis=1)
        return float(hit.mean())

    def touchdown(self) -> dict[str, float] | None:
        """Time, vertical speed, horizontal position and miss distance at touchdown.

        Touchdown is the first sample at or below 2 cm after the vehicle has
        been above 0.5 m; ``None`` if it never lands.
        """
        z = self.position[:, 2]
        up = np.flatnonzero(z > 0.5)
        if up.size == 0:
            return None
        down = np.flatnonzero(z[up[0]:] <= 0.02)
        if down.size == 0:
            return None
        i = up[0] + down[0]
        ref_xy = np.asarray(self.reference(self.time[-1]).pos)[0:2]
        return {
            "time": float(self.time[i]),
            "vertical_speed": float(self.velocity[i, 2]),
            "horizontal_error": float(np.linalg.norm(self.position[i, 0:2] - ref_xy)),
            "x": float(self.position[i, 0]),
            "y": float(self.position[i, 1]),
        }

    def metrics(self) -> dict[str, float]:
        """Headline numbers for a flight."""
        err = self.tracking_error()
        out = {
            "rms_error": float(np.sqrt(np.mean(err**2))),
            "max_error": float(err.max()),
            "max_tilt_deg": float(np.degrees(self.tilt().max())),
            "saturation_fraction": self.saturation_fraction(),
            "energy_wh": self.energy_wh(),
            "duration": float(self.time[-1] - self.time[0]),
        }
        td = self.touchdown()
        if td is not None:
            out["touchdown_speed"] = td["vertical_speed"]
            out["landing_error"] = td["horizontal_error"]
        return out

    def phase_rms(self) -> dict[str, float]:
        """RMS tracking error per mission phase (needs a :class:`Mission`)."""
        if not isinstance(self.reference, Mission):
            raise TypeError("phase_rms needs the reference to be a Mission.")
        err = self.tracking_error()
        out: dict[str, list[float]] = {}
        names = np.array([self.reference.phase(t) for t in self.time])
        for name in dict.fromkeys(names):
            out[name] = err[names == name]
        return {str(k): float(np.sqrt(np.mean(v**2))) for k, v in out.items()}


def fly(
    vehicle: Quadrotor,
    mission: Callable[[float], RefPoint],
    *,
    gains: ControllerGains | None = None,
    dt: float = 0.004,
    control_dt: float = 0.008,
    t_end: float | None = None,
    method: str = "rk4",
) -> FlightResult:
    """Fly ``vehicle`` along ``mission`` with a :class:`CascadedController`.

    The controller models the vehicle with ``vehicle.params``, so changing
    ``vehicle.mass`` (a payload) or motor effectiveness afterwards tests the
    controller against model mismatch. ``vehicle`` may also be a wrapper with a
    ``.system`` attribute, such as a :class:`~simweave.faults.FaultInjector`.
    ``t_end`` defaults to the mission duration.
    """
    if t_end is None:
        t_end = getattr(mission, "duration", None)
        if t_end is None:
            raise ValueError("Pass t_end for a reference without a duration.")
    quad = vehicle
    while not isinstance(quad, Quadrotor):  # e.g. a FaultInjector wrapping the quad
        quad = getattr(quad, "system", None)
        if quad is None:
            raise TypeError("fly needs a Quadrotor or a wrapper exposing .system.")
    controller = CascadedController(mission, quad.params, gains)
    result = simulate(
        vehicle, (0.0, float(t_end)), dt=dt, method=method,
        controller=controller, control_dt=control_dt,
    )
    return FlightResult(result=result, reference=mission, vehicle=quad)
