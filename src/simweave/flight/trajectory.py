"""Reference trajectories and missions.

A *reference* is a callable ``ref(t) -> RefPoint`` giving the desired position,
velocity, acceleration (all world frame, Z up) and heading at time ``t``.
The building blocks are segments with a ``duration``; a :class:`Mission`
chains them and guarantees continuity by joining them with minimum-jerk
quintic transitions that match position, velocity and acceleration at both
ends.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np

from simweave.flight.frames import wrap_angle


class RefPoint(NamedTuple):
    """Desired state at one instant (``yaw_rate`` is optional feed-forward)."""

    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    yaw: float
    yaw_rate: float = 0.0


def _vec3(v, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have three components.")
    return arr


def _smoothstep5(u: float) -> float:
    """Quintic 0 -> 1 with zero first and second derivative at both ends."""
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _smoothstep5_rate(u: float) -> float:
    """Derivative of :func:`_smoothstep5` with respect to ``u``."""
    return 30.0 * u * u * (1.0 - u) ** 2


class Hold:
    """Stay at a point (zero velocity and acceleration) for ``duration`` s."""

    def __init__(self, position, yaw: float = 0.0, duration: float = 1.0, name: str = "hover") -> None:
        if duration <= 0:
            raise ValueError("duration must be positive.")
        self.position = _vec3(position, "position")
        self.yaw = float(yaw)
        self.duration = float(duration)
        self.name = name

    def __call__(self, tau: float) -> RefPoint:
        zero = np.zeros(3)
        return RefPoint(self.position.copy(), zero, zero, self.yaw)


class Quintic:
    """Minimum-jerk move between two :class:`RefPoint` states.

    Each axis follows the quintic polynomial that meets the position,
    velocity and acceleration of ``start`` and ``end`` after ``duration``
    seconds. Heading changes along a smooth quintic in the shortest direction.
    """

    def __init__(self, start: RefPoint, end: RefPoint, duration: float, name: str = "transition") -> None:
        if duration <= 0:
            raise ValueError("duration must be positive.")
        T = float(duration)
        p0, v0, a0 = (np.asarray(x, dtype=float) for x in (start.pos, start.vel, start.acc))
        p1, v1, a1 = (np.asarray(x, dtype=float) for x in (end.pos, end.vel, end.acc))
        dp = p1 - p0
        self._c = np.stack(
            [
                p0,
                v0,
                a0 / 2.0,
                (20 * dp - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * T**2) / (2 * T**3),
                (-30 * dp + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * T**2) / (2 * T**4),
                (12 * dp - 6 * (v1 + v0) * T - (a0 - a1) * T**2) / (2 * T**5),
            ]
        )
        self.duration = T
        self.name = name
        self._yaw0 = float(start.yaw)
        self._dyaw = float(wrap_angle(end.yaw - start.yaw))

    def __call__(self, tau: float) -> RefPoint:
        tau = min(max(tau, 0.0), self.duration)
        c = self._c
        pos = c[0] + tau * (c[1] + tau * (c[2] + tau * (c[3] + tau * (c[4] + tau * c[5]))))
        vel = c[1] + tau * (2 * c[2] + tau * (3 * c[3] + tau * (4 * c[4] + tau * 5 * c[5])))
        acc = 2 * c[2] + tau * (6 * c[3] + tau * (12 * c[4] + tau * 20 * c[5]))
        u = tau / self.duration
        yaw = self._yaw0 + self._dyaw * _smoothstep5(u)
        yaw_rate = self._dyaw * _smoothstep5_rate(u) / self.duration
        return RefPoint(pos, vel, acc, yaw, yaw_rate)


class Figure8:
    """A figure-of-eight (lemniscate) flown at constant altitude.

    ``x = a sin(w t)``, ``y = b sin(2 w t)`` about ``center`` with
    ``w = 2 pi / period``; the path starts at the crossing point moving
    diagonally. An optional vertical oscillation ``z_amp sin(w t)`` makes one
    lobe climb and the other dip.

    ``yaw_mode`` is ``"fixed"`` (hold ``yaw``) or ``"heading"`` (point along the
    direction of travel).
    """

    def __init__(
        self,
        center=(0.0, 0.0, 3.0),
        a: float = 4.0,
        b: float = 2.0,
        period: float = 20.0,
        laps: float = 1.0,
        z_amp: float = 0.0,
        yaw_mode: str = "fixed",
        yaw: float = 0.0,
        name: str = "figure8",
    ) -> None:
        if period <= 0 or laps <= 0:
            raise ValueError("period and laps must be positive.")
        if yaw_mode not in ("fixed", "heading"):
            raise ValueError("yaw_mode must be 'fixed' or 'heading'.")
        self.center = _vec3(center, "center")
        self.a, self.b, self.z_amp = float(a), float(b), float(z_amp)
        self.period = float(period)
        self.laps = float(laps)
        self.yaw_mode = yaw_mode
        self.yaw = float(yaw)
        self.duration = self.period * self.laps
        self.name = name
        self._w = 2.0 * math.pi / self.period

    def __call__(self, tau: float) -> RefPoint:
        tau = min(max(tau, 0.0), self.duration)
        w, a, b, za = self._w, self.a, self.b, self.z_amp
        s1, c1 = math.sin(w * tau), math.cos(w * tau)
        s2, c2 = math.sin(2 * w * tau), math.cos(2 * w * tau)
        pos = self.center + np.array([a * s1, b * s2, za * s1])
        vel = np.array([a * w * c1, 2 * b * w * c2, za * w * c1])
        acc = np.array([-a * w * w * s1, -4 * b * w * w * s2, -za * w * w * s1])
        if self.yaw_mode == "heading":
            yaw = math.atan2(vel[1], vel[0])
            speed2 = vel[0] ** 2 + vel[1] ** 2
            yaw_rate = (vel[0] * acc[1] - vel[1] * acc[0]) / speed2
        else:
            yaw, yaw_rate = self.yaw, 0.0
        return RefPoint(pos, vel, acc, yaw, yaw_rate)


class Mission:
    """A chain of segments flown back to back.

    ``mission(t)`` returns the :class:`RefPoint` at time ``t`` (held at the
    first/last point outside the mission). ``phase(t)`` names the active
    segment. Use :meth:`figure8` for the standard take off, fly a figure-8,
    land sequence.
    """

    def __init__(self, segments: Sequence) -> None:
        if not segments:
            raise ValueError("A mission needs at least one segment.")
        self.segments = list(segments)
        durations = np.array([s.duration for s in self.segments], dtype=float)
        self._starts = np.concatenate([[0.0], np.cumsum(durations)])
        self.duration = float(self._starts[-1])

    def _locate(self, t: float) -> tuple[int, float]:
        if t <= 0.0:
            return 0, 0.0
        if t >= self.duration:
            return len(self.segments) - 1, self.segments[-1].duration
        i = int(np.searchsorted(self._starts, t, side="right")) - 1
        return i, t - self._starts[i]

    def __call__(self, t: float) -> RefPoint:
        i, tau = self._locate(t)
        return self.segments[i](tau)

    def phase(self, t: float) -> str:
        return self.segments[self._locate(t)[0]].name

    @property
    def phases(self) -> list[tuple[str, float, float]]:
        """``(name, start, end)`` for each segment."""
        return [
            (s.name, float(self._starts[i]), float(self._starts[i + 1]))
            for i, s in enumerate(self.segments)
        ]

    @classmethod
    def figure8(
        cls,
        height: float = 3.0,
        a: float = 4.0,
        b: float = 2.0,
        period: float = 20.0,
        laps: float = 1.0,
        center=(0.0, 0.0),
        start=(0.0, 0.0),
        yaw_mode: str = "fixed",
        z_amp: float = 0.0,
        climb_time: float = 4.0,
        settle_time: float = 2.0,
        blend_time: float = 3.0,
        hover_time: float = 2.0,
        land_time: float = 6.0,
        land_speed: float = 0.3,
        ground_time: float = 2.0,
    ) -> "Mission":
        """Take off at ``start``, fly ``laps`` figure-8s about ``center``, land.

        The transitions into and out of the figure-8 are quintic blends, so
        position, velocity and acceleration are continuous throughout.
        ``land_speed`` is the descent rate at touchdown.
        """
        if height <= 0:
            raise ValueError("height must be positive.")
        s_xy = np.asarray(start, dtype=float)
        c_xy = np.asarray(center, dtype=float)
        ground = np.array([s_xy[0], s_xy[1], 0.0])
        hover_a = np.array([s_xy[0], s_xy[1], height])
        hover_b = np.array([c_xy[0], c_xy[1], height])
        zero = np.zeros(3)

        eight = Figure8(
            center=hover_b, a=a, b=b, period=period, laps=laps, z_amp=z_amp,
            yaw_mode=yaw_mode, name="figure8",
        )
        entry = eight(0.0)
        exit_ = eight(eight.duration)
        touchdown = RefPoint(
            np.array([c_xy[0], c_xy[1], 0.0]),
            np.array([0.0, 0.0, -abs(land_speed)]),
            zero,
            0.0,
        )
        hover_pt = RefPoint(hover_b, zero, zero, entry.yaw)
        segments = [
            Quintic(RefPoint(ground, zero, zero, 0.0), RefPoint(hover_a, zero, zero, 0.0),
                    climb_time, name="takeoff"),
            Hold(hover_a, 0.0, settle_time, name="hover"),
            Quintic(RefPoint(hover_a, zero, zero, 0.0), entry, blend_time, name="transition"),
            eight,
            Quintic(exit_, hover_pt, blend_time, name="transition"),
            Hold(hover_b, entry.yaw, hover_time, name="hover"),
            Quintic(hover_pt, RefPoint(touchdown.pos, touchdown.vel, zero, entry.yaw),
                    land_time, name="land"),
            Hold(touchdown.pos, entry.yaw, ground_time, name="landed"),
        ]
        return cls(segments)
