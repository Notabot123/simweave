"""Wind models: callables ``wind(t, position) -> ndarray(3)`` in the world frame.

All models are deterministic functions of time (and position), so they are safe
to evaluate several times per RK4 step. Random models draw their whole
realisation up front from a seeded generator.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Wind(Protocol):
    """Velocity of the air (m/s, world frame) at time ``t`` and ``position``."""

    def __call__(self, t: float, position) -> np.ndarray: ...


def _vec3(v, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have three components.")
    return arr


class ConstantWind:
    """Steady wind ``velocity = (wx, wy, wz)``."""

    def __init__(self, velocity=(0.0, 0.0, 0.0)) -> None:
        self.velocity = _vec3(velocity, "velocity")

    def __call__(self, t: float, position=None) -> np.ndarray:
        return self.velocity


class GustWind:
    """A steady wind plus one smooth "1 - cos" gust.

    The gust adds ``amplitude * 0.5 * (1 - cos(2 pi (t - start) / duration))``
    for ``start <= t <= start + duration``.
    """

    def __init__(self, amplitude, start: float, duration: float, base=(0.0, 0.0, 0.0)) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive.")
        self.amplitude = _vec3(amplitude, "amplitude")
        self.start = float(start)
        self.duration = float(duration)
        self.base = _vec3(base, "base")

    def __call__(self, t: float, position=None) -> np.ndarray:
        if self.start <= t <= self.start + self.duration:
            shape = 0.5 * (1.0 - math.cos(2.0 * math.pi * (t - self.start) / self.duration))
            return self.base + shape * self.amplitude
        return self.base


class TurbulentWind:
    """Steady wind plus first-order Gauss-Markov turbulence.

    Each axis is an Ornstein-Uhlenbeck process with standard deviation
    ``sigma`` (scalar or per-axis) and correlation time ``correlation_time``.
    The realisation is sampled once on a grid of spacing ``dt`` covering
    ``duration`` seconds and linearly interpolated, so repeated evaluations at
    the same time agree. After ``duration`` the last value is held.
    """

    def __init__(
        self,
        sigma,
        correlation_time: float,
        duration: float,
        mean=(0.0, 0.0, 0.0),
        dt: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> None:
        if correlation_time <= 0 or duration <= 0 or dt <= 0:
            raise ValueError("correlation_time, duration and dt must be positive.")
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (3,))
        if np.any(sigma < 0):
            raise ValueError("sigma must be non-negative.")
        rng = rng if rng is not None else np.random.default_rng()
        self.mean = _vec3(mean, "mean")
        self.dt = float(dt)
        n = int(math.ceil(duration / dt)) + 1
        a = math.exp(-dt / correlation_time)
        noise = rng.standard_normal((n, 3))
        gust = np.zeros((n, 3))
        gust[0] = sigma * noise[0]
        b = math.sqrt(1.0 - a * a)
        for k in range(1, n):
            gust[k] = a * gust[k - 1] + b * sigma * noise[k]
        self._gust = gust

    def __call__(self, t: float, position=None) -> np.ndarray:
        x = max(t, 0.0) / self.dt
        i = int(x)
        if i >= self._gust.shape[0] - 1:
            return self.mean + self._gust[-1]
        frac = x - i
        return self.mean + (1.0 - frac) * self._gust[i] + frac * self._gust[i + 1]
