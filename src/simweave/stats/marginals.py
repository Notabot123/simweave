"""Marginal distributions with a quantile function.

A copula couples uniforms; a *marginal* turns each uniform into a value via
its inverse CDF ``ppf``. Every :class:`Marginal` here is also a callable
``(rng) -> float``, so it can be used anywhere the discrete-event helpers
expect a ``Distribution`` (``interarrival``, ``EntityProperties.service_time``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from simweave.stats.special import clip_unit, ndtr, ndtri


class Marginal(ABC):
    """Base class: subclasses implement ``ppf`` and ``cdf``."""

    mean: float = float("nan")

    @abstractmethod
    def ppf(self, u): ...

    @abstractmethod
    def cdf(self, x): ...

    def rvs(self, rng: np.random.Generator, size=None):
        """Independent draws (inverse-transform of uniforms)."""
        return self.ppf(rng.random(size))

    def __call__(self, rng: np.random.Generator) -> float:
        return float(self.ppf(rng.random()))


class Exponential(Marginal):
    """Exponential with the given **mean** (same convention as ``sw.exponential``)."""

    def __init__(self, mean: float) -> None:
        if mean <= 0:
            raise ValueError("Exponential mean must be positive.")
        self.mean = float(mean)

    def ppf(self, u):
        return -self.mean * np.log1p(-clip_unit(u))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x < 0, 0.0, -np.expm1(-np.maximum(x, 0.0) / self.mean))


class Uniform(Marginal):
    def __init__(self, low: float, high: float) -> None:
        if high <= low:
            raise ValueError("Uniform requires high > low.")
        self.low, self.high = float(low), float(high)
        self.mean = 0.5 * (self.low + self.high)

    def ppf(self, u):
        return self.low + (self.high - self.low) * np.asarray(u, dtype=float)

    def cdf(self, x):
        return np.clip((np.asarray(x, dtype=float) - self.low) / (self.high - self.low), 0.0, 1.0)


class Normal(Marginal):
    """Normal; clipped at zero by default, like ``sw.normal``."""

    def __init__(self, mean: float, std: float, *, clip_nonnegative: bool = True) -> None:
        if std <= 0:
            raise ValueError("Normal std must be positive.")
        self.mean, self.std = float(mean), float(std)
        self.clip_nonnegative = clip_nonnegative

    def ppf(self, u):
        x = self.mean + self.std * ndtri(clip_unit(u))
        return np.maximum(x, 0.0) if self.clip_nonnegative else x

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        p = ndtr((x - self.mean) / self.std)
        return np.where(x < 0, 0.0, p) if self.clip_nonnegative else p


class LogNormal(Marginal):
    """``exp(N(mu, sigma^2))``."""

    def __init__(self, mu: float, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError("LogNormal sigma must be positive.")
        self.mu, self.sigma = float(mu), float(sigma)
        self.mean = math.exp(self.mu + 0.5 * self.sigma**2)

    @classmethod
    def from_mean_cv(cls, mean: float, cv: float) -> "LogNormal":
        if mean <= 0 or cv <= 0:
            raise ValueError("mean and cv must be positive.")
        s2 = math.log1p(cv * cv)
        return cls(math.log(mean) - 0.5 * s2, math.sqrt(s2))

    def ppf(self, u):
        return np.exp(self.mu + self.sigma * ndtri(clip_unit(u)))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore"):
            z = (np.log(np.maximum(x, 1e-300)) - self.mu) / self.sigma
        return np.where(x <= 0, 0.0, ndtr(z))


class Weibull(Marginal):
    def __init__(self, shape: float, scale: float) -> None:
        if shape <= 0 or scale <= 0:
            raise ValueError("Weibull shape and scale must be positive.")
        self.shape, self.scale = float(shape), float(scale)
        self.mean = self.scale * math.gamma(1.0 + 1.0 / self.shape)

    def ppf(self, u):
        return self.scale * (-np.log1p(-clip_unit(u))) ** (1.0 / self.shape)

    def cdf(self, x):
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        return -np.expm1(-((x / self.scale) ** self.shape))


class Deterministic(Marginal):
    def __init__(self, value: float) -> None:
        if value < 0:
            raise ValueError("Deterministic value cannot be negative.")
        self.value = float(value)
        self.mean = self.value

    def ppf(self, u):
        return np.full(np.shape(u), self.value) if np.ndim(u) else self.value

    def cdf(self, x):
        return (np.asarray(x, dtype=float) >= self.value).astype(float)


class Empirical(Marginal):
    """Empirical marginal: linear-interpolated quantile function of a sample."""

    def __init__(self, data) -> None:
        d = np.sort(np.asarray(data, dtype=float).ravel())
        if d.size < 2:
            raise ValueError("Empirical needs at least two observations.")
        if not np.all(np.isfinite(d)):
            raise ValueError("Empirical data must be finite.")
        self._sorted = d
        self._grid = (np.arange(d.size) + 0.5) / d.size
        self.mean = float(d.mean())

    def ppf(self, u):
        return np.interp(np.asarray(u, dtype=float), self._grid, self._sorted)

    def cdf(self, x):
        return np.searchsorted(self._sorted, np.asarray(x, dtype=float), side="right") / self._sorted.size


class ScipyMarginal(Marginal):
    """Wrap a frozen ``scipy.stats`` distribution."""

    def __init__(self, frozen) -> None:
        self._d = frozen
        try:
            self.mean = float(frozen.mean())
        except Exception:  # pragma: no cover
            self.mean = float("nan")

    def ppf(self, u):
        return self._d.ppf(clip_unit(u))

    def cdf(self, x):
        return self._d.cdf(x)

    def rvs(self, rng, size=None):
        return self._d.rvs(size=size, random_state=rng)


class _DuckMarginal(Marginal):
    """Anything exposing ``ppf`` (e.g. the ``sw.exponential`` closures)."""

    def __init__(self, obj) -> None:
        self._obj = obj
        self.mean = float(getattr(obj, "mean", float("nan")))

    def ppf(self, u):
        return self._obj.ppf(u)

    def cdf(self, x):
        if not hasattr(self._obj, "cdf"):
            raise AttributeError("This marginal has no cdf.")
        return self._obj.cdf(x)


def as_marginal(obj) -> Marginal:
    """Coerce ``obj`` to a :class:`Marginal`.

    Accepts a :class:`Marginal`, a frozen ``scipy.stats`` distribution, a raw
    array of observations (-> :class:`Empirical`), or any object with a
    ``ppf`` attribute, including ``sw.exponential(...)`` and friends.
    """
    if isinstance(obj, Marginal):
        return obj
    if hasattr(obj, "ppf") and hasattr(obj, "dist"):
        return ScipyMarginal(obj)
    if hasattr(obj, "ppf"):
        return _DuckMarginal(obj)
    if isinstance(obj, (list, tuple, np.ndarray)):
        return Empirical(obj)
    raise TypeError(f"Cannot interpret {type(obj).__name__} as a marginal (no ppf).")
