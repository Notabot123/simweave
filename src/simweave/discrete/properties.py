"""Service-time distributions and per-entity sim properties.

The previous ``simProperites`` class conflated the distribution configuration
with a ``dt`` attribute that gets overwritten whenever ``next_service_time()``
is called. Here we split the two: ``ServiceTime`` knows how to *draw*; an
:class:`EntityProperties` record carries the distribution plus any static
attributes the modeller wants to track.

Every distribution accepts an optional ``rng`` (``numpy.random.Generator``)
so that Monte Carlo runs can be seeded independently without touching the
legacy global ``np.random`` state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


RNG = np.random.Generator
Distribution = Callable[[RNG], float]


def exponential(mean: float) -> Distribution:
    """Exponential inter-arrival / service time."""
    if mean <= 0:
        raise ValueError("Exponential mean must be positive.")

    def draw(rng: RNG) -> float:
        return float(rng.exponential(mean))

    def ppf(u):
        return -mean * np.log1p(-np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12))

    def cdf(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < 0, 0.0, -np.expm1(-np.maximum(x, 0.0) / mean))

    # Quantile/CDF attributes let these helpers act as copula marginals.
    draw.mean, draw.ppf, draw.cdf = float(mean), ppf, cdf
    return draw


def uniform(low: float, high: float) -> Distribution:
    if high <= low:
        raise ValueError("Uniform requires high > low.")

    def draw(rng: RNG) -> float:
        return float(rng.uniform(low, high))

    def ppf(u):
        return low + (high - low) * np.asarray(u, dtype=float)

    def cdf(x):
        return np.clip((np.asarray(x, dtype=float) - low) / (high - low), 0.0, 1.0)

    draw.mean, draw.ppf, draw.cdf = 0.5 * (low + high), ppf, cdf
    return draw


def normal(mean: float, std: float, *, clip_nonnegative: bool = True) -> Distribution:
    """Normal distribution; by default clips at 0 since negative service times are nonsense."""
    if std <= 0:
        raise ValueError("normal std must be positive.")

    def draw(rng: RNG) -> float:
        x = float(rng.normal(mean, std))
        return max(0.0, x) if clip_nonnegative else x

    def ppf(u):
        from simweave.stats.special import clip_unit, ndtri

        x = mean + std * ndtri(clip_unit(u))
        return np.maximum(x, 0.0) if clip_nonnegative else x

    def cdf(x):
        from simweave.stats.special import ndtr

        x = np.asarray(x, dtype=float)
        p = ndtr((x - mean) / std)
        return np.where(x < 0, 0.0, p) if clip_nonnegative else p

    draw.ppf, draw.cdf = ppf, cdf
    return draw


def deterministic(value: float) -> Distribution:
    if value < 0:
        raise ValueError("deterministic value cannot be negative.")

    def draw(rng: RNG) -> float:
        return float(value)

    def ppf(u):
        return np.full(np.shape(u), float(value)) if np.ndim(u) else float(value)

    def cdf(x):
        return (np.asarray(x, dtype=float) >= value).astype(float)

    draw.mean, draw.ppf, draw.cdf = float(value), ppf, cdf
    return draw


@dataclass
class EntityProperties:
    """Per-entity sim properties. Pass the same instance to many entities.

    Attributes
    ----------
    entity_type:
        Free-form tag, useful for stratified summaries.
    service_time:
        Distribution drawn whenever a Service pulls this entity into a work
        channel.
    balk_on_length:
        If the waiting queue has more than this many items on arrival, the
        entity refuses to join. ``None`` disables balking.
    renege_after:
        If the entity's ``current_wait_time`` exceeds this threshold, it
        leaves the queue. ``None`` disables reneging.
    extras:
        Arbitrary additional fields the modeller wants to track.
    """

    entity_type: str = "default"
    service_time: Distribution = field(default_factory=lambda: exponential(1.0))
    balk_on_length: int | None = None
    renege_after: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_draw(
        cls,
        draw: dict[str, float],
        service_key: str | None = "service_time",
        **kwargs: Any,
    ) -> "EntityProperties":
        """Build per-entity properties from a joint draw (see ``JointSampler``).

        ``draw[service_key]`` becomes a fixed service time; every other drawn
        value is stored in ``extras``. Extra keyword arguments (``entity_type``,
        ``balk_on_length``, ``renege_after``, ``extras``) pass through, with
        ``extras`` merged over the drawn values.
        """
        extras = {k: v for k, v in draw.items() if k != service_key}
        extras.update(kwargs.pop("extras", {}))
        if service_key is not None:
            if service_key not in draw:
                raise KeyError(f"draw has no {service_key!r}; keys: {sorted(draw)}")
            kwargs["service_time"] = deterministic(draw[service_key])
        return cls(extras=extras, **kwargs)

    def draw_service_time(self, rng: RNG | None = None) -> float:
        if rng is None:
            rng = _default_rng()
        return float(self.service_time(rng))


# A shared default generator so users can omit ``rng`` in tests / quick demos.
_DEFAULT_RNG = np.random.default_rng()


def _default_rng() -> RNG:
    return _DEFAULT_RNG


def set_default_seed(seed: int) -> None:
    """Replace the module-level generator with a seeded one."""
    global _DEFAULT_RNG
    _DEFAULT_RNG = np.random.default_rng(seed)
