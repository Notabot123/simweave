"""Simulation clock.

simeng uses a fixed-step *atomic* clock as its time authority. Every process
registered with the :class:`~simeng.core.environment.SimEnvironment` is
invoked once per tick with the clock's current ``dt``.

Atomic time is a design choice that makes Monte Carlo aggregation trivial:
every replicate shares the same time grid, so summaries collapse to array
reductions rather than ragged joins. Where sparse events would otherwise waste
cycles, the environment can cooperate with the optional ``EventQueue`` to
fast-forward idle gaps (see :mod:`simeng.core.scheduler`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Clock:
    """Fixed-step simulation clock.

    Parameters
    ----------
    start:
        Initial simulation time.
    dt:
        Fixed timestep. Units are whatever the rest of the model uses
        (typically seconds, but can be hours, days, or dimensionless ticks).
    end:
        Optional end time. If provided, :meth:`is_finished` returns ``True``
        once ``t >= end``.
    """

    start: float = 0.0
    dt: float = 1.0
    end: float | None = None
    t: float = 0.0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("Clock.dt must be positive.")
        if self.end is not None and self.end <= self.start:
            raise ValueError("Clock.end must be strictly greater than start.")
        self.t = self.start

    def advance(self, dt: float | None = None) -> None:
        """Advance the clock by ``dt`` (defaults to ``self.dt``)."""
        step = self.dt if dt is None else dt
        if step < 0:
            raise ValueError("Cannot advance clock by a negative interval.")
        self.t += step

    def jump_to(self, t: float) -> None:
        """Jump the clock forward to absolute time ``t`` (no backward jumps)."""
        if t < self.t:
            raise ValueError("Cannot rewind the clock.")
        self.t = t

    def reset(self) -> None:
        self.t = self.start

    def is_finished(self) -> bool:
        return self.end is not None and self.t >= self.end
