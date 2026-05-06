"""Road network vehicles and arrival processes.

A :class:`Vehicle` is a lightweight :class:`~simweave.core.entity.Entity`
that travels through roads and intersections.  A
:class:`VehicleArrivalProcess` generates vehicles according to an
inter-arrival distribution and enters them onto a target road.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from simweave.core.entity import Entity

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment
    from simweave.roads.road import Road


class Vehicle(Entity):
    """A single road vehicle travelling through a road network.

    Parameters
    ----------
    speed:
        Optional own-speed override (m/s, or whatever unit your simulation
        uses).  If ``None``, the road's ``speed_limit`` governs travel time.
        A vehicle faster than the speed limit is still capped at the limit.
    name:
        Optional display name.  Auto-generated if omitted.

    Attributes
    ----------
    speed : float | None
    roads_traversed : int
        Number of road segments completed so far.
    total_travel_time : float
        Cumulative in-road travel time (does not include intersection wait).
    """

    def __init__(
        self,
        speed: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.speed: float | None = speed
        self.roads_traversed: int = 0
        self.total_travel_time: float = 0.0


class VehicleArrivalProcess(Entity):
    """Generates :class:`Vehicle` instances and enters them onto a road.

    Parameters
    ----------
    interarrival:
        Callable ``(rng) -> float`` returning the next inter-arrival gap.
        Use ``simweave.exponential(rate=λ)`` for a Poisson process.
    road:
        The :class:`~simweave.roads.road.Road` that receives new vehicles.
    rng:
        Numpy random generator.  Defaults to ``np.random.default_rng()``.
    speed:
        Fixed speed assigned to every generated vehicle.  ``None`` uses the
        road's speed limit.
    name:
        Optional display name.

    Attributes
    ----------
    generated : int
        Cumulative vehicles successfully placed on the road.
    rejected : int
        Vehicles that could not enter (should not occur for free-flow roads).
    """

    def __init__(
        self,
        interarrival: Callable[[np.random.Generator], float],
        road: "Road",
        rng: np.random.Generator | None = None,
        speed: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.interarrival = interarrival
        self.road = road
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )
        self.speed = speed
        self._countdown: float = float(self.interarrival(self.rng))
        self.generated: int = 0
        self.rejected: int = 0

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        super().tick(dt, env)
        remaining = dt
        while remaining >= self._countdown:
            remaining -= self._countdown
            v = Vehicle(speed=self.speed)
            if self.road.enter(v, env):
                self.generated += 1
            else:
                self.rejected += 1
            self._countdown = float(self.interarrival(self.rng))
        self._countdown -= remaining

    def has_work(self, env: "SimEnvironment") -> bool:
        return True


__all__ = ["Vehicle", "VehicleArrivalProcess"]
