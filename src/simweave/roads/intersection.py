"""Signalised and give-way road intersections.

An :class:`Intersection` has one or more approach roads (each with its own
FIFO queue) and one or more exit roads.  On every tick it dispatches waiting
vehicles at green approaches to exit roads chosen by weighted random
selection.

If no :class:`~simweave.roads.signal.TrafficSignal` is attached all
approaches are treated as permanently green — this models a give-way /
all-way-stop junction where the first-in-first-out rule is the only
constraint.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from simweave.core.entity import Entity

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment
    from simweave.roads.road import Road
    from simweave.roads.signal import TrafficSignal
    from simweave.roads.vehicle import Vehicle


class Intersection(Entity):
    """Signalised or give-way road intersection.

    Parameters
    ----------
    signal:
        Optional :class:`~simweave.roads.signal.TrafficSignal`.  ``None``
        means all approaches are always green (give-way / all-way stop).
    rng:
        Numpy random generator for exit-road selection.
    approach_capacity:
        Maximum vehicles per approach queue.  Further arrivals are dropped.
    name:
        Optional display name.

    Attributes
    ----------
    total_vehicles : int
        Cumulative vehicles dispatched through this intersection.
    total_delayed : int
        Cumulative vehicles that had to queue (arrived at a red approach).
    total_dropped : int
        Cumulative vehicles dropped because an approach queue was full.

    Notes
    -----
    **Registration order**: if a :class:`TrafficSignal` is attached, register
    it **before** this intersection so the signal's state is updated before
    approach queues are released each tick.

    **Throughput**: at most one vehicle per approach road per tick is
    dispatched.  Reduce ``dt`` for higher saturation flows.
    """

    def __init__(
        self,
        signal: "TrafficSignal | None" = None,
        rng: np.random.Generator | None = None,
        approach_capacity: int = 50,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.signal = signal
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )
        self.approach_capacity = approach_capacity

        # Keyed by road.id
        self._approach_queues: dict[int, deque["Vehicle"]] = {}
        self._approach_roads: dict[int, "Road"] = {}

        self._exit_roads: list["Road"] = []
        self._exit_weights: list[float] = []

        self.total_vehicles: int = 0
        self.total_delayed: int = 0
        self.total_dropped: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_approach(self, road: "Road") -> None:
        """Register an approach road.

        Called automatically on the first arrival from an unknown road, but
        you can also call this explicitly to pre-create the queue.
        """
        if road.id not in self._approach_queues:
            self._approach_queues[road.id] = deque(
                maxlen=self.approach_capacity
            )
            self._approach_roads[road.id] = road

    def add_exit(self, road: "Road", weight: float = 1.0) -> None:
        """Register an exit road with a routing weight.

        Weights are normalised to probabilities; absolute values do not matter,
        only relative proportions.
        """
        self._exit_roads.append(road)
        self._exit_weights.append(float(weight))

    # ------------------------------------------------------------------
    # Arrival callback — called by Road._deliver
    # ------------------------------------------------------------------

    def arrive(
        self, vehicle: "Vehicle", from_road: "Road", env: "SimEnvironment"
    ) -> None:
        """Receive a vehicle arriving from *from_road*."""
        if from_road.id not in self._approach_queues:
            self.add_approach(from_road)

        is_green = self.signal is None or self.signal.road_is_green(from_road)

        if is_green and self._exit_roads:
            # Dispatch immediately — no queuing delay.
            self._dispatch(vehicle, env)
        else:
            q = self._approach_queues[from_road.id]
            if len(q) < self.approach_capacity:
                q.append(vehicle)
                self.total_delayed += 1
            else:
                self.total_dropped += 1

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        super().tick(dt, env)
        if not self._exit_roads:
            return
        for road_id, q in self._approach_queues.items():
            if not q:
                continue
            approach_road = self._approach_roads[road_id]
            is_green = (
                self.signal is None
                or self.signal.road_is_green(approach_road)
            )
            if is_green:
                vehicle = q.popleft()
                self._dispatch(vehicle, env)

    def has_work(self, env: "SimEnvironment") -> bool:
        return any(len(q) > 0 for q in self._approach_queues.values())

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _dispatch(self, vehicle: "Vehicle", env: "SimEnvironment") -> None:
        """Route *vehicle* to one exit road by weighted random selection."""
        weights = np.asarray(self._exit_weights, dtype=float)
        weights /= weights.sum()
        idx = int(self.rng.choice(len(self._exit_roads), p=weights))
        self._exit_roads[idx].enter(vehicle, env)
        self.total_vehicles += 1

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def queue_lengths(self) -> dict[str, int]:
        """Current queue length per approach road name."""
        return {
            self._approach_roads[rid].name: len(q)
            for rid, q in self._approach_queues.items()
        }

    @property
    def total_queued(self) -> int:
        """Total vehicles currently waiting across all approaches."""
        return sum(len(q) for q in self._approach_queues.values())


__all__ = ["Intersection"]
