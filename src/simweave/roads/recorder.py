"""Time-sampling recorders for road-network primitives.

:class:`RoadOccupancyRecorder` snapshots the number of vehicles in transit
on a set of roads each tick.

:class:`IntersectionQueueRecorder` snapshots total and per-approach queue
lengths at a single intersection each tick.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simweave.viz.recorders import _Recorder

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment
    from simweave.roads.intersection import Intersection
    from simweave.roads.road import Road


class RoadOccupancyRecorder(_Recorder):
    """Snapshot in-transit vehicle counts for a set of roads each tick.

    Attributes
    ----------
    times : list[float]
        Simulation times of each sample.
    occupancy : list[list[int]]
        ``(n_samples, n_roads)`` in-transit counts per road.
    road_names : list[str]
        Names of the roads being recorded (same column order as
        ``occupancy``).
    """

    def __init__(
        self,
        roads: "list[Road]",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "road_occupancy")
        self.roads = list(roads)
        self.road_names: list[str] = [r.name for r in self.roads]
        self.occupancy: list[list[int]] = []

    def _sample(self, env: "SimEnvironment", t: float) -> None:
        self.times.append(t)
        self.occupancy.append([r.in_transit for r in self.roads])


class IntersectionQueueRecorder(_Recorder):
    """Snapshot approach-queue lengths at one intersection each tick.

    Attributes
    ----------
    times : list[float]
        Simulation times.
    total_queued : list[int]
        Total vehicles waiting across all approaches.
    per_approach : list[dict[str, int]]
        Per-approach breakdown ``{road_name: queue_length}`` at each sample.
    """

    def __init__(
        self,
        intersection: "Intersection",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"queue({intersection.name})")
        self.intersection = intersection
        self.total_queued: list[int] = []
        self.per_approach: list[dict[str, int]] = []

    def _sample(self, env: "SimEnvironment", t: float) -> None:
        self.times.append(t)
        self.total_queued.append(self.intersection.total_queued)
        self.per_approach.append(dict(self.intersection.queue_lengths))


__all__ = ["IntersectionQueueRecorder", "RoadOccupancyRecorder"]
