"""Roundabout (rotary / traffic circle) model.

:class:`Roundabout` implements a *priority-to-circulating-traffic* model:
vehicles arriving at an arm wait until the roundabout has capacity before
entering.  Once admitted, they spend ``transit_time`` simulation-time units
circulating and are then routed to an exit road.

Handedness (``LEFT`` for UK / AU / JP, ``RIGHT`` for continental Europe /
North America) is recorded on the object and used in visualisation labels
but does not affect the mathematical model — the circulatory behaviour is
symmetric.
"""

from __future__ import annotations

import enum
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from simweave.core.entity import Entity

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment
    from simweave.roads.road import Road
    from simweave.roads.vehicle import Vehicle


class Handedness(enum.Enum):
    """Driving-side convention for a roundabout.

    ``LEFT``
        Traffic circulates **clockwise** (viewed from above).  Entering
        vehicles give way to traffic from the **right**.  Used in the UK,
        Australia, Japan, and other left-hand-traffic countries.

    ``RIGHT``
        Traffic circulates **anti-clockwise**.  Entering vehicles give way
        to traffic from the **left**.  Used in continental Europe, North
        America, and most of the rest of the world.
    """

    LEFT = "left"    # clockwise; UK / AU / JP
    RIGHT = "right"  # anti-clockwise; EU / US


class Roundabout(Entity):
    """Priority-to-circulating roundabout.

    Parameters
    ----------
    max_circulating:
        Maximum number of vehicles allowed in the roundabout simultaneously.
        Models the finite capacity of the circulatory carriageway.
    transit_time:
        Time a vehicle spends in the roundabout from entry to exit (same
        units as the simulation ``dt``).
    handedness:
        Driving-side convention — recorded for labelling / visualisation.
    rng:
        Numpy random generator for exit-road selection.
    approach_capacity:
        Maximum vehicles per arm queue before arrivals are dropped.
    name:
        Optional display name.

    Attributes
    ----------
    handedness : Handedness
    circulating : int
        Number of vehicles currently in the roundabout.
    total_entered : int
        Cumulative vehicles admitted to the roundabout.
    total_exited : int
        Cumulative vehicles that have left onto an exit road.
    total_delayed : int
        Cumulative vehicles that had to queue at an arm entry.
    total_dropped : int
        Cumulative vehicles dropped because an arm queue was full.
    """

    def __init__(
        self,
        max_circulating: int = 8,
        transit_time: float = 5.0,
        handedness: Handedness = Handedness.LEFT,
        rng: np.random.Generator | None = None,
        approach_capacity: int = 30,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.max_circulating = int(max_circulating)
        self.transit_time = float(transit_time)
        self.handedness = handedness
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )
        self.approach_capacity = approach_capacity

        # Keyed by entry road.id
        self._arm_queues: dict[int, deque["Vehicle"]] = {}
        self._arm_roads: dict[int, "Road"] = {}

        self._exit_roads: list["Road"] = []
        self._exit_weights: list[float] = []

        self.circulating: int = 0
        self.total_entered: int = 0
        self.total_exited: int = 0
        self.total_delayed: int = 0
        self.total_dropped: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_entry(self, road: "Road") -> None:
        """Register an entry arm road.

        Called automatically on first arrival from an unknown road, but can
        also be called explicitly to pre-create the arm queue.
        """
        if road.id not in self._arm_queues:
            self._arm_queues[road.id] = deque(maxlen=self.approach_capacity)
            self._arm_roads[road.id] = road

    def add_exit(self, road: "Road", weight: float = 1.0) -> None:
        """Register an exit road with a routing weight."""
        self._exit_roads.append(road)
        self._exit_weights.append(float(weight))

    # ------------------------------------------------------------------
    # Arrival callback — called by Road._deliver
    # ------------------------------------------------------------------

    def arrive(
        self, vehicle: "Vehicle", from_road: "Road", env: "SimEnvironment"
    ) -> None:
        """Receive a vehicle at the entry of arm *from_road*."""
        if from_road.id not in self._arm_queues:
            self.add_entry(from_road)

        if self.circulating < self.max_circulating:
            self._admit(vehicle, env)
        else:
            q = self._arm_queues[from_road.id]
            if len(q) < self.approach_capacity:
                q.append(vehicle)
                self.total_delayed += 1
            else:
                self.total_dropped += 1

    # ------------------------------------------------------------------
    # Tick — try to admit queued vehicles
    # ------------------------------------------------------------------

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        super().tick(dt, env)
        # Attempt to admit one vehicle per arm per tick when capacity allows.
        for road_id, q in self._arm_queues.items():
            if self.circulating >= self.max_circulating:
                break
            if q:
                vehicle = q.popleft()
                self._admit(vehicle, env)

    def has_work(self, env: "SimEnvironment") -> bool:
        return self.circulating > 0 or any(
            len(q) > 0 for q in self._arm_queues.values()
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _admit(self, vehicle: "Vehicle", env: "SimEnvironment") -> None:
        """Admit *vehicle* into the roundabout and schedule its exit."""
        self.circulating += 1
        self.total_entered += 1

        def _exit(
            _v: "Vehicle" = vehicle,
            _rb: "Roundabout" = self,
            _env: "SimEnvironment" = env,
        ) -> None:
            _rb.circulating -= 1
            _rb.total_exited += 1
            _rb._route_exit(_v, _env)

        env.schedule_after(self.transit_time, _exit)

    def _route_exit(self, vehicle: "Vehicle", env: "SimEnvironment") -> None:
        """Send *vehicle* to an exit road (weighted random choice)."""
        if not self._exit_roads:
            return  # vehicle leaves the system
        weights = np.asarray(self._exit_weights, dtype=float)
        weights /= weights.sum()
        idx = int(self.rng.choice(len(self._exit_roads), p=weights))
        self._exit_roads[idx].enter(vehicle, env)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def queue_lengths(self) -> dict[str, int]:
        """Current queue length per entry arm road name."""
        return {
            self._arm_roads[rid].name: len(q)
            for rid, q in self._arm_queues.items()
        }

    @property
    def total_queued(self) -> int:
        """Total vehicles currently waiting at all arms."""
        return sum(len(q) for q in self._arm_queues.values())


__all__ = ["Handedness", "Roundabout"]
