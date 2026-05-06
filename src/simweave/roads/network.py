"""RoadNetwork -- container and bulk-registration helper.

:class:`RoadNetwork` keeps references to every component in a road-network
scenario and registers them with a :class:`~simweave.core.environment.SimEnvironment`
in the correct tick order:

1. :class:`~simweave.roads.signal.TrafficSignal` — must tick first so
   intersection logic sees up-to-date phase state.
2. :class:`~simweave.roads.intersection.Intersection` — dispatches queued
   vehicles on green transitions.
3. :class:`~simweave.roads.roundabout.Roundabout` — admits queued vehicles.
4. :class:`~simweave.roads.road.Road` — advances in-transit vehicles (no-op
   per tick; delivery is event-driven).
5. :class:`~simweave.roads.vehicle.VehicleArrivalProcess` — generates new
   vehicles at the end of each tick.
6. Recorders — snapshot state after all entities have ticked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment
    from simweave.roads.intersection import Intersection
    from simweave.roads.road import DualCarriageway, Road
    from simweave.roads.roundabout import Roundabout
    from simweave.roads.signal import TrafficSignal
    from simweave.roads.vehicle import VehicleArrivalProcess


class RoadNetwork:
    """Container for all road-network components in a scenario.

    Parameters
    ----------
    name:
        Optional label for this network.

    Example
    -------
    ::

        net = sw.RoadNetwork(name="town_centre")
        net.add_signal(signal)
        net.add_intersection(crossroads)
        net.add_road(road_north)
        net.add_arrival_process(arrivals_north)
        net.register_all(env)
    """

    def __init__(self, name: str = "road_network") -> None:
        self.name = name
        self._signals: list["TrafficSignal"] = []
        self._intersections: list["Intersection"] = []
        self._roundabouts: list["Roundabout"] = []
        self._roads: list["Road"] = []
        self._arrival_processes: list["VehicleArrivalProcess"] = []
        self._recorders: list = []

    # ------------------------------------------------------------------
    # Component registration
    # ------------------------------------------------------------------

    def add_signal(self, signal: "TrafficSignal") -> "TrafficSignal":
        """Add a :class:`~simweave.roads.signal.TrafficSignal`."""
        self._signals.append(signal)
        return signal

    def add_intersection(
        self, intersection: "Intersection"
    ) -> "Intersection":
        """Add an :class:`~simweave.roads.intersection.Intersection`."""
        self._intersections.append(intersection)
        return intersection

    def add_roundabout(self, roundabout: "Roundabout") -> "Roundabout":
        """Add a :class:`~simweave.roads.roundabout.Roundabout`."""
        self._roundabouts.append(roundabout)
        return roundabout

    def add_road(self, road: "Road") -> "Road":
        """Add a single :class:`~simweave.roads.road.Road`."""
        self._roads.append(road)
        return road

    def add_dual_carriageway(
        self, dc: "DualCarriageway"
    ) -> "DualCarriageway":
        """Register both constituent roads of a
        :class:`~simweave.roads.road.DualCarriageway`."""
        self._roads.append(dc.forward)
        self._roads.append(dc.backward)
        return dc

    def add_arrival_process(
        self, proc: "VehicleArrivalProcess"
    ) -> "VehicleArrivalProcess":
        """Add a :class:`~simweave.roads.vehicle.VehicleArrivalProcess`."""
        self._arrival_processes.append(proc)
        return proc

    def add_recorder(self, recorder: object) -> object:
        """Add a recorder (e.g.
        :class:`~simweave.roads.recorder.RoadOccupancyRecorder`)."""
        self._recorders.append(recorder)
        return recorder

    # ------------------------------------------------------------------
    # Bulk registration
    # ------------------------------------------------------------------

    def register_all(self, env: "SimEnvironment") -> None:
        """Register every component with *env* in the correct tick order.

        Order: signals → intersections → roundabouts → roads → arrival
        processes → recorders.
        """
        for s in self._signals:
            env.register(s)
        for inx in self._intersections:
            env.register(inx)
        for rb in self._roundabouts:
            env.register(rb)
        for r in self._roads:
            env.register(r)
        for ap in self._arrival_processes:
            env.register(ap)
        for rec in self._recorders:
            env.register(rec)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def all_roads(self) -> list["Road"]:
        return list(self._roads)

    @property
    def all_intersections(self) -> list["Intersection"]:
        return list(self._intersections)

    @property
    def all_roundabouts(self) -> list["Roundabout"]:
        return list(self._roundabouts)


__all__ = ["RoadNetwork"]
