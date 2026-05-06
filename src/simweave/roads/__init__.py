"""``simweave.roads`` -- road network simulation primitives.

Classes
-------
Vehicle
    A lightweight entity that travels through roads and intersections.
VehicleArrivalProcess
    Generates vehicles at a given inter-arrival rate and enters them on a road.
Road
    Free-flow road segment (event-driven conveyor-belt travel model).
DualCarriageway
    Two opposing :class:`Road` instances sharing the same corridor.
SignalPhase
    One phase in a traffic-signal cycle (green roads + duration).
TrafficSignal
    Fixed-time signal controller that cycles through :class:`SignalPhase` objects.
Intersection
    Signalised or give-way intersection with per-approach queues.
Handedness
    Enum for driving-side convention (LEFT = UK/AU, RIGHT = EU/US).
Roundabout
    Priority-to-circulating roundabout model.
RoadNetwork
    Container for all network components with a :meth:`register_all` helper.
RoadOccupancyRecorder
    Snapshots in-transit vehicle counts for a set of roads each tick.
IntersectionQueueRecorder
    Snapshots approach-queue lengths at a single intersection each tick.
"""

from simweave.roads.vehicle import Vehicle, VehicleArrivalProcess
from simweave.roads.road import Road, DualCarriageway
from simweave.roads.signal import SignalPhase, TrafficSignal
from simweave.roads.intersection import Intersection
from simweave.roads.roundabout import Handedness, Roundabout
from simweave.roads.network import RoadNetwork
from simweave.roads.recorder import RoadOccupancyRecorder, IntersectionQueueRecorder

__all__ = [
    "Vehicle",
    "VehicleArrivalProcess",
    "Road",
    "DualCarriageway",
    "SignalPhase",
    "TrafficSignal",
    "Intersection",
    "Handedness",
    "Roundabout",
    "RoadNetwork",
    "RoadOccupancyRecorder",
    "IntersectionQueueRecorder",
]
