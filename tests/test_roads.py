"""Unit tests for simweave.roads.

Tests cover:
- Road: travel-time calculation, event-driven delivery, in-transit tracking.
- DualCarriageway: two constituent roads.
- Vehicle / VehicleArrivalProcess: generation and counting.
- TrafficSignal / SignalPhase: phase cycling, road_is_green.
- Intersection: immediate dispatch (green), queuing (red), tick release.
- Roundabout: admission, capacity limit, transit time, exit routing.
- RoadNetwork: register_all ordering.
- Recorders: RoadOccupancyRecorder, IntersectionQueueRecorder.
- Top-level exports in simweave namespace.
"""

from __future__ import annotations

import sys
import os

# Ensure the source tree is preferred over any installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import simweave as sw
from simweave.roads import (
    DualCarriageway,
    Handedness,
    Intersection,
    IntersectionQueueRecorder,
    Road,
    RoadNetwork,
    RoadOccupancyRecorder,
    Roundabout,
    SignalPhase,
    TrafficSignal,
    Vehicle,
    VehicleArrivalProcess,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(dt: float = 1.0, end: float = 100.0) -> sw.SimEnvironment:
    return sw.SimEnvironment(dt=dt, end=end)


def make_rng(seed: int = 0):
    import numpy as np
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------


class TestVehicle:
    def test_defaults(self):
        v = Vehicle()
        assert v.speed is None
        assert v.roads_traversed == 0
        assert v.total_travel_time == 0.0

    def test_speed_override(self):
        v = Vehicle(speed=10.0)
        assert v.speed == 10.0

    def test_name_auto_generated(self):
        v = Vehicle()
        assert "Vehicle" in v.name or v.name  # just has a name

    def test_name_explicit(self):
        v = Vehicle(name="car_01")
        assert v.name == "car_01"


# ---------------------------------------------------------------------------
# Road
# ---------------------------------------------------------------------------


class TestRoad:
    def test_travel_time_no_vehicle(self):
        r = Road(length=100.0, speed_limit=10.0)
        assert r.travel_time() == 10.0

    def test_travel_time_vehicle_slower(self):
        r = Road(length=100.0, speed_limit=10.0)
        v = Vehicle(speed=5.0)
        assert r.travel_time(v) == 20.0  # vehicle is slower

    def test_travel_time_vehicle_faster_capped(self):
        r = Road(length=100.0, speed_limit=10.0)
        v = Vehicle(speed=50.0)
        assert r.travel_time(v) == 10.0  # capped at speed limit

    def test_enter_increments_in_transit(self):
        r = Road(length=100.0, speed_limit=10.0)
        env = make_env(dt=1.0, end=200.0)
        env.register(r)
        v = Vehicle()
        r.enter(v, env)
        assert r.in_transit == 1
        assert r.total_entered == 1

    def test_delivery_decrements_in_transit(self):
        r = Road(length=1.0, speed_limit=1.0)  # 1 s travel
        env = make_env(dt=1.0, end=10.0)
        env.register(r)
        v = Vehicle()
        r.enter(v, env)
        env.run(until=5.0)
        assert r.in_transit == 0
        assert r.total_exited == 1
        assert v.roads_traversed == 1
        assert v.total_travel_time == 1.0

    def test_delivery_calls_outlet(self):
        delivered = []
        r = Road(length=1.0, speed_limit=1.0)

        class FakeOutlet:
            def arrive(self, vehicle, from_road, env):
                delivered.append(vehicle)

        r.outlet = FakeOutlet()
        env = make_env(dt=1.0, end=10.0)
        env.register(r)
        v = Vehicle()
        r.enter(v, env)
        env.run(until=5.0)
        assert v in delivered

    def test_multiple_vehicles_in_transit(self):
        r = Road(length=10.0, speed_limit=10.0)  # 1 s each
        env = make_env(dt=1.0, end=20.0)
        env.register(r)
        for _ in range(5):
            r.enter(Vehicle(), env)
        assert r.in_transit == 5
        env.run(until=5.0)
        assert r.in_transit == 0
        assert r.total_exited == 5

    def test_invalid_length_raises(self):
        import pytest
        with pytest.raises(ValueError):
            Road(length=0.0, speed_limit=10.0)

    def test_invalid_speed_raises(self):
        import pytest
        with pytest.raises(ValueError):
            Road(length=100.0, speed_limit=0.0)

    def test_has_work_false_when_empty(self):
        r = Road(length=100.0, speed_limit=10.0)
        env = make_env()
        assert not r.has_work(env)

    def test_has_work_true_when_in_transit(self):
        r = Road(length=100.0, speed_limit=10.0)
        env = make_env(dt=1.0, end=200.0)
        env.register(r)
        r.enter(Vehicle(), env)
        assert r.has_work(env)


# ---------------------------------------------------------------------------
# DualCarriageway
# ---------------------------------------------------------------------------


class TestDualCarriageway:
    def test_has_forward_and_backward(self):
        dc = DualCarriageway(length=500.0, speed_limit=20.0, name="high_st")
        assert isinstance(dc.forward, Road)
        assert isinstance(dc.backward, Road)

    def test_roads_property(self):
        dc = DualCarriageway(length=500.0, speed_limit=20.0)
        assert len(dc.roads) == 2

    def test_name_suffix(self):
        dc = DualCarriageway(length=100.0, speed_limit=10.0, name="main_rd")
        assert "forward" in dc.forward.name
        assert "backward" in dc.backward.name

    def test_independent_in_transit(self):
        dc = DualCarriageway(length=1.0, speed_limit=1.0)
        env = make_env(dt=1.0, end=20.0)
        env.register(dc.forward)
        env.register(dc.backward)
        dc.forward.enter(Vehicle(), env)
        dc.forward.enter(Vehicle(), env)
        dc.backward.enter(Vehicle(), env)
        assert dc.forward.in_transit == 2
        assert dc.backward.in_transit == 1


# ---------------------------------------------------------------------------
# TrafficSignal / SignalPhase
# ---------------------------------------------------------------------------


class TestTrafficSignal:
    def test_initial_phase(self):
        r1 = Road(100.0, 10.0, name="north")
        r2 = Road(100.0, 10.0, name="east")
        p1 = SignalPhase(green_roads=[r1], duration=30.0, name="NS")
        p2 = SignalPhase(green_roads=[r2], duration=20.0, name="EW")
        sig = TrafficSignal([p1, p2])
        assert sig.current_phase is p1
        assert sig.road_is_green(r1)
        assert not sig.road_is_green(r2)

    def test_phase_advances_after_duration(self):
        r1 = Road(100.0, 10.0, name="n")
        r2 = Road(100.0, 10.0, name="e")
        p1 = SignalPhase(green_roads=[r1], duration=5.0)
        p2 = SignalPhase(green_roads=[r2], duration=5.0)
        sig = TrafficSignal([p1, p2])
        env = make_env(dt=1.0, end=20.0)
        env.register(sig)
        env.run(until=6.0)
        # After 5 ticks of dt=1 the phase should have advanced.
        assert sig.current_phase is p2
        assert sig.road_is_green(r2)
        assert not sig.road_is_green(r1)

    def test_phase_wraps_around(self):
        r1 = Road(100.0, 10.0, name="n")
        p1 = SignalPhase(green_roads=[r1], duration=3.0)
        sig = TrafficSignal([p1])
        env = make_env(dt=1.0, end=20.0)
        env.register(sig)
        env.run(until=10.0)
        assert sig.cycle_count > 0
        assert sig.current_phase is p1  # single phase always wraps to itself

    def test_empty_phases_raises(self):
        import pytest
        with pytest.raises(ValueError):
            TrafficSignal([])

    def test_has_work(self):
        sig = TrafficSignal([SignalPhase(duration=10.0)])
        env = make_env()
        assert sig.has_work(env)

    def test_phase_index_property(self):
        p1 = SignalPhase(duration=2.0)
        p2 = SignalPhase(duration=2.0)
        sig = TrafficSignal([p1, p2])
        env = make_env(dt=1.0, end=10.0)
        env.register(sig)
        assert sig.phase_index == 0
        env.run(until=3.0)
        assert sig.phase_index == 1


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------


class TestIntersection:
    def _make_approach_and_exit(self):
        approach = Road(length=10.0, speed_limit=10.0, name="approach")
        exit_road = Road(length=10.0, speed_limit=10.0, name="exit")
        ix = Intersection(name="junction")
        ix.add_approach(approach)
        ix.add_exit(exit_road, weight=1.0)
        approach.outlet = ix
        return approach, exit_road, ix

    def test_immediate_dispatch_no_signal(self):
        """Without a signal all vehicles pass straight through."""
        approach, exit_road, ix = self._make_approach_and_exit()
        env = make_env(dt=1.0, end=50.0)
        env.register(ix)
        env.register(approach)
        env.register(exit_road)

        v = Vehicle()
        approach.enter(v, env)
        env.run(until=5.0)
        # Vehicle should have completed the approach road and been dispatched.
        assert ix.total_vehicles >= 1

    def test_queues_on_red(self):
        r_ns = Road(10.0, 10.0, name="ns")
        r_ew = Road(10.0, 10.0, name="ew")
        exit_road = Road(10.0, 10.0, name="exit")

        p_ns = SignalPhase(green_roads=[r_ns], duration=100.0)
        p_ew = SignalPhase(green_roads=[r_ew], duration=100.0)
        sig = TrafficSignal([p_ns, p_ew])  # starts on NS green

        ix = Intersection(signal=sig, name="lights")
        ix.add_approach(r_ew)
        ix.add_exit(exit_road, weight=1.0)
        r_ew.outlet = ix

        env = make_env(dt=1.0, end=50.0)
        env.register(sig)
        env.register(ix)
        env.register(r_ew)
        env.register(exit_road)

        v = Vehicle()
        r_ew.enter(v, env)
        env.run(until=5.0)
        # EW is red during NS phase; vehicle should be in the queue.
        assert ix.total_queued >= 1 or ix.total_delayed >= 1

    def test_releases_on_green(self):
        r_approach = Road(1.0, 1.0, name="approach")  # travel time = 1 s
        exit_road = Road(10.0, 10.0, name="exit")

        p_green = SignalPhase(green_roads=[r_approach], duration=5.0)
        p_red = SignalPhase(green_roads=[], duration=100.0)
        # Start on green so vehicles are released, then switch to red.
        sig = TrafficSignal([p_green, p_red])

        ix = Intersection(signal=sig, name="lights")
        ix.add_approach(r_approach)
        ix.add_exit(exit_road, weight=1.0)
        r_approach.outlet = ix

        env = make_env(dt=1.0, end=20.0)
        env.register(sig)
        env.register(ix)
        env.register(r_approach)
        env.register(exit_road)

        # Vehicle arrives during green phase.
        r_approach.enter(Vehicle(), env)
        env.run(until=10.0)
        assert ix.total_vehicles >= 1

    def test_has_work_reflects_queued(self):
        ix = Intersection(name="ix")
        env = make_env()
        assert not ix.has_work(env)

    def test_drop_when_approach_full(self):
        ix = Intersection(approach_capacity=1, name="tiny")
        exit_road = Road(10.0, 10.0, name="exit")
        r_approach = Road(1.0, 1.0, name="approach")
        # Use a signal that is permanently red so vehicles queue.
        p_red = SignalPhase(green_roads=[], duration=1000.0)
        sig = TrafficSignal([p_red])
        ix.signal = sig
        ix.add_approach(r_approach)
        ix.add_exit(exit_road, weight=1.0)
        r_approach.outlet = ix

        env = make_env(dt=1.0, end=50.0)
        env.register(sig)
        env.register(ix)
        env.register(r_approach)

        for _ in range(5):
            r_approach.enter(Vehicle(), env)
        env.run(until=5.0)
        assert ix.total_dropped >= 1

    def test_queue_lengths_property(self):
        ix = Intersection(name="ix")
        approach = Road(10.0, 10.0, name="road_a")
        ix.add_approach(approach)
        ql = ix.queue_lengths
        assert "road_a" in ql


# ---------------------------------------------------------------------------
# Roundabout
# ---------------------------------------------------------------------------


class TestRoundabout:
    def test_admits_vehicle_when_capacity_available(self):
        rb = Roundabout(max_circulating=4, transit_time=5.0, name="rb")
        exit_road = Road(10.0, 10.0, name="exit")
        entry_road = Road(1.0, 1.0, name="entry")
        rb.add_entry(entry_road)
        rb.add_exit(exit_road, weight=1.0)
        entry_road.outlet = rb

        env = make_env(dt=1.0, end=30.0)
        env.register(rb)
        env.register(entry_road)
        env.register(exit_road)

        entry_road.enter(Vehicle(), env)
        env.run(until=3.0)
        # After entry road delivers vehicle (1 s), roundabout should have it.
        assert rb.total_entered >= 1

    def test_queues_when_at_capacity(self):
        rb = Roundabout(max_circulating=1, transit_time=100.0, name="rb")
        exit_road = Road(10.0, 10.0, name="exit")
        entry_road = Road(1.0, 1.0, name="entry")
        rb.add_entry(entry_road)
        rb.add_exit(exit_road, weight=1.0)
        entry_road.outlet = rb

        env = make_env(dt=1.0, end=20.0)
        env.register(rb)
        env.register(entry_road)
        env.register(exit_road)

        # Enter two vehicles; second should queue.
        for _ in range(2):
            entry_road.enter(Vehicle(), env)
        env.run(until=5.0)
        assert rb.total_entered == 1
        assert rb.total_delayed == 1

    def test_exits_after_transit_time(self):
        rb = Roundabout(max_circulating=4, transit_time=5.0, name="rb")
        exit_road = Road(10.0, 10.0, name="exit")
        entry_road = Road(1.0, 1.0, name="entry")
        rb.add_entry(entry_road)
        rb.add_exit(exit_road, weight=1.0)
        entry_road.outlet = rb

        env = make_env(dt=1.0, end=30.0)
        env.register(rb)
        env.register(entry_road)
        env.register(exit_road)

        entry_road.enter(Vehicle(), env)
        env.run(until=20.0)
        # 1s to traverse entry road + 5s transit = exited by t=7
        assert rb.total_exited >= 1
        assert rb.circulating == 0

    def test_handedness_stored(self):
        rb = Roundabout(handedness=Handedness.RIGHT)
        assert rb.handedness == Handedness.RIGHT

    def test_handedness_default_left(self):
        rb = Roundabout()
        assert rb.handedness == Handedness.LEFT

    def test_has_work_false_when_idle(self):
        rb = Roundabout()
        env = make_env()
        assert not rb.has_work(env)

    def test_drops_when_arm_full(self):
        rb = Roundabout(max_circulating=1, transit_time=1000.0,
                        approach_capacity=1, name="tiny_rb")
        exit_road = Road(10.0, 10.0, name="exit")
        entry_road = Road(1.0, 1.0, name="entry")
        rb.add_entry(entry_road)
        rb.add_exit(exit_road, weight=1.0)
        entry_road.outlet = rb

        env = make_env(dt=1.0, end=20.0)
        env.register(rb)
        env.register(entry_road)
        env.register(exit_road)

        for _ in range(5):
            entry_road.enter(Vehicle(), env)
        env.run(until=10.0)
        assert rb.total_dropped >= 1

    def test_add_entry_auto_on_arrive(self):
        """An unknown road is registered automatically on first arrive()."""
        rb = Roundabout(max_circulating=4, transit_time=2.0)
        exit_road = Road(10.0, 10.0, name="exit")
        rb.add_exit(exit_road)
        entry_road = Road(1.0, 1.0, name="auto_entry")
        entry_road.outlet = rb

        env = make_env(dt=1.0, end=20.0)
        env.register(rb)
        env.register(entry_road)
        env.register(exit_road)
        entry_road.enter(Vehicle(), env)
        env.run(until=5.0)
        assert rb.total_entered >= 1


# ---------------------------------------------------------------------------
# VehicleArrivalProcess
# ---------------------------------------------------------------------------


class TestVehicleArrivalProcess:
    def test_generates_vehicles(self):
        import numpy as np
        road = Road(length=1000.0, speed_limit=10.0, name="main")
        rng = np.random.default_rng(42)
        proc = VehicleArrivalProcess(
            interarrival=lambda r: r.exponential(1.0),
            road=road,
            rng=rng,
        )
        env = make_env(dt=1.0, end=100.0)
        env.register(road)
        env.register(proc)
        env.run(until=100.0)
        assert proc.generated > 0

    def test_has_work(self):
        road = Road(100.0, 10.0)
        proc = VehicleArrivalProcess(
            interarrival=lambda r: 1.0,
            road=road,
        )
        env = make_env()
        assert proc.has_work(env)

    def test_speed_assigned_to_vehicles(self):
        received_speeds: list = []

        class SpeedCapture:
            def arrive(self, vehicle, from_road, env):
                received_speeds.append(vehicle.speed)

        road = Road(length=5.0, speed_limit=5.0, name="road")
        road.outlet = SpeedCapture()
        import numpy as np
        proc = VehicleArrivalProcess(
            interarrival=lambda r: 1.0,
            road=road,
            rng=np.random.default_rng(0),
            speed=3.0,
        )
        env = make_env(dt=1.0, end=10.0)
        env.register(road)
        env.register(proc)
        env.run(until=10.0)
        assert all(s == 3.0 for s in received_speeds)


# ---------------------------------------------------------------------------
# RoadNetwork
# ---------------------------------------------------------------------------


class TestRoadNetwork:
    def test_register_all_in_order(self):
        """Signals must precede intersections in the registered process list."""
        r_approach = Road(10.0, 10.0, name="approach")
        r_exit = Road(10.0, 10.0, name="exit")
        sig = TrafficSignal(
            [SignalPhase(green_roads=[r_approach], duration=30.0)]
        )
        ix = Intersection(signal=sig, name="junction")
        ix.add_approach(r_approach)
        ix.add_exit(r_exit, weight=1.0)
        r_approach.outlet = ix

        net = RoadNetwork(name="test_net")
        net.add_signal(sig)
        net.add_intersection(ix)
        net.add_road(r_approach)
        net.add_road(r_exit)

        env = make_env(dt=1.0, end=100.0)
        net.register_all(env)

        procs = list(env.processes)
        sig_idx = procs.index(sig)
        ix_idx = procs.index(ix)
        assert sig_idx < ix_idx, "Signal must be registered before intersection"

    def test_add_dual_carriageway_registers_both_roads(self):
        dc = DualCarriageway(100.0, 10.0, name="dc")
        net = RoadNetwork()
        net.add_dual_carriageway(dc)
        assert dc.forward in net.all_roads
        assert dc.backward in net.all_roads

    def test_add_roundabout(self):
        rb = Roundabout(name="rb")
        net = RoadNetwork()
        net.add_roundabout(rb)
        assert rb in net.all_roundabouts

    def test_register_all_runs_without_error(self):
        net = RoadNetwork()
        r = Road(100.0, 10.0, name="r")
        net.add_road(r)
        env = make_env(dt=1.0, end=10.0)
        net.register_all(env)
        env.run()


# ---------------------------------------------------------------------------
# Recorders
# ---------------------------------------------------------------------------


class TestRoadOccupancyRecorder:
    def test_records_occupancy(self):
        road = Road(length=10.0, speed_limit=10.0, name="rd")
        rec = RoadOccupancyRecorder([road], name="rec")
        env = make_env(dt=1.0, end=20.0)
        env.register(road)
        env.register(rec)

        road.enter(Vehicle(), env)
        road.enter(Vehicle(), env)
        env.run(until=5.0)

        assert len(rec.times) > 0
        assert len(rec.occupancy) == len(rec.times)
        assert rec.road_names == ["rd"]

    def test_occupancy_drops_after_delivery(self):
        road = Road(length=1.0, speed_limit=1.0, name="short")
        rec = RoadOccupancyRecorder([road])
        env = make_env(dt=1.0, end=20.0)
        env.register(road)
        env.register(rec)

        road.enter(Vehicle(), env)
        env.run(until=15.0)
        # Last occupancy sample should be 0 after vehicle exits.
        assert rec.occupancy[-1][0] == 0


class TestIntersectionQueueRecorder:
    def test_records_queue_lengths(self):
        r_approach = Road(1.0, 1.0, name="approach")
        r_exit = Road(10.0, 10.0, name="exit")
        # Permanently red signal so vehicles queue.
        p_red = SignalPhase(green_roads=[], duration=1000.0)
        sig = TrafficSignal([p_red])
        ix = Intersection(signal=sig, name="ix")
        ix.add_approach(r_approach)
        ix.add_exit(r_exit, weight=1.0)
        r_approach.outlet = ix

        rec = IntersectionQueueRecorder(ix)

        env = make_env(dt=1.0, end=20.0)
        env.register(sig)
        env.register(ix)
        env.register(r_approach)
        env.register(r_exit)
        env.register(rec)

        for _ in range(3):
            r_approach.enter(Vehicle(), env)
        env.run(until=10.0)

        assert len(rec.times) > 0
        assert any(q > 0 for q in rec.total_queued)


# ---------------------------------------------------------------------------
# Top-level namespace exports
# ---------------------------------------------------------------------------


class TestTopLevelExports:
    def test_all_road_names_exported(self):
        expected = [
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
            "plot_road_occupancy",
            "plot_intersection_queues",
        ]
        for name in expected:
            assert hasattr(sw, name), f"sw.{name} not found"

    def test_handedness_enum_values(self):
        assert sw.Handedness.LEFT.value == "left"
        assert sw.Handedness.RIGHT.value == "right"
