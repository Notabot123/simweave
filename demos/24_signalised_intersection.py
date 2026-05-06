"""Signalised crossroads -- 4-way junction with a traffic signal.

Scenario
--------
A busy town-centre crossroads with two arterial roads crossing at right
angles (North/South and East/West).  A two-phase fixed-time traffic signal
allocates 45 s of green to NS traffic and 30 s to EW traffic.

Road geometry::

              North (200 m)
                  |
    West ----[JUNCTION]---- East (200 m)
                  |
              South (200 m)

Arrival rates (vehicles/second):
  North approach  : 0.20 veh/s   (Poisson)
  South approach  : 0.15 veh/s
  East approach   : 0.25 veh/s
  West approach   : 0.18 veh/s

Exit routing from the junction: traffic is distributed equally across the
three exits that are not the approach road (i.e. one-third each to left,
ahead, right).

Three plots are produced:

1. **Road occupancy** -- in-transit vehicle counts for each of the 4
   approach roads over the simulation.
2. **Intersection queue length** -- total vehicles waiting and per-approach
   breakdown.
3. **Signal phase timeline** -- which phase is active at each tick.

Run::

    python demos/24_signalised_intersection.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import simweave as sw
from simweave.roads import (
    Intersection,
    IntersectionQueueRecorder,
    Road,
    RoadNetwork,
    RoadOccupancyRecorder,
    SignalPhase,
    TrafficSignal,
    VehicleArrivalProcess,
)

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

SIM_SECONDS = 3_600       # 1 hour
DT = 1.0                  # 1-second ticks
ROAD_LENGTH = 200.0       # metres each approach
SPEED_LIMIT = 13.9        # m/s ≈ 50 km/h

GREEN_NS = 45.0           # seconds of NS green
GREEN_EW = 30.0           # seconds of EW green

# Arrival rates for each approach (vehicles / second)
RATE_NORTH = 0.20
RATE_SOUTH = 0.15
RATE_EAST  = 0.25
RATE_WEST  = 0.18


def main() -> None:
    print("=" * 60)
    print("Demo 24 -- Signalised Crossroads")
    print(f"  Simulation duration : {SIM_SECONDS} s ({SIM_SECONDS / 60:.0f} min)")
    print(f"  NS green : {GREEN_NS:.0f} s    EW green : {GREEN_EW:.0f} s")
    print(f"  Cycle time : {GREEN_NS + GREEN_EW:.0f} s")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # Road segments (approach + exit shared; all 4 arms serve both ways)  #
    # ------------------------------------------------------------------ #
    road_north = Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name="north_approach")
    road_south = Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name="south_approach")
    road_east  = Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name="east_approach")
    road_west  = Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name="west_approach")

    # Exit roads (vehicles leaving the junction onto the far side of each arm)
    exit_north = Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name="north_exit")
    exit_south = Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name="south_exit")
    exit_east  = Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name="east_exit")
    exit_west  = Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name="west_exit")

    # ------------------------------------------------------------------ #
    # Traffic signal                                                       #
    # ------------------------------------------------------------------ #
    phase_ns = SignalPhase(
        green_roads=[road_north, road_south],
        duration=GREEN_NS,
        name="NS_green",
    )
    phase_ew = SignalPhase(
        green_roads=[road_east, road_west],
        duration=GREEN_EW,
        name="EW_green",
    )
    signal = TrafficSignal([phase_ns, phase_ew], name="junction_signal")

    # ------------------------------------------------------------------ #
    # Intersection                                                         #
    # ------------------------------------------------------------------ #
    junction = Intersection(
        signal=signal,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
        name="crossroads",
    )
    # Approaches
    for road in (road_north, road_south, road_east, road_west):
        junction.add_approach(road)
        road.outlet = junction

    # Exits: each approach distributes roughly evenly across the 3 other arms.
    junction.add_exit(exit_north, weight=1.0)
    junction.add_exit(exit_south, weight=1.0)
    junction.add_exit(exit_east,  weight=1.0)
    junction.add_exit(exit_west,  weight=1.0)

    # ------------------------------------------------------------------ #
    # Arrival processes                                                    #
    # ------------------------------------------------------------------ #
    def poisson(rate: float, seed: int) -> VehicleArrivalProcess:
        return VehicleArrivalProcess(
            interarrival=lambda r: r.exponential(1.0 / rate),
            road={
                0.20: road_north,
                0.15: road_south,
                0.25: road_east,
                0.18: road_west,
            }[rate],
            rng=np.random.default_rng(seed),
        )

    arrivals_north = VehicleArrivalProcess(
        interarrival=lambda r: r.exponential(1.0 / RATE_NORTH),
        road=road_north,
        rng=np.random.default_rng(1),
    )
    arrivals_south = VehicleArrivalProcess(
        interarrival=lambda r: r.exponential(1.0 / RATE_SOUTH),
        road=road_south,
        rng=np.random.default_rng(2),
    )
    arrivals_east = VehicleArrivalProcess(
        interarrival=lambda r: r.exponential(1.0 / RATE_EAST),
        road=road_east,
        rng=np.random.default_rng(3),
    )
    arrivals_west = VehicleArrivalProcess(
        interarrival=lambda r: r.exponential(1.0 / RATE_WEST),
        road=road_west,
        rng=np.random.default_rng(4),
    )

    # ------------------------------------------------------------------ #
    # Recorders                                                            #
    # ------------------------------------------------------------------ #
    occ_rec = RoadOccupancyRecorder(
        [road_north, road_south, road_east, road_west],
        name="approach_occupancy",
    )
    q_rec = IntersectionQueueRecorder(junction, name="junction_queues")

    # ------------------------------------------------------------------ #
    # Assemble network and run                                             #
    # ------------------------------------------------------------------ #
    net = RoadNetwork(name="town_centre")
    net.add_signal(signal)
    net.add_intersection(junction)
    for r in (road_north, road_south, road_east, road_west,
              exit_north, exit_south, exit_east, exit_west):
        net.add_road(r)
    for ap in (arrivals_north, arrivals_south, arrivals_east, arrivals_west):
        net.add_arrival_process(ap)
    net.add_recorder(occ_rec)
    net.add_recorder(q_rec)

    env = sw.SimEnvironment(dt=DT, end=SIM_SECONDS)
    net.register_all(env)
    env.run(until=SIM_SECONDS)

    # ------------------------------------------------------------------ #
    # Summary statistics                                                   #
    # ------------------------------------------------------------------ #
    total_generated = sum(
        ap.generated
        for ap in (arrivals_north, arrivals_south, arrivals_east, arrivals_west)
    )
    print(f"\nTotal vehicles generated : {total_generated:,}")
    print(f"Total cleared junction   : {junction.total_vehicles:,}")
    print(f"Total delayed at red     : {junction.total_delayed:,}")
    print(f"Total dropped (q full)   : {junction.total_dropped:,}")

    pct_delayed = (
        100.0 * junction.total_delayed / total_generated
        if total_generated > 0
        else 0.0
    )
    print(f"Delay rate               : {pct_delayed:.1f}%")

    # Peak queue lengths per approach
    print("\nPeak queue lengths:")
    per_approach_arr = q_rec.per_approach
    for road_name in (junction.queue_lengths or occ_rec.road_names):
        peak = max((d.get(road_name, 0) for d in per_approach_arr), default=0)
        print(f"  {road_name:<25} : {peak} vehicles")

    signal_cycles = signal.cycle_count
    print(f"\nSignal cycles completed  : {signal_cycles}")

    try:
        import plotly  # noqa: F401
    except ImportError:
        print("\nplotly not installed — skipping plots.")
        return

    print("\nGenerating plots ...")

    # 1. Road occupancy
    fig_occ = sw.plot_road_occupancy(
        occ_rec,
        title="Signalised Crossroads — Approach Road Occupancy",
    )
    fig_occ.write_html("intersection_occupancy.html")
    print("  Saved: intersection_occupancy.html")

    # 2. Intersection queue lengths
    fig_q = sw.plot_intersection_queues(
        q_rec,
        title="Signalised Crossroads — Queue Lengths",
    )
    fig_q.write_html("intersection_queues.html")
    print("  Saved: intersection_queues.html")

    # 3. Phase timeline — show which phase is green at each recorded tick
    import plotly.graph_objects as go
    import numpy as _np

    times = _np.asarray(q_rec.times)
    # Reconstruct approximate phase from the cycle length
    cycle = GREEN_NS + GREEN_EW
    phase_flag = (times % cycle) < GREEN_NS  # True = NS green, False = EW green

    fig_phase = go.Figure()
    fig_phase.add_trace(
        go.Scatter(
            x=times,
            y=phase_flag.astype(float),
            mode="lines",
            name="NS phase (1=green)",
            line={"color": "green"},
        )
    )
    fig_phase.add_trace(
        go.Scatter(
            x=times,
            y=(~phase_flag).astype(float),
            mode="lines",
            name="EW phase (1=green)",
            line={"color": "orange"},
        )
    )
    fig_phase.update_layout(
        title="Signal Phase Timeline",
        xaxis_title="time (s)",
        yaxis_title="green (1) / red (0)",
        yaxis={"range": [-0.1, 1.3]},
    )
    fig_phase.write_html("intersection_phases.html")
    print("  Saved: intersection_phases.html")

    print("\nDone.")


if __name__ == "__main__":
    main()
