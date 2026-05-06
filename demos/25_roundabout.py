"""Roundabout vs signalised junction comparison.

Scenario
--------
The same 4-arm junction from demo 24 is modelled in two configurations:

A. **Signalised crossroads** -- a two-phase signal (45 s NS / 30 s EW).
B. **Roundabout** -- a 4-arm UK-style (LEFT handedness) roundabout with
   capacity for 6 circulating vehicles and a 6-second transit time.

Both scenarios share identical arrival rates and road geometry.  Running
them under the same simulation clock lets us compare:

* Total throughput (vehicles cleared)
* Delay rate (vehicles that had to queue)
* Peak queue length

Three plots are produced:

1. **Roundabout occupancy** -- circulating and queued vehicle counts over time.
2. **Side-by-side throughput bar chart** -- junction vs roundabout.
3. **Approach queue comparison** -- queue lengths for both configurations.

Run::

    python demos/25_roundabout.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import simweave as sw
from simweave.roads import (
    Handedness,
    Intersection,
    IntersectionQueueRecorder,
    Road,
    RoadNetwork,
    RoadOccupancyRecorder,
    Roundabout,
    SignalPhase,
    TrafficSignal,
    VehicleArrivalProcess,
)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

SIM_SECONDS = 3_600
DT = 1.0
ROAD_LENGTH = 200.0
SPEED_LIMIT = 13.9  # m/s  ≈ 50 km/h

ARRIVAL_RATES = {
    "north": 0.20,
    "south": 0.15,
    "east":  0.25,
    "west":  0.18,
}

GREEN_NS = 45.0
GREEN_EW = 30.0

RB_MAX_CIRCULATING = 6
RB_TRANSIT_TIME    = 6.0


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def build_signalised(rng_seed: int = 0) -> dict:
    """Build and run the signalised-junction scenario."""
    rng = np.random.default_rng(rng_seed)

    # Approach roads
    roads = {
        arm: Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name=f"{arm}_approach")
        for arm in ARRIVAL_RATES
    }
    exits = {
        arm: Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name=f"{arm}_exit")
        for arm in ARRIVAL_RATES
    }

    phase_ns = SignalPhase(
        green_roads=[roads["north"], roads["south"]],
        duration=GREEN_NS, name="NS_green",
    )
    phase_ew = SignalPhase(
        green_roads=[roads["east"], roads["west"]],
        duration=GREEN_EW, name="EW_green",
    )
    signal = TrafficSignal([phase_ns, phase_ew], name="signal")

    junction = Intersection(
        signal=signal,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
        name="junction",
    )
    for arm, road in roads.items():
        junction.add_approach(road)
        road.outlet = junction
    for exit_road in exits.values():
        junction.add_exit(exit_road, weight=1.0)

    arrivals = {
        arm: VehicleArrivalProcess(
            interarrival=lambda r, rate=rate: r.exponential(1.0 / rate),
            road=roads[arm],
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        )
        for arm, rate in ARRIVAL_RATES.items()
    }

    occ_rec = RoadOccupancyRecorder(list(roads.values()), name="sig_occ")
    q_rec   = IntersectionQueueRecorder(junction, name="sig_q")

    net = RoadNetwork(name="signalised")
    net.add_signal(signal)
    net.add_intersection(junction)
    for r in list(roads.values()) + list(exits.values()):
        net.add_road(r)
    for ap in arrivals.values():
        net.add_arrival_process(ap)
    net.add_recorder(occ_rec)
    net.add_recorder(q_rec)

    env = sw.SimEnvironment(dt=DT, end=SIM_SECONDS)
    net.register_all(env)
    env.run(until=SIM_SECONDS)

    return {
        "junction": junction,
        "arrivals": arrivals,
        "occ_rec": occ_rec,
        "q_rec": q_rec,
        "total_generated": sum(ap.generated for ap in arrivals.values()),
    }


def build_roundabout(rng_seed: int = 99) -> dict:
    """Build and run the roundabout scenario."""
    rng = np.random.default_rng(rng_seed)

    # Approach roads
    roads = {
        arm: Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name=f"{arm}_approach")
        for arm in ARRIVAL_RATES
    }
    exits = {
        arm: Road(ROAD_LENGTH, SPEED_LIMIT, lanes=2, name=f"{arm}_exit")
        for arm in ARRIVAL_RATES
    }

    rb = Roundabout(
        max_circulating=RB_MAX_CIRCULATING,
        transit_time=RB_TRANSIT_TIME,
        handedness=Handedness.LEFT,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
        name="roundabout",
    )
    for arm, road in roads.items():
        rb.add_entry(road)
        road.outlet = rb
    for exit_road in exits.values():
        rb.add_exit(exit_road, weight=1.0)

    arrivals = {
        arm: VehicleArrivalProcess(
            interarrival=lambda r, rate=rate: r.exponential(1.0 / rate),
            road=roads[arm],
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        )
        for arm, rate in ARRIVAL_RATES.items()
    }

    occ_rec = RoadOccupancyRecorder(list(roads.values()), name="rb_occ")

    # Roundabout-specific recorder using IntersectionQueueRecorder duck-typing.
    # We build a thin adapter so the recorder works with Roundabout's interface.
    class _RbAdapter:
        """Adapts Roundabout to IntersectionQueueRecorder's expected interface."""
        def __init__(self, roundabout: Roundabout, name: str = "rb_q") -> None:
            self._rb = roundabout
            self.name = name

        @property
        def total_queued(self) -> int:
            return self._rb.total_queued

        @property
        def queue_lengths(self) -> dict:
            return self._rb.queue_lengths

    # We snapshot manually using RoadOccupancyRecorder for occupancy; for
    # roundabout queue we define a simple custom recorder.
    class _RbQueueRecorder(sw.Entity):
        def __init__(self, roundabout: Roundabout, name: str = "rb_queue_rec") -> None:
            super().__init__(name=name)
            self._rb = roundabout
            self.times: list[float] = []
            self.circulating: list[int] = []
            self.queued: list[int] = []

        def on_register(self, env: sw.SimEnvironment) -> None:
            super().on_register(env)
            self._snap(env.clock.t)

        def tick(self, dt: float, env: sw.SimEnvironment) -> None:
            super().tick(dt, env)
            self._snap(env.clock.t + dt)

        def _snap(self, t: float) -> None:
            self.times.append(t)
            self.circulating.append(self._rb.circulating)
            self.queued.append(self._rb.total_queued)

    rb_rec = _RbQueueRecorder(rb)

    net = RoadNetwork(name="roundabout_net")
    net.add_roundabout(rb)
    for r in list(roads.values()) + list(exits.values()):
        net.add_road(r)
    for ap in arrivals.values():
        net.add_arrival_process(ap)
    net.add_recorder(occ_rec)
    net.add_recorder(rb_rec)

    env = sw.SimEnvironment(dt=DT, end=SIM_SECONDS)
    net.register_all(env)
    env.run(until=SIM_SECONDS)

    return {
        "roundabout": rb,
        "arrivals": arrivals,
        "occ_rec": occ_rec,
        "rb_rec": rb_rec,
        "total_generated": sum(ap.generated for ap in arrivals.values()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Demo 25 -- Roundabout vs Signalised Junction Comparison")
    print(f"  Simulation : {SIM_SECONDS} s ({SIM_SECONDS / 60:.0f} min)")
    print(f"  Arrival rates : {ARRIVAL_RATES}")
    print("=" * 60)

    print("\n[1/2] Running signalised junction ...")
    sig_res = build_signalised(rng_seed=42)

    print("[2/2] Running roundabout ...")
    rb_res = build_roundabout(rng_seed=42)

    # ------------------------------------------------------------------ #
    # Comparison summary                                                   #
    # ------------------------------------------------------------------ #
    sig_jn  = sig_res["junction"]
    rb_obj  = rb_res["roundabout"]

    sig_gen  = sig_res["total_generated"]
    rb_gen   = rb_res["total_generated"]

    sig_thru = sig_jn.total_vehicles
    rb_thru  = rb_obj.total_exited

    sig_dly  = sig_jn.total_delayed
    rb_dly   = rb_obj.total_delayed

    sig_drp  = sig_jn.total_dropped
    rb_drp   = rb_obj.total_dropped

    print("\n" + "-" * 55)
    print(f"{'Metric':<35} {'Signal':>9} {'Roundabout':>9}")
    print("-" * 55)
    print(f"{'Vehicles generated':<35} {sig_gen:>9,} {rb_gen:>9,}")
    print(f"{'Vehicles cleared':<35} {sig_thru:>9,} {rb_thru:>9,}")
    print(f"{'Vehicles delayed at red/capacity':<35} {sig_dly:>9,} {rb_dly:>9,}")
    print(f"{'Vehicles dropped (queue full)':<35} {sig_drp:>9,} {rb_drp:>9,}")

    sig_delay_pct = 100.0 * sig_dly / sig_gen if sig_gen else 0.0
    rb_delay_pct  = 100.0 * rb_dly  / rb_gen  if rb_gen  else 0.0
    print(f"{'Delay rate (%)':<35} {sig_delay_pct:>8.1f}% {rb_delay_pct:>8.1f}%")
    print("-" * 55)

    try:
        import plotly  # noqa: F401
    except ImportError:
        print("\nplotly not installed — skipping plots.")
        return

    print("\nGenerating plots ...")
    import plotly.graph_objects as go
    import numpy as _np

    # 1. Roundabout occupancy (circulating + queued)
    rb_rec = rb_res["rb_rec"]
    times_rb = _np.asarray(rb_rec.times)

    fig_rb = go.Figure()
    fig_rb.add_trace(go.Scatter(
        x=times_rb, y=rb_rec.circulating,
        mode="lines", name="circulating",
        line={"color": "steelblue"},
    ))
    fig_rb.add_trace(go.Scatter(
        x=times_rb, y=rb_rec.queued,
        mode="lines", name="queued at arms",
        line={"color": "tomato", "dash": "dot"},
    ))
    fig_rb.update_layout(
        title=f"Roundabout Occupancy ({RB_MAX_CIRCULATING} max circulating)",
        xaxis_title="time (s)",
        yaxis_title="vehicles",
        legend={"orientation": "h", "y": -0.2},
    )
    fig_rb.write_html("roundabout_occupancy.html")
    print("  Saved: roundabout_occupancy.html")

    # 2. Side-by-side throughput bar
    fig_bar = go.Figure(data=[
        go.Bar(name="Signal", x=["Cleared", "Delayed", "Dropped"],
               y=[sig_thru, sig_dly, sig_drp], marker_color="steelblue"),
        go.Bar(name="Roundabout", x=["Cleared", "Delayed", "Dropped"],
               y=[rb_thru, rb_dly, rb_drp], marker_color="darkorange"),
    ])
    fig_bar.update_layout(
        barmode="group",
        title="Signal vs Roundabout — Throughput Comparison",
        yaxis_title="vehicles",
        legend={"orientation": "h", "y": -0.2},
    )
    fig_bar.write_html("roundabout_comparison_bar.html")
    print("  Saved: roundabout_comparison_bar.html")

    # 3. Approach road occupancy for roundabout
    fig_occ = sw.plot_road_occupancy(
        rb_res["occ_rec"],
        title="Roundabout Approach Road Occupancy",
    )
    fig_occ.write_html("roundabout_approach_occupancy.html")
    print("  Saved: roundabout_approach_occupancy.html")

    print("\nDone.")


if __name__ == "__main__":
    main()
