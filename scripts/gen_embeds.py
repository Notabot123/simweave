"""Generate Plotly embed HTML files for the SimWeave docs site.

Runs at ``mkdocs build`` time via the ``mkdocs-gen-files`` plugin. Each
``mkdocs_gen_files.open(...)`` call writes a virtual file into the
rendered site under the path given. Iframes in the docs pages reference
these files via relative paths, e.g. ``embeds/<name>.html``.

The figures here mirror ``demos/14_viz_tour.py`` so the docs always show
exactly what the demo produces. Keep each scenario short (sub-second) so
``mkdocs build`` stays fast.
"""

from __future__ import annotations

import numpy as np
import mkdocs_gen_files

import simweave as sw


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _emit(fig, name: str) -> None:
    """Write a Plotly figure to ``site/embeds/{name}.html``."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    with mkdocs_gen_files.open(f"embeds/{name}.html", "w") as f:
        f.write(html)


# --------------------------------------------------------------------------- #
# Continuous: state trajectories + phase portrait                             #
# --------------------------------------------------------------------------- #


def build_continuous() -> None:
    msd = sw.MassSpringDamper(mass=1.0, damping=0.4, stiffness=4.0)
    res = sw.simulate(
        msd, t_span=(0.0, 12.0), dt=0.01, x0=np.array([1.0, 0.0])
    )
    _emit(
        sw.plot_state_trajectories(res, title="Damped MSD trajectories"),
        "msd_states",
    )
    _emit(
        sw.plot_phase_portrait(res, title="Damped MSD phase portrait"),
        "msd_phase",
    )


# --------------------------------------------------------------------------- #
# Discrete: queue length + service utilisation                                #
# --------------------------------------------------------------------------- #


def build_discrete() -> None:
    rng = np.random.default_rng(7)
    sink = sw.Queue(maxlen=10_000, name="sink")
    svc = sw.Service(
        capacity=2,
        buffer_size=50,
        next_q=sink,
        default_service_time=1.0,
        rng=rng,
        name="svc",
    )
    gen = sw.ArrivalGenerator(
        interarrival=lambda r: r.exponential(0.6),
        factory=lambda env: sw.Entity(),
        target=svc,
        rng=rng,
        name="gen",
    )
    qrec = sw.QueueLengthRecorder(svc, name="svc_qlen")
    urec = sw.ServiceUtilisationRecorder(svc, name="svc_util")

    env = sw.SimEnvironment(dt=0.1, end=200.0)
    for proc in (gen, svc, sink, qrec, urec):
        env.register(proc)
    env.run()

    _emit(sw.plot_queue_length(qrec, title="M/M/2 buffer length"), "queue_length")
    _emit(
        sw.plot_service_utilisation(urec, title="Service utilisation (2 channels)"),
        "service_util",
    )


# --------------------------------------------------------------------------- #
# Supply chain: warehouse stock                                               #
# --------------------------------------------------------------------------- #


def build_supplychain() -> None:
    inv = sw.InventoryItems(
        part_names=["widget", "gizmo", "sprocket"],
        unit_cost=[1.0, 2.5, 0.7],
        stock_level=[20.0, 10.0, 50.0],
        batchsize=[20.0, 10.0, 50.0],
        reorder_points=[5.0, 3.0, 15.0],
        repairable_prc=[0.0, 0.0, 0.0],
        repair_times=[1.0, 1.0, 1.0],
        newbuy_leadtimes=[3.0, 5.0, 2.0],
    )
    wh = sw.Warehouse(inventory=inv, name="store")
    rec = sw.WarehouseStockRecorder(wh, name="store_stock")

    env = sw.SimEnvironment(dt=1.0, end=80.0)
    env.register(wh)
    env.register(rec)
    rng = np.random.default_rng(0)
    for t in range(80):
        env.run(until=float(t + 1))
        # Stochastic per-tick demand.
        wh.decrement_vector(rng.poisson([0.7, 0.4, 1.5]).astype(float))

    _emit(
        sw.plot_warehouse_stock(rec, title="Warehouse stock vs reorder points"),
        "warehouse_stock",
    )


# --------------------------------------------------------------------------- #
# Monte Carlo: percentile fan                                                 #
# --------------------------------------------------------------------------- #


def build_monte_carlo() -> None:
    rng = np.random.default_rng(42)
    n_runs, n_t = 200, 80
    increments = rng.normal(0.05, 0.4, size=(n_runs, n_t))
    trajectories = np.cumsum(increments, axis=1)
    times = np.arange(n_t, dtype=float)
    _emit(
        sw.plot_mc_fan((times, trajectories), title="Random-walk-with-drift fan"),
        "mc_fan",
    )


# --------------------------------------------------------------------------- #
# Agents: A* path on a grid                                                   #
# --------------------------------------------------------------------------- #


def build_agents() -> None:
    g = sw.grid_graph(8, 12, diagonal=True)
    agent = sw.Agent(
        graph=g,
        start_node=(0, 0),
        tasks=[(7, 11), (3, 4), (0, 11)],
        speed=2.0,
        heuristic=sw.manhattan,
        name="rover",
    )
    env = sw.SimEnvironment(dt=0.5, end=50.0)
    env.register(agent)
    env.run(until=50.0)
    _emit(
        sw.plot_agent_path(agent, graph=g, title="A* agent over an 8x12 grid"),
        "agent_path",
    )


# --------------------------------------------------------------------------- #
# Full Car Model: Show Acceleration, Suspension Travel & Tyre Forces          #
# --------------------------------------------------------------------------- #


def build_vehicle() -> None:
    car = sw.FullCarModel(
        sprung_mass=1200,
        pitch_inertia=1500,
        roll_inertia=800,
        unsprung_mass=40,
        k_s=20000,
        c_s=1500,
        k_t=200000,
        a=1.2,
        b=1.3,
        track_width=1.6,
    )

    res = sw.simulate(
        car,
        t_span=(0.0, 5.0),
        dt=0.005,
        inputs=lambda t: np.array([0.01, 0.01, 0.0, 0.0]) if t > 1 else np.zeros(4),
    )

    fig = sw.plot_vehicle_metrics(
        res,
        title="Full car response (bump input)",
        model=car
    )

    _emit(fig, "full_car_metrics")


# --------------------------------------------------------------------------- #
# Reliability: fleet availability stacked area                                #
# --------------------------------------------------------------------------- #


def build_fleet_availability() -> None:
    from simweave.reliability import (
        Fleet,
        FleetAvailabilityRecorder,
        RepairCentre,
        ReliableEntity,
        SubsystemSpec,
    )

    rng = np.random.default_rng(42)

    specs = [
        SubsystemSpec(
            "engine", 1 / 120, 0,
            consumable=False, beyond_economic_repair_prc=0.10,
            repair_time=5.0, unit_cost=8_000.0, repair_cost=2_500.0,
        ),
        SubsystemSpec(
            "transmission", 1 / 90, 1,
            consumable=False, beyond_economic_repair_prc=0.05,
            repair_time=3.0, unit_cost=5_000.0, repair_cost=1_200.0,
        ),
        SubsystemSpec(
            "tyres", 1 / 45, 2,
            consumable=True, repair_time=0.5, unit_cost=400.0,
        ),
    ]

    inv = sw.InventoryItems(
        part_names=["engine", "transmission", "tyres"],
        unit_cost=[8_000.0, 5_000.0, 400.0],
        stock_level=[3.0, 4.0, 10.0],
        batchsize=[2.0, 2.0, 4.0],
        reorder_points=[1.0, 1.0, 2.0],
        repairable_prc=[0.90, 0.95, 0.0],
        repair_times=[5.0, 3.0, 0.0],
        newbuy_leadtimes=[14.0, 10.0, 3.0],
    )
    wh = sw.Warehouse(inventory=inv, name="parts_depot")

    pool = sw.ResourcePool(maxlen=3, name="technicians")
    for i in range(3):
        pool.deposit(sw.Resource(name=f"tech_{i}"))
    rc = RepairCentre(capacity=2, resources=pool)

    vehicles = [
        ReliableEntity(
            specs, wh, rc, name=f"taxi_{i:02d}",
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        )
        for i in range(8)
    ]
    fleet = Fleet(vehicles, name="taxi_fleet")
    recorder = FleetAvailabilityRecorder(fleet)

    env = sw.SimEnvironment(dt=1.0, end=365.0)
    env.register(wh)
    env.register(rc)
    for v in vehicles:
        env.register(v)
    env.register(recorder)
    env.run(until=365.0)

    # With calendar time axis (1 tick = 1 day from 1 Jan 2027)
    tax = sw.SimTimeAxis(start="2027-01-01", tick_unit="days")

    _emit(
        sw.plot_fleet_availability(
            recorder,
            time_axis=tax,
            title="Taxi Fleet Availability -- 2027",
        ),
        "fleet_availability",
    )


# --------------------------------------------------------------------------- #
# Roads: signalised intersection + roundabout occupancy                       #
# --------------------------------------------------------------------------- #


def build_roads() -> None:
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

    rng = np.random.default_rng(7)

    # --- Signalised intersection occupancy --------------------------------- #
    road_ns = Road(200.0, 13.9, lanes=2, name="NS_approach")
    road_ew = Road(200.0, 13.9, lanes=2, name="EW_approach")
    exit_ns = Road(200.0, 13.9, name="NS_exit")
    exit_ew = Road(200.0, 13.9, name="EW_exit")

    signal = TrafficSignal([
        SignalPhase(green_roads=[road_ns], duration=45.0, name="NS_green"),
        SignalPhase(green_roads=[road_ew], duration=30.0, name="EW_green"),
    ])
    junction = Intersection(
        signal=signal,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
        name="junction",
    )
    for road in (road_ns, road_ew):
        junction.add_approach(road)
        road.outlet = junction
    junction.add_exit(exit_ns, weight=1.0)
    junction.add_exit(exit_ew, weight=1.0)

    arrivals_ns = VehicleArrivalProcess(
        interarrival=lambda r: r.exponential(5.0),
        road=road_ns,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    arrivals_ew = VehicleArrivalProcess(
        interarrival=lambda r: r.exponential(4.0),
        road=road_ew,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )
    occ_rec = RoadOccupancyRecorder([road_ns, road_ew], name="ix_occ")
    q_rec   = IntersectionQueueRecorder(junction)

    net = RoadNetwork(name="intersection")
    net.add_signal(signal)
    net.add_intersection(junction)
    for r in (road_ns, road_ew, exit_ns, exit_ew):
        net.add_road(r)
    net.add_arrival_process(arrivals_ns)
    net.add_arrival_process(arrivals_ew)
    net.add_recorder(occ_rec)
    net.add_recorder(q_rec)

    env = sw.SimEnvironment(dt=1.0, end=3600.0)
    net.register_all(env)
    env.run()

    _emit(
        sw.plot_road_occupancy(
            occ_rec,
            title="Signalised Crossroads — Approach Road Occupancy",
        ),
        "intersection_occupancy",
    )
    _emit(
        sw.plot_intersection_queues(
            q_rec,
            title="Signalised Crossroads — Queue Lengths",
        ),
        "intersection_queues",
    )

    # --- Roundabout occupancy ---------------------------------------------- #
    rb_rng = np.random.default_rng(13)

    roads_rb = {
        arm: Road(200.0, 13.9, lanes=2, name=f"{arm}_approach")
        for arm in ("north", "south", "east", "west")
    }
    exits_rb = {
        arm: Road(200.0, 13.9, name=f"{arm}_exit")
        for arm in ("north", "south", "east", "west")
    }
    rb = Roundabout(
        max_circulating=6,
        transit_time=6.0,
        handedness=Handedness.LEFT,
        rng=np.random.default_rng(rb_rng.integers(0, 2**32)),
        name="roundabout",
    )
    RATES = {"north": 0.20, "south": 0.15, "east": 0.25, "west": 0.18}
    for arm, road in roads_rb.items():
        rb.add_entry(road)
        road.outlet = rb
    for exit_road in exits_rb.values():
        rb.add_exit(exit_road, weight=1.0)

    arrivals_rb = {
        arm: VehicleArrivalProcess(
            interarrival=lambda r, rate=rate: r.exponential(1.0 / rate),
            road=roads_rb[arm],
            rng=np.random.default_rng(rb_rng.integers(0, 2**32)),
        )
        for arm, rate in RATES.items()
    }

    occ_rb = RoadOccupancyRecorder(list(roads_rb.values()), name="rb_occ")

    class _RbRec(sw.Entity):
        def __init__(self, roundabout):
            super().__init__(name="rb_rec")
            self._rb = roundabout
            self.times: list = []
            self.circulating: list = []
            self.queued: list = []

        def on_register(self, env):
            super().on_register(env)
            self._snap(env.clock.t)

        def tick(self, dt, env):
            super().tick(dt, env)
            self._snap(env.clock.t + dt)

        def _snap(self, t):
            self.times.append(t)
            self.circulating.append(self._rb.circulating)
            self.queued.append(self._rb.total_queued)

    rb_rec = _RbRec(rb)

    net_rb = RoadNetwork(name="roundabout_net")
    net_rb.add_roundabout(rb)
    for r in list(roads_rb.values()) + list(exits_rb.values()):
        net_rb.add_road(r)
    for ap in arrivals_rb.values():
        net_rb.add_arrival_process(ap)
    net_rb.add_recorder(occ_rb)
    net_rb.add_recorder(rb_rec)

    env_rb = sw.SimEnvironment(dt=1.0, end=3600.0)
    net_rb.register_all(env_rb)
    env_rb.run()

    import plotly.graph_objects as go
    times_rb = np.asarray(rb_rec.times)
    fig_rb = go.Figure()
    fig_rb.add_trace(go.Scatter(
        x=times_rb, y=rb_rec.circulating,
        mode="lines", name="circulating", line={"color": "steelblue"},
    ))
    fig_rb.add_trace(go.Scatter(
        x=times_rb, y=rb_rec.queued,
        mode="lines", name="queued at arms",
        line={"color": "tomato", "dash": "dot"},
    ))
    fig_rb.update_layout(
        title="Roundabout — Circulating and Queued Vehicles",
        xaxis_title="time (s)",
        yaxis_title="vehicles",
        legend={"orientation": "h", "y": -0.2},
    )
    _emit(fig_rb, "roundabout_occupancy")


# --------------------------------------------------------------------------- #
# Run all                                                                     #
# --------------------------------------------------------------------------- #


def main() -> None:
    build_continuous()
    build_discrete()
    build_supplychain()
    build_monte_carlo()
    build_agents()
    build_vehicle()
    build_fleet_availability()
    build_roads()


# mkdocs-gen-files imports this module at build time and runs whatever
# executes at top level, so we invoke main() unconditionally here.
main()
