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
# Run all                                                                     #
# -------------------------------------------------------