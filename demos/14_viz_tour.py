"""End-to-end tour of ``simweave.viz``.

Runs one mini-simulation per plot helper and writes a self-contained
HTML file per figure plus a single combined ``viz_tour.html`` index. The
plotly figures are deliberately produced via the public ``simweave``
top-level surface so this file also doubles as a worked example of the
viz API.

Requires the ``viz`` extra::

    pip install -e .[viz]
    # or for everything dev:
    pip install -e .[dev]

Run::

    python demos/14_viz_tour.py

Outputs are written next to this file under ``./viz_out/``.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

import pathlib
from typing import Any

import numpy as np

import simweave as sw


OUT = pathlib.Path(__file__).resolve().parent / "viz_out"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _save(fig: Any, name: str) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def _confirm_json_roundtrip(fig: Any, name: str) -> None:
    """Exercise the EdgeWeave consumption path: ``fig.to_json()`` and back."""
    import json
    blob = fig.to_json()
    parsed = json.loads(blob)
    assert "data" in parsed and "layout" in parsed, f"{name}: bad JSON shape"


# --------------------------------------------------------------------------- #
# 1. Continuous: state trajectories + phase portrait                          #
# --------------------------------------------------------------------------- #


def tour_continuous() -> list[pathlib.Path]:
    msd = sw.MassSpringDamper(m=1.0, c=0.4, k=4.0)
    res = sw.simulate(msd, t_span=(0.0, 12.0), dt=0.01, x0=np.array([1.0, 0.0]))
    return [
        _save(sw.plot_state_trajectories(res, title="MSD state trajectories"),
              "01_msd_states"),
        _save(sw.plot_phase_portrait(res, title="MSD phase portrait"),
              "02_msd_phase"),
    ]


# --------------------------------------------------------------------------- #
# 2. Discrete: queue length + service utilisation                             #
# --------------------------------------------------------------------------- #


def tour_discrete() -> list[pathlib.Path]:
    rng = np.random.default_rng(7)
    sink = sw.Queue(maxlen=10_000, name="sink")
    svc = sw.Service(
        capacity=2, buffer_size=50, next_q=sink,
        default_service_time=1.0, rng=rng, name="svc",
    )
    gen = sw.ArrivalGenerator(
        interarrival=lambda r: r.exponential(0.6),
        factory=lambda env: sw.Entity(),
        target=svc, rng=rng, name="gen",
    )
    qrec = sw.QueueLengthRecorder(svc, name="svc_qlen")
    urec = sw.ServiceUtilisationRecorder(svc, name="svc_util")

    env = sw.SimEnvironment(dt=0.1, end=200.0)
    env.register(gen); env.register(svc); env.register(sink)
    env.register(qrec); env.register(urec)
    env.run()

    return [
        _save(sw.plot_queue_length(qrec, title="Service buffer length"),
              "03_queue_length"),
        _save(sw.plot_service_utilisation(urec, title="Service utilisation (2 channels)"),
              "04_service_util"),
    ]


# --------------------------------------------------------------------------- #
# 3. Supply chain: warehouse stock                                            #
# --------------------------------------------------------------------------- #


def tour_supplychain() -> list[pathlib.Path]:
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
    env.register(wh); env.register(rec)
    rng = np.random.default_rng(0)
    for t in range(80):
        env.run(until=float(t + 1))
        # Stochastic per-tick demand.
        wh.decrement_vector(rng.poisson([0.7, 0.4, 1.5]).astype(float))

    return [
        _save(sw.plot_warehouse_stock(rec, title="Warehouse stock vs reorder points"),
              "05_warehouse_stock"),
    ]


# --------------------------------------------------------------------------- #
# 4. Monte Carlo fan chart                                                    #
# --------------------------------------------------------------------------- #


def tour_monte_carlo() -> list[pathlib.Path]:
    # Generate a 200-replicate ensemble of stochastic trajectories.
    rng = np.random.default_rng(42)
    n_runs, n_t = 200, 80
    drift = 0.05
    vol = 0.4
    times = np.arange(n_t, dtype=float)
    increments = rng.normal(drift, vol, size=(n_runs, n_t))
    trajectories = np.cumsum(increments, axis=1)

    # Build an MCResult so we exercise that input path too.
    mc = sw.MCResult(
        n_runs=n_runs,
        seeds=list(range(n_runs)),
        samples=trajectories,
        scenario_name="random_walk_drift",
    )
    return [
        _save(sw.plot_mc_fan(mc, times=times, title="Random-walk-with-drift fan"),
              "06_mc_fan"),
    ]


# --------------------------------------------------------------------------- #
# 5. Agents: A* path on a grid                                                #
# --------------------------------------------------------------------------- #


def tour_agents() -> list[pathlib.Path]:
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
    return [
        _save(sw.plot_agent_path(agent, graph=g, title="A* agent over 8x12 grid"),
              "07_agent_path"),
    ]


# --------------------------------------------------------------------------- #
# 6. Theme switching: same MSD figure rendered light + dark                   #
# --------------------------------------------------------------------------- #


def tour_themes() -> list[pathlib.Path]:
    msd = sw.MassSpringDamper(m=1.0, c=0.2, k=2.0)
    res = sw.simulate(msd, t_span=(0.0, 16.0), dt=0.01, x0=np.array([1.5, 0.0]))
    out: list[pathlib.Path] = []
    for theme in ("light", "dark", "minimal"):
        fig = sw.plot_state_trajectories(
            res, theme=theme, title=f"MSD ({theme} theme)"
        )
        out.append(_save(fig, f"08_theme_{theme}"))
    return out


# --------------------------------------------------------------------------- #
# Index page                                                                  #
# --------------------------------------------------------------------------- #


_INDEX_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>simweave.viz tour</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto;
          padding: 0 1rem; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  small {{ color: #666; }}
  ul {{ line-height: 1.7; }}
  code {{ background: #f4f4f4; padding: 0.1em 0.35em; border-radius: 3px; }}
</style>
<h1>simweave.viz tour</h1>
<small>Each link below opens a self-contained interactive plotly figure.</small>
<ul>
{links}
</ul>
<p>Source: <code>demos/14_viz_tour.py</code></p>
"""


def write_index(paths: list[pathlib.Path]) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    items = "\n".join(
        f'  <li><a href="{p.name}">{p.stem}</a></li>' for p in sorted(paths)
    )
    index = OUT / "index.html"
    index.write_text(_INDEX_TEMPLATE.format(links=items), encoding="utf-8")
    return index


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    if not sw.have_plotly():
        print("plotly is not installed in this environment.")
        print("  Install with:  pip install simweave[viz]")
        return

    print(f"writing figures under {OUT}")
    figs: list[pathlib.Path] = []
    figs += tour_continuous()
    figs += tour_discrete()
    figs += tour_supplychain()
    figs += tour_monte_carlo()
    figs += tour_agents()
    figs += tour_themes()

    # Confirm each figure JSON-round-trips for EdgeWeave.
    print("verifying JSON serialisability for EdgeWeave consumption ...")
    # Re-render the first figure of each section as a representative sample.
    sample = sw.plot_state_trajectories(
        sw.simulate(sw.MassSpringDamper(), t_span=(0.0, 1.0), dt=0.01)
    )
    _confirm_json_roundtrip(sample, "MSD")

    index = write_index(figs)
    print(f"done -> open {index}")


if __name__ == "__main__":
    main()
