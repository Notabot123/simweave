"""Sensitivity analysis for a reliable fleet -- 2-D parameter sweep.

Scenario
--------
Using the same taxi fleet model from demo 21, this script sweeps two
independent parameters simultaneously:

* **Axis 1 -- repair bays** (1 → 4): how many parallel repair channels the
  depot has.
* **Axis 2 -- initial stock level multiplier** (0.5 → 2.0): a scalar applied
  to the base warehouse stock levels, representing investment in spares
  holdings.

For each (bays, stock) combination, 20 Monte Carlo replicates are run and
the mean operational availability (Ao) is recorded.  The result is displayed
as:

1. A smooth 3-D surface plot.
2. A 2-D heatmap (useful for reading off exact values).
3. A grouped bar chart (discrete view by repair bay count).

Run::

    python demos/22_sensitivity_analysis.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import simweave as sw
from simweave.reliability import (
    SubsystemSpec,
    ReliableEntity,
    RepairCentre,
    Fleet,
    FleetAvailabilityRecorder,
    sensitivity_sweep,
)

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

FLEET_SIZE = 10
SIM_DAYS = 180          # shorter run so the sweep is quick
DT = 1.0
N_TECHNICIANS = 4       # held constant; bays is varied

SKU_ENGINE = 0
SKU_TRANS = 1
SKU_TYRE = 2

BASE_STOCK = np.array([3.0, 4.0, 10.0])  # engines, transmissions, tyre sets


def make_subsystem_specs() -> list[SubsystemSpec]:
    return [
        SubsystemSpec(
            name="engine",
            failure_rate=1 / 120,
            sku_index=SKU_ENGINE,
            consumable=False,
            beyond_economic_repair_prc=0.10,
            repair_time=5.0,
            unit_cost=8_000.0,
            repair_cost=2_500.0,
        ),
        SubsystemSpec(
            name="transmission",
            failure_rate=1 / 90,
            sku_index=SKU_TRANS,
            consumable=False,
            beyond_economic_repair_prc=0.05,
            repair_time=3.0,
            unit_cost=5_000.0,
            repair_cost=1_200.0,
        ),
        SubsystemSpec(
            name="tyres",
            failure_rate=1 / 45,
            sku_index=SKU_TYRE,
            consumable=True,
            repair_time=0.5,
            unit_cost=400.0,
        ),
    ]


def make_warehouse(stock_multiplier: float) -> sw.Warehouse:
    stock = (BASE_STOCK * stock_multiplier).clip(min=1.0)
    inv = sw.InventoryItems(
        part_names=["engine", "transmission", "tyres"],
        unit_cost=[8_000.0, 5_000.0, 400.0],
        stock_level=stock.tolist(),
        batchsize=[2.0, 2.0, 4.0],
        reorder_points=[1.0, 1.0, 2.0],
        repairable_prc=[0.90, 0.95, 0.0],
        repair_times=[5.0, 3.0, 0.0],
        newbuy_leadtimes=[14.0, 10.0, 3.0],
    )
    return sw.Warehouse(inventory=inv, name="parts_depot")


# ---------------------------------------------------------------------------
# Scenario builder for sensitivity_sweep
# ---------------------------------------------------------------------------

def run_scenario(n_bays: float, stock_mult: float, seed: int) -> float:
    """Return mean operational availability for given parameters."""
    n_bays = max(1, int(round(n_bays)))
    rng_master = np.random.default_rng(seed)

    warehouse = make_warehouse(stock_mult)
    technicians = sw.ResourcePool(maxlen=N_TECHNICIANS, name="technicians")
    for i in range(N_TECHNICIANS):
        technicians.deposit(sw.Resource(name=f"tech_{i}"))
    repair_centre = RepairCentre(
        capacity=n_bays,
        buffer_size=100,
        resources=technicians,
        name="repair_bay",
    )

    specs = make_subsystem_specs()
    vehicles = [
        ReliableEntity(
            subsystems=specs,
            warehouse=warehouse,
            repair_centre=repair_centre,
            name=f"taxi_{i:02d}",
            rng=np.random.default_rng(rng_master.integers(0, 2**32)),
        )
        for i in range(FLEET_SIZE)
    ]

    fleet = Fleet(vehicles, name="taxi_fleet")
    recorder = FleetAvailabilityRecorder(fleet)

    env = sw.SimEnvironment(dt=DT, end=SIM_DAYS)
    env.register(warehouse)
    env.register(repair_centre)
    for v in vehicles:
        env.register(v)
    env.register(recorder)

    env.run(until=SIM_DAYS)
    return recorder.mean_operational_availability


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Demo 22 -- Sensitivity Analysis (2-D sweep)")
    print("=" * 60)

    bays_values = [1, 2, 3, 4]
    stock_values = [0.5, 0.75, 1.0, 1.5, 2.0]
    n_runs = 15  # replicates per grid cell

    print(
        f"\nSweeping {len(bays_values)} repair-bay levels × "
        f"{len(stock_values)} stock-multiplier levels "
        f"× {n_runs} MC replicates = "
        f"{len(bays_values)*len(stock_values)*n_runs} total runs..."
    )
    print("(This may take a minute on slower machines.)")

    result = sensitivity_sweep(
        run_scenario,
        param1_name="repair_bays",
        param1_values=bays_values,
        param2_name="stock_multiplier",
        param2_values=stock_values,
        metric_name="Operational Availability (Ao)",
        n_runs=n_runs,
        seed=0,
        executor="serial",
    )

    print("\nMean Ao grid (rows = repair_bays, cols = stock_multiplier):")
    header = "          " + "".join(f"  stk×{v:.2f}" for v in stock_values)
    print(header)
    for i, bays in enumerate(bays_values):
        row = f"  bays={bays}  " + "".join(
            f"    {result.metric_mean[i, j]:.3f}" for j in range(len(stock_values))
        )
        print(row)

    try:
        import plotly  # noqa: F401
    except ImportError:
        print("\nplotly not installed; skipping plots (pip install simweave[viz]).")
        return

    print("\nGenerating plots...")

    # 3-D surface
    fig_surf = sw.plot_sensitivity_surface(
        result,
        chart_type="surface",
        title="Fleet Ao Sensitivity -- 3-D Surface",
    )
    fig_surf.write_html("sensitivity_surface.html")
    print("  Saved: sensitivity_surface.html")

    # Heatmap
    fig_heat = sw.plot_sensitivity_surface(
        result,
        chart_type="heatmap",
        title="Fleet Ao Sensitivity -- Heatmap",
    )
    fig_heat.write_html("sensitivity_heatmap.html")
    print("  Saved: sensitivity_heatmap.html")

    # Grouped bar chart with error bars
    fig_bar = sw.plot_sensitivity_surface(
        result,
        chart_type="bar",
        show_std=True,
        title="Fleet Ao Sensitivity -- Grouped Bars (±1σ)",
    )
    fig_bar.write_html("sensitivity_bar.html")
    print("  Saved: sensitivity_bar.html")

    # 1-D slice: vary bays with stock_mult = 1.0 (baseline)
    print("\nRunning 1-D slice: vary repair_bays at baseline stock...")
    result_1d = sensitivity_sweep(
        lambda bays, seed: run_scenario(bays, 1.0, seed),
        param1_name="repair_bays",
        param1_values=list(range(1, 7)),
        metric_name="Operational Availability (Ao)",
        n_runs=20,
        seed=100,
    )
    fig_1d = sw.plot_sensitivity_surface(
        result_1d,
        show_std=True,
        title="Ao vs Repair Bays (baseline stock, ±1σ)",
    )
    fig_1d.write_html("sensitivity_1d_bays.html")
    print("  Saved: sensitivity_1d_bays.html")

    print("\nDone.")


if __name__ == "__main__":
    main()
