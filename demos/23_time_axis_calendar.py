"""Calendar time axis -- mapping simulation ticks to real-world dates.

Scenario
--------
A small taxi fleet begins operations on 1 January 2027.  Each simulation
tick represents one day.  The demo shows how :class:`~simweave.core.time_axis.SimTimeAxis`
transforms numeric tick labels on every plot into calendar months so that
a manager reading the chart immediately sees "June 2027" rather than "180".

Three plots are produced:

1. **Fleet availability (stacked area)** -- green/amber/red bands with
   calendar months on the x-axis.
2. **Parts depot stock** -- warehouse stock levels over the same period
   showing real resupply dates.
3. **Availability fan chart** -- Monte Carlo percentile fan for 20
   replicates, again with calendar dates.

Run::

    python demos/23_time_axis_calendar.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import simweave as sw
from simweave.reliability import (
    Fleet,
    FleetAvailabilityRecorder,
    RepairCentre,
    ReliableEntity,
    SubsystemSpec,
)

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

SIM_DAYS = 365
DT = 1.0
FLEET_SIZE = 8
N_MC = 20

# Real-world calendar anchor: 1 tick = 1 day, starting 1 Jan 2027.
TAX = sw.SimTimeAxis(start="2027-01-01", tick_unit="days")

SKU_ENGINE = 0
SKU_TRANS = 1
SKU_TYRE = 2


def make_specs() -> list[SubsystemSpec]:
    return [
        SubsystemSpec(
            "engine",
            failure_rate=1 / 120,
            sku_index=SKU_ENGINE,
            consumable=False,
            beyond_economic_repair_prc=0.10,
            repair_time=5.0,
            unit_cost=8_000.0,
            repair_cost=2_500.0,
        ),
        SubsystemSpec(
            "transmission",
            failure_rate=1 / 90,
            sku_index=SKU_TRANS,
            consumable=False,
            beyond_economic_repair_prc=0.05,
            repair_time=3.0,
            unit_cost=5_000.0,
            repair_cost=1_200.0,
        ),
        SubsystemSpec(
            "tyres",
            failure_rate=1 / 45,
            sku_index=SKU_TYRE,
            consumable=True,
            repair_time=0.5,
            unit_cost=400.0,
        ),
    ]


def make_warehouse() -> sw.Warehouse:
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
    return sw.Warehouse(inventory=inv, name="parts_depot")


def build_scenario(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    wh = make_warehouse()
    wh_rec = sw.WarehouseStockRecorder(wh)

    pool = sw.ResourcePool(maxlen=3, name="technicians")
    for i in range(3):
        pool.deposit(sw.Resource(name=f"tech_{i}"))
    rc = RepairCentre(capacity=2, resources=pool)

    specs = make_specs()
    vehicles = [
        ReliableEntity(
            specs, wh, rc,
            name=f"taxi_{i:02d}",
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        )
        for i in range(FLEET_SIZE)
    ]
    fleet = Fleet(vehicles, name="taxi_fleet")
    fl_rec = FleetAvailabilityRecorder(fleet)

    env = sw.SimEnvironment(dt=DT, end=SIM_DAYS)
    env.register(wh)
    env.register(wh_rec)
    env.register(rc)
    for v in vehicles:
        env.register(v)
    env.register(fl_rec)
    env.run(until=SIM_DAYS)

    return {
        "fleet": fleet,
        "fleet_recorder": fl_rec,
        "wh_recorder": wh_rec,
        "ao": fl_rec.mean_operational_availability,
    }


def main() -> None:
    print("=" * 60)
    print("Demo 23 -- Calendar Time Axis")
    print(f"  Simulation start : {TAX.start.strftime('%d %B %Y')}")
    print("  Tick unit        : 1 day")
    print(f"  Horizon          : {SIM_DAYS} days "
          f"({TAX.label(SIM_DAYS)})")
    print("=" * 60)

    # ---- Single run -------------------------------------------------------
    result = build_scenario(seed=42)
    fleet: Fleet = result["fleet"]
    fl_rec: FleetAvailabilityRecorder = result["fleet_recorder"]
    wh_rec = result["wh_recorder"]

    print(f"\nSingle-run Ao  : {result['ao']:.3f}")
    print(f"Total cost     : £{fleet.total_cost:>10,.0f}")

    # Quick sanity-check: show a few date conversions
    print("\nDate conversions:")
    for t in [0, 30, 90, 180, 365]:
        print(f"  tick {t:>3}  →  {TAX.label(float(t))}")

    try:
        import plotly  # noqa: F401
    except ImportError:
        print("\nplotly not installed; skipping plots.")
        return

    print("\nGenerating plots...")

    # 1. Fleet availability with calendar x-axis
    fig_fleet = sw.plot_fleet_availability(
        fl_rec,
        time_axis=TAX,
        title="Taxi Fleet Availability -- 2027 (calendar months)",
    )
    fig_fleet.write_html("fleet_availability_calendar.html")
    print("  Saved: fleet_availability_calendar.html")

    # 2. Warehouse stock with calendar x-axis
    fig_stock = sw.plot_warehouse_stock(
        wh_rec,
        time_axis=TAX,
        title="Parts Depot Stock Level -- 2027",
    )
    fig_stock.write_html("warehouse_stock_calendar.html")
    print("  Saved: warehouse_stock_calendar.html")

    # 3. MC fan of fleet availability with calendar dates
    print(f"  Running {N_MC} MC replicates for fan chart...")
    n_ticks = int(SIM_DAYS / DT)
    mc_traces = np.zeros((N_MC, n_ticks), dtype=float)
    for r in range(N_MC):
        res = build_scenario(seed=r)
        op = np.asarray(res["fleet_recorder"].operational, dtype=float)
        mc_traces[r, : len(op)] = op / FLEET_SIZE

    times = np.arange(n_ticks, dtype=float)
    fig_fan = sw.plot_mc_fan(
        (times, mc_traces),
        time_axis=TAX,
        title=f"Taxi Fleet Ao Fan ({N_MC} replicates) -- 2027",
    )
    fig_fan.update_layout(yaxis_title="operational fraction")
    fig_fan.write_html("fleet_mc_fan_calendar.html")
    print("  Saved: fleet_mc_fan_calendar.html")

    # 4. Demonstrate post-hoc application on an existing figure
    fig_post = sw.plot_fleet_availability(fl_rec)  # built without time_axis
    TAX.apply_to_figure(fig_post, title="Calendar date")          # applied after
    fig_post.update_layout(title="Fleet Availability (post-hoc date axis)")
    fig_post.write_html("fleet_availability_posthoc.html")
    print("  Saved: fleet_availability_posthoc.html")

    # 5. tick_for_date utility -- schedule events at named dates
    q2_start = TAX.tick_for_date("2027-04-01")
    q3_start = TAX.tick_for_date("2027-07-01")
    print(f"\n  Q2 starts at tick {q2_start:.0f}  ({TAX.label(q2_start)})")
    print(f"  Q3 starts at tick {q3_start:.0f}  ({TAX.label(q3_start)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
