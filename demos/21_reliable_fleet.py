"""Reliable fleet simulation -- taxi company operational availability.

Scenario
--------
A taxi company runs a fleet of 10 vehicles.  Each vehicle has three
subsystems with distinct failure characteristics:

* **Engine** -- repairable; low failure rate; long repair time; ~10% BER.
* **Transmission** -- repairable; moderate failure rate; medium repair time.
* **Tyres** (set)-- consumable; moderate failure rate; quick fit time.

A shared parts warehouse holds stock for each SKU.  Two repair bays are
staffed by a pool of 3 technicians (modelled via a ResourcePool).

The demo runs:
1. A single deterministic run with a stacked area availability plot.
2. A 30-replicate Monte Carlo run whose mean ± percentile fan is overlaid.

Run::

    python demos/21_reliable_fleet.py
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
)

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

FLEET_SIZE = 10
SIM_DAYS = 365          # simulation time units = days
DT = 1.0                # one tick per day
N_MC_RUNS = 30
REPAIR_BAYS = 2
N_TECHNICIANS = 3

# SKU indices in the warehouse
SKU_ENGINE = 0
SKU_TRANS = 1
SKU_TYRE = 2


def make_subsystem_specs() -> list[SubsystemSpec]:
    return [
        SubsystemSpec(
            name="engine",
            failure_rate=1 / 120,       # MTBF ≈ 120 days
            sku_index=SKU_ENGINE,
            consumable=False,
            beyond_economic_repair_prc=0.10,
            repair_time=5.0,            # 5 days in repair bay
            unit_cost=8_000.0,
            repair_cost=2_500.0,
        ),
        SubsystemSpec(
            name="transmission",
            failure_rate=1 / 90,        # MTBF ≈ 90 days
            sku_index=SKU_TRANS,
            consumable=False,
            beyond_economic_repair_prc=0.05,
            repair_time=3.0,
            unit_cost=5_000.0,
            repair_cost=1_200.0,
        ),
        SubsystemSpec(
            name="tyres",
            failure_rate=1 / 45,        # MTBF ≈ 45 days (puncture / wear out)
            sku_index=SKU_TYRE,
            consumable=True,
            repair_time=0.5,            # half a day to fit
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
    """Build and run one replicate; return summary dict."""
    rng_master = np.random.default_rng(seed)

    warehouse = make_warehouse()
    technicians = sw.ResourcePool(maxlen=N_TECHNICIANS, name="technicians")
    for i in range(N_TECHNICIANS):
        technicians.deposit(sw.Resource(name=f"tech_{i}"))
    repair_centre = RepairCentre(
        capacity=REPAIR_BAYS,
        buffer_size=50,
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

    return {
        "recorder": recorder,
        "fleet": fleet,
        "repair_centre": repair_centre,
        "operational_availability": recorder.mean_operational_availability,
        "total_cost": fleet.total_cost,
        "total_newbuy_cost": fleet.total_newbuy_cost,
        "total_repair_cost": fleet.total_repair_cost,
        "total_repairs": repair_centre.total_repairs,
        "total_newbuys": repair_centre.total_newbuys,
    }


# ---------------------------------------------------------------------------
# Monte Carlo helper
# ---------------------------------------------------------------------------

def run_mc_scenario(seed: int) -> float:
    """Return operational availability for a single MC replicate."""
    return build_scenario(seed)["operational_availability"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Demo 21 -- Reliable Fleet (Taxi Company)")
    print("=" * 60)

    # ---- Single deterministic run ----------------------------------------
    print(f"\nRunning single deterministic simulation ({SIM_DAYS} days)...")
    single = build_scenario(seed=42)
    fleet: Fleet = single["fleet"]
    recorder: FleetAvailabilityRecorder = single["recorder"]

    print(f"  Fleet size             : {fleet.size}")
    print(f"  Mean operational Ao    : {single['operational_availability']:.3f}")
    print(f"  Total cost             : £{single['total_cost']:>10,.0f}")
    print(f"    of which new buys    : £{single['total_newbuy_cost']:>10,.0f}")
    print(f"    of which repairs     : £{single['total_repair_cost']:>10,.0f}")
    print(f"  Repairs completed      : {single['total_repairs']}")
    print(f"  New buys completed     : {single['total_newbuys']}")
    print()
    print("  Per-vehicle summary:")
    for v in fleet.entities:
        s = v.summary()
        print(
            f"    {v.name}  Ao={s['availability']:.3f}"
            f"  cost=£{s['total_cost']:>7,.0f}"
            f"  failures={s['subsystem_failures']}"
        )

    # ---- Monte Carlo run --------------------------------------------------
    print(f"\nRunning {N_MC_RUNS} Monte Carlo replicates...")
    mc = sw.run_monte_carlo(
        run_mc_scenario,
        n_runs=N_MC_RUNS,
        seeds=list(range(N_MC_RUNS)),
        executor="serial",
        scenario_name="taxi_fleet_mc",
    )
    mc_mean = float(mc.mean())
    mc_std = float(mc.std())
    mc_5, mc_95 = mc.quantile([0.05, 0.95])
    print(f"  MC mean Ao             : {mc_mean:.3f}")
    print(f"  MC std  Ao             : {mc_std:.4f}")
    print(f"  MC P5 / P95            : {float(mc_5):.3f} / {float(mc_95):.3f}")

    # ---- Plots -----------------------------------------------------------
    try:
        import plotly  # noqa: F401
    except ImportError:
        print("\nplotly not installed; skipping plots (pip install simweave[viz]).")
        return

    print("\nGenerating plots...")

    # Stacked area chart -- single run
    fig_avail = sw.plot_fleet_availability(
        recorder,
        title="Taxi Fleet Availability (single run, 365 days)",
    )
    fig_avail.write_html("fleet_availability_single.html")
    print("  Saved: fleet_availability_single.html")

    # Normalised (fraction) version
    fig_norm = sw.plot_fleet_availability(
        recorder,
        normalize=True,
        title="Taxi Fleet Availability -- normalised (single run)",
    )
    fig_norm.write_html("fleet_availability_normalised.html")
    print("  Saved: fleet_availability_normalised.html")

    # MC fan of operational count over the single-run time axis
    # (Collect time-series availability fraction per replicate for the fan)
    print("  Collecting MC time-series for fan chart...")
    n_ticks = int(SIM_DAYS / DT)
    mc_traces = np.zeros((N_MC_RUNS, n_ticks), dtype=float)
    for r in range(N_MC_RUNS):
        res = build_scenario(seed=r)
        op_arr = np.asarray(res["recorder"].operational, dtype=float)
        mc_traces[r, : len(op_arr)] = op_arr / FLEET_SIZE

    times = np.arange(n_ticks, dtype=float) * DT
    fig_fan = sw.plot_mc_fan(
        (times, mc_traces),
        title=f"Taxi Fleet Ao Fan ({N_MC_RUNS} replicates)",
    )
    fig_fan.update_layout(yaxis_title="operational fraction")
    fig_fan.write_html("fleet_mc_fan.html")
    print("  Saved: fleet_mc_fan.html")

    print("\nDone.")


if __name__ == "__main__":
    main()
