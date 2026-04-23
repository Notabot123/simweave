"""End-to-end inventory optimisation from observed demand.

The cost optimisers in :mod:`simweave.supplychain.optimization` need a
per-SKU demand-rate estimate to size each part. Demo 05 hard-codes that
rate to keep the example minimal. This demo does the realistic thing:

1. Build a warehouse with naive starting reorder points.
2. Run a simulation in which a stochastic Poisson-demand generator
   consumes from the warehouse every tick.
3. Call :meth:`Warehouse.estimate_demand_rate` to recover a demand-rate
   estimate from the observed order history.
4. Hand the estimate to the Poisson heuristic and the differential-
   evolution cost optimiser, comparing the two recommendations.

Run::

    pip install simweave[optim]
    python demos/16_inventory_optimisation.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

import numpy as np

import simweave as sw
from simweave.supplychain.optimization import (
    cost_optimise_stock,
    poisson_reorder_points,
)


# True (unobserved) per-tick demand intensities -- the optimiser must
# rediscover something close to these from history alone.
TRUE_LAMBDAS = np.array([2.0, 0.6, 0.15, 0.05])


def build_warehouse() -> sw.Warehouse:
    inv = sw.InventoryItems(
        part_names=["bolt", "bracket", "actuator", "ECU"],
        unit_cost=[0.5, 4.0, 80.0, 600.0],
        stock_level=[20.0, 8.0, 4.0, 2.0],
        batchsize=[10.0, 4.0, 2.0, 1.0],
        reorder_points=[5.0, 2.0, 1.0, 1.0],
        repairable_prc=[0.0, 0.2, 0.6, 0.85],
        repair_times=[0.0, 2.0, 5.0, 12.0],
        newbuy_leadtimes=[3.0, 5.0, 14.0, 60.0],
    )
    return sw.Warehouse(inventory=inv, name="depot")


def simulate_demand(warehouse: sw.Warehouse, sim_time: float, seed: int = 0) -> None:
    """Run a fixed-horizon sim with stochastic per-tick demand."""
    env = sw.SimEnvironment(dt=1.0, end=sim_time)
    env.register(warehouse)
    rng = np.random.default_rng(seed)
    for t in range(int(sim_time)):
        env.run(until=float(t + 1))
        # Stochastic per-SKU demand each tick.
        warehouse.decrement_vector(rng.poisson(TRUE_LAMBDAS).astype(float))


def main() -> None:
    sim_time = 365.0
    print(f"--- Phase 1: simulate {int(sim_time)} ticks of stochastic demand ---")
    w = build_warehouse()
    simulate_demand(w, sim_time)

    # Recover demand rates from observed orders. This is the bit we
    # would not have in a real planning context unless we had logged
    # historical consumption.
    estimated = w.estimate_demand_rate(sim_time)
    print()
    print("Estimated vs true per-tick demand rate:")
    print(f"  {'SKU':<10}{'true':>8}{'estimated':>14}")
    for name, true_rate, est_rate in zip(w.inv.part_names, TRUE_LAMBDAS, estimated):
        print(f"  {name:<10}{true_rate:>8.3f}{est_rate:>14.3f}")

    try:
        import scipy  # noqa: F401
    except ImportError:
        print("\nscipy not installed; skipping cost-optimal comparison.")
        print("Install with `pip install simweave[optim]` to see the full demo.")
        return

    print()
    print("--- Phase 2: optimise reorder points against availability=0.95 ---")
    target_avail = 0.95
    k_p, cost_p = poisson_reorder_points(w, target_availability=target_avail)
    k_de, cost_de = cost_optimise_stock(
        w, target_availability=target_avail, maxiter=120, seed=0
    )

    print(f"  {'SKU':<10}{'unit_cost':>12}{'poisson_r':>12}{'DE_r':>10}")
    for i, name in enumerate(w.inv.part_names):
        print(
            f"  {name:<10}{w.inv.unit_cost[i]:>12.2f}"
            f"{int(k_p[i]):>12d}{int(k_de[i]):>10d}"
        )

    print()
    print(f"Poisson heuristic budgeted cost  : {cost_p:>10,.0f}")
    print(f"Cost-optimised (DE) budgeted cost: {cost_de:>10,.0f}")
    if cost_p > 0:
        saving = 100.0 * (1.0 - cost_de / cost_p)
        print(f"Saving vs heuristic               : {saving:>9.1f}%")


if __name__ == "__main__":
    main()
