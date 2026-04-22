"""Planning-stage optimisation of reorder points.

Demonstrates the two complementary approximations before running the
full simulation:

1. Poisson-based per-SKU heuristic (closed form, fast).
2. Whole-warehouse differential-evolution cost minimiser under an
   availability constraint (scipy required).

Run:
    pip install simeng[optim]   # one-off
    python demos/05_supply_chain_optimise.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

import numpy as np

from simeng.supplychain.inventory import InventoryItems
from simeng.supplychain.warehouse import Warehouse
from simeng.supplychain.optimization import (
    poisson_reorder_points,
    cost_optimise_stock,
)


def build_warehouse(n=6):
    inv = InventoryItems(
        part_names=[f"sku_{i}" for i in range(n)],
        unit_cost=[1.0, 3.0, 12.0, 50.0, 200.0, 0.5],
        stock_level=[10.0] * n,
        batchsize=[5.0, 5.0, 5.0, 2.0, 1.0, 10.0],
        reorder_points=[1.0] * n,
        repairable_prc=[0.1, 0.0, 0.3, 0.5, 0.8, 0.0],
        repair_times=[3.0, 0.0, 5.0, 10.0, 14.0, 0.0],
        newbuy_leadtimes=[7.0, 5.0, 10.0, 21.0, 40.0, 2.0],
    )
    w = Warehouse(inv, name="plant")
    # Pretend we already have a demand-rate estimate from a previous run.
    w._demand_rate = np.array([2.0, 0.5, 0.1, 0.05, 0.01, 10.0])
    return w


def main() -> None:
    w = build_warehouse()

    try:
        import scipy  # noqa: F401
    except ImportError:
        print("scipy not installed; running Poisson heuristic only.")
        k, cost = poisson_reorder_points(w, target_availability=0.9)
        print("Poisson ROPs :", k.astype(int).tolist())
        print(f"Budgeted cost: {cost:.0f}")
        return

    k_p, cost_p = poisson_reorder_points(w, target_availability=0.9)
    k_de, cost_de = cost_optimise_stock(w, target_availability=0.9,
                                          quantize_by_batchsize=False,
                                          maxiter=100, seed=0)
    print(f"{'sku':<8}{'unit_cost':>10}{'poisson_r':>12}{'DE_r':>8}")
    for i, name in enumerate(w.inv.part_names):
        print(f"{name:<8}{w.inv.unit_cost[i]:>10.2f}"
              f"{int(k_p[i]):>12d}{int(k_de[i]):>8d}")
    print()
    print(f"Poisson heuristic cost     : {cost_p:,.0f}")
    print(f"Cost-optimised (DE) cost   : {cost_de:,.0f}")
    print(f"Saving vs heuristic        : {100 * (1 - cost_de / cost_p):.1f}%")


if __name__ == "__main__":
    main()
