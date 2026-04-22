"""Two-echelon supply chain with deterministic daily demand.

    industry --> regional_dc --> store

We synthesise five SKUs, spin up daily consumption and let the system
evolve over 90 days. Prints end-of-run stock levels, order book and
lifetime backorder counts.

Run:
    python demos/04_supply_chain_basic.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

import numpy as np

from simeng.core.entity import Entity
from simeng.core.environment import SimEnvironment
from simeng.supplychain.inventory import InventoryItems
from simeng.supplychain.warehouse import Warehouse


def _inv(stock, rop, batch, lt, unit_cost, n=5):
    return InventoryItems(
        part_names=[f"sku_{i}" for i in range(n)],
        unit_cost=unit_cost,
        stock_level=stock,
        batchsize=batch,
        reorder_points=rop,
        repairable_prc=[0.0] * n,
        repair_times=[0.0] * n,
        newbuy_leadtimes=lt,
    )


class DailyDemand(Entity):
    """Simple entity that decrements a downstream warehouse each day."""

    def __init__(self, target: Warehouse, rate: np.ndarray, name="demand"):
        super().__init__(name=name)
        self.target = target
        self.rate = np.asarray(rate, dtype=float)
        self.lost_sales = np.zeros_like(self.rate)

    def tick(self, dt, env):
        super().tick(dt, env)
        draws = self.rate * dt
        mask = self.target.decrement_vector(draws)
        self.lost_sales += (~mask) * draws

    def has_work(self, env):
        return True


def main(days: float = 90.0, dt: float = 1.0, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    n = 5

    regional_dc = Warehouse(
        _inv(stock=[500.0] * n,
             rop=[100.0] * n,
             batch=[300.0] * n,
             lt=[7.0] * n,
             unit_cost=[10.0, 20.0, 5.0, 50.0, 15.0],
             n=n),
        name="regional_dc",
    )
    store = Warehouse(
        _inv(stock=[50.0] * n,
             rop=[20.0] * n,
             batch=[30.0] * n,
             lt=[2.0] * n,
             unit_cost=[10.0, 20.0, 5.0, 50.0, 15.0],
             n=n),
        name="store",
        parent_warehouse=regional_dc,
    )

    demand_rate = np.array([5.0, 2.0, 10.0, 1.0, 3.0])
    demand = DailyDemand(target=store, rate=demand_rate)

    env = SimEnvironment(dt=dt, end=days)
    env.register(demand); env.register(store); env.register(regional_dc)
    env.run()

    print("End-of-run store stock :", store.inv.stock_level.astype(int).tolist())
    print("End-of-run DC stock    :", regional_dc.inv.stock_level.astype(int).tolist())
    print("Store orders in flight :", store._reorders_volume.astype(int).tolist())
    print("Store backorders (qty) :", store._lifetime_backorders.astype(int).tolist())
    print("Lost sales at the till :", demand.lost_sales.astype(int).tolist())
    print("Est. store demand rate :", store.estimate_demand_rate(env.clock.t).round(3).tolist())


if __name__ == "__main__":
    main()
