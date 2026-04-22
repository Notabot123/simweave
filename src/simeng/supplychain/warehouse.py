"""Warehouse entity that consumes inventory and replenishes from a parent.

Simulation model (per tick):
  1. Count down any outstanding reorder lead times; receive completed orders.
  2. If a SKU's stock level is at or below its reorder point AND no order is
     already in flight, raise a new order for ``batchsize``. If the parent
     warehouse has stock, decrement them; otherwise record a backorder.

Use :meth:`decrement_vector` (vectorised consumption) each tick to represent
demand. Wire two Warehouses via ``parent_warehouse=`` to form multi-echelon
supply chains; the outermost one uses the sentinel string ``"industry"``
for infinite replenishment.
"""
from __future__ import annotations

import numpy as np

from simeng.core.entity import Entity
from simeng.core.logging import get_logger
from simeng.supplychain.inventory import InventoryItems

log = get_logger("supplychain.warehouse")


class Warehouse(Entity):
    """Multi-SKU warehouse with (R, Q) reorder policy."""

    def __init__(self,
                 inventory: InventoryItems,
                 name: str | None = None,
                 parent_warehouse: "Warehouse | str" = "industry") -> None:
        super().__init__(name=name)
        self.inv = inventory
        self.parent: "Warehouse | str" = parent_warehouse

        n = inventory.n_items
        self._reorder_time_remaining = np.zeros(n, dtype=float)
        self._orders_placed_bool = np.zeros(n, dtype=bool)
        self._reorders_volume = np.zeros(n, dtype=float)
        self._lifetime_total_orders = inventory.stock_level.copy()
        self._lifetime_backorders = np.zeros(n, dtype=float)
        self._demand_rate: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Stock ops
    # ------------------------------------------------------------------
    def increment_by_idx(self, idx: int, amount: float = 1.0) -> None:
        self.inv.stock_level[idx] += amount

    def increment_vector(self, vec: np.ndarray) -> None:
        self.inv.stock_level = self.inv.stock_level + np.asarray(vec, dtype=float)

    def decrement_by_idx(self, idx: int, amount: float = 1.0) -> bool:
        if self.inv.stock_level[idx] >= amount:
            self.inv.stock_level[idx] -= amount
            return True
        log.debug("%s: item %s unavailable (have %.2f want %.2f).",
                  self.name, self.inv.part_names[idx], self.inv.stock_level[idx], amount)
        return False

    def decrement_vector(self, vec: np.ndarray) -> np.ndarray:
        """Vectorised consumption; returns a boolean mask of which SKUs succeeded."""
        vec = np.asarray(vec, dtype=float)
        avail = self.inv.stock_level >= vec
        self.inv.stock_level = self.inv.stock_level - vec * avail.astype(float)
        if not avail.all():
            log.debug("%s: %d SKUs unavailable this tick.", self.name, (~avail).sum())
        return avail

    def check_items_available(self, requested: np.ndarray) -> np.ndarray:
        return self.inv.stock_level >= np.asarray(requested, dtype=float)

    # ------------------------------------------------------------------
    # Reorder bookkeeping
    # ------------------------------------------------------------------
    def _receive_pending(self, elapsed: float = 1.0) -> None:
        self._reorder_time_remaining = np.maximum(0.0, self._reorder_time_remaining - elapsed)
        arrived = (self._reorder_time_remaining == 0) & (self._reorders_volume > 0)
        if arrived.any():
            deliveries = self._reorders_volume * arrived
            self.increment_vector(deliveries)
            self._reorders_volume = self._reorders_volume - deliveries
            self._orders_placed_bool[arrived] = False

    def _place_reorders(self) -> None:
        need_reorder = (self.inv.stock_level <= self.inv.reorder_points) & (~self._orders_placed_bool)
        if not need_reorder.any():
            return
        reorders = self.inv.batchsize * need_reorder

        if self.parent == "industry":
            # Infinite replenishment.
            pass
        else:
            assert isinstance(self.parent, Warehouse)
            avail = self.parent.check_items_available(reorders)
            # Decrement parent for the subset that is available.
            self.parent.inv.stock_level = self.parent.inv.stock_level - reorders * avail.astype(float)
            if not avail.all():
                missing = ~avail & need_reorder
                if missing.any():
                    self._lifetime_backorders[missing] += reorders[missing]
                need_reorder = need_reorder & avail
                reorders = reorders * avail.astype(float)

        self._orders_placed_bool = self._orders_placed_bool | need_reorder
        self._reorders_volume = self._reorders_volume + reorders
        self._reorder_time_remaining[need_reorder] = self.inv.newbuy_leadtimes[need_reorder]
        self._lifetime_total_orders = self._lifetime_total_orders + reorders

    def process_orders(self, elapsed: float = 1.0) -> None:
        """Tick the reorder logic by ``elapsed`` time units."""
        self._receive_pending(elapsed)
        self._place_reorders()

    # ------------------------------------------------------------------
    # Simulation tick integration
    # ------------------------------------------------------------------
    def tick(self, dt: float, env) -> None:
        super().tick(dt, env)
        # Default tick just counts down pending orders + places new.
        # Consumption is driven by external demand -- users call
        # decrement_vector() from a demand generator process.
        self.process_orders(elapsed=dt)

    def has_work(self, env) -> bool:
        return bool((self._reorders_volume > 0).any() or self._orders_placed_bool.any())

    # ------------------------------------------------------------------
    # Post-hoc estimates
    # ------------------------------------------------------------------
    def estimate_demand_rate(self, total_sim_time: float) -> np.ndarray:
        """Crude demand rate = lifetime orders / total_sim_time."""
        if total_sim_time <= 0:
            raise ValueError("total_sim_time must be positive.")
        self._demand_rate = self._lifetime_total_orders / total_sim_time
        return self._demand_rate
