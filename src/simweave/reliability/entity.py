"""ReliableEntity -- an Entity composed of failable subsystems.

Each :class:`ReliableEntity` owns a list of
:class:`~simweave.reliability.subsystem.SubsystemSpec` descriptors and at
runtime maintains a corresponding list of
:class:`~simweave.reliability.subsystem.SubsystemStatus` objects.  On every
simulation tick it:

1. Checks each UP subsystem for a random failure event (exponential and/or
   cycle-based failure distributions).
2. For newly failed subsystems, attempts to draw a spare part from the
   associated :class:`~simweave.supplychain.warehouse.Warehouse`.
3. Re-checks AWAITING_PART subsystems each tick in case stock has been
   replenished.
4. Submits a :class:`~simweave.reliability.repair.RepairJob` to the
   :class:`~simweave.reliability.repair.RepairCentre` once parts are in hand.
5. Tracks cumulative operational time, downtime, and costs.

The entity is considered *operational* only when **all** of its subsystems
are in the UP state.  If you want partial-availability semantics (e.g. an
aircraft that can fly with one engine out) subclass and override
:meth:`is_operational`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from simweave.core.entity import Entity
from simweave.reliability.subsystem import SubsystemSpec, SubsystemState, SubsystemStatus

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment
    from simweave.supplychain.warehouse import Warehouse
    from simweave.reliability.repair import RepairCentre


class ReliableEntity(Entity):
    """An entity composed of subsystems that can fail and require repair.

    Parameters
    ----------
    subsystems:
        One :class:`~simweave.reliability.subsystem.SubsystemSpec` per
        subsystem fitted to this entity.
    warehouse:
        Parts warehouse.  When a subsystem fails, one unit of its SKU is
        consumed from here.  When a repairable unit is returned to service,
        one unit is added back.
    repair_centre:
        Optional :class:`~simweave.reliability.repair.RepairCentre`.  If
        supplied, failed subsystems are queued here for repair/fitting.  If
        ``None``, repairs are instantaneous (the subsystem is restored on the
        same tick the part is obtained).
    name:
        Display name.
    rng:
        Numpy random generator.  Defaults to ``np.random.default_rng()``.

    Attributes
    ----------
    subsystems : list[SubsystemStatus]
        Live state of each fitted subsystem.
    operational_cycles : float
        Cumulative operating cycles.  Increment this in your scenario script
        (e.g. each km driven, each sortie flown) to activate cycle-based
        failure rates.
    total_operational_time : float
        Simulation time spent fully operational.
    total_downtime : float
        Simulation time spent with at least one subsystem not UP.
    cost_newbuy : float
        Cumulative spend on new part purchases.
    cost_repair : float
        Cumulative spend on repairs.
    """

    def __init__(
        self,
        subsystems: Sequence[SubsystemSpec],
        warehouse: "Warehouse",
        repair_centre: "RepairCentre | None" = None,
        name: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(name=name)
        self.warehouse = warehouse
        self.repair_centre = repair_centre
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        self.subsystems: list[SubsystemStatus] = [
            SubsystemStatus(spec=s) for s in subsystems
        ]

        # Metrics
        self.total_operational_time: float = 0.0
        self.total_downtime: float = 0.0
        self.cost_newbuy: float = 0.0
        self.cost_repair: float = 0.0
        self.operational_cycles: float = 0.0
        self._prev_cycles: float = 0.0  # for cycle-delta computation

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_operational(self) -> bool:
        """``True`` when every subsystem is in the UP state."""
        return all(s.state == SubsystemState.UP for s in self.subsystems)

    @property
    def availability(self) -> float:
        """Empirical operational availability (time-based)."""
        total = self.total_operational_time + self.total_downtime
        return self.total_operational_time / total if total > 0 else 1.0

    @property
    def total_cost(self) -> float:
        return self.cost_newbuy + self.cost_repair

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        super().tick(dt, env)

        delta_cycles = self.operational_cycles - self._prev_cycles
        self._prev_cycles = self.operational_cycles

        for idx, status in enumerate(self.subsystems):
            if status.state == SubsystemState.UP:
                self._check_failure(idx, dt, delta_cycles)
            elif status.state == SubsystemState.AWAITING_PART:
                # Retry warehouse in case stock has arrived.
                self._try_get_part(idx)

            if status.state != SubsystemState.UP:
                status.total_downtime += dt
            status.time_in_state += dt

        if self.is_operational:
            self.total_operational_time += dt
        else:
            self.total_downtime += dt

    def has_work(self, env: "SimEnvironment") -> bool:
        return True  # always potentially experiencing failure or recovery

    # ------------------------------------------------------------------
    # Internal failure / repair logic
    # ------------------------------------------------------------------

    def _check_failure(self, idx: int, dt: float, delta_cycles: float) -> None:
        """Draw failure events for a UP subsystem."""
        spec = self.subsystems[idx].spec

        # Time-based failure: P(fail) = 1 − exp(−λ·dt)
        failed = False
        if spec.failure_rate > 0.0:
            p_time = 1.0 - np.exp(-spec.failure_rate * dt)
            if self.rng.random() < p_time:
                failed = True

        # Cycle-based failure: P(fail) = 1 − exp(−λ_c·Δcycles)
        if not failed and spec.failure_rate_per_cycle > 0.0 and delta_cycles > 0.0:
            p_cyc = 1.0 - np.exp(-spec.failure_rate_per_cycle * delta_cycles)
            if self.rng.random() < p_cyc:
                failed = True

        if failed:
            self.subsystems[idx].total_failures += 1
            self.subsystems[idx].state = SubsystemState.AWAITING_PART
            self.subsystems[idx].time_in_state = 0.0
            self._try_get_part(idx)

    def _try_get_part(self, idx: int) -> None:
        """Attempt to draw a spare part from the warehouse and submit a job."""
        spec = self.subsystems[idx].spec
        got = self.warehouse.decrement_by_idx(spec.sku_index, 1.0)
        if not got:
            # Remain AWAITING_PART; will retry next tick.
            return

        # Determine new-buy vs repair.
        is_ber = (
            not spec.consumable
            and spec.beyond_economic_repair_prc > 0.0
            and self.rng.random() < spec.beyond_economic_repair_prc
        )
        is_new_buy = spec.consumable or is_ber
        return_to_stock = not is_new_buy  # repaired units go back to stock

        cost = spec.unit_cost if is_new_buy else spec.repair_cost

        self.subsystems[idx].state = SubsystemState.IN_REPAIR
        self.subsystems[idx].time_in_state = 0.0

        if self.repair_centre is not None:
            from simweave.reliability.repair import RepairJob

            job = RepairJob(
                owner=self,
                subsystem_idx=idx,
                is_new_buy=is_new_buy,
                return_to_stock=return_to_stock,
                repair_time=spec.repair_time,
                cost=cost,
                name=f"job_{self.name}_{spec.name}",
            )
            self.repair_centre.enqueue(job)
        else:
            # No queuing: restore subsystem immediately.
            self._on_repair_complete(idx, cost, is_new_buy)

    def _on_repair_complete(
        self, idx: int, cost: float, is_new_buy: bool
    ) -> None:
        """Callback invoked by :class:`~simweave.reliability.repair.RepairCentre`
        (or directly when no repair centre is configured) when a job finishes."""
        self.subsystems[idx].state = SubsystemState.UP
        self.subsystems[idx].time_in_state = 0.0

        if is_new_buy:
            self.cost_newbuy += cost
            self.subsystems[idx].cost_newbuy += cost
        else:
            self.cost_repair += cost
            self.subsystems[idx].cost_repair += cost

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a snapshot dict suitable for logging / MC aggregation."""
        return {
            "name": self.name,
            "operational": self.is_operational,
            "availability": self.availability,
            "total_downtime": self.total_downtime,
            "cost_newbuy": self.cost_newbuy,
            "cost_repair": self.cost_repair,
            "total_cost": self.total_cost,
            "subsystem_failures": {
                s.spec.name: s.total_failures for s in self.subsystems
            },
        }


__all__ = ["ReliableEntity"]
