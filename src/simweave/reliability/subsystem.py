"""Subsystem specification and live-state types for ReliableEntity.

A :class:`SubsystemSpec` is an immutable description of one fitted subsystem
(engine, gearbox, tyre, sensor, etc.) that you attach to a
:class:`~simweave.reliability.entity.ReliableEntity`.  At runtime each fitted
subsystem is tracked via a :class:`SubsystemStatus` instance that records
whether it is currently up, waiting for a spare part, or being repaired.

Failure model
-------------
Failures are drawn from an exponential (memoryless) distribution.  Given a
failure rate ``λ`` (failures per unit simulation time), the probability of at
least one failure occurring in a tick of width ``dt`` is::

    P(fail) = 1 − exp(−λ · dt)

For cycle-based wear set ``failure_rate_per_cycle`` and increment the owning
entity's ``operational_cycles`` counter each tick.  Both mechanisms can be
active simultaneously; either one can trigger a failure event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class SubsystemState(Enum):
    """Operational state of a single subsystem."""

    UP = auto()
    """Subsystem is fully functional."""

    AWAITING_PART = auto()
    """Subsystem has failed; waiting for a spare part to become available."""

    IN_REPAIR = auto()
    """Part obtained; repair / fit job is queued or active at the RepairCentre."""


@dataclass
class SubsystemSpec:
    """Immutable description of one subsystem fitted to a ReliableEntity.

    Parameters
    ----------
    name:
        Human-readable label (e.g. ``"engine"``, ``"front_tyre"``).
    failure_rate:
        Time-based failure rate ``λ`` in failures per unit simulation time.
        Set to ``0.0`` if only cycle-based failure is used.
    sku_index:
        Index into the associated :class:`~simweave.supplychain.warehouse.Warehouse`'s
        :attr:`~simweave.supplychain.inventory.InventoryItems.part_names` list.
        When the subsystem fails, one unit of this SKU is consumed.
    consumable:
        If ``True`` the failed unit is discarded and a new one is bought.
        If ``False`` the failed unit is returned to a
        :class:`~simweave.reliability.repair.RepairCentre` for repair.
    beyond_economic_repair_prc:
        Fraction of failures on a *repairable* subsystem that are beyond
        economic repair and therefore require a new buy instead.  Ignored
        when ``consumable=True``.
    repair_time:
        Nominal repair / fit time in simulation time units.  This becomes the
        ``remaining_service_time`` of the :class:`~simweave.reliability.repair.RepairJob`
        submitted to the repair centre.
    unit_cost:
        Cost charged per new unit purchased (new buy or BER replacement).
    repair_cost:
        Cost charged per repair (non-BER repairable failure).
    failure_rate_per_cycle:
        Cycle-based failure rate in failures per operational cycle.  Set to
        ``0.0`` to disable cycle-based failures.
    """

    name: str
    failure_rate: float
    sku_index: int
    consumable: bool = True
    beyond_economic_repair_prc: float = 0.0
    repair_time: float = 1.0
    unit_cost: float = 0.0
    repair_cost: float = 0.0
    failure_rate_per_cycle: float = 0.0


@dataclass
class SubsystemStatus:
    """Live state of one subsystem on a specific entity.

    Created automatically by :class:`~simweave.reliability.entity.ReliableEntity`
    for each :class:`SubsystemSpec` it is initialised with.
    """

    spec: SubsystemSpec
    state: SubsystemState = field(default=SubsystemState.UP)
    time_in_state: float = 0.0
    total_failures: int = 0
    total_downtime: float = 0.0
    cost_newbuy: float = 0.0
    cost_repair: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.cost_newbuy + self.cost_repair


__all__ = ["SubsystemState", "SubsystemSpec", "SubsystemStatus"]
