"""Inventory items dataclass.

A thin record type that captures the per-SKU attributes needed by a
:class:`~simeng.supplychain.warehouse.Warehouse`. Lengths of every field
must match; constructor validates this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class InventoryItems:
    """A collection of SKUs that share a storage location.

    All list-shaped fields must be the same length (one entry per SKU).
    Numeric fields are coerced to ``np.ndarray`` on ``__post_init__`` so
    vectorised warehouse operations don't have to re-check types at runtime.
    """

    part_names: Sequence[str]
    unit_cost: Sequence[float]
    stock_level: Sequence[float]
    batchsize: Sequence[float]
    reorder_points: Sequence[float]
    repairable_prc: Sequence[float]
    repair_times: Sequence[float]
    newbuy_leadtimes: Sequence[float]
    shelf_life: Sequence[float] = field(default_factory=list)
    failure_rate: Sequence[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.part_names)
        # Validate and coerce.
        checked = [
            ("unit_cost", self.unit_cost),
            ("stock_level", self.stock_level),
            ("batchsize", self.batchsize),
            ("reorder_points", self.reorder_points),
            ("repairable_prc", self.repairable_prc),
            ("repair_times", self.repair_times),
            ("newbuy_leadtimes", self.newbuy_leadtimes),
        ]
        for label, seq in checked:
            if len(seq) != n:
                raise ValueError(
                    f"InventoryItems: field {label!r} has length {len(seq)} but expected {n}."
                )

        # Optional fields: allow empty -> defaults to zeros.
        if len(self.shelf_life) == 0:
            self.shelf_life = [float("inf")] * n
        elif len(self.shelf_life) != n:
            raise ValueError(f"shelf_life length {len(self.shelf_life)} != {n}")
        if len(self.failure_rate) == 0:
            self.failure_rate = [0.0] * n
        elif len(self.failure_rate) != n:
            raise ValueError(f"failure_rate length {len(self.failure_rate)} != {n}")

        # Cast numeric fields to np.ndarray.
        self.unit_cost = np.asarray(self.unit_cost, dtype=float)
        self.stock_level = np.asarray(self.stock_level, dtype=float)
        self.batchsize = np.asarray(self.batchsize, dtype=float)
        self.reorder_points = np.asarray(self.reorder_points, dtype=float)
        self.repairable_prc = np.asarray(self.repairable_prc, dtype=float)
        self.repair_times = np.asarray(self.repair_times, dtype=float)
        self.newbuy_leadtimes = np.asarray(self.newbuy_leadtimes, dtype=float)
        self.shelf_life = np.asarray(self.shelf_life, dtype=float)
        self.failure_rate = np.asarray(self.failure_rate, dtype=float)

    @property
    def n_items(self) -> int:
        return len(self.part_names)
