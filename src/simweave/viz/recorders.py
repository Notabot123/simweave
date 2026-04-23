"""Time-sampling recorders for simweave primitives.

simweave's Queue, Service and Warehouse track aggregate scalars
(``cumulative_length_time``, ``busy_time``, ``inv.stock_level``) but do
**not** keep per-tick time series. The viz helpers need a time axis, so a
recorder is a small :class:`Entity` that registers with the
:class:`SimEnvironment` and snapshots the quantity of interest each tick.

Recorders are intentionally separate from the primitives themselves so:

* Users who don't visualise pay no memory cost.
* Adding a new recorder later (e.g. ``ResourcePoolRecorder``) does not
  touch the simulation cores.
* Recorders can be subclassed in user code if they want to capture
  additional fields.

Each recorder exposes:

* ``times`` -- list of simulation times at which a sample was taken.
* one or more value attributes documented per class.

A baseline sample is captured at registration time so the resulting series
starts at ``t = env.clock.start`` rather than ``t = start + dt``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from simweave.core.entity import Entity

if TYPE_CHECKING:  # pragma: no cover
    from simweave.core.environment import SimEnvironment
    from simweave.discrete.queues import Queue
    from simweave.discrete.services import Service
    from simweave.supplychain.warehouse import Warehouse


class _Recorder(Entity):
    """Internal base class.

    Subclasses must implement :meth:`_sample(env, t)` and append both the
    time and their per-class values inside it.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)
        self.times: list[float] = []

    def on_register(self, env: "SimEnvironment") -> None:
        super().on_register(env)
        self._sample(env, env.clock.t)

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        super().tick(dt, env)
        # Record the *post-tick* time so the trace lines up with the state
        # of upstream entities after they've ticked. Order of registration
        # matters: register the recorder *after* the entity it records for
        # the cleanest semantics.
        self._sample(env, env.clock.t + dt)

    def has_work(self, env: "SimEnvironment") -> bool:  # noqa: D401
        return False

    def _sample(self, env: "SimEnvironment", t: float) -> None:  # pragma: no cover
        raise NotImplementedError


class QueueLengthRecorder(_Recorder):
    """Sample ``len(queue)`` each tick.

    Attributes
    ----------
    times:
        Simulation times of each sample.
    lengths:
        Queue length at each sample (matches ``times`` element-wise).
    """

    def __init__(self, queue: "Queue", name: str | None = None) -> None:
        super().__init__(name=name or f"qlen({queue.name})")
        self.queue = queue
        self.lengths: list[int] = []

    def _sample(self, env: "SimEnvironment", t: float) -> None:
        self.times.append(t)
        self.lengths.append(len(self.queue))


class ServiceUtilisationRecorder(_Recorder):
    """Sample a Service's per-channel and aggregate utilisation each tick.

    The simweave :class:`~simweave.discrete.services.Service` keeps a
    monotonic ``busy_time`` per channel. We capture that snapshot and also
    derive an aggregate running-mean utilisation
    ``sum(busy_time) / (capacity * elapsed)``.

    Attributes
    ----------
    times:
        Simulation times of each sample.
    busy_time:
        ``(n_samples, n_channels)`` cumulative busy time per channel.
    utilisation:
        Running aggregate utilisation in ``[0, 1]`` at each sample.
    instantaneous_busy:
        ``(n_samples, n_channels)`` boolean array; True where the channel
        was busy at the moment of sampling. Useful for Gantt-style views.
    """

    def __init__(self, service: "Service", name: str | None = None) -> None:
        super().__init__(name=name or f"util({service.name})")
        self.service = service
        self.busy_time: list[np.ndarray] = []
        self.utilisation: list[float] = []
        self.instantaneous_busy: list[np.ndarray] = []
        self._t0: float | None = None

    def _sample(self, env: "SimEnvironment", t: float) -> None:
        if self._t0 is None:
            self._t0 = t
        self.times.append(t)
        bt = np.asarray([ch.busy_time for ch in self.service.channels], dtype=float)
        self.busy_time.append(bt)
        instbusy = np.asarray(
            [ch.is_busy() for ch in self.service.channels], dtype=bool
        )
        self.instantaneous_busy.append(instbusy)
        elapsed = max(t - self._t0, 1e-12)
        self.utilisation.append(
            float(bt.sum() / (self.service.capacity * elapsed))
            if self.service.capacity
            else 0.0
        )


class WarehouseStockRecorder(_Recorder):
    """Sample ``warehouse.inv.stock_level`` each tick.

    Attributes
    ----------
    times:
        Simulation times of each sample.
    stock:
        ``(n_samples, n_skus)`` stock-level history.
    sku_names:
        Names of the SKUs (snapshot at registration; stable thereafter).
    reorder_points:
        Reorder points snapshot at registration.
    """

    def __init__(self, warehouse: "Warehouse", name: str | None = None) -> None:
        super().__init__(name=name or f"stock({warehouse.name})")
        self.warehouse = warehouse
        self.stock: list[np.ndarray] = []
        self.sku_names: tuple[str, ...] = tuple(warehouse.inv.part_names)
        self.reorder_points: np.ndarray = np.asarray(
            warehouse.inv.reorder_points, dtype=float
        ).copy()

    def _sample(self, env: "SimEnvironment", t: float) -> None:
        self.times.append(t)
        self.stock.append(self.warehouse.inv.stock_level.copy())


__all__ = [
    "QueueLengthRecorder",
    "ServiceUtilisationRecorder",
    "WarehouseStockRecorder",
]
