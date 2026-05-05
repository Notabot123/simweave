"""Repair centre and repair-job entity.

A :class:`RepairJob` is a lightweight :class:`~simweave.core.entity.Entity`
that flows through a :class:`RepairCentre`.  When the job completes the
centre calls back into the owning :class:`~simweave.reliability.entity.ReliableEntity`
to restore the subsystem to UP and log costs.

A :class:`RepairCentre` is a thin subclass of
:class:`~simweave.discrete.services.Service`.  It inherits all of that class's
queuing, multi-channel, and resource-pool machinery.  To model a repair team
under the operator's employment, pass a
:class:`~simweave.discrete.resources.ResourcePool` whose size equals the
number of technicians.  For a third-party maintenance contract simply choose a
``capacity`` and ``buffer_size`` that reflect the contracted throughput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simweave.core.entity import Entity
from simweave.discrete.services import Service

if TYPE_CHECKING:
    from simweave.reliability.entity import ReliableEntity


# ---------------------------------------------------------------------------
# RepairJob
# ---------------------------------------------------------------------------


class RepairJob(Entity):
    """A work item representing a repair or new-unit-fit operation.

    Parameters
    ----------
    owner:
        The :class:`~simweave.reliability.entity.ReliableEntity` whose
        subsystem has failed.
    subsystem_idx:
        Position of the failed subsystem in ``owner.subsystems``.
    is_new_buy:
        ``True`` if this job represents fitting a brand-new part (consumable
        failure or beyond-economic-repair).  ``False`` if it is a repair of
        the existing unit.
    return_to_stock:
        ``True`` when the repaired unit should be returned to the warehouse
        stock on job completion (only applicable to non-BER repairable items).
    repair_time:
        How long the job takes at the repair centre (simulation time units).
        Stored in ``remaining_service_time`` so the
        :class:`~simweave.discrete.services._WorkChannel` can pick it up
        without needing ``sim_properties``.
    cost:
        Financial cost charged to ``owner`` upon completion.
    """

    def __init__(
        self,
        owner: "ReliableEntity",
        subsystem_idx: int,
        is_new_buy: bool,
        return_to_stock: bool,
        repair_time: float,
        cost: float,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.owner = owner
        self.subsystem_idx = subsystem_idx
        self.is_new_buy = is_new_buy
        self.return_to_stock = return_to_stock
        self.remaining_service_time = float(repair_time)
        self.cost = float(cost)


# ---------------------------------------------------------------------------
# RepairCentre
# ---------------------------------------------------------------------------


class RepairCentre(Service):
    """A repair facility; a :class:`~simweave.discrete.services.Service` whose
    completions restore failed subsystems on
    :class:`~simweave.reliability.entity.ReliableEntity` instances.

    The centre accepts :class:`RepairJob` items in its queue.  On completion:

    1. If ``job.return_to_stock`` the repaired part is returned to the owning
       entity's warehouse (incrementing stock by one unit).
    2. The owning entity's subsystem is transitioned back to UP.
    3. Cost and counter metrics on this centre are updated.

    Parameters
    ----------
    capacity:
        Number of parallel work channels (repair bays / technicians when no
        explicit resource pool is used).
    buffer_size:
        Maximum number of jobs that can wait in the pre-repair queue.
    resources:
        Optional :class:`~simweave.discrete.resources.ResourcePool`.  Attach a
        pool of *n* technicians here to gate each repair bay on staff
        availability.  If ``None`` the centre is unconstrained by personnel.
    rng:
        Random number generator forwarded to the parent ``Service``.
    name:
        Display name.
    """

    def __init__(
        self,
        capacity: int = 1,
        buffer_size: int = 100,
        resources=None,
        rng=None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            capacity=capacity,
            buffer_size=buffer_size,
            next_q="terminus",
            resources=resources,
            rng=rng,
            name=name or "RepairCentre",
        )
        # Aggregate metrics
        self.total_cost: float = 0.0
        self.total_newbuys: int = 0
        self.total_repairs: int = 0

    # ------------------------------------------------------------------
    # Override completion hook
    # ------------------------------------------------------------------

    def _record_completion(self, item: Entity) -> None:
        super()._record_completion(item)
        if not isinstance(item, RepairJob):
            return
        job: RepairJob = item

        # Charge cost to the centre's aggregate ledger.
        self.total_cost += job.cost

        if job.is_new_buy:
            self.total_newbuys += 1
        else:
            self.total_repairs += 1

        # Return repaired unit to warehouse stock if applicable.
        if job.return_to_stock:
            sku_idx = job.owner.subsystems[job.subsystem_idx].spec.sku_index
            job.owner.warehouse.increment_by_idx(sku_idx, 1.0)

        # Restore subsystem and log cost back to the owning entity.
        job.owner._on_repair_complete(job.subsystem_idx, job.cost, job.is_new_buy)


__all__ = ["RepairJob", "RepairCentre"]
