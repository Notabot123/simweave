"""Multi-channel Service and arrival generator.

A :class:`Service` is a :class:`Queue` (the pre-service buffer) plus a set of
independent :class:`_WorkChannel` instances. Each channel processes one item
at a time; if a completed item cannot be forwarded downstream (because the
next queue is full) it sits in a "blocked-completion" slot until it clears.
"""

from __future__ import annotations

import weakref
from typing import Callable

import numpy as np

from simweave.core.entity import Entity
from simweave.core.logging import get_logger
from simweave.discrete.queues import Queue
from simweave.discrete.resources import ResourcePool

log = get_logger("discrete.service")


class _WorkChannel:
    """Internal single-item processing slot within a Service."""

    __slots__ = ("parent", "idx", "current", "blocked", "held_resource", "busy_time")

    def __init__(self, parent: "Service", idx: int) -> None:
        self.parent = parent
        self.idx = idx
        self.current: Entity | None = None
        self.blocked: Entity | None = None  # completed but next_q was full
        self.held_resource = None
        self.busy_time: float = 0.0

    def is_busy(self) -> bool:
        return self.current is not None or self.blocked is not None

    def tick(self, dt: float, env) -> None:
        # 1. Try to clear any blocked completed item first.
        if self.blocked is not None:
            if self._try_forward(self.blocked):
                self._release_resource()
                self.blocked = None

        # 2. If idle, pull new work from the buffer.
        if self.current is None and self.blocked is None:
            self._try_start(env)

        # 3. Process the current item, if any.
        if self.current is not None:
            step = min(self.current.remaining_service_time, dt)
            self.current.remaining_service_time -= step
            self.current.age += step
            self.busy_time += step
            if self.current.remaining_service_time <= 1e-12:
                item = self.current
                self.current = None
                if self._try_forward(item):
                    self._release_resource()
                else:
                    self.blocked = item

    def _try_start(self, env) -> None:
        if not self.parent._deque:
            return
        if self.parent.resources is not None:
            r = self.parent.resources.try_acquire()
            if r is None:
                return
            self.held_resource = r
        item = self.parent.dequeue()
        rng = getattr(self.parent, "rng", None)
        sim_props = getattr(item, "sim_properties", None)
        if sim_props is not None and hasattr(sim_props, "draw_service_time"):
            item.remaining_service_time = sim_props.draw_service_time(rng)
        elif item.remaining_service_time <= 0:
            item.remaining_service_time = self.parent.default_service_time
        self.current = item

    def _try_forward(self, item: Entity) -> bool:
        target = self.parent.next_q
        if target == "terminus":
            self.parent._record_completion(item)
            return True
        assert isinstance(target, Queue)
        if target.is_full:
            return False
        accepted = target.enqueue(item)
        if accepted:
            self.parent._record_completion(item)
        return accepted

    def _release_resource(self) -> None:
        if self.held_resource is not None:
            self.parent.resources.release(self.held_resource)
            self.held_resource = None


class Service(Queue):
    """Multi-channel server with optional resource pool.

    Parameters
    ----------
    capacity:
        Number of parallel work channels (servers).
    buffer_size:
        Max queue length *before* service. Further arrivals are dropped.
    next_q:
        Where to forward completed items.
    resources:
        Optional :class:`ResourcePool` from which to acquire one resource per
        item served. If ``None``, the service is unconstrained.
    default_service_time:
        Fallback when an arriving entity has no ``sim_properties``.
    rng:
        Optional numpy Generator used when drawing service times.
    """

    def __init__(
        self,
        capacity: int = 1,
        buffer_size: int = 10,
        next_q: "Queue | str" = "terminus",
        resources: ResourcePool | None = None,
        default_service_time: float = 1.0,
        rng: np.random.Generator | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(maxlen=buffer_size, name=name, next_q=next_q)
        if capacity < 1:
            raise ValueError("Service capacity must be >= 1.")
        self.capacity = capacity
        self.resources = resources
        self.default_service_time = float(default_service_time)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.channels = [_WorkChannel(self, i) for i in range(capacity)]
        self.completed_count: int = 0
        self.completed_total_time: float = (
            0.0  # sum of (wait + service) across completed
        )
        self._entry_age: "weakref.WeakKeyDictionary[Entity, float]" = (
            weakref.WeakKeyDictionary()
        )

    # ------------------------------------------------------------------
    # next_q propagation to channels is implicit: they read parent.next_q.
    # ------------------------------------------------------------------

    def tick(self, dt: float, env) -> None:
        # First tick the work channels (they pull from the buffer).
        for ch in self.channels:
            ch.tick(dt, env)
        # Then tick the buffer's own age/renege bookkeeping (parent Queue).
        super().tick(dt, env)

    def has_work(self, env) -> bool:
        return any(ch.is_busy() for ch in self.channels) or bool(self._deque)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def utilisation(self, elapsed: float) -> float:
        """Average utilisation across all channels over ``elapsed`` time."""
        if elapsed <= 0 or self.capacity == 0:
            return 0.0
        busy = sum(ch.busy_time for ch in self.channels)
        return busy / (self.capacity * elapsed)

    def average_residence(self) -> float:
        """Mean time in this service (buffer wait + service) per completed item.

        This is ``W`` in Little's law (``L = lambda * W``); for M/M/1 it is
        ``Wq + 1/mu = 1/(mu - lambda)``. Compare :meth:`Queue.average_wait`,
        which is the buffer-only wait ``Wq``.
        """
        if self.completed_count == 0:
            return 0.0
        return self.completed_total_time / self.completed_count

    def enqueue(self, item: Entity) -> bool:
        # ``age`` accrues both in the buffer and in service, so residence at
        # this stage is the age delta between entry and completion. Recording
        # the entry age keeps this correct for chained services too.
        age_in = item.age
        accepted = super().enqueue(item)
        if accepted:
            self._entry_age[item] = age_in
        return accepted

    def _record_completion(self, item: Entity) -> None:
        self.completed_count += 1
        self.completed_total_time += item.age - self._entry_age.pop(item, 0.0)


# ---------------------------------------------------------------------------
# Arrival generator
# ---------------------------------------------------------------------------


class ArrivalGenerator(Entity):
    """Generates new entities according to an inter-arrival distribution.

    On each tick it adds to an internal clock and, whenever the clock passes
    the next scheduled arrival, invokes ``factory(env)`` to mint a new
    entity and pushes it into ``target``. Multiple arrivals within a single
    tick are handled correctly.

    Parameters
    ----------
    interarrival:
        ``(rng) -> float`` gap sampler. Omit (``None``) when ``joint`` is given.
    factory:
        ``factory(env) -> Entity``; with ``joint`` it is called as
        ``factory(env, draw)`` where ``draw`` is the ``{name: value}`` dict of
        the joint draw (see :class:`~simweave.stats.JointSampler`).
    joint:
        Optional :class:`~simweave.stats.JointSampler` that draws the gap
        (under ``interarrival_key``) together with entity attributes, so they
        can be dependent. It uses a child generator seeded from ``rng``.
    interarrival_key:
        Name of the gap in the joint draw.
    gap_alignment:
        ``"before"`` (default): the attributes drawn with a gap belong to the
        entity that arrives after that gap (attribute vs. time since the last
        arrival). ``"after"``: attributes belong to the entity just created and
        the gap is the time until the next arrival.
    """

    def __init__(
        self,
        interarrival: Callable[[np.random.Generator], float] | None,
        factory: Callable[..., Entity],
        target: Queue,
        rng: np.random.Generator | None = None,
        name: str | None = None,
        *,
        joint=None,
        interarrival_key: str = "interarrival",
        gap_alignment: str = "before",
    ) -> None:
        super().__init__(name=name)
        if gap_alignment not in ("before", "after"):
            raise ValueError("gap_alignment must be 'before' or 'after'.")
        if joint is None and interarrival is None:
            raise ValueError("Provide interarrival, or a joint sampler.")
        if joint is not None and interarrival is not None:
            raise ValueError("Pass either interarrival or joint, not both.")
        if joint is not None and interarrival_key not in joint.names:
            raise ValueError(f"joint has no {interarrival_key!r} variable; has {joint.names}.")
        self.interarrival = interarrival
        self.factory = factory
        self.target = target
        self.rng = rng if rng is not None else np.random.default_rng()
        self.joint = joint
        self.interarrival_key = interarrival_key
        self.gap_alignment = gap_alignment
        self._pending: dict[str, float] | None = None
        if joint is None:
            self._countdown: float = float(self.interarrival(self.rng))
        else:
            # Dedicated stream: adding a copula must not perturb other draws on rng.
            self._joint_rng = np.random.default_rng(int(self.rng.integers(2**63 - 1)))
            if gap_alignment == "before":
                self._pending = joint.draw(self._joint_rng)
                self._countdown = float(self._pending[interarrival_key])
            else:
                first = joint.marginals[interarrival_key].rvs(self._joint_rng)
                self._countdown = float(first)
        self.generated: int = 0
        self.rejected: int = 0

    def _next(self, env) -> Entity:
        """Create the arriving entity and schedule the following arrival."""
        if self.joint is None:
            entity = self.factory(env)
            self._countdown = float(self.interarrival(self.rng))
            return entity
        if self.gap_alignment == "before":
            entity = self.factory(env, self._pending)
            self._pending = self.joint.draw(self._joint_rng)
            self._countdown = float(self._pending[self.interarrival_key])
        else:
            draw = self.joint.draw(self._joint_rng)
            entity = self.factory(env, draw)
            self._countdown = float(draw[self.interarrival_key])
        return entity

    def tick(self, dt: float, env) -> None:
        super().tick(dt, env)
        remaining = dt
        while remaining >= self._countdown:
            remaining -= self._countdown
            entity = self._next(env)
            if self.target.enqueue(entity):
                self.generated += 1
            else:
                self.rejected += 1
        self._countdown -= remaining

    def has_work(self, env) -> bool:
        # The generator always has "work" in the sense that another arrival
        # is pending.
        return True
