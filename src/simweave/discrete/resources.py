"""Reusable resources and resource pools.

A :class:`Resource` is an :class:`Entity` that a :class:`Service` may acquire
before serving an item and release afterwards. Resources are held in a
:class:`ResourcePool`, which is itself a :class:`Queue` so it enjoys the
same tick semantics and bookkeeping.
"""

from __future__ import annotations

from simweave.core.entity import Entity
from simweave.discrete.queues import Queue


class Resource(Entity):
    """A reusable resource that can be checked out of a pool."""

    def __init__(
        self, name: str | None = None, home_pool: "ResourcePool | str" = "terminus"
    ) -> None:
        super().__init__(name=name)
        self.home_pool: "ResourcePool | str" = home_pool
        self.times_acquired: int = 0
        self.busy_time: float = 0.0
        self._is_busy: bool = False

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    def release(self, to: "ResourcePool | None" = None) -> None:
        """Return the resource to ``to`` or its home pool."""
        target = to if to is not None else self.home_pool
        self._is_busy = False
        if isinstance(target, ResourcePool):
            target.deposit(self)
        elif target == "terminus":
            return
        else:
            raise TypeError(
                "Resource.release target must be a ResourcePool or 'terminus'."
            )

    def tick(self, dt: float, env) -> None:
        super().tick(dt, env)
        if self._is_busy:
            self.busy_time += dt


class ResourcePool(Queue):
    """Pool of interchangeable resources that a Service can check out."""

    def __init__(self, maxlen: int = 10, name: str | None = None) -> None:
        super().__init__(maxlen=maxlen, name=name, next_q="terminus")

    def deposit(self, resource: Resource) -> None:
        """Return a resource to the pool. Raises if the pool is at capacity."""
        if not isinstance(resource, Resource):
            raise TypeError("ResourcePool only accepts Resource instances.")
        resource.home_pool = self
        if self.is_full:
            raise RuntimeError(f"{self.name}: pool full when returning {resource.name}")
        self._deque.append(resource)
        self.arrivals += 1

    # Alias for legacy naming.
    add_resource = deposit

    def try_acquire(self) -> Resource | None:
        """Non-blocking acquire. Returns a Resource or ``None`` if empty."""
        if not self._deque:
            return None
        resource = self._deque.popleft()
        assert isinstance(resource, Resource)
        resource.home_pool = self
        resource._is_busy = True
        resource.times_acquired += 1
        self.departures += 1
        return resource

    def release(self, resource: Resource) -> None:
        """Release a previously acquired resource back to this pool."""
        resource.release(to=self)

    def tick(self, dt: float, env) -> None:
        super().tick(dt, env)
        # Propagate tick to each idle resource so busy_time stays accurate
        # when they are NOT in the pool (those will be ticked by the Service).
        # Idle resources sitting here don't accrue busy_time.
