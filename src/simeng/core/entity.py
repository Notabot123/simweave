"""Entity base class.

Everything that persists across ticks in simeng -- items, resources, agents,
queues, services -- inherits from :class:`Entity`. The contract is small:

* ``tick(dt, env)`` is called once per simulation tick for registered entities
  (processes). The default implementation just ages the entity.
* ``has_work(env)`` reports whether this entity has pending work *right now*;
  used by the environment's skip-idle-gaps fast path.

Items flowing through queues do not need to be registered themselves; they
ride along with whatever process owns them. Queues and services age their
contained items in their own ``tick``.
"""
from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simeng.core.environment import SimEnvironment


class Entity:
    """Base class for simulation entities."""

    _id_counter = count(0)

    def __init__(self, name: str | None = None) -> None:
        self.id: int = next(Entity._id_counter)
        self.name: str = name if name is not None else f"{type(self).__name__}_{self.id}"
        self.created_at: float | None = None
        self.age: float = 0.0
        # Queue residency bookkeeping.
        self.current_wait_time: float = 0.0
        self.total_wait_time: float = 0.0
        # Populated by a Service when pulling this entity for processing.
        self.remaining_service_time: float = 0.0

    def on_register(self, env: "SimEnvironment") -> None:
        """Hook called when this entity is registered with an environment."""
        if self.created_at is None:
            self.created_at = env.clock.t

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        """Advance this entity by ``dt`` simulation seconds.

        Subclasses should call ``super().tick(dt, env)`` first (to keep the
        age counter correct) before doing their own logic.
        """
        self.age += dt

    def has_work(self, env: "SimEnvironment") -> bool:
        """Report whether this entity has pending work.

        The default is ``False`` -- plain items waiting in a queue are not
        themselves "doing work" in the skip-idle-gaps sense; the queue/service
        holding them is.
        """
        return False

    # ------------------------------------------------------------------
    # Dunder conveniences
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, id={self.id})"

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the shared ID counter. Useful between replicates in tests."""
        cls._id_counter = count(0)
