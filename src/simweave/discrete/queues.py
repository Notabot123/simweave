"""FIFO queue and priority queue primitives.

Queues are :class:`~simweave.core.entity.Entity` subclasses so they register
with the environment and receive ``tick(dt, env)``. During a tick a queue
ages the items it contains. Balking and reneging (via
:class:`EntityProperties <simweave.discrete.properties.EntityProperties>`) are
handled on :meth:`enqueue` and :meth:`tick` respectively.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from simweave.core.entity import Entity
from simweave.core.logging import get_logger

log = get_logger("discrete.queue")


class Queue(Entity):
    """Bounded FIFO queue.

    Parameters
    ----------
    maxlen:
        Maximum number of items held. Extra arrivals are dropped (and the
        drop counter is incremented).
    name:
        Human-readable name.
    next_q:
        Where forwarded items go -- another Queue/Service or the sentinel
        string ``"terminus"`` meaning "remove from the system".
    """

    def __init__(
        self,
        maxlen: int = 10,
        name: str | None = None,
        next_q: "Queue | str" = "terminus",
    ) -> None:
        super().__init__(name=name)
        if maxlen <= 0:
            raise ValueError("Queue maxlen must be positive.")
        self._deque: deque[Entity] = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self._next_q: "Queue | str" = "terminus"
        self.next_q = next_q  # validate via setter
        self.dropped_count: int = 0
        self.reneged_count: int = 0
        self.balked_count: int = 0
        # Metrics accumulated over the lifetime of the queue.
        self.cumulative_length_time: float = 0.0  # integral of len(q) * dt
        self.cumulative_wait_time: float = 0.0  # sum of total_wait_time on departure
        self.arrivals: int = 0
        self.departures: int = 0

    # ------------------------------------------------------------------
    # next_q management
    # ------------------------------------------------------------------
    @property
    def next_q(self) -> "Queue | str":
        return self._next_q

    @next_q.setter
    def next_q(self, value: "Queue | str") -> None:
        if value == "terminus" or isinstance(value, Queue):
            self._next_q = value
        else:
            raise TypeError(
                f"next_q must be a Queue or 'terminus'; got {type(value).__name__}"
            )

    # ------------------------------------------------------------------
    # Basic container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._deque)

    def __iter__(self):
        return iter(self._deque)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        nq = self._next_q if isinstance(self._next_q, str) else self._next_q.name
        return f"{type(self).__name__}(name={self.name!r}, len={len(self)}/{self.maxlen}, next={nq})"

    @property
    def is_full(self) -> bool:
        return len(self._deque) >= self.maxlen

    @property
    def is_empty(self) -> bool:
        return len(self._deque) == 0

    def peek(self) -> Entity | None:
        return self._deque[0] if self._deque else None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def enqueue(self, item: Entity) -> bool:
        """Attempt to append an item. Returns ``True`` on success.

        Honours ``item.sim_properties.balk_on_length`` if present.
        """
        sim_props = getattr(item, "sim_properties", None)
        balk = (
            getattr(sim_props, "balk_on_length", None)
            if sim_props is not None
            else None
        )
        if balk is not None and len(self._deque) >= balk:
            self.balked_count += 1
            log.debug(
                "%s: %s balked (len %d >= %d).",
                self.name,
                item.name,
                len(self._deque),
                balk,
            )
            return False

        if self.is_full:
            self.dropped_count += 1
            log.debug("%s: dropped %s (full).", self.name, item.name)
            return False

        item.current_wait_time = 0.0
        self._deque.append(item)
        self.arrivals += 1
        return True

    def dequeue(self) -> Entity:
        if not self._deque:
            raise IndexError(f"{self.name}: dequeue from empty queue")
        item = self._deque.popleft()
        self.departures += 1
        self.cumulative_wait_time += item.total_wait_time
        return item

    def renege_tail(self) -> Entity:
        if not self._deque:
            raise IndexError(f"{self.name}: renege from empty queue")
        item = self._deque.pop()
        self.reneged_count += 1
        return item

    def forward(self, override_target: "Queue | str | None" = None) -> bool:
        """Dequeue head and enqueue into ``override_target`` (or ``self.next_q``).

        Returns ``True`` if forwarded, ``False`` if downstream is blocked (in
        which case the head item is *not* removed).
        """
        target = override_target if override_target is not None else self._next_q
        if not self._deque:
            return False
        if target == "terminus":
            self.dequeue()
            return True
        assert isinstance(target, Queue)
        if target.is_full:
            return False
        item = self._deque.popleft()
        self.departures += 1
        self.cumulative_wait_time += item.total_wait_time
        accepted = target.enqueue(item)
        if not accepted:
            # Shouldn't normally happen because we checked is_full, but handle
            # balking on the downstream queue: silently drop.
            self.dropped_count += 1
        return True

    # ------------------------------------------------------------------
    # Tick-loop integration
    # ------------------------------------------------------------------
    def tick(self, dt: float, env) -> None:
        super().tick(dt, env)
        # Time-integrate length for Little's law calculations.
        self.cumulative_length_time += len(self._deque) * dt
        # Age waiting items and handle reneging.
        to_renege: list[Entity] = []
        for item in list(self._deque):
            item.age += dt
            item.current_wait_time += dt
            item.total_wait_time += dt
            renege_after = None
            sim_props = getattr(item, "sim_properties", None)
            if sim_props is not None:
                renege_after = getattr(sim_props, "renege_after", None)
            if renege_after is not None and item.current_wait_time >= renege_after:
                to_renege.append(item)
        for item in to_renege:
            try:
                self._deque.remove(item)
                self.reneged_count += 1
                log.debug(
                    "%s: %s reneged after %.2f.",
                    self.name,
                    item.name,
                    item.current_wait_time,
                )
            except ValueError:
                pass

    def has_work(self, env) -> bool:
        # A plain queue does not perform work; items just wait.
        return False

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def average_length(self, elapsed: float) -> float:
        """Mean queue length over the elapsed simulation time (L in Little's law)."""
        if elapsed <= 0:
            return 0.0
        return self.cumulative_length_time / elapsed

    def average_wait(self) -> float:
        """Mean residence time of completed items (W in Little's law)."""
        if self.departures == 0:
            return 0.0
        return self.cumulative_wait_time / self.departures


# ---------------------------------------------------------------------------
# Priority queue
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _PItem:
    priority: float
    seq: int
    item: Any = field(compare=False)


class PriorityQueue(Queue):
    """Min-heap priority queue. Lower priority values dequeue first.

    Items are wrapped in ``(priority, seq, item)`` tuples internally so the
    payload doesn't need to be comparable.
    """

    def __init__(
        self,
        maxlen: int = 10,
        name: str | None = None,
        next_q: "Queue | str" = "terminus",
    ) -> None:
        super().__init__(maxlen=maxlen, name=name, next_q=next_q)
        self._heap: list[_PItem] = []
        self._seq_counter = 0
        # Drop the inherited deque; we store in _heap. We keep len() consistent
        # by overriding __len__.

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self):
        return (p.item for p in self._heap)

    def peek(self) -> Entity | None:
        return self._heap[0].item if self._heap else None

    @property
    def is_full(self) -> bool:
        return len(self._heap) >= self.maxlen

    @property
    def is_empty(self) -> bool:
        return not self._heap

    def enqueue(self, item: Entity, priority: float = 0.0) -> bool:  # type: ignore[override]
        if self.is_full:
            self.dropped_count += 1
            return False
        item.current_wait_time = 0.0
        heapq.heappush(
            self._heap, _PItem(priority=priority, seq=self._seq_counter, item=item)
        )
        self._seq_counter += 1
        self.arrivals += 1
        return True

    def dequeue(self) -> Entity:  # type: ignore[override]
        if not self._heap:
            raise IndexError(f"{self.name}: dequeue from empty priority queue")
        wrapped = heapq.heappop(self._heap)
        self.departures += 1
        self.cumulative_wait_time += wrapped.item.total_wait_time
        return wrapped.item

    def forward(self, override_target=None) -> bool:  # type: ignore[override]
        target = override_target if override_target is not None else self._next_q
        if not self._heap:
            return False
        if target == "terminus":
            self.dequeue()
            return True
        assert isinstance(target, Queue)
        if target.is_full:
            return False
        item = self.dequeue()
        target.enqueue(item)
        return True

    def tick(self, dt: float, env) -> None:  # type: ignore[override]
        # Skip Queue.tick deque iteration; operate on our heap.
        Entity.tick(self, dt, env)
        self.cumulative_length_time += len(self._heap) * dt
        for wrapper in self._heap:
            item = wrapper.item
            item.age += dt
            item.current_wait_time += dt
            item.total_wait_time += dt
