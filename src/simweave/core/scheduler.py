"""Optional heap-based event queue.

The atomic-clock core ticks every process each timestep. Sometimes that's
wasteful -- e.g. a warehouse that reorders once a week does not want to be
polled 10080 times in between. The :class:`EventQueue` lets processes post
callbacks to fire at (or after) a given simulation time. The environment
drains any events whose scheduled time has been reached at the start of each
tick.

When combined with ``SimEnvironment.run(skip_idle_gaps=True)``, this also
powers a DEVS-style fast path: if every registered process reports
``has_work() is False`` and an event is pending at ``t_next``, the clock
fast-forwards directly to ``t_next`` rather than grinding through empty
ticks.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class ScheduledEvent:
    """Heap-ordered event record.

    ``seq`` is a tiebreaker so that two events at the same ``time`` fire in
    insertion order, giving deterministic behaviour for simultaneous events.
    """

    time: float
    seq: int
    callback: Callable[..., Any] = field(compare=False)
    args: tuple = field(compare=False, default=())
    kwargs: dict = field(compare=False, default_factory=dict)
    cancelled: bool = field(compare=False, default=False)


class EventQueue:
    """Min-heap priority queue of scheduled callbacks."""

    def __init__(self) -> None:
        self._heap: list[ScheduledEvent] = []
        self._seq: int = 0

    def __len__(self) -> int:
        return sum(1 for e in self._heap if not e.cancelled)

    def __bool__(self) -> bool:
        return len(self) > 0

    def schedule(
        self, time: float, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> ScheduledEvent:
        """Schedule ``callback(*args, **kwargs)`` to fire at simulation ``time``."""
        evt = ScheduledEvent(
            time=float(time), seq=self._seq, callback=callback, args=args, kwargs=kwargs
        )
        heapq.heappush(self._heap, evt)
        self._seq += 1
        return evt

    def peek_time(self) -> float | None:
        """Return the time of the next non-cancelled event, or ``None``."""
        self._drop_cancelled()
        return self._heap[0].time if self._heap else None

    def pop_due(self, now: float):
        """Yield every non-cancelled event whose time is ``<= now``."""
        self._drop_cancelled()
        while self._heap and self._heap[0].time <= now:
            evt = heapq.heappop(self._heap)
            if evt.cancelled:
                continue
            yield evt
            self._drop_cancelled()

    def cancel(self, event: ScheduledEvent) -> None:
        """Flag an event as cancelled without paying heap-removal cost."""
        event.cancelled = True

    def _drop_cancelled(self) -> None:
        while self._heap and self._heap[0].cancelled:
            heapq.heappop(self._heap)
