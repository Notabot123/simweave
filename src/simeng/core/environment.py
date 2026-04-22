"""Simulation environment: clock + event queue + process registry.

A :class:`SimEnvironment` owns the authoritative clock and an optional event
queue. Processes (anything with a ``tick(dt, env)`` method) are registered so
they step together each tick. An optional ``graph`` attribute exists as a
convenience for spatial/agent simulations that need a shared topology.

Run styles
----------
* ``env.run(until)`` -- step the clock by ``dt`` until ``t >= until``.
* ``env.run(until, skip_idle_gaps=True)`` -- if nothing has work in the
  current tick and the next pending event is strictly in the future, jump the
  clock straight to that event. This degenerates to pure fixed-step when
  events are dense, and approaches DEVS efficiency when they are sparse.
* ``env.step()`` -- advance exactly one tick (useful in tests).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from simeng.core.clock import Clock
from simeng.core.scheduler import EventQueue, ScheduledEvent
from simeng.core.logging import get_logger

log = get_logger("env")


@runtime_checkable
class Process(Protocol):
    """A process that participates in the tick loop."""

    def tick(self, dt: float, env: "SimEnvironment") -> None: ...


class SimEnvironment:
    """Container for a clock, an event queue, and registered processes."""

    def __init__(
        self,
        start: float = 0.0,
        dt: float = 1.0,
        end: float | None = None,
        graph: Any = None,
    ) -> None:
        self.clock = Clock(start=start, dt=dt, end=end)
        self.events = EventQueue()
        self.graph = graph
        self._processes: list[Process] = []

    # ------------------------------------------------------------------
    # Process registry
    # ------------------------------------------------------------------
    def register(self, process: Process) -> Process:
        """Register a process so it is ticked each simulation step."""
        if not hasattr(process, "tick"):
            raise TypeError("Registered process must implement tick(dt, env).")
        self._processes.append(process)
        on_register = getattr(process, "on_register", None)
        if callable(on_register):
            on_register(self)
        return process

    def register_all(self, processes: Iterable[Process]) -> None:
        for p in processes:
            self.register(p)

    @property
    def processes(self) -> tuple[Process, ...]:
        return tuple(self._processes)

    # ------------------------------------------------------------------
    # Event scheduling helpers
    # ------------------------------------------------------------------
    def schedule_at(
        self, time: float, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> ScheduledEvent:
        return self.events.schedule(time, callback, *args, **kwargs)

    def schedule_after(
        self, delay: float, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> ScheduledEvent:
        return self.events.schedule(self.clock.t + delay, callback, *args, **kwargs)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Process events due at ``t``, tick every process, then advance."""
        for evt in self.events.pop_due(self.clock.t):
            evt.callback(*evt.args, **evt.kwargs)
        for p in self._processes:
            p.tick(self.clock.dt, self)
        self.clock.advance()

    def run(self, until: float | None = None, skip_idle_gaps: bool = False) -> None:
        """Run the simulation until ``until`` or ``clock.end``."""
        if until is None:
            until = self.clock.end
        if until is None:
            raise ValueError("Pass `until` or set clock.end before calling run().")
        if until <= self.clock.t:
            return

        while self.clock.t < until:
            if skip_idle_gaps and not self._any_has_work():
                next_evt = self.events.peek_time()
                if next_evt is not None and next_evt > self.clock.t:
                    self.clock.jump_to(min(next_evt, until))
                    continue
                # Nothing has work and no future events -> nothing will change.
                if next_evt is None:
                    log.debug(
                        "No work and no pending events; halting early at t=%s.",
                        self.clock.t,
                    )
                    break
            self.step()

    def _any_has_work(self) -> bool:
        return any(p.has_work(self) for p in self._processes if hasattr(p, "has_work"))
