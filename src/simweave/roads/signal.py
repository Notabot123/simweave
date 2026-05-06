"""Traffic signal controller.

A :class:`TrafficSignal` cycles through a list of :class:`SignalPhase`
objects, spending ``phase.duration`` simulation-time units in each phase
before advancing to the next.  The :class:`~simweave.roads.intersection.Intersection`
queries :meth:`TrafficSignal.road_is_green` each tick to decide which
approach queues to release.

Register the signal **before** any intersections it controls so that its
state is updated before vehicles are dispatched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from simweave.core.entity import Entity

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment
    from simweave.roads.road import Road


@dataclass
class SignalPhase:
    """One phase in a traffic-signal cycle.

    Parameters
    ----------
    green_roads:
        Roads whose approach queues are released during this phase.
    duration:
        Phase duration in simulation-time units.
    name:
        Optional label (e.g. ``"NS_green"``).
    """

    green_roads: list["Road"] = field(default_factory=list)
    duration: float = 30.0
    name: str = "phase"


class TrafficSignal(Entity):
    """Fixed-time traffic signal controller.

    Cycles sequentially through :class:`SignalPhase` objects.  When the
    current phase timer expires the controller advances to the next phase
    (wrapping around to the first after the last).

    Parameters
    ----------
    phases:
        Ordered list of signal phases.  Must contain at least one entry.
    name:
        Optional display name.

    Attributes
    ----------
    current_phase : SignalPhase
        The currently active phase.
    phase_timer : float
        Simulation time remaining in the current phase.
    phase_index : int
        Zero-based index of the current phase.
    """

    def __init__(
        self,
        phases: list[SignalPhase],
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not phases:
            raise ValueError("TrafficSignal requires at least one phase.")
        self.phases = list(phases)
        self._phase_idx: int = 0
        self.phase_timer: float = float(self.phases[0].duration)
        self.cycle_count: int = 0  # incremented each time we wrap back to phase 0

    @property
    def current_phase(self) -> SignalPhase:
        return self.phases[self._phase_idx]

    @property
    def phase_index(self) -> int:
        return self._phase_idx

    def road_is_green(self, road: "Road") -> bool:
        """Return ``True`` if *road* is in the currently active green set."""
        return road in self.current_phase.green_roads

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        super().tick(dt, env)
        self.phase_timer -= dt
        if self.phase_timer <= 0.0:
            next_idx = (self._phase_idx + 1) % len(self.phases)
            if next_idx == 0:
                self.cycle_count += 1
            self._phase_idx = next_idx
            self.phase_timer = float(self.phases[self._phase_idx].duration)

    def has_work(self, env: "SimEnvironment") -> bool:
        return True


__all__ = ["SignalPhase", "TrafficSignal"]
