"""Fleet container and availability recorder.

A :class:`Fleet` is a thin wrapper around a list of
:class:`~simweave.reliability.entity.ReliableEntity` instances that provides
aggregate metrics (operational count, mean availability, total cost).

A :class:`FleetAvailabilityRecorder` registers with the environment and
snapshots fleet state at every simulation tick so the time-series can later
be fed into :func:`~simweave.viz.plots.plot_fleet_availability`.

Typical usage::

    fleet = Fleet(vehicles, name="taxi_fleet")
    recorder = FleetAvailabilityRecorder(fleet)
    env.register(recorder)
    env.run(until=365.0)

    fig = plot_fleet_availability(recorder)
    fig.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from simweave.reliability.entity import ReliableEntity
from simweave.reliability.subsystem import SubsystemState

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment


class Fleet:
    """A collection of :class:`~simweave.reliability.entity.ReliableEntity`
    instances with aggregate operational metrics.

    Parameters
    ----------
    entities:
        The vehicles / platforms that make up the fleet.
    name:
        Display name used in plot titles.
    """

    def __init__(self, entities: Sequence[ReliableEntity], name: str = "fleet") -> None:
        self.entities: list[ReliableEntity] = list(entities)
        self.name = name

    # ------------------------------------------------------------------
    # Counts / fractions
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.entities)

    @property
    def operational_count(self) -> int:
        """Number of entities that are fully operational right now."""
        return sum(1 for e in self.entities if e.is_operational)

    @property
    def operational_availability(self) -> float:
        """Instantaneous operational availability (0–1)."""
        if not self.entities:
            return 0.0
        return self.operational_count / self.size

    @property
    def mean_availability(self) -> float:
        """Mean of each entity's time-based empirical availability."""
        if not self.entities:
            return 0.0
        return sum(e.availability for e in self.entities) / self.size

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return sum(e.total_cost for e in self.entities)

    @property
    def total_newbuy_cost(self) -> float:
        return sum(e.cost_newbuy for e in self.entities)

    @property
    def total_repair_cost(self) -> float:
        return sum(e.cost_repair for e in self.entities)

    # ------------------------------------------------------------------
    # Status breakdown
    # ------------------------------------------------------------------

    def status_counts(self) -> dict[str, int]:
        """Classify every entity into one of three broad states.

        Returns
        -------
        dict with keys ``"operational"``, ``"in_repair"``, ``"awaiting_part"``.
        An entity is *awaiting_part* if any subsystem is in that state.
        An entity is *in_repair* if it has at least one subsystem IN_REPAIR
        and none AWAITING_PART.
        """
        operational = in_repair = awaiting_part = 0
        for e in self.entities:
            if e.is_operational:
                operational += 1
            elif any(
                s.state == SubsystemState.AWAITING_PART for s in e.subsystems
            ):
                awaiting_part += 1
            else:
                in_repair += 1
        return {
            "operational": operational,
            "in_repair": in_repair,
            "awaiting_part": awaiting_part,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Aggregate snapshot suitable for Monte Carlo result dicts."""
        counts = self.status_counts()
        return {
            "fleet_name": self.name,
            "fleet_size": self.size,
            "operational_count": counts["operational"],
            "operational_availability": self.operational_availability,
            "mean_entity_availability": self.mean_availability,
            "total_cost": self.total_cost,
            "total_newbuy_cost": self.total_newbuy_cost,
            "total_repair_cost": self.total_repair_cost,
        }


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class FleetAvailabilityRecorder:
    """Records fleet state at each simulation tick.

    Register with the environment *after* all
    :class:`~simweave.reliability.entity.ReliableEntity` instances so the
    snapshot captures the state *after* each tick's failures and repairs.

    Parameters
    ----------
    fleet:
        The :class:`Fleet` to monitor.

    Attributes
    ----------
    times : list[float]
        Simulation clock value at each snapshot.
    operational : list[int]
        Count of operational entities at each snapshot.
    in_repair : list[int]
        Count of entities in repair (part available) at each snapshot.
    awaiting_part : list[int]
        Count of entities waiting for parts at each snapshot.
    """

    def __init__(self, fleet: Fleet) -> None:
        self.fleet = fleet
        self.times: list[float] = []
        self.operational: list[int] = []
        self.in_repair: list[int] = []
        self.awaiting_part: list[int] = []

    def snapshot(self, t: float) -> None:
        counts = self.fleet.status_counts()
        self.times.append(t)
        self.operational.append(counts["operational"])
        self.in_repair.append(counts["in_repair"])
        self.awaiting_part.append(counts["awaiting_part"])

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        self.snapshot(env.clock.t)

    def has_work(self, env: "SimEnvironment") -> bool:
        return True

    @property
    def mean_operational_availability(self) -> float:
        """Time-averaged fraction of the fleet that was operational."""
        if not self.operational:
            return 0.0
        import numpy as np
        return float(np.mean(self.operational)) / self.fleet.size


__all__ = ["Fleet", "FleetAvailabilityRecorder"]
