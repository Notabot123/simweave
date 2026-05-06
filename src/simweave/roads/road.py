"""Road segments -- event-driven conveyor-belt travel model.

A :class:`Road` is a free-flow segment: vehicles enter at one end and are
delivered to an outlet (intersection, roundabout, or the system boundary)
after ``length / effective_speed`` simulation-time units.  Multiple vehicles
travel simultaneously; there is no blocking or overtaking in this model.

:class:`DualCarriageway` is a convenience wrapper around two opposing
:class:`Road` instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simweave.core.entity import Entity

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment
    from simweave.roads.vehicle import Vehicle


class Road(Entity):
    """Free-flow road segment.

    Vehicles enter at one end via :meth:`enter` and are asynchronously
    delivered to the :attr:`outlet` after ``length / effective_speed``
    simulation-time units.  The outlet must expose an
    ``arrive(vehicle, from_road, env)`` method (implemented by
    :class:`~simweave.roads.intersection.Intersection` and
    :class:`~simweave.roads.roundabout.Roundabout`).

    Parameters
    ----------
    length:
        Road length in metres (or whatever distance unit your scenario uses).
    speed_limit:
        Free-flow speed in m/s.  Travel time = ``length / speed_limit``.
    lanes:
        Number of lanes (informational; does not limit throughput in the
        free-flow model but is exposed for recorders and visualisation).
    outlet:
        Where vehicles go after traversing this road.  Pass ``None`` for a
        terminal road (vehicles exit the system silently).
    name:
        Optional display name.

    Attributes
    ----------
    in_transit : int
        Number of vehicles currently travelling on this road.
    total_entered : int
        Cumulative vehicles that have entered this road.
    total_exited : int
        Cumulative vehicles that have completed this road.
    """

    def __init__(
        self,
        length: float,
        speed_limit: float,
        lanes: int = 1,
        outlet: Any = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if length <= 0:
            raise ValueError("Road length must be positive.")
        if speed_limit <= 0:
            raise ValueError("speed_limit must be positive.")
        self.length = float(length)
        self.speed_limit = float(speed_limit)
        self.lanes = int(lanes)
        self._outlet = outlet
        self.in_transit: int = 0
        self.total_entered: int = 0
        self.total_exited: int = 0

    # ------------------------------------------------------------------
    # Outlet management
    # ------------------------------------------------------------------

    @property
    def outlet(self) -> Any:
        """Where vehicles go after traversing this road."""
        return self._outlet

    @outlet.setter
    def outlet(self, value: Any) -> None:
        self._outlet = value

    # ------------------------------------------------------------------
    # Travel time
    # ------------------------------------------------------------------

    def travel_time(self, vehicle: "Vehicle | None" = None) -> float:
        """Return travel time in simulation-time units.

        Vehicles are capped at the speed limit; slower vehicles use their
        own speed.
        """
        spd = self.speed_limit
        if vehicle is not None and vehicle.speed is not None:
            spd = min(vehicle.speed, self.speed_limit)
        return self.length / spd

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def enter(self, vehicle: "Vehicle", env: "SimEnvironment") -> bool:
        """Place a vehicle onto this road.

        Schedules a delivery event after ``travel_time(vehicle)`` simulation
        units.  Returns ``True`` always (free-flow roads have no capacity
        limit).
        """
        tt = self.travel_time(vehicle)
        self.in_transit += 1
        self.total_entered += 1

        def _deliver(
            _v: "Vehicle" = vehicle,
            _road: "Road" = self,
            _env: "SimEnvironment" = env,
            _tt: float = tt,
        ) -> None:
            _road.in_transit -= 1
            _road.total_exited += 1
            _v.roads_traversed += 1
            _v.total_travel_time += _tt
            if _road._outlet is not None:
                _road._outlet.arrive(_v, from_road=_road, env=_env)
            # else vehicle exits the system silently

        env.schedule_after(tt, _deliver)
        return True

    def tick(self, dt: float, env: "SimEnvironment") -> None:
        super().tick(dt, env)

    def has_work(self, env: "SimEnvironment") -> bool:
        return self.in_transit > 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Road(name={self.name!r}, length={self.length}m, "
            f"speed={self.speed_limit}m/s, lanes={self.lanes}, "
            f"in_transit={self.in_transit})"
        )


class DualCarriageway:
    """Two opposing :class:`Road` instances sharing the same corridor.

    Models a dual-carriageway (divided highway) where traffic flows in
    both directions simultaneously.  Each direction is an independent
    :class:`Road` with its own outlet and arrival process — simply wire
    them up separately.

    Parameters
    ----------
    length:
        Carriageway length (m) — shared by both directions.
    speed_limit:
        Free-flow speed (m/s) — shared by both directions.
    lanes_each:
        Lanes per direction.  Default 1.
    name:
        Base name.  ``"_forward"`` / ``"_backward"`` suffixes are appended.
    """

    def __init__(
        self,
        length: float,
        speed_limit: float,
        lanes_each: int = 1,
        name: str = "dual_carriageway",
    ) -> None:
        self.name = name
        self.forward = Road(
            length=length,
            speed_limit=speed_limit,
            lanes=lanes_each,
            name=f"{name}_forward",
        )
        self.backward = Road(
            length=length,
            speed_limit=speed_limit,
            lanes=lanes_each,
            name=f"{name}_backward",
        )

    @property
    def roads(self) -> tuple[Road, Road]:
        """Both constituent roads as ``(forward, backward)``."""
        return (self.forward, self.backward)

    def __repr__(self) -> str:  # pragma: no cover
        return f"DualCarriageway(name={self.name!r})"


__all__ = ["Road", "DualCarriageway"]
