"""SimTimeAxis -- map simulation ticks to real-world calendar dates.

Simulation clocks are dimensionless floats.  A tick of ``dt=1.0`` might mean
one second, one day, one month, or any other real-world interval depending on
the modelled scenario.  :class:`SimTimeAxis` makes that mapping explicit and
lets every plot helper substitute calendar dates for bare numbers on the
x-axis.

Supported tick units
--------------------
``"seconds"``, ``"minutes"``, ``"hours"``, ``"days"``, ``"weeks"``
are exact (``datetime.timedelta``-based).  ``"months"`` (≈ 30.4375 days)
and ``"years"`` (≈ 365.25 days) are mean Gregorian approximations,
which is the standard for planning-horizon simulations.

Basic usage::

    from simweave.core.time_axis import SimTimeAxis

    # 1 tick = 1 day, simulation starts 1 January 2027
    tax = SimTimeAxis(start="2027-01-01", tick_unit="days")

    # Convert a scalar
    tax.to_datetime(30.0)        # datetime(2027, 1, 31)
    tax.label(30.0)              # "2027-01-31"

    # Convert an array ready for a plotly x-axis
    tax.to_datetimes([0, 7, 14, 30])

    # Post-hoc: rewrite all numeric x-data in an existing figure
    fig = plot_fleet_availability(recorder)
    tax.apply_to_figure(fig)
    fig.show()

Plot-helper integration::

    fig = plot_fleet_availability(recorder, time_axis=tax)
    fig.show()
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECONDS_PER_UNIT: dict[str, float] = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3_600.0,
    "days": 86_400.0,
    "weeks": 7 * 86_400.0,
    "months": 30.4375 * 86_400.0,   # mean Gregorian month
    "years": 365.25 * 86_400.0,     # mean Gregorian year
}

_DEFAULT_FMT: dict[str, str] = {
    "seconds": "%Y-%m-%d %H:%M:%S",
    "minutes": "%Y-%m-%d %H:%M",
    "hours":   "%Y-%m-%d %H:%M",
    "days":    "%Y-%m-%d",
    "weeks":   "%d %b %Y",
    "months":  "%b %Y",
    "years":   "%Y",
}


# ---------------------------------------------------------------------------
# SimTimeAxis
# ---------------------------------------------------------------------------


class SimTimeAxis:
    """Map simulation tick time to real-world calendar dates.

    Parameters
    ----------
    start:
        Calendar date/time that corresponds to simulation ``t = 0``.
        Strings are parsed via :meth:`datetime.fromisoformat`; a bare date
        string such as ``"2027-01-01"`` works on Python 3.7+.
    tick_unit:
        Duration of **one simulation time unit**.  Must be one of
        ``"seconds"``, ``"minutes"``, ``"hours"``, ``"days"``,
        ``"weeks"``, ``"months"``, or ``"years"``.
    tick_size:
        Scale factor: how many ``tick_unit`` durations one simulation unit
        represents.  Defaults to ``1.0``.  For example, setting
        ``tick_unit="hours"`` and ``tick_size=4`` makes each sim unit
        equal to 4 hours.
    date_format:
        :func:`strftime` format string for :meth:`label` and :meth:`labels`.
        Defaults to a sensible format for the chosen ``tick_unit``.

    Examples
    --------
    >>> tax = SimTimeAxis("2027-06-01", tick_unit="weeks")
    >>> tax.label(4.0)
    '29 Jun 2027'
    """

    def __init__(
        self,
        start: str | datetime,
        tick_unit: str = "days",
        tick_size: float = 1.0,
        date_format: str | None = None,
    ) -> None:
        if isinstance(start, str):
            # Accept bare dates ("2027-01-01") and full ISO strings.
            start = datetime.fromisoformat(start)
        self.start: datetime = start

        tick_unit = tick_unit.lower()
        if tick_unit not in _SECONDS_PER_UNIT:
            raise ValueError(
                f"tick_unit {tick_unit!r} not recognised. "
                f"Choose one of: {sorted(_SECONDS_PER_UNIT)}"
            )
        self.tick_unit: str = tick_unit
        self.tick_size: float = float(tick_size)
        self._secs_per_tick: float = _SECONDS_PER_UNIT[tick_unit] * self.tick_size
        self.date_format: str = date_format or _DEFAULT_FMT[tick_unit]

    # ------------------------------------------------------------------
    # Core converters
    # ------------------------------------------------------------------

    def to_datetime(self, t: float) -> datetime:
        """Convert a scalar simulation time to a :class:`~datetime.datetime`."""
        return self.start + timedelta(seconds=float(t) * self._secs_per_tick)

    def to_datetimes(self, times: Sequence[float] | np.ndarray) -> list[datetime]:
        """Convert an iterable of simulation times to :class:`~datetime.datetime` objects.

        The returned list is suitable for Plotly's ``x`` parameter; Plotly
        renders datetime objects natively with automatic axis formatting.
        """
        return [self.to_datetime(float(t)) for t in times]

    def label(self, t: float) -> str:
        """Format a simulation time as a human-readable date string."""
        return self.to_datetime(t).strftime(self.date_format)

    def labels(self, times: Sequence[float] | np.ndarray) -> list[str]:
        """Vectorised :meth:`label`."""
        return [self.label(float(t)) for t in times]

    # ------------------------------------------------------------------
    # Plotly figure integration
    # ------------------------------------------------------------------

    def apply_to_figure(
        self,
        fig: Any,
        axis: str = "x",
        title: str | None = None,
    ) -> Any:
        """Replace numeric tick values on ``axis`` with calendar dates.

        Iterates through every trace in ``fig.data`` and, wherever the
        nominated axis data is a numeric array, substitutes
        :class:`~datetime.datetime` objects.  Plotly then renders the axis
        as a date axis with its own smart tick-label formatter.

        Parameters
        ----------
        fig:
            A ``plotly.graph_objects.Figure``.
        axis:
            ``"x"`` (default) or ``"y"`` -- which axis to reformat.
        title:
            Optional replacement axis title.  If ``None``, the existing
            title is kept (or set to ``"date"`` if it was empty).

        Returns
        -------
        The same figure object (modified in-place), so calls can be chained::

            fig = plot_fleet_availability(rec)
            fig = time_axis.apply_to_figure(fig, title="Calendar date")
            fig.show()
        """
        attr = axis  # "x" or "y"
        for trace in fig.data:
            data = getattr(trace, attr, None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.dtype.kind in ("f", "i", "u"):
                setattr(trace, attr, self.to_datetimes(arr))

        # Update the layout axis title.
        layout_axis = f"{axis}axis"
        existing = getattr(fig.layout, layout_axis, None)
        current_title = ""
        if existing is not None:
            t_obj = getattr(existing, "title", None)
            if t_obj is not None:
                current_title = getattr(t_obj, "text", "") or ""
        new_title = title if title is not None else (current_title or "date")
        fig.update_layout(**{layout_axis: {"title": new_title}})

        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def tick_for_date(self, dt: str | datetime) -> float:
        """Return the simulation tick corresponding to a given calendar date.

        Useful for scheduling events at specific real-world dates::

            t_start = tax.tick_for_date("2027-03-01")
            env.schedule_at(t_start, my_callback)
        """
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        delta_secs = (dt - self.start).total_seconds()
        return delta_secs / self._secs_per_tick

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SimTimeAxis(start={self.start.isoformat()!r}, "
            f"tick_unit={self.tick_unit!r}, tick_size={self.tick_size})"
        )


__all__ = ["SimTimeAxis"]
