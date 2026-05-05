"""Unit tests for SimTimeAxis.

Coverage
--------
* Construction -- valid tick_unit values, string and datetime start,
  tick_size scaling, bad tick_unit rejection.
* to_datetime / to_datetimes -- correct offsets for every tick_unit.
* label / labels -- strftime formatting.
* tick_for_date -- inverse of to_datetime.
* apply_to_figure -- numeric x-values replaced with datetime objects,
  axis title updated.
* plot helper integration -- time_axis= kwarg accepted by all relevant
  plot helpers without error.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from simweave.core.time_axis import SimTimeAxis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

START_STR = "2027-01-01"
START_DT = datetime(2027, 1, 1)


@pytest.fixture
def daily() -> SimTimeAxis:
    return SimTimeAxis(start=START_STR, tick_unit="days")


@pytest.fixture
def weekly() -> SimTimeAxis:
    return SimTimeAxis(start=START_STR, tick_unit="weeks")


@pytest.fixture
def hourly() -> SimTimeAxis:
    return SimTimeAxis(start=START_STR, tick_unit="hours")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_string_start_parsed(self, daily):
        assert daily.start == START_DT

    def test_datetime_start_stored_directly(self):
        tax = SimTimeAxis(start=START_DT, tick_unit="days")
        assert tax.start == START_DT

    def test_default_tick_unit_is_days(self):
        tax = SimTimeAxis(start=START_STR)
        assert tax.tick_unit == "days"

    def test_tick_size_stored(self):
        tax = SimTimeAxis(start=START_STR, tick_unit="hours", tick_size=4.0)
        assert tax.tick_size == pytest.approx(4.0)

    def test_invalid_tick_unit_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            SimTimeAxis(start=START_STR, tick_unit="fortnights")

    def test_custom_date_format_stored(self):
        tax = SimTimeAxis(start=START_STR, tick_unit="days", date_format="%d/%m/%Y")
        assert tax.date_format == "%d/%m/%Y"

    @pytest.mark.parametrize("unit", [
        "seconds", "minutes", "hours", "days", "weeks", "months", "years"
    ])
    def test_all_valid_units_accepted(self, unit):
        tax = SimTimeAxis(start=START_STR, tick_unit=unit)
        assert tax.tick_unit == unit


# ---------------------------------------------------------------------------
# to_datetime
# ---------------------------------------------------------------------------


class TestToDatetime:
    def test_zero_tick_returns_start(self, daily):
        assert daily.to_datetime(0.0) == START_DT

    def test_one_day_tick(self, daily):
        assert daily.to_datetime(1.0) == START_DT + timedelta(days=1)

    def test_thirty_day_ticks(self, daily):
        assert daily.to_datetime(30.0) == START_DT + timedelta(days=30)

    def test_one_week_tick(self, weekly):
        assert weekly.to_datetime(1.0) == START_DT + timedelta(weeks=1)

    def test_four_week_ticks(self, weekly):
        assert weekly.to_datetime(4.0) == START_DT + timedelta(weeks=4)

    def test_one_hour_tick(self, hourly):
        assert hourly.to_datetime(1.0) == START_DT + timedelta(hours=1)

    def test_tick_size_scaling(self):
        """tick_size=4 with hours means 1 sim unit = 4 real hours."""
        tax = SimTimeAxis(start=START_STR, tick_unit="hours", tick_size=4.0)
        assert tax.to_datetime(1.0) == START_DT + timedelta(hours=4)
        assert tax.to_datetime(3.0) == START_DT + timedelta(hours=12)

    def test_fractional_tick(self, daily):
        result = daily.to_datetime(0.5)
        assert result == START_DT + timedelta(hours=12)

    def test_seconds_unit(self):
        tax = SimTimeAxis(start=START_STR, tick_unit="seconds")
        assert tax.to_datetime(3600.0) == START_DT + timedelta(hours=1)

    def test_minutes_unit(self):
        tax = SimTimeAxis(start=START_STR, tick_unit="minutes")
        assert tax.to_datetime(60.0) == START_DT + timedelta(hours=1)

    def test_months_approximate(self):
        """months uses 30.4375-day approximation; just check it's in the right ballpark."""
        tax = SimTimeAxis(start=START_STR, tick_unit="months")
        result = tax.to_datetime(1.0)
        # Should be approximately 30-31 days after start.
        delta = (result - START_DT).total_seconds() / 86400
        assert 30.0 < delta < 31.0

    def test_years_approximate(self):
        tax = SimTimeAxis(start=START_STR, tick_unit="years")
        result = tax.to_datetime(1.0)
        delta = (result - START_DT).total_seconds() / 86400
        assert 365.0 < delta < 366.0


# ---------------------------------------------------------------------------
# to_datetimes (vectorised)
# ---------------------------------------------------------------------------


class TestToDatetimes:
    def test_returns_list_of_datetimes(self, daily):
        result = daily.to_datetimes([0.0, 1.0, 2.0])
        assert isinstance(result, list)
        assert all(isinstance(d, datetime) for d in result)

    def test_correct_length(self, daily):
        result = daily.to_datetimes(range(10))
        assert len(result) == 10

    def test_correct_values(self, daily):
        result = daily.to_datetimes([0.0, 7.0, 14.0])
        assert result[0] == START_DT
        assert result[1] == START_DT + timedelta(weeks=1)
        assert result[2] == START_DT + timedelta(weeks=2)

    def test_accepts_numpy_array(self, daily):
        arr = np.arange(5, dtype=float)
        result = daily.to_datetimes(arr)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# label / labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_label_zero(self, daily):
        assert daily.label(0.0) == "2027-01-01"

    def test_label_thirty_days(self, daily):
        # 30 days after 1 Jan = 31 Jan
        assert daily.label(30.0) == "2027-01-31"

    def test_labels_list(self, daily):
        result = daily.labels([0.0, 30.0])
        assert result == ["2027-01-01", "2027-01-31"]

    def test_custom_format(self):
        tax = SimTimeAxis(start=START_STR, tick_unit="days", date_format="%d/%m/%Y")
        assert tax.label(0.0) == "01/01/2027"

    def test_months_default_format(self):
        tax = SimTimeAxis(start=START_STR, tick_unit="months")
        lbl = tax.label(0.0)
        assert "Jan" in lbl or "01" in lbl  # month is represented

    def test_years_default_format(self):
        tax = SimTimeAxis(start=START_STR, tick_unit="years")
        assert "2027" in tax.label(0.0)


# ---------------------------------------------------------------------------
# tick_for_date (inverse)
# ---------------------------------------------------------------------------


class TestTickForDate:
    def test_start_date_returns_zero(self, daily):
        assert daily.tick_for_date(START_STR) == pytest.approx(0.0)

    def test_one_week_ahead(self, daily):
        assert daily.tick_for_date("2027-01-08") == pytest.approx(7.0)

    def test_datetime_input(self, daily):
        dt = datetime(2027, 2, 1)
        # 31 days after 1 Jan
        assert daily.tick_for_date(dt) == pytest.approx(31.0)

    def test_roundtrip(self, daily):
        """to_datetime(tick_for_date(d)) should reproduce d (for exact units)."""
        target = datetime(2027, 6, 15)
        t = daily.tick_for_date(target)
        recovered = daily.to_datetime(t)
        assert recovered == target

    def test_weekly_roundtrip(self, weekly):
        target = datetime(2027, 4, 12)  # 14 weeks after Jan 1
        t = weekly.tick_for_date(target)
        recovered = weekly.to_datetime(t)
        # Weeks are exact so roundtrip should be exact.
        assert recovered == target


# ---------------------------------------------------------------------------
# apply_to_figure
# ---------------------------------------------------------------------------


class TestApplyToFigure:
    """These tests use a minimal mock figure so plotly is not required."""

    class _MockTrace:
        def __init__(self, x):
            self.x = x
            self.y = list(range(len(x)))

    class _MockAxis:
        def __init__(self):
            self.title = type("T", (), {"text": "time"})()

    class _MockLayout:
        def __init__(self):
            self.xaxis = TestApplyToFigure._MockAxis()

        def __getattr__(self, name):
            if name == "xaxis":
                return self.__dict__.get("xaxis", TestApplyToFigure._MockAxis())
            raise AttributeError(name)

    class _MockFig:
        def __init__(self, *traces):
            self.data = list(traces)
            self.layout = TestApplyToFigure._MockLayout()
            self._updates: list[dict] = []

        def update_layout(self, **kwargs):
            self._updates.append(kwargs)

    def test_numeric_x_replaced_with_datetimes(self, daily):
        trace = self._MockTrace([0.0, 1.0, 2.0])
        fig = self._MockFig(trace)
        daily.apply_to_figure(fig)
        assert all(isinstance(d, datetime) for d in fig.data[0].x)

    def test_correct_datetime_values(self, daily):
        trace = self._MockTrace([0.0, 7.0])
        fig = self._MockFig(trace)
        daily.apply_to_figure(fig)
        assert fig.data[0].x[0] == START_DT
        assert fig.data[0].x[1] == START_DT + timedelta(weeks=1)

    def test_layout_title_updated(self, daily):
        trace = self._MockTrace([0.0])
        fig = self._MockFig(trace)
        daily.apply_to_figure(fig, title="Calendar date")
        assert any("xaxis" in u for u in fig._updates)

    def test_returns_same_figure(self, daily):
        trace = self._MockTrace([0.0])
        fig = self._MockFig(trace)
        returned = daily.apply_to_figure(fig)
        assert returned is fig

    def test_non_numeric_x_untouched(self, daily):
        """Traces with string x values should not be modified."""
        trace = self._MockTrace.__new__(self._MockTrace)
        trace.x = ["a", "b", "c"]
        trace.y = [1, 2, 3]
        fig = self._MockFig(trace)
        daily.apply_to_figure(fig)
        # Should remain strings.
        assert fig.data[0].x == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# plot helper integration (import-only; no plotly required)
# ---------------------------------------------------------------------------


def test_simtimeaxis_importable_from_simweave():
    import simweave as sw
    assert hasattr(sw, "SimTimeAxis")
    tax = sw.SimTimeAxis("2027-01-01", tick_unit="weeks")
    assert tax.tick_unit == "weeks"


def test_plot_helpers_accept_time_axis_kwarg():
    """Verify signature compatibility -- helpers must not raise TypeError on
    time_axis=None even without plotly installed."""
    import inspect
    import simweave.viz.plots as plots

    for fn_name in (
        "plot_queue_length",
        "plot_service_utilisation",
        "plot_warehouse_stock",
        "plot_mc_fan",
        "plot_fleet_availability",
    ):
        fn = getattr(plots, fn_name)
        sig = inspect.signature(fn)
        assert "time_axis" in sig.parameters, (
            f"{fn_name} is missing the time_axis= parameter"
        )
