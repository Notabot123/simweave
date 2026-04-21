import pytest

from simeng.units import (
    SIUnit,
    Distance,
    Velocity,
    Acceleration,
    Mass,
    Area,
    Force,
    TimeUnit,
)


def test_distance_velocity_acceleration():
    d = Distance(10.0)
    t = TimeUnit(2.0)
    v = d / t
    assert isinstance(v, Velocity)
    assert v.value == pytest.approx(5.0)

    a = v / t
    assert isinstance(a, Acceleration)
    assert a.value == pytest.approx(2.5)


def test_area_from_distance_squared():
    a = Distance(3.0) * Distance(4.0)
    assert isinstance(a, Area)
    assert a.value == pytest.approx(12.0)


def test_force_from_mass_times_acceleration():
    m = Mass(2.0)
    a = Acceleration(9.81)
    f = m * a
    assert isinstance(f, Force)
    assert f.value == pytest.approx(19.62)


def test_add_same_dim():
    total = Distance(3.0) + Distance(4.0)
    assert isinstance(total, Distance)
    assert total.value == pytest.approx(7.0)


def test_add_different_dim_raises():
    with pytest.raises(TypeError):
        _ = Distance(1.0) + Mass(1.0)


def test_time_unit_scaling():
    assert TimeUnit(5, "mins").value == pytest.approx(300.0)
    assert TimeUnit(2, "hrs").value == pytest.approx(7200.0)
    with pytest.raises(ValueError):
        TimeUnit(1, "fortnight")


def test_generic_siunit_for_unknown_product():
    # Mass * Distance has no registered class.
    result = Mass(1.0) * Distance(1.0)
    assert isinstance(result, SIUnit)
    assert tuple(result.exponents) == (1, 1, 0, 0, 0, 0, 0)
