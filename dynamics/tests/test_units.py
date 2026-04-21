import pytest

from simeng.units import Acceleration, Distance, Force, Mass, SIUnit, TimeUnit, Velocity


def test_unit_passthrough():
    m = Mass(5)
    m2 = Mass(m)
    assert m2.value == 5.0


def test_force_from_mass_times_acceleration():
    f = Mass(2) * Acceleration(3)
    assert isinstance(f, Force)
    assert f.value == 6.0
    assert f.unit == 'N'


def test_velocity_from_distance_over_time():
    v = Distance(10) / TimeUnit(2)
    assert isinstance(v, Velocity)
    assert v.value == 5.0


def test_acceleration_from_velocity_over_time():
    a = Velocity(10) / TimeUnit(2)
    assert isinstance(a, Acceleration)
    assert a.value == 5.0


def test_dimensionless_result():
    ratio = Distance(10) / Distance(2)
    assert isinstance(ratio, SIUnit)
    assert ratio.unit == 'dimensionless'
    assert ratio.value == 5.0


def test_invalid_addition_raises():
    with pytest.raises(TypeError):
        _ = Distance(1) + Velocity(2)


def test_time_unit_conversion_hours_to_seconds():
    t = TimeUnit(1, 'hrs')
    assert t.value == 3600.0
