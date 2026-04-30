import pytest
import numpy as np

from simweave.units.si import (
    SIUnit,
    Distance,
    Velocity,
    Acceleration,
    Mass,
    Area,
    Volume,
    Force,
    TimeUnit,
    Energy,
    Pressure,
    Power,
    Frequency,
    Temperature,
    TemperatureDelta,
    Voltage,
    Current,
    Resistance,
    Capacitance,
    Resistivity
)


def test_distance_velocity_acceleration():
    d = Distance(10.0)
    t = TimeUnit(2.0)
    v = d / t
    assert isinstance(v, Velocity)
    assert v.value == pytest.approx(5.0, rel=1e-9)

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

# temp tests
def test_temperature_conversion():
    t = Temperature(0, "C")
    assert t.value == pytest.approx(273.15)
    assert t.to("C") == pytest.approx(0.0)


def test_temperature_kelvin_roundtrip():
    t = Temperature(300, "K")
    assert t.to("K") == pytest.approx(300.0)

def test_temperature_subtraction_returns_delta():
    t1 = Temperature(30, "C")
    t2 = Temperature(20, "C")

    delta = t1 - t2

    assert isinstance(delta, TemperatureDelta)
    assert delta.value == pytest.approx(10.0)

def test_temperature_add_delta():
    t = Temperature(20, "C")
    delta = TemperatureDelta(10, "C")

    result = t + delta

    assert isinstance(result, Temperature)
    assert result.to("C") == pytest.approx(30.0)


def test_temperature_add_temperature_raises():
    with pytest.raises(TypeError):
        Temperature(10, "C") + Temperature(10, "C")

# conversion tests
def test_distance_conversion():
    d = Distance(10, "ft")
    assert d.to("m") == pytest.approx(3.048)
    assert d.to("ft") == pytest.approx(10.0)


def test_velocity_conversion():
    v = Velocity(60, "mph")
    assert v.to("m/s") == pytest.approx(26.8224, rel=1e-4)

# format methods
def test_format_default_and_unit():
    d = Distance(1.0)
    assert d.format() == "1.0 [m]"
    assert d.format("cm") == "100.0 [cm]"


def test_auto_format_energy():
    e = Energy(1500)
    assert "kJ" in e.auto_format()

# alias handling
def test_distance_aliases():
    d1 = Distance(1, "ft")
    d2 = Distance(1, "foot")
    d3 = Distance(1, "feet")

    assert d1.value == pytest.approx(d2.value)
    assert d1.value == pytest.approx(d3.value)

# exponentiation
def test_power_operator():
    v = Velocity(3.0)
    v2 = v ** 2

    assert isinstance(v2, SIUnit) or isinstance(v2, Area)
    assert v2.value == pytest.approx(9.0)
    assert tuple(v2.exponents) == (2, 0, 0, 0, 0, 0, -2)

# to unit
def test_to_unit_returns_instance():
    d = Distance(100, "cm")
    d2 = d.to_unit("m")

    assert isinstance(d2, Distance)
    assert d2.value == pytest.approx(1.0)

# newly added units
def test_pressure_units():
    p = Pressure(1, "bar")
    assert p.value == pytest.approx(100000.0)


def test_energy_units():
    e = Energy(1, "kWh")
    assert e.value == pytest.approx(3.6e6)


def test_power_units():
    p = Power(1, "hp")
    assert p.value == pytest.approx(745.7)


def test_frequency_units():
    f = Frequency(60, "rpm")
    assert f.value == pytest.approx(1.0)

# derived units
def test_energy_from_force_times_distance():
    e = Force(10) * Distance(2)
    assert isinstance(e, Energy)
    assert e.value == pytest.approx(20.0)

# error handling
def test_invalid_conversion_unit():
    d = Distance(1.0)
    with pytest.raises(ValueError):
        d.to("invalid_unit")


def test_conversion_not_supported():
    a = Acceleration(9.81)
    with pytest.raises(ValueError):
        a.to("ft/s^2")

# covering _retype we discovered as edge case
def test_inline_energy_retyping():
    m = Distance(1)
    s = TimeUnit(1)

    c = 299_792_458 * m / s
    energy = Mass(1.0) * c**2

    assert isinstance(energy, Energy)

# control test, already works but helps as regression test
def test_energy_from_constants():
    from simweave.units.constants import c

    energy = Mass(1.0) * c**2

    assert isinstance(energy, Energy)

def test_energy_exponents_consistency():
    m = Distance(1)
    s = TimeUnit(1)

    c = 299_792_458 * m / s
    energy = Mass(1.0) * c**2

    assert tuple(energy.exponents) == (2, 1, 0, 0, 0, 0, -2)

# numpy arrays with siunits
def test_array_support():
    d = Distance(np.array([1.0, 2.0, 3.0]))
    t = TimeUnit(2.0)

    v = d / t

    assert isinstance(v, Velocity)
    assert np.allclose(v.value, np.array([0.5, 1.0, 1.5]))

def test_array_conversion():
    import numpy as np

    d = Distance(np.array([1.0, 2.0]), "m")
    result = d.to("cm")

    assert np.allclose(result, np.array([100.0, 200.0]))

def test_sqrt_area():
    a = Area(9.0)
    d = a ** 0.5

    assert isinstance(d, Distance)
    assert d.value == pytest.approx(3.0)

def test_cuberoot_volume():
    v = Volume(8.0)
    d = v ** (1/3)

    assert isinstance(d, Distance)
    assert d.value == pytest.approx(2.0)

def test_invalid_fractional_power():
    d = Distance(4.0)

    with pytest.raises(TypeError):
        _ = d ** 0.5

# use of method .sqrt and .cdbrt
def test_sqrt_area():
    a = Area(9.0)
    d = a.sqrt()

    assert isinstance(d, Distance)
    assert d.value == pytest.approx(3.0)

def test_cbrt_volume():
    v = Volume(8.0)
    d = v.cbrt()

    assert isinstance(d, Distance)
    assert d.value == pytest.approx(2.0, rel=1e-9)

def test_cbrt_volume_array():
    import numpy as np

    v = Volume(np.array([8.0]))
    d = v.cbrt()

    assert isinstance(d, Distance)
    assert np.allclose(d.value, np.array([2.0]))

def test_invalid_sqrt_distance():
    d = Distance(4.0)

    with pytest.raises(TypeError):
        _ = d.sqrt()

def test_array_sqrt():
    import numpy as np

    a = Area(np.array([1, 4, 9]))
    d = a.sqrt()

    assert isinstance(d, Distance)
    assert np.allclose(d.value, np.array([1, 2, 3]))

# electricity
def test_ohms_law():
    I = Current(2.0)
    R = Resistance(5.0)

    V = I * R

    assert isinstance(V, Voltage)
    assert V.value == pytest.approx(10.0)

def test_resistance_from_voltage_current():
    V = Voltage(10.0)
    I = Current(2.0)

    R = V / I

    assert isinstance(R, Resistance)