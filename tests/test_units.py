import pytest
import numpy as np

from simweave.units.si import (
    SIUnit,
    Angle,
    AngularVelocity,
    AngularAcceleration,
    Distance,
    Velocity,
    Acceleration,
    Mass,
    Inertia,
    Area,
    Volume,
    Force,
    Torque,
    SpringStiffness,
    Damping,
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
    Charge,
    Capacitance,
    Resistivity,
    Inductance,
    ThermalResistance,
    ThermalCapacitance
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
def test_sqrt_area_method():
    a = Area(9.0)
    d = a.sqrt()

    assert isinstance(d, Distance)
    assert d.value == pytest.approx(3.0)

def test_cbrt_volume_method():
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
    current = Current(2.0)
    resistance = Resistance(5.0)

    voltage = current * resistance

    assert isinstance(voltage, Voltage)
    assert voltage.value == pytest.approx(10.0)

def test_resistance_from_voltage_current():
    voltage = Voltage(10.0)
    current = Current(2.0)

    resistance = voltage / current

    assert isinstance(resistance, Resistance)

def test_charge_basic():
    q = Charge(1.0)
    assert q.value == pytest.approx(1.0)
    assert q.unit == "C"

def test_charge_scaling():
    q = Charge(1000.0, "mC")
    assert q.value == pytest.approx(1.0)

    q = Charge(1_000_000.0, "uC")
    assert q.value == pytest.approx(1.0)

def test_charge_exponents():
    q = Charge(1.0)
    assert tuple(q.exponents) == (0, 0, 1, 0, 0, 0, 1)

def test_charge_from_current_time():
    q = Current(2.0) * TimeUnit(3.0)
    assert tuple(q.exponents) == (0, 0, 1, 0, 0, 0, 1)

def test_capacitance_from_charge_voltage():
    Q = Current(2.0) * TimeUnit(3.0)  # charge = I * t
    V = Voltage(6.0)

    C = Q / V

    assert isinstance(C, Capacitance)

def test_resistivity_relation():
    rho = Resistivity(1.0)
    L = Distance(2.0)
    A = Area(1.0)

    R = rho * L / A

    assert isinstance(R, Resistance)

def test_inductance_scaling():
    L = Inductance(1.0, "mH")
    assert L.value == pytest.approx(1e-3)

    L = Inductance(1000.0, "uH")
    assert L.value == pytest.approx(1e-3)

    with pytest.raises(ValueError):
        Inductance(1.0, "invalid")

def test_inductance_exponents():
    L = Inductance(1.0)
    assert tuple(L.exponents) == (2, 1, -2, 0, 0, 0, -2)

def test_inductance_array_support():
    L = Inductance(np.array([1.0, 2.0]), "mH")
    result = L.to("H")

    assert np.allclose(result, np.array([1e-3, 2e-3]))

def test_inductance_from_voltage_current():
    # V = L di/dt → L = V / (di/dt)
    V = Voltage(10.0)
    di_dt = Current(2.0) / TimeUnit(1.0)

    L = V / di_dt

    assert isinstance(L, SIUnit)
    assert tuple(L.exponents) == (2, 1, -2, 0, 0, 0, -2)

# angles
def test_angle_conversion():
    a = Angle(np.pi)

    assert a.to("deg") == pytest.approx(180.0, rel=1e-9)

def test_angle_from_degrees():
    a = Angle(180.0, "deg")

    assert a.value == pytest.approx(np.pi, rel=1e-9)

def test_angle_round_trip():
    a = Angle(45.0, "deg")

    assert a.to("deg") == pytest.approx(45.0, rel=1e-9)

def test_angle_array():
    import numpy as np

    a = Angle(np.array([0, np.pi / 2, np.pi]))

    deg = a.to("deg")

    assert np.allclose(deg, np.array([0, 90, 180]))

def test_angle_invalid_unit():
    with pytest.raises(ValueError):
        Angle(1.0, "radians")

def test_angle_format():
    a = Angle(np.pi)

    s = a.format("deg", precision=1)

    assert "180.0 [deg]" in s

# angular velocity
def test_angular_velocity_scaling():
    av = AngularVelocity(180.0, "deg/s")
    assert av.value == pytest.approx(np.pi, rel=1e-9)

    av = AngularVelocity(60.0, "rpm")
    assert av.value == pytest.approx(2 * np.pi, rel=1e-9)

    with pytest.raises(ValueError):
        AngularVelocity(1.0, "invalid")

def test_angular_velocity_conversion():
    av = AngularVelocity(np.pi)

    assert av.to("deg/s") == pytest.approx(180.0, rel=1e-9)
    assert av.to("rpm") == pytest.approx(30.0, rel=1e-9)

def test_angular_velocity_array_support():
    av = AngularVelocity(np.array([0.0, np.pi]))
    result = av.to("deg/s")

    assert np.allclose(result, np.array([0.0, 180.0]))

# angular acceleration
def test_angular_acceleration_scaling():
    aa = AngularAcceleration(180.0, "deg/s^2")
    assert aa.value == pytest.approx(np.pi, rel=1e-9)

    with pytest.raises(ValueError):
        AngularAcceleration(1.0, "invalid")

def test_angular_acceleration_conversion():
    aa = AngularAcceleration(np.pi)

    assert aa.to("deg/s^2") == pytest.approx(180.0, rel=1e-9)

def test_angular_acceleration_array_support():
    aa = AngularAcceleration(np.array([0.0, np.pi]))
    result = aa.to("deg/s^2")

    assert np.allclose(result, np.array([0.0, 180.0]))

def test_angular_velocity_exponents():
    av = AngularVelocity(1.0)
    assert tuple(av.exponents) == (0, 0, 0, 0, 0, 0, -1)


def test_angular_acceleration_exponents():
    aa = AngularAcceleration(1.0)
    assert tuple(aa.exponents) == (0, 0, 0, 0, 0, 0, -2)

# inertia and torque
def test_inertia_basic():
    I = Inertia(10.0)
    assert I.value == pytest.approx(10.0)

    with pytest.raises(ValueError):
        Inertia(1.0, "invalid")

def test_torque_scaling():
    t = Torque(10.0, "Nm")
    assert t.value == pytest.approx(10.0)

def test_torque_exponents():
    t = Torque(1.0)
    assert tuple(t.exponents) == (2, 1, 0, 0, 0, 0, -2)

"""
# will return Energy - we'd need vector quantities to know perpendicular to pivot etc
def test_torque_from_force_distance():
    t = Force(10.0) * Distance(2.0)
    assert isinstance(t, Torque)
"""

# stiffness and damping
def test_stiffness_scaling():
    k = SpringStiffness(1.0, "kN/m")
    assert k.value == pytest.approx(1000.0)

def test_stiffness_exponents():
    k = SpringStiffness(1.0)
    assert tuple(k.exponents) == (0, 1, 0, 0, 0, 0, -2)

def test_damping_basic():
    c = Damping(10.0)
    assert c.value == pytest.approx(10.0)

def test_damping_exponents():
    c = Damping(1.0)
    assert tuple(c.exponents) == (0, 1, 0, 0, 0, 0, -1)

# thermal
def test_thermal_resistance_basic():
    R = ThermalResistance(2.0)
    assert R.value == pytest.approx(2.0)

    with pytest.raises(ValueError):
        ThermalResistance(1.0, "invalid")

def test_thermal_resistance_exponents():
    R = ThermalResistance(1.0)
    assert tuple(R.exponents) == (-2, -1, 0, 1, 0, 0, 3)


def test_thermal_capacitance_scaling():
    C = ThermalCapacitance(1.0, "kJ/K")
    assert C.value == pytest.approx(1000.0)

def test_thermal_capacitance_exponents():
    C = ThermalCapacitance(1.0)
    assert tuple(C.exponents) == (2, 1, 0, -1, 0, 0, -2)