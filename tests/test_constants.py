import pytest
from simweave.units.constants import g, c, h, k_B
from simweave.units.si import SIUnit, Acceleration, Velocity, Energy, Temperature


def test_gravity_constant():
    assert isinstance(g, Acceleration)
    assert g.value > 9 and g.value < 10


def test_speed_of_light():
    assert isinstance(c, Velocity)
    assert c.value == 299_792_458


def test_planck_constant():
    # Should be Energy * Time dimensionally
    assert isinstance(h, Energy) or isinstance(h, SIUnit)
    

def test_temperature_conversion():
    t = Temperature(0, "C")
    assert t.value == pytest.approx(273.15)
    assert t.to("C") == pytest.approx(0.0)


def test_temperature_kelvin_roundtrip():
    t = Temperature(300, "K")
    assert t.to("K") == pytest.approx(300.0)


def test_boltzmann_usage():
    T = Temperature(300, "K")
    energy = k_B * T
    assert isinstance(energy, Energy)