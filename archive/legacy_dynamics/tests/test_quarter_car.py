import numpy as np

from simeng.solver import simulate
from simeng.systems import QuarterCarModel


def test_quarter_car_static_equilibrium():
    model = QuarterCarModel(
        sprung_mass=250.0,
        unsprung_mass=40.0,
        suspension_stiffness=15000.0,
        damping=1500.0,
        tyre_stiffness=200000.0,
    )
    result = simulate(model, (0.0, 0.2), dt=0.001, inputs=lambda t: 0.0)
    assert np.allclose(result.state, 0.0, atol=1e-12)


def test_quarter_car_bump_response():
    model = QuarterCarModel(
        sprung_mass=250.0,
        unsprung_mass=40.0,
        suspension_stiffness=15000.0,
        damping=1500.0,
        tyre_stiffness=200000.0,
    )
    road = lambda t: 0.05 if t >= 0.02 else 0.0
    result = simulate(model, (0.0, 0.5), dt=0.001, inputs=road)
    assert np.max(np.abs(result.state[:, 0])) > 0.0
    assert np.max(np.abs(result.state[:, 2])) > 0.0
