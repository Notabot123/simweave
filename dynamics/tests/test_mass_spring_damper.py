import numpy as np

from simeng.solver import simulate
from simeng.systems import MassSpringDamper


def test_rest_state_stays_at_rest_without_force():
    system = MassSpringDamper(mass=1.0, damping=0.2, stiffness=10.0)
    result = simulate(system, (0.0, 1.0), dt=0.01)
    assert np.allclose(result.state, 0.0, atol=1e-12)


def test_forced_response_moves_mass():
    system = MassSpringDamper(mass=1.0, damping=0.5, stiffness=5.0)
    result = simulate(system, (0.0, 1.0), dt=0.01, inputs=lambda t: 1.0)
    assert np.max(np.abs(result.state[:, 0])) > 0.0
