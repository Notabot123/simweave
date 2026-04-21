import numpy as np

from simeng.solver import simulate
from simeng.systems import SimplePendulum


def test_pendulum_rest_state_stays_at_rest():
    pendulum = SimplePendulum(length=1.0, x0=(0.0, 0.0))
    result = simulate(pendulum, (0.0, 1.0), dt=0.01)
    assert np.allclose(result.state, 0.0, atol=1e-12)


def test_pendulum_small_angle_moves_toward_zero():
    pendulum = SimplePendulum(length=1.0, damping=0.05, x0=(0.1, 0.0))
    result = simulate(pendulum, (0.0, 1.0), dt=0.001)
    assert abs(result.state[-1, 0]) < 0.1
