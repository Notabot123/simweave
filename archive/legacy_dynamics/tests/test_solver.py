import numpy as np
import pytest

from simeng.solver import DynamicSystem, simulate


class ExponentialDecay(DynamicSystem):
    def __init__(self, x0: float = 1.0, rate: float = 2.0):
        self.x0 = x0
        self.rate = rate

    def initial_state(self) -> np.ndarray:
        return np.array([self.x0], dtype=float)

    def state_labels(self) -> tuple[str]:
        return ('x',)

    def derivatives(self, t: float, state: np.ndarray, inputs=None) -> np.ndarray:
        return np.array([-self.rate * state[0]], dtype=float)


def test_rk4_matches_exponential_decay():
    system = ExponentialDecay(x0=1.0, rate=2.0)
    result = simulate(system, (0.0, 1.0), dt=0.01, method='rk4')
    expected = np.exp(-2.0 * result.time[-1])
    assert np.isclose(result.state[-1, 0], expected, atol=1e-5)


def test_euler_runs():
    system = ExponentialDecay(x0=1.0, rate=2.0)
    result = simulate(system, (0.0, 0.5), dt=0.01, method='euler')
    assert result.state.shape[1] == 1
    assert result.state[-1, 0] < 1.0


def test_invalid_dt_raises():
    system = ExponentialDecay()
    with pytest.raises(ValueError):
        simulate(system, (0.0, 1.0), dt=0.0)
