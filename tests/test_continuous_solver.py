import numpy as np
import pytest

from simeng.continuous.solver import DynamicSystem, simulate, ContinuousProcess
from simeng.continuous.systems import (
    MassSpringDamper,
    SimplePendulum,
    QuarterCarModel,
    SeriesRLC,
)
from simeng.core.environment import SimEnvironment


class ExponentialDecay(DynamicSystem):
    def __init__(self, x0=1.0, rate=2.0):
        self.x0 = x0
        self.rate = rate

    def initial_state(self):
        return np.array([self.x0], dtype=float)

    def state_labels(self):
        return ("x",)

    def derivatives(self, t, state, inputs=None):
        return np.array([-self.rate * state[0]], dtype=float)


def test_rk4_matches_exponential_decay():
    system = ExponentialDecay()
    result = simulate(system, (0.0, 1.0), dt=0.01, method="rk4")
    expected = np.exp(-2.0 * result.time[-1])
    assert np.isclose(result.state[-1, 0], expected, atol=1e-5)


def test_euler_runs():
    system = ExponentialDecay()
    result = simulate(system, (0.0, 0.5), dt=0.01, method="euler")
    assert result.state.shape[1] == 1


def test_invalid_dt_raises():
    with pytest.raises(ValueError):
        simulate(ExponentialDecay(), (0.0, 1.0), dt=0.0)


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        simulate(ExponentialDecay(), (0.0, 1.0), dt=0.01, method="trapezoid")


def test_msd_oscillates_and_damps():
    m = MassSpringDamper(mass=1.0, damping=1.0, stiffness=4.0, x0=(1.0, 0.0))
    r = simulate(m, (0.0, 20.0), dt=0.01)
    # With moderate damping (zeta ~= 0.25) the response decays to ~0 by t=20.
    assert abs(r.state[-1, 0]) < 0.01
    # Ensure we captured at least one zero-crossing along the way (oscillation).
    signs = r.state[:, 0] > 0
    assert (signs[:-1] != signs[1:]).any()


def test_pendulum_state_labels():
    p = SimplePendulum(length=1.0)
    r = simulate(p, (0.0, 0.1), dt=0.001)
    assert r.state_labels == ("theta", "theta_dot")


def test_quarter_car_runs():
    m = QuarterCarModel(250, 40, 15000, 1500, 200000)
    r = simulate(m, (0.0, 0.5), dt=0.001, inputs=lambda t: 0.01)
    assert r.state.shape[1] == 4


def test_series_rlc_underdamped():
    sys = SeriesRLC(resistance=1.0, inductance=1.0, capacitance=1.0, x0=(1.0, 0.0))
    r = simulate(sys, (0.0, 10.0), dt=0.001)
    # Energy should decay.
    q0, i0 = r.state[0]
    q_end, i_end = r.state[-1]
    e0 = 0.5 * i0**2 + 0.5 * q0**2
    e_end = 0.5 * i_end**2 + 0.5 * q_end**2
    assert e_end < e0


def test_continuous_process_integrates_with_env():
    env = SimEnvironment(dt=0.01, end=1.0)
    proc = ContinuousProcess(ExponentialDecay())
    env.register(proc)
    env.run()
    r = proc.result()
    expected = np.exp(-2.0 * r.time[-1])
    assert np.isclose(r.state[-1, 0], expected, atol=1e-3)
