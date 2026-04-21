import numpy as np

from simeng.solver import simulate
from simeng.systems import SeriesRLC


def test_rlc_rest_state_stays_at_rest():
    system = SeriesRLC(resistance=1.0, inductance=0.5, capacitance=0.1)
    result = simulate(system, (0.0, 0.5), dt=0.001)
    assert np.allclose(result.state, 0.0, atol=1e-12)


def test_rlc_step_input_changes_state():
    system = SeriesRLC(resistance=2.0, inductance=1.0, capacitance=0.5)
    result = simulate(system, (0.0, 0.5), dt=0.001, inputs=lambda t: 1.0)
    assert np.max(np.abs(result.state[:, 0])) > 0.0 or np.max(np.abs(result.state[:, 1])) > 0.0
