"""ThermalRC's embedded PID must not depend on the solver step size.

Before this was fixed, ``ThermalRC(controller=pid)`` called the PID inside ``derivatives`` with a
hard-coded dt of 1e-3, so the integral term advanced by an amount tied to how many times the
integrator evaluated the derivative, not to time. The same gains ended at 343.5 K for dt=0.05
and 350.6 K for dt=0.01.
"""
import warnings

import numpy as np
import pytest

from simweave.continuous.control import PIDController
from simweave.continuous.solver import ContinuousProcess, simulate
from simweave.continuous.systems import ThermalRC
from simweave.core.environment import SimEnvironment

SETPOINT = 350.0


def _pid():
    return PIDController(Kp=8.0, Ki=0.5, setpoint=SETPOINT)


def _embedded():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ThermalRC(0.5, 100.0, 293.15, 293.15, controller=_pid())


def _plant():
    return ThermalRC(0.5, 100.0, 293.15, 293.15)


def test_embedded_controller_result_is_independent_of_dt():
    finals = []
    for dt in (0.1, 0.05, 0.02, 0.01):
        T = simulate(_embedded(), (0.0, 100.0), dt=dt).state[:, 0]
        finals.append(T[-1])
    assert max(finals) - min(finals) < 0.05


def test_embedded_route_matches_the_sampled_data_hook():
    dt = 0.05
    emb = simulate(_embedded(), (0.0, 100.0), dt=dt).state[:, 0]
    hook = simulate(_plant(), (0.0, 100.0), dt=dt, controller=_pid(), control_dt=dt).state[:, 0]
    assert np.allclose(emb, hook, atol=1e-6)


def test_closed_loop_reaches_the_setpoint():
    T = simulate(_embedded(), (0.0, 200.0), dt=0.05).state[:, 0]
    assert abs(T[-1] - SETPOINT) < 0.5          # integral action removes the offset


def test_repeated_runs_are_identical():
    model = _embedded()
    a = simulate(model, (0.0, 50.0), dt=0.05).state[:, 0]
    b = simulate(model, (0.0, 50.0), dt=0.05).state[:, 0]
    assert np.array_equal(a, b)                 # controller state is reset between runs


def test_external_heat_input_is_still_added():
    base = simulate(_embedded(), (0.0, 100.0), dt=0.05).state[:, 0]
    pushed = simulate(_embedded(), (0.0, 100.0), dt=0.05, inputs=lambda t: 40.0 if t > 20 else 0.0).state[:, 0]
    assert np.abs(pushed - base).max() > 1.0


def test_deprecation_warning_only_with_a_controller():
    with pytest.warns(DeprecationWarning, match="simulate"):
        ThermalRC(0.5, 100.0, controller=_pid())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ThermalRC(0.5, 100.0)                   # no controller, no warning


def test_continuous_process_runs_the_embedded_controller():
    dt = 0.05
    ref = simulate(_embedded(), (0.0, 100.0), dt=dt).state[-1, 0]
    env = SimEnvironment(dt=dt, end=100.0)
    proc = ContinuousProcess(_embedded(), method="rk4")
    env.register(proc)
    env.run()
    assert proc.result().state[-1, 0] == pytest.approx(ref, abs=0.1)


def test_control_update_hook_is_generic():
    """Any system may supply control_update/reset_controls; simulate calls them with the true dt."""
    class Probe(ThermalRC):
        def __init__(self):
            super().__init__(0.5, 100.0, 293.15, 293.15)
            self.calls = []
            self.resets = 0

        def reset_controls(self):
            self.resets += 1

        def control_update(self, t, state, dt):
            self.calls.append((round(t, 6), round(dt, 6)))

    p = Probe()
    simulate(p, (0.0, 1.0), dt=0.25)
    assert p.resets == 1
    assert p.calls == [(0.0, 0.25), (0.25, 0.25), (0.5, 0.25), (0.75, 0.25)]
