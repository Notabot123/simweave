"""Sampled-data controller hook, post_step hook and PID upgrades."""

import numpy as np
import pytest

from simweave.continuous.control import PIDController
from simweave.continuous.solver import ContinuousProcess, DynamicSystem, simulate
from simweave.core.environment import SimEnvironment


class Integrator(DynamicSystem):
    """x' = u (u supplied through ``inputs``)."""

    def __init__(self, x0=0.0):
        self.x0 = x0

    def initial_state(self):
        return np.array([self.x0], dtype=float)

    def state_labels(self):
        return ("x",)

    def derivatives(self, t, state, inputs=None):
        u = 0.0 if inputs is None else float(np.atleast_1d(inputs)[0])
        return np.array([u], dtype=float)


class PController:
    def __init__(self, kp, setpoint):
        self.kp = kp
        self.setpoint = setpoint
        self.calls = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def update(self, t, state, dt):
        self.calls.append((t, dt))
        return self.kp * (self.setpoint - state[0])


def test_zoh_matches_analytic_discrete_solution():
    # x' = u, u = kp (r - x) held for Tc. Over one period the exact flow is
    # x+ = x + Tc * kp (r - x), so the error shrinks by (1 - kp*Tc) per tick.
    kp, r, Tc = 2.0, 1.0, 0.1
    ctrl = PController(kp, r)
    res = simulate(Integrator(0.0), (0.0, 1.0), dt=0.01, controller=ctrl, control_dt=Tc)
    for n in range(0, 11):
        t = n * Tc
        idx = int(round(t / 0.01))
        expected = r * (1.0 - (1.0 - kp * Tc) ** n)
        assert res.state[idx, 0] == pytest.approx(expected, abs=1e-9)


def test_controller_called_once_per_period_with_reset():
    ctrl = PController(1.0, 1.0)
    simulate(Integrator(), (0.0, 1.0), dt=0.01, controller=ctrl, control_dt=0.1)
    # ticks at t = 0, 0.1, ... 0.9 (the last sample at t=1.0 is never stepped from)
    assert len(ctrl.calls) == 10
    assert ctrl.reset_count == 1
    assert all(dt == pytest.approx(0.1) for _, dt in ctrl.calls)
    assert [t for t, _ in ctrl.calls] == pytest.approx([0.1 * n for n in range(10)])


def test_default_control_period_is_every_step():
    ctrl = PController(1.0, 1.0)
    simulate(Integrator(), (0.0, 0.1), dt=0.01, controller=ctrl)
    assert len(ctrl.calls) == 10


def test_control_signal_is_logged_and_held():
    ctrl = PController(1.0, 1.0)
    res = simulate(Integrator(), (0.0, 0.5), dt=0.01, controller=ctrl, control_dt=0.1)
    u = res.inputs[:, 0]
    # constant within each control period
    assert np.allclose(u[1:11], u[1])
    assert not np.isclose(u[1], u[11])


def test_control_dt_must_be_multiple_of_dt():
    with pytest.raises(ValueError, match="multiple"):
        simulate(
            Integrator(), (0.0, 1.0), dt=0.01,
            controller=PController(1.0, 1.0), control_dt=0.015,
        )
    with pytest.raises(ValueError, match="positive"):
        simulate(
            Integrator(), (0.0, 1.0), dt=0.01,
            controller=PController(1.0, 1.0), control_dt=-0.1,
        )


def test_controller_and_inputs_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        simulate(
            Integrator(), (0.0, 1.0), dt=0.01,
            controller=PController(1.0, 1.0), inputs=lambda t: 0.0,
        )


def test_no_controller_path_unchanged():
    res = simulate(Integrator(), (0.0, 1.0), dt=0.01, inputs=lambda t: 1.0)
    assert res.state[-1, 0] == pytest.approx(1.0)


def test_post_step_applied_each_step():
    class Floor(Integrator):
        calls = 0

        def post_step(self, state):
            Floor.calls += 1
            return np.maximum(state, 0.0)

    Floor.calls = 0
    res = simulate(Floor(0.5), (0.0, 1.0), dt=0.1, inputs=lambda t: -1.0)
    assert Floor.calls == res.time.size - 1
    assert res.state.min() == 0.0  # clamped, never negative


def test_continuous_process_with_controller():
    ctrl = PController(2.0, 1.0)
    env = SimEnvironment(dt=0.1)
    proc = ContinuousProcess(
        Integrator(0.0), n_substeps=5, controller=ctrl, control_dt=0.1
    )
    env.register(proc)
    env.run(until=1.0)
    # 5 substeps of 0.02 s; control period 0.1 s = 5 substeps = once per tick
    assert ctrl.reset_count == 1
    assert len(ctrl.calls) == len(proc.history_time) - 1
    assert proc.state[0] == pytest.approx(1.0 - 0.8 ** len(ctrl.calls), abs=1e-9)


def test_continuous_process_rejects_both():
    with pytest.raises(ValueError):
        ContinuousProcess(
            Integrator(), controller=PController(1.0, 1.0), inputs=lambda t: 0.0
        )


# --------------------------------------------------------------------- PID


def test_pid_defaults_match_legacy_behaviour():
    pid = PIDController(Kp=2.0, Ki=0.5, Kd=0.1, setpoint=1.0)
    out1 = pid(0.0, 0.1)  # e=1, integral=0.1, derivative 0 on first call
    assert out1 == pytest.approx(2.0 + 0.5 * 0.1)
    out2 = pid(0.5, 0.1)  # e=0.5, integral=0.15, de/dt=-5
    assert out2 == pytest.approx(2.0 * 0.5 + 0.5 * 0.15 + 0.1 * -5.0)


def test_pid_output_limits_clamp():
    pid = PIDController(Kp=10.0, setpoint=1.0, output_limits=(-2.0, 3.0))
    assert pid(0.0, 0.1) == 3.0
    assert pid(5.0, 0.1) == -2.0
    half = PIDController(Kp=10.0, setpoint=1.0, output_limits=(None, 3.0))
    assert half(5.0, 0.1) == -40.0


def test_pid_anti_windup_stops_integral_growth():
    kw = dict(Kp=1.0, Ki=1.0, setpoint=10.0, output_limits=(-1.0, 1.0))
    clamped = PIDController(**kw)
    wound = PIDController(**kw, anti_windup="none")
    for _ in range(100):
        clamped(0.0, 0.1)
        wound(0.0, 0.1)
    assert clamped._integral < 1e-9 + 1.0  # bounded
    assert wound._integral == pytest.approx(100.0)
    # after the error flips sign the clamped controller recovers immediately
    assert clamped(20.0, 0.1) < 0.0
    assert wound(20.0, 0.1) > 0.0


def test_pid_derivative_on_measurement_has_no_setpoint_kick():
    err = PIDController(Kp=0.0, Kd=1.0, setpoint=0.0)
    meas = PIDController(Kp=0.0, Kd=1.0, setpoint=0.0, derivative_on_measurement=True)
    for p in (err, meas):
        p(0.0, 0.1)
        p.setpoint = 10.0
    assert abs(err(0.0, 0.1)) == pytest.approx(100.0)
    assert meas(0.0, 0.1) == pytest.approx(0.0)


def test_pid_derivative_filter_smooths_step():
    raw = PIDController(Kp=0.0, Kd=1.0)
    filt = PIDController(Kp=0.0, Kd=1.0, derivative_tau=0.5)
    for p in (raw, filt):
        p(0.0, 0.1)
    assert abs(filt(1.0, 0.1)) < abs(raw(1.0, 0.1))


def test_pid_validation_and_reset():
    with pytest.raises(ValueError):
        PIDController(1.0, anti_windup="bogus")
    with pytest.raises(ValueError):
        PIDController(1.0, output_limits=(2.0, 1.0))
    pid = PIDController(1.0)
    with pytest.raises(ValueError):
        pid(0.0, 0.0)
    pid = PIDController(1.0, 1.0, 1.0, setpoint=1.0)
    pid(0.0, 0.1)
    pid.reset()
    assert (pid._integral, pid._prev_error, pid._derivative) == (0.0, None, 0.0)


def test_pid_as_simulate_controller_regulates_first_order_plant():
    class Lag(DynamicSystem):
        def initial_state(self):
            return np.array([0.0])

        def derivatives(self, t, state, inputs=None):
            u = 0.0 if inputs is None else float(np.atleast_1d(inputs)[0])
            return np.array([-state[0] + u])

    pid = PIDController(Kp=4.0, Ki=2.0, setpoint=1.0, output_limits=(-10, 10))
    res = simulate(Lag(), (0.0, 20.0), dt=0.01, controller=pid, control_dt=0.02)
    assert res.state[-1, 0] == pytest.approx(1.0, abs=1e-3)
