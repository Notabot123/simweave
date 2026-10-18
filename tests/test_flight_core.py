"""Quadrotor plant, mixer, frames and wind models."""

import math

import numpy as np
import pytest

from simweave.continuous.solver import simulate
from simweave.faults import FaultInjector, FaultProfile, ParameterFault
from simweave.flight import (
    ConstantWind,
    GustWind,
    Mixer,
    Quadrotor,
    QuadrotorParams,
    TurbulentWind,
    euler_from_rotation,
    euler_to_quaternion,
    quaternion_to_euler,
    rotation_matrix,
    wrap_angle,
)
from simweave.units.si import Mass


class Hold:
    """Controller returning a fixed vector of rotor thrusts."""

    def __init__(self, thrusts):
        self.thrusts = np.asarray(thrusts, dtype=float)

    def update(self, t, state, dt):
        return self.thrusts


# ------------------------------------------------------------------ frames


@pytest.mark.parametrize(
    "angles",
    [(0.0, 0.0, 0.0), (0.3, -0.4, 1.2), (-1.0, 0.8, -2.5), (0.0, 1.2, 3.0)],
)
def test_rotation_matrix_is_proper_and_round_trips(angles):
    R = rotation_matrix(*angles)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert euler_from_rotation(R) == pytest.approx(angles, abs=1e-9)


def test_euler_quaternion_round_trip_and_unit_norm():
    angles = (0.4, -0.7, 2.0)
    q = euler_to_quaternion(*angles)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert quaternion_to_euler(q) == pytest.approx(angles, abs=1e-9)


def test_yaw_is_counter_clockwise_and_pitch_is_nose_down():
    R_yaw = rotation_matrix(0.0, 0.0, math.pi / 2)
    assert R_yaw @ [1, 0, 0] == pytest.approx([0, 1, 0], abs=1e-12)
    R_pitch = rotation_matrix(0.0, 0.3, 0.0)
    assert (R_pitch @ [1, 0, 0])[2] < 0  # nose points down
    R_roll = rotation_matrix(0.3, 0.0, 0.0)
    assert (R_roll @ [0, -1, 0])[2] < 0  # right side goes down


def test_wrap_angle():
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(np.array([-4.0, 4.0])) == pytest.approx(
        [-4.0 + 2 * math.pi, 4.0 - 2 * math.pi]
    )


# ------------------------------------------------------------------ params


def test_params_defaults_and_units():
    p = QuadrotorParams()
    assert p.thrust_to_weight > 2.0
    assert p.hover_thrust == pytest.approx(p.mass * p.gravity / 4)
    assert QuadrotorParams(mass=Mass(2.0)).mass == 2.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": -1.0},
        {"inertia": (0.01, 0.01)},
        {"inertia": (0.01, 0.0, 0.01)},
        {"motor_tau": 0.0},
        {"drag_lin": -0.1},
        {"figure_of_merit": 1.5},
        {"max_thrust": 1.0},  # cannot lift off
    ],
)
def test_params_validation(kwargs):
    with pytest.raises(ValueError):
        QuadrotorParams(**kwargs)


# ------------------------------------------------------------------- mixer


def test_mixer_round_trip_and_collective_split():
    mixer = Mixer()
    f = mixer.thrusts(10.0, (0.0, 0.0, 0.0))
    assert f == pytest.approx(np.full(4, 2.5))
    wrench = [11.0, 0.05, -0.04, 0.01]
    assert mixer.wrench(mixer.thrusts(wrench[0], wrench[1:])) == pytest.approx(wrench)


def test_mixer_torque_signs():
    mixer = Mixer()
    rear_up = mixer.thrusts(10.0, (0.0, 0.1, 0.0))  # +tau_y: nose down
    assert rear_up[2] > rear_up[0] and rear_up[3] > rear_up[1]
    left_up = mixer.thrusts(10.0, (0.1, 0.0, 0.0))  # +tau_x: bank right
    assert left_up[1] > left_up[0] and left_up[2] > left_up[3]


def test_allocate_feasible_request_is_exact():
    mixer = Mixer()
    f = mixer.allocate(11.0, (0.05, -0.03, 0.002))
    assert mixer.wrench(f) == pytest.approx([11.0, 0.05, -0.03, 0.002])


def test_allocate_sacrifices_collective_before_torque():
    mixer = Mixer()
    fmax = mixer.params.max_thrust
    torques = (0.3, 0.0, 0.0)
    f = mixer.allocate(4 * fmax, torques)  # collective at the very limit
    assert f.min() >= -1e-12 and f.max() <= fmax + 1e-12
    w = mixer.wrench(f)
    assert w[1:] == pytest.approx(torques)
    assert w[0] < 4 * fmax


def test_allocate_scales_infeasible_torque():
    mixer = Mixer()
    fmax = mixer.params.max_thrust
    huge = (50.0, 0.0, 0.0)
    f = mixer.allocate(10.0, huge)
    assert f.min() >= -1e-12 and f.max() <= fmax + 1e-12
    w = mixer.wrench(f)
    assert 0 < w[1] < huge[0]
    assert w[2] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------- plant


def make_quad(**params):
    return Quadrotor(QuadrotorParams(**params), hover=True, position=(0, 0, 5))


def test_state_shape_labels_and_units():
    quad = make_quad()
    assert quad.initial_state().shape == (16,)
    assert len(quad.state_labels()) == 16
    res = simulate(quad, (0.0, 0.1), dt=0.01, controller=Hold(np.full(4, quad.hover_thrust)))
    wrapped = quad.wrap_states(res)
    assert wrapped["z"].value[0] == pytest.approx(5.0)


def test_hover_holds_position():
    quad = make_quad()
    res = simulate(
        quad, (0.0, 5.0), dt=0.005, controller=Hold(np.full(4, quad.hover_thrust))
    )
    assert np.allclose(res.state[:, 2], 5.0, atol=1e-9)
    assert np.allclose(res.state[:, 3:12], 0.0, atol=1e-9)


def test_free_fall_matches_analytic():
    quad = make_quad(drag_lin=0.0)
    x0 = quad.initial_state()
    x0[2] = 100.0
    x0[12:16] = 0.0
    res = simulate(quad, (0.0, 2.0), dt=0.01, x0=x0, controller=Hold(np.zeros(4)))
    g = quad.gravity
    assert res.state[-1, 2] == pytest.approx(100.0 - 0.5 * g * 4.0, abs=1e-9)
    assert res.state[-1, 5] == pytest.approx(-g * 2.0, abs=1e-9)


def test_terminal_velocity_under_linear_drag():
    quad = make_quad(drag_lin=0.1)
    x0 = quad.initial_state()
    x0[2] = 1e5
    x0[12:16] = 0.0
    res = simulate(quad, (0.0, 150.0), dt=0.02, x0=x0, controller=Hold(np.zeros(4)))
    assert res.state[-1, 5] == pytest.approx(-quad.mass * quad.gravity / 0.1, rel=1e-4)


def test_quadratic_drag_terminal_velocity():
    quad = make_quad(drag_lin=0.0, drag_quad=0.05)
    x0 = quad.initial_state()
    x0[2] = 1e5
    x0[12:16] = 0.0
    res = simulate(quad, (0.0, 80.0), dt=0.02, x0=x0, controller=Hold(np.zeros(4)))
    expected = -math.sqrt(quad.mass * quad.gravity / 0.05)
    assert res.state[-1, 5] == pytest.approx(expected, rel=1e-3)


def test_pure_torques_give_expected_angular_accelerations():
    quad = make_quad()
    mixer = Mixer(quad.params)
    p = quad.params
    for axis, inertia in enumerate(p.inertia):
        tau = np.zeros(3)
        tau[axis] = 0.02
        thrusts = mixer.thrusts(p.weight, tau)
        x0 = quad.hover_state((0, 0, 50))
        x0[12:16] = thrusts  # rotors already at the commanded values
        res = simulate(quad, (0.0, 0.05), dt=0.001, x0=x0, controller=Hold(thrusts))
        rate = res.state[-1, 9 + axis]
        assert rate == pytest.approx(tau[axis] / inertia * 0.05, rel=1e-6)
        others = [res.state[-1, 9 + i] for i in range(3) if i != axis]
        assert np.allclose(others, 0.0, atol=1e-12)


def test_motor_lag_first_order_response():
    quad = make_quad()
    h = quad.hover_thrust
    cmd = np.full(4, 2 * h)
    res = simulate(quad, (0.0, quad.motor_tau), dt=0.0005, controller=Hold(cmd))
    expected = h + (2 * h - h) * (1 - math.exp(-1))
    assert res.state[-1, 12] == pytest.approx(expected, rel=1e-6)


def test_commands_are_clipped_to_thrust_limits():
    quad = make_quad()
    res = simulate(quad, (0.0, 1.0), dt=0.005, controller=Hold(np.full(4, 1e6)))
    assert res.state[:, 12:16].max() <= quad.max_thrust + 1e-9
    res = simulate(quad, (0.0, 1.0), dt=0.005, controller=Hold(np.full(4, -5.0)))
    assert res.state[:, 12:16].min() >= -1e-9


def test_wrong_input_size_raises():
    quad = make_quad()
    with pytest.raises(ValueError, match="four"):
        simulate(quad, (0.0, 0.1), dt=0.01, controller=Hold([1.0, 2.0]))


def test_constant_wind_pushes_a_stationary_vehicle():
    quad = Quadrotor(hover=True, position=(0, 0, 5), wind=ConstantWind((5.0, 0.0, 0.0)))
    res = simulate(
        quad, (0.0, 0.1), dt=0.01, controller=Hold(np.full(4, quad.hover_thrust))
    )
    expected_ax = quad.drag_lin * 5.0 / quad.mass
    assert res.state[-1, 3] == pytest.approx(expected_ax * 0.1, rel=0.02)
    assert res.state[-1, 4] == pytest.approx(0.0, abs=1e-12)


def test_dead_motor_rolls_and_pitches_the_right_way():
    quad = make_quad()
    quad.motor_eff_1 = 0.0  # front-right rotor lost
    res = simulate(
        quad, (0.0, 0.1), dt=0.005, controller=Hold(np.full(4, quad.hover_thrust))
    )
    p_rate, q_rate = res.state[-1, 9], res.state[-1, 10]
    assert p_rate > 0 and q_rate > 0  # front-right drops: bank right, nose down


def test_fault_injector_degrades_one_rotor():
    quad = make_quad()
    profile = FaultProfile(onset_time=0.0, failure_time=1.0, mode="motor", shape="linear")
    fault = ParameterFault(param="motor_eff_2", profile=profile, max_delta=-1.0)
    injector = FaultInjector(quad, [fault])
    hold = Hold(np.full(4, quad.hover_thrust))
    healthy = simulate(quad, (0.0, 1.0), dt=0.005, controller=hold)
    faulty = simulate(injector, (0.0, 1.0), dt=0.005, controller=hold)
    assert abs(healthy.state[-1, 8]) < 1e-9
    assert abs(faulty.state[-1, 8]) > 1e-3  # yaw drifts without compensation
    assert quad.motor_eff_2 == 1.0  # nominal restored


def test_ground_contact_keeps_a_grounded_vehicle_still():
    quad = Quadrotor()  # on the ground, rotors off
    res = simulate(quad, (0.0, 1.0), dt=0.01, controller=Hold(np.zeros(4)))
    assert np.allclose(res.state[:, 2:12], 0.0, atol=1e-12)


def test_ground_holds_attitude_until_liftoff_then_climbs():
    quad = Quadrotor()
    thrusts = np.full(4, 1.5 * quad.hover_thrust)
    thrusts[0] *= 0.9  # slight asymmetry that would tilt a free body
    res = simulate(quad, (0.0, 1.0), dt=0.005, controller=Hold(thrusts))
    assert res.state[:, 2].max() > 0.05  # it did lift off (open loop, it then tumbles)
    # while the thrust was still below weight the attitude stayed level
    grounded = np.flatnonzero(res.state[:, 2] <= 1e-9)
    assert grounded.size > 3
    # (the final grounded sample is the liftoff step itself, which may tilt slightly)
    assert np.allclose(res.state[grounded[:-1]][:, 6:8], 0.0)


def test_never_goes_below_ground():
    quad = make_quad()
    x0 = quad.initial_state()
    x0[2] = 1.0
    x0[12:16] = 0.0
    res = simulate(quad, (0.0, 3.0), dt=0.005, x0=x0, controller=Hold(np.zeros(4)))
    assert res.state[:, 2].min() >= 0.0
    assert res.state[-1, 2] == 0.0


def test_power_matches_momentum_theory():
    quad = make_quad()
    p = quad.params
    f = quad.hover_thrust
    expected = 4 * f**1.5 / (math.sqrt(2 * p.air_density * p.disk_area) * p.figure_of_merit)
    assert quad.power(np.full(4, f)) == pytest.approx(expected)
    assert quad.power(np.zeros((3, 4))).shape == (3,)


# -------------------------------------------------------------------- wind


def test_constant_and_gust_wind():
    assert ConstantWind((1, 2, 3))(0.0, None) == pytest.approx([1, 2, 3])
    gust = GustWind(amplitude=(4, 0, 0), start=1.0, duration=2.0, base=(1, 0, 0))
    assert gust(0.5, None)[0] == pytest.approx(1.0)
    assert gust(2.0, None)[0] == pytest.approx(1.0 + 4.0)  # peak at mid gust
    assert gust(1.0, None)[0] == pytest.approx(1.0)
    assert gust(3.5, None)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        GustWind((1, 0, 0), 0.0, 0.0)
    with pytest.raises(ValueError):
        ConstantWind((1, 2))


def test_turbulence_is_seeded_and_statistically_sane():
    def make(seed):
        return TurbulentWind(
            sigma=2.0, correlation_time=1.0, duration=2000.0, mean=(3, 0, 0),
            rng=np.random.default_rng(seed),
        )

    a, b, c = make(1), make(1), make(2)
    ts = np.linspace(0, 100, 50)
    assert np.array_equal([a(t) for t in ts], [b(t) for t in ts])
    assert not np.array_equal([a(t) for t in ts], [c(t) for t in ts])
    samples = np.array([a(t) for t in np.arange(0, 2000, 0.5)])
    assert samples[:, 0].mean() == pytest.approx(3.0, abs=0.3)
    assert samples[:, 0].std() == pytest.approx(2.0, rel=0.1)
    assert a(2000.0)[0] == a(5000.0)[0]  # held after the end
    assert a(0.123) == pytest.approx(a(0.123))  # repeatable evaluation


def test_turbulent_wind_runs_inside_a_simulation_deterministically():
    def run(seed):
        wind = TurbulentWind(1.5, 0.8, 10.0, rng=np.random.default_rng(seed))
        quad = Quadrotor(hover=True, position=(0, 0, 5), wind=wind)
        res = simulate(
            quad, (0.0, 5.0), dt=0.01, controller=Hold(np.full(4, quad.hover_thrust))
        )
        return res.state[-1]

    assert np.array_equal(run(3), run(3))
    assert not np.array_equal(run(3), run(4))
