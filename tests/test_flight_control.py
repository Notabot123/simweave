"""Trajectories, missions, the cascaded controller and FlightResult."""

import math

import numpy as np
import pytest

from simweave.continuous.solver import simulate
from simweave.flight import (
    CascadedController,
    ConstantWind,
    ControllerGains,
    Figure8,
    GustWind,
    Hold,
    Mission,
    Quadrotor,
    Quintic,
    RefPoint,
    TurbulentWind,
    fly,
    rotation_matrix,
)


def pt(pos, vel=(0, 0, 0), acc=(0, 0, 0), yaw=0.0):
    return RefPoint(np.array(pos, float), np.array(vel, float), np.array(acc, float), yaw)


# -------------------------------------------------------------- trajectories


def test_quintic_meets_boundary_conditions():
    start = pt((1, 2, 3), (0.5, -0.2, 0.1), (0.1, 0.0, -0.3), yaw=0.2)
    end = pt((4, -1, 6), (-0.3, 0.4, 0.0), (0.0, 0.2, 0.1), yaw=1.0)
    seg = Quintic(start, end, 5.0)
    for tau, ref in ((0.0, start), (5.0, end)):
        got = seg(tau)
        assert got.pos == pytest.approx(ref.pos, abs=1e-10)
        assert got.vel == pytest.approx(ref.vel, abs=1e-10)
        assert got.acc == pytest.approx(ref.acc, abs=1e-10)
        assert got.yaw == pytest.approx(ref.yaw, abs=1e-10)


def test_quintic_derivatives_are_consistent():
    seg = Quintic(pt((0, 0, 0)), pt((3, 1, 2), (1, 0, 0)), 4.0)
    h = 1e-5
    for tau in (0.5, 2.0, 3.5):
        a, b = seg(tau - h), seg(tau + h)
        assert (b.pos - a.pos) / (2 * h) == pytest.approx(seg(tau).vel, abs=1e-6)
        assert (b.vel - a.vel) / (2 * h) == pytest.approx(seg(tau).acc, abs=1e-6)


def test_quintic_yaw_takes_the_short_way_round():
    seg = Quintic(pt((0, 0, 0), yaw=3.0), pt((1, 0, 0), yaw=-3.0), 2.0)
    mid = seg(1.0).yaw
    assert abs(mid) > 3.0 or mid == pytest.approx(math.pi, abs=0.2)  # crosses +/-pi
    assert seg(2.0).yaw == pytest.approx(3.0 + (2 * math.pi - 6.0), abs=1e-9)


def test_quintic_and_hold_validate():
    with pytest.raises(ValueError):
        Quintic(pt((0, 0, 0)), pt((1, 0, 0)), 0.0)
    with pytest.raises(ValueError):
        Hold((0, 0, 0), duration=-1)


def test_figure8_is_periodic_with_consistent_derivatives():
    f = Figure8(center=(1, 2, 3), a=4, b=2, period=12.0, laps=2, z_amp=0.5)
    assert f.duration == pytest.approx(24.0)
    s, e = f(0.0), f(12.0)
    assert e.pos == pytest.approx(s.pos, abs=1e-9)
    assert e.vel == pytest.approx(s.vel, abs=1e-9)
    assert f(24.0).acc == pytest.approx(s.acc, abs=1e-9)
    h = 1e-5
    for tau in (1.0, 4.4, 9.1):
        a, b = f(tau - h), f(tau + h)
        assert (b.pos - a.pos) / (2 * h) == pytest.approx(f(tau).vel, abs=1e-6)
        assert (b.vel - a.vel) / (2 * h) == pytest.approx(f(tau).acc, abs=1e-6)


def test_figure8_crosses_itself_and_has_two_lobes():
    f = Figure8(center=(0, 0, 3), a=4, b=2, period=10.0)
    xy = np.array([f(t).pos[:2] for t in np.linspace(0, 10, 400)])
    assert xy[:, 0].min() == pytest.approx(-4, abs=0.01)
    assert xy[:, 0].max() == pytest.approx(4, abs=0.01)
    assert xy[:, 1].max() == pytest.approx(2, abs=0.01)
    # the path passes through the centre at the start and halfway round
    assert f(5.0).pos[:2] == pytest.approx([0, 0], abs=1e-9)


def test_yaw_rate_feed_forward_matches_finite_difference():
    f = Figure8(yaw_mode="heading")
    q = Quintic(pt((0, 0, 0), yaw=0.2), pt((1, 0, 0), yaw=1.4), 3.0)
    h = 1e-6
    for seg, taus in ((f, (1.0, 6.0, 13.0)), (q, (0.5, 1.5, 2.5))):
        for tau in taus:
            num = (seg(tau + h).yaw - seg(tau - h).yaw) / (2 * h)
            assert seg(tau).yaw_rate == pytest.approx(num, abs=1e-5)
    assert Hold((0, 0, 0), duration=1.0)(0.0).yaw_rate == 0.0


def test_figure8_heading_mode_follows_velocity():
    f = Figure8(yaw_mode="heading")
    for tau in np.linspace(0, 20, 30):
        r = f(tau)
        assert r.yaw == pytest.approx(math.atan2(r.vel[1], r.vel[0]))
        assert np.hypot(*r.vel[:2]) > 0.1  # never degenerate
    with pytest.raises(ValueError):
        Figure8(yaw_mode="sideways")


def test_mission_is_continuous_through_every_join():
    m = Mission.figure8(yaw_mode="heading", z_amp=1.0, center=(2.0, -1.0), start=(0.5, 0.5))
    eps = 1e-7
    for name, t0, t1 in m.phases[1:]:
        before, after = m(t0 - eps), m(t0 + eps)
        assert after.pos == pytest.approx(before.pos, abs=1e-5), name
        if name != "landed":  # touchdown speed drops to zero on the ground, by design
            assert after.vel == pytest.approx(before.vel, abs=1e-5), name
            assert after.acc == pytest.approx(before.acc, abs=1e-4), name
        assert wrap_diff(after.yaw, before.yaw) < 1e-4, name


def wrap_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def test_mission_phases_and_endpoints():
    m = Mission.figure8()
    names = [p[0] for p in m.phases]
    assert names == [
        "takeoff", "hover", "transition", "figure8",
        "transition", "hover", "land", "landed",
    ]
    assert m.phases[-1][2] == pytest.approx(m.duration)
    assert m(0.0).pos == pytest.approx([0, 0, 0])
    assert m(-5.0).pos == pytest.approx([0, 0, 0])
    assert m(m.duration + 10).pos[2] == pytest.approx(0.0)
    assert m.phase(m.phases[3][1] + 1.0) == "figure8"
    assert m.phase(1e9) == "landed"
    with pytest.raises(ValueError):
        Mission([])
    with pytest.raises(ValueError):
        Mission.figure8(height=0.0)


# --------------------------------------------------------------- controller


def test_controller_hovers_at_weight_with_no_error():
    quad = Quadrotor(hover=True, position=(0, 0, 3))
    ref = Hold((0, 0, 3), duration=10.0)
    ctrl = CascadedController(ref, quad.params)
    f = ctrl.update(0.0, quad.initial_state(), 0.01)
    assert f == pytest.approx(np.full(4, quad.hover_thrust), abs=1e-9)


def test_controller_is_disarmed_on_the_ground():
    quad = Quadrotor()
    ctrl = CascadedController(Hold((0, 0, 0), duration=5.0), quad.params)
    assert np.array_equal(ctrl.update(0.0, quad.initial_state(), 0.01), np.zeros(4))
    assert not ctrl.armed


def test_controller_desired_heading_is_respected():
    # a vehicle yawed away from the reference heading must turn towards it
    quad = Quadrotor(hover=True, position=(0, 0, 3), yaw=0.5)
    ref = Hold((0, 0, 3), yaw=0.0, duration=10.0)
    ctrl = CascadedController(ref, quad.params)
    f = ctrl.update(0.0, quad.initial_state(), 0.01)
    yaw_torque = quad.torque_ratio * (-f[0] + f[1] - f[2] + f[3])
    assert yaw_torque < 0  # negative torque reduces a positive yaw error


def test_controller_pushes_towards_the_reference_and_respects_tilt_limit():
    quad = Quadrotor(hover=True, position=(0, 0, 3))
    ref = Hold((50.0, 0, 3), duration=10.0)  # far away: demands max tilt
    ctrl = CascadedController(ref, quad.params)
    res = simulate(quad, (0.0, 6.0), dt=0.004, controller=ctrl, control_dt=0.008)
    tilt = np.arccos(np.cos(res.state[:, 6]) * np.cos(res.state[:, 7]))
    assert tilt.max() <= ctrl.gains.max_tilt + math.radians(6.0)
    assert res.state[-1, 0] > 5.0 and res.state[-1, 3] > 0.5


def test_hover_recovers_from_an_offset():
    quad = Quadrotor(hover=True, position=(1.0, -1.0, 4.0))
    ref = Hold((0.0, 0.0, 3.0), duration=20.0)
    ctrl = CascadedController(ref, quad.params)
    res = simulate(quad, (0.0, 15.0), dt=0.004, controller=ctrl, control_dt=0.008)
    err = np.linalg.norm(res.state[:, 0:3] - np.array([0, 0, 3.0]), axis=1)
    assert err[-1] < 0.02
    assert err.max() < 1.8  # never wanders off by much more than the initial offset
    assert abs(res.state[:, 2].min() - 3.0) < 0.5  # modest altitude undershoot


def test_controller_reset_clears_state():
    quad = Quadrotor(hover=True, position=(1.0, 0, 3.0))
    ctrl = CascadedController(Hold((0, 0, 3), duration=5.0), quad.params)
    ctrl.update(0.0, quad.initial_state(), 0.01)
    assert ctrl.armed and ctrl._vel_pid[0]._integral != 0.0
    ctrl.reset()
    assert not ctrl.armed and ctrl._vel_pid[0]._integral == 0.0


# ------------------------------------------------------------- closed loop


@pytest.fixture(scope="module")
def nominal():
    return fly(Quadrotor(), Mission.figure8())


def test_nominal_mission_tracks_closely_and_lands_softly(nominal):
    m = nominal.metrics()
    assert not np.isnan(nominal.result.state).any()
    assert m["rms_error"] < 0.08
    assert m["max_error"] < 0.3
    assert m["saturation_fraction"] == 0.0
    assert m["max_tilt_deg"] < 25.0
    assert -0.8 < m["touchdown_speed"] < 0.0
    assert m["landing_error"] < 0.1
    assert 0.5 < m["energy_wh"] < 2.0  # about 90 W hover for 42 s
    assert nominal.position[-1, 2] == pytest.approx(0.0, abs=1e-6)


def test_nominal_mission_reaches_cruise_height_and_figure8_extent(nominal):
    pos, t = nominal.position, nominal.time
    in_eight = np.array([nominal.reference.phase(x) == "figure8" for x in t])
    assert pos[in_eight, 2].mean() == pytest.approx(3.0, abs=0.05)
    assert pos[in_eight, 0].max() > 3.8 and pos[in_eight, 0].min() < -3.8
    assert pos[in_eight, 1].max() > 1.8


def test_phase_rms_names_every_phase(nominal):
    rms = nominal.phase_rms()
    assert set(rms) == {"takeoff", "hover", "transition", "figure8", "land", "landed"}
    assert rms["figure8"] < 0.05
    with pytest.raises(TypeError):
        fly(Quadrotor(), Hold((0, 0, 0), duration=1.0), t_end=1.0).phase_rms()


@pytest.mark.parametrize(
    "wind",
    [
        ConstantWind((5.0, 0.0, 0.0)),
        ConstantWind((6.0, -5.0, 0.0)),
        GustWind((6.0, 0.0, 0.0), start=15.0, duration=3.0),
    ],
)
def test_tracking_survives_wind(wind):
    r = fly(Quadrotor(wind=wind), Mission.figure8())
    m = r.metrics()
    assert m["rms_error"] < 0.1
    assert m["landing_error"] < 0.1
    assert -0.8 < m["touchdown_speed"] < 0.0


def test_integral_action_removes_steady_wind_error():
    quad = Quadrotor(hover=True, position=(0, 0, 3), wind=ConstantWind((6.0, 0.0, 0.0)))
    ctrl = CascadedController(Hold((0, 0, 3), duration=60.0), quad.params)
    res = simulate(quad, (0.0, 40.0), dt=0.004, controller=ctrl, control_dt=0.008)
    assert np.linalg.norm(res.state[-1, 0:3] - [0, 0, 3]) < 0.02


def test_payload_mismatch_is_absorbed():
    quad = Quadrotor()
    quad.mass *= 1.25  # controller still believes the nominal mass
    r = fly(quad, Mission.figure8())
    m = r.metrics()
    assert m["rms_error"] < 0.15
    assert -0.8 < m["touchdown_speed"] < 0.0


def test_weakened_motor_is_compensated():
    quad = Quadrotor()
    quad.motor_eff_1 = 0.8
    r = fly(quad, Mission.figure8())
    assert r.metrics()["rms_error"] < 0.1
    assert r.metrics()["saturation_fraction"] < 0.05


def test_demanding_mission_saturates_but_stays_finite():
    r = fly(Quadrotor(), Mission.figure8(period=8.0, a=6.0, b=3.0))
    assert not np.isnan(r.result.state).any()
    assert r.metrics()["max_tilt_deg"] <= 40.0


def test_turbulence_run_is_deterministic():
    def run(seed):
        wind = TurbulentWind(1.5, 1.0, 50.0, rng=np.random.default_rng(seed))
        return fly(Quadrotor(wind=wind), Mission.figure8()).metrics()["rms_error"]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_heading_mode_points_along_the_path():
    r = fly(Quadrotor(), Mission.figure8(yaw_mode="heading"))
    t = r.time
    sel = np.array([r.reference.phase(x) == "figure8" for x in t])
    sel &= (t > r.reference.phases[3][1] + 1.0)
    yaw_err = np.abs((r.euler[sel, 2] - np.array([r.reference(x).yaw for x in t[sel]])
                      + math.pi) % (2 * math.pi) - math.pi)
    assert yaw_err.max() < 0.08


def test_fly_needs_t_end_for_open_ended_reference():
    with pytest.raises(ValueError, match="t_end"):
        fly(Quadrotor(), lambda t: pt((0, 0, 0)))


# ------------------------------------------------------------------ results


def test_to_dict_columns_and_lengths(nominal):
    d = nominal.to_dict()
    assert list(d)[:7] == ["time", "x", "y", "z", "roll", "pitch", "yaw"]
    assert all(len(v) == len(nominal.time) for v in d.values())
    assert d["z"] == pytest.approx(nominal.position[:, 2])


def test_to_frame_matches_dict(nominal):
    pd = pytest.importorskip("pandas")
    df = nominal.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns)[:7] == ["time", "x", "y", "z", "roll", "pitch", "yaw"]


def test_to_viewer_mapping_is_a_consistent_rotation():
    """Viewer (X, Y, Z) = (y, z, x); angles keep their values with YXZ order."""
    P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)  # sim -> viewer axes
    assert np.linalg.det(P) == pytest.approx(1.0)  # proper, so right-handed
    roll, pitch, yaw = 0.3, -0.2, 1.1
    R_sim = rotation_matrix(roll, pitch, yaw)

    def rot(axis, a):
        c, s = math.cos(a), math.sin(a)
        return {
            "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
        }[axis]

    # three.js 'YXZ': yaw about Y (up), pitch about X (left), roll about Z (forward)
    R_view = rot("y", yaw) @ rot("x", pitch) @ rot("z", roll)
    assert P @ R_sim @ P.T == pytest.approx(R_view, abs=1e-12)


def test_to_viewer_columns(nominal):
    v = nominal.to_viewer()
    assert v["y"] == pytest.approx(nominal.position[:, 2])  # up
    assert v["z"] == pytest.approx(nominal.position[:, 0])  # forward
    assert v["x"] == pytest.approx(nominal.position[:, 1])
    assert v["yaw"] == pytest.approx(nominal.euler[:, 2])


def test_touchdown_is_none_if_it_never_lands():
    r = fly(Quadrotor(), Hold((0, 0, 3), duration=6.0), t_end=6.0)
    assert r.touchdown() is None
    assert "touchdown_speed" not in r.metrics()
    ground = fly(Quadrotor(), Hold((0, 0, 0), duration=2.0), t_end=2.0)
    assert ground.touchdown() is None
    assert ground.saturation_fraction() == 0.0


def test_gains_are_overridable():
    soft = ControllerGains(kp_pos_xy=0.5, kp_vel_xy=1.5)
    base = fly(Quadrotor(), Mission.figure8()).metrics()["rms_error"]
    slow = fly(Quadrotor(), Mission.figure8(), gains=soft).metrics()["rms_error"]
    assert slow > base
