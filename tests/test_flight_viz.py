"""Flight plots, faults through ``fly`` and the Monte Carlo landing workflow."""
from __future__ import annotations

import json

import numpy as np
import pytest

from simweave.continuous.solver import DynamicSystem, simulate
from simweave.faults import FaultInjector, FaultProfile, ParameterFault
from simweave.flight import ConstantWind, Hold, Mission, Quadrotor, TurbulentWind, fly
from simweave.mc import run_monte_carlo


@pytest.fixture(scope="module")
def flight():
    return fly(Quadrotor(wind=ConstantWind((3.0, 0.0, 0.0))), Mission.figure8())


# ------------------------------------------------------------------- plots


def test_plot_trajectory_3d(flight):
    pytest.importorskip("plotly")
    from simweave.viz import plot_trajectory_3d

    fig = plot_trajectory_3d(flight)
    names = [t.name for t in fig.data]
    assert names[:2] == ["reference", "flown"]
    flown = fig.data[1]
    assert len(flown.x) == len(flight.time)
    assert fig.layout.scene.aspectmode == "data"
    json.loads(fig.to_json())  # serialisable for EdgeWeave

    bare = plot_trajectory_3d(flight, show_reference=False, color_by_speed=False,
                              title="mine", theme="dark")
    assert [t.name for t in bare.data][0] == "flown"
    assert bare.layout.title.text == "mine"


def test_plot_trajectory_3d_marks_start_and_end_separately_for_one_way_flights():
    pytest.importorskip("plotly")
    from simweave.viz import plot_trajectory_3d

    one_way = fly(Quadrotor(hover=True, position=(0, 0, 3)),
                  Hold((6.0, 0.0, 3.0), duration=12.0), t_end=12.0)
    fig = plot_trajectory_3d(one_way)
    assert list(fig.data[-1].text) == ["start", "end"]


def test_plot_flight_summary_has_four_panels_and_phase_labels(flight):
    pytest.importorskip("plotly")
    from simweave.viz import plot_flight_summary

    fig = plot_flight_summary(flight)
    assert len([t for t in fig.data if t.name and t.name.startswith("rotor ")]) == 4
    labels = [a.text for a in fig.layout.annotations]
    for phase in ("takeoff", "figure8", "land", "landed"):
        assert phase in labels
    for title in ("Position error (m)", "Tilt (deg)", "Rotor thrust (N)", "Rotor power (W)"):
        assert title in labels
    json.loads(fig.to_json())


def test_plot_flight_summary_without_a_mission():
    pytest.importorskip("plotly")
    from simweave.viz import plot_flight_summary

    plain = fly(Quadrotor(hover=True, position=(0, 0, 3)),
                Hold((0.0, 0.0, 3.0), duration=4.0), t_end=4.0)
    fig = plot_flight_summary(plain, title="hover")
    assert fig.layout.title.text == "hover"
    labels = [a.text for a in fig.layout.annotations]
    # panel titles and the two thrust reference lines, but no mission phases
    assert sorted(labels) == sorted([
        "Position error (m)", "Tilt (deg)", "Rotor thrust (N)", "Rotor power (W)",
        "max thrust", "hover",
    ])


def test_plot_landing_dispersion():
    pytest.importorskip("plotly")
    from simweave.viz import plot_landing_dispersion

    pts = np.random.default_rng(0).normal(0, 0.1, (50, 2))
    fig = plot_landing_dispersion(pts, target=(0.1, -0.1))
    names = [t.name for t in fig.data]
    assert names[:2] == ["landings", "target"]
    assert any(n.startswith("50% within") for n in names)
    assert any(n.startswith("95% within") for n in names)
    assert fig.layout.yaxis.scaleanchor == "x"
    with pytest.raises(ValueError, match="shape"):
        plot_landing_dispersion(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="containment"):
        plot_landing_dispersion(pts, containment=(1.5,))


def test_flight_plots_reexported_at_top_level():
    import simweave as sw

    for name in ("plot_trajectory_3d", "plot_flight_summary", "plot_landing_dispersion"):
        assert callable(getattr(sw, name)) and name in sw.__all__


# ------------------------------------------------------------------ faults


def degrade(quad, loss, onset=12.0, failure=26.0, param="motor_eff_1"):
    profile = FaultProfile(onset_time=onset, failure_time=failure, mode="rotor", shape="linear")
    return FaultInjector(quad, [ParameterFault(param, profile, max_delta=-loss)])


def test_fly_accepts_a_fault_injector_and_compensates_moderate_loss():
    quad = Quadrotor()
    r = fly(degrade(quad, 0.4), Mission.figure8())
    assert r.vehicle is quad
    m = r.metrics()
    assert m["rms_error"] < 0.1
    assert m["saturation_fraction"] == 0.0
    healthy = fly(Quadrotor(), Mission.figure8()).metrics()
    assert m["energy_wh"] > healthy["energy_wh"]  # the other rotors work harder
    assert quad.motor_eff_1 == 1.0  # nominal restored


def test_losing_more_than_the_thrust_margin_loses_the_vehicle():
    quad = Quadrotor()
    limit = 1.0 - quad.hover_thrust / quad.max_thrust  # about 58 %
    r = fly(degrade(Quadrotor(), limit + 0.1), Mission.figure8())
    airborne = (r.time > 8) & (r.time < 34)
    assert r.position[airborne, 2].min() < 0.5  # came down
    assert r.metrics()["saturation_fraction"] > 0.2
    assert not np.isnan(r.result.state).any()


def test_fault_injector_forwards_post_step_so_the_ground_still_stops_it():
    """Regression: a wrapped system's post_step hook used to be dropped."""
    r = fly(degrade(Quadrotor(), 0.7, onset=0.0, failure=1.0), Mission.figure8())
    assert r.position[:, 2].min() >= 0.0


def test_fault_injector_post_step_is_a_no_op_without_a_hook():
    class Plain(DynamicSystem):
        value = 1.0

        def initial_state(self):
            return np.array([1.0])

        def derivatives(self, t, state, inputs=None):
            return -self.value * state

    profile = FaultProfile(onset_time=0.0, failure_time=1.0, mode="x", shape="linear")
    inj = FaultInjector(Plain(), [ParameterFault("value", profile, max_delta=1.0)])
    state = np.array([2.0])
    assert inj.post_step(state) is state
    res = simulate(inj, (0.0, 1.0), dt=0.01)
    assert res.state[-1, 0] < 1.0


def test_fly_rejects_unknown_vehicle():
    with pytest.raises(TypeError, match="Quadrotor"):
        fly(object(), Hold((0, 0, 0), duration=1.0), t_end=1.0)


# ----------------------------------------------------------------- touchdown


def test_touchdown_reports_where_the_vehicle_landed(flight):
    td = flight.touchdown()
    assert set(td) == {"time", "vertical_speed", "horizontal_error", "x", "y"}
    assert np.hypot(td["x"], td["y"]) == pytest.approx(td["horizontal_error"])
    assert 34.0 < td["time"] < 41.0


# ---------------------------------------------------------- Monte Carlo flow


def _landing_scenario(seed: int):
    rng = np.random.default_rng(seed)
    mission = Mission.figure8(laps=0.5, climb_time=3.0, settle_time=1.0,
                              blend_time=2.0, hover_time=1.0, land_time=4.0)
    wind = TurbulentWind(1.0, 1.5, mission.duration + 1.0,
                         mean=(rng.uniform(-5, 5), rng.uniform(-5, 5), 0.0), rng=rng)
    quad = Quadrotor(wind=wind)
    quad.mass *= rng.uniform(0.9, 1.2)
    td = fly(quad, mission).touchdown()
    return [td["x"], td["y"], td["vertical_speed"]]


def test_monte_carlo_landing_dispersion_is_reproducible_and_sane():
    a = run_monte_carlo(_landing_scenario, n_runs=4, executor="serial")
    b = run_monte_carlo(_landing_scenario, n_runs=4, executor="serial")
    assert np.array_equal(a.samples, b.samples)
    samples = np.asarray(a.samples)
    assert samples.shape == (4, 3)
    assert np.all(np.hypot(samples[:, 0], samples[:, 1]) < 0.3)
    assert np.all(samples[:, 2] < 0) and np.all(samples[:, 2] > -1.0)
    assert len({tuple(row) for row in samples}) == 4  # seeds give different flights
