"""Tests for ``simweave.viz``.

The recorder tests run unconditionally because they only depend on numpy.
The plot-helper tests are skipped automatically when plotly is not
installed via ``pytest.importorskip``, so the test suite stays green for
contributors who haven't pulled the ``[viz]`` extra.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import simweave as sw
from simweave.viz import themes as themes_mod
from simweave.viz._plotly import have_plotly


# --------------------------------------------------------------------------- #
# Module-import sanity                                                        #
# --------------------------------------------------------------------------- #


def test_top_level_reexports_viz_symbols():
    for name in (
        "QueueLengthRecorder",
        "ServiceUtilisationRecorder",
        "WarehouseStockRecorder",
        "set_default_theme",
        "register_theme",
        "available_themes",
        "have_plotly",
        "plot_state_trajectories",
        "plot_phase_portrait",
        "plot_queue_length",
        "plot_service_utilisation",
        "plot_warehouse_stock",
        "plot_mc_fan",
        "plot_agent_path",
    ):
        assert hasattr(sw, name), f"top-level simweave is missing {name!r}"


def test_have_plotly_returns_bool():
    assert isinstance(have_plotly(), bool)


# --------------------------------------------------------------------------- #
# Themes                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _restore_default_theme():
    """Each test runs against a known default; restore afterwards."""
    saved = themes_mod.get_default_theme()
    yield
    themes_mod.set_default_theme(saved)


def test_built_in_themes_present():
    names = sw.available_themes()
    for expected in ("light", "dark", "presentation", "minimal"):
        assert expected in names


def test_set_default_theme_round_trip():
    sw.set_default_theme("dark")
    assert sw.get_default_theme() == "dark"
    sw.set_default_theme("light")
    assert sw.get_default_theme() == "light"


def test_set_default_theme_rejects_unknown():
    with pytest.raises(KeyError):
        sw.set_default_theme("does-not-exist")


def test_register_theme_then_select():
    sw.register_theme(
        "test-brand",
        template="plotly_white",
        palette=("#000", "#fff"),
        layout_overrides={"font": {"family": "Comic Sans MS"}},
    )
    assert "test-brand" in sw.available_themes()
    sw.set_default_theme("test-brand")
    assert sw.get_default_theme() == "test-brand"


def test_register_theme_duplicate_without_overwrite():
    sw.register_theme("dup", template="plotly_white")
    with pytest.raises(KeyError):
        sw.register_theme("dup", template="plotly_dark")
    # overwrite=True replaces silently
    sw.register_theme("dup", template="plotly_dark", overwrite=True)


# --------------------------------------------------------------------------- #
# Recorders -- no plotly required                                             #
# --------------------------------------------------------------------------- #


def test_queue_length_recorder_tracks_length():
    env = sw.SimEnvironment(dt=1.0, end=5.0)
    q = sw.Queue(maxlen=10, name="q")
    rec = sw.QueueLengthRecorder(q)
    env.register(q)
    env.register(rec)
    q.enqueue(sw.Entity())
    q.enqueue(sw.Entity())
    env.run(until=2.0)
    q.dequeue()
    env.run(until=5.0)
    # Times: 0, 1, 2, 3, 4, 5 (registration + 5 ticks)
    assert rec.times[0] == 0.0
    assert rec.times[-1] == pytest.approx(5.0)
    assert len(rec.times) == len(rec.lengths)
    # First sample sees 0; after enqueues, length goes to >= 1; after dequeue, 1.
    assert max(rec.lengths) >= 2
    assert rec.lengths[-1] == 1


def test_service_utilisation_recorder_shapes():
    env = sw.SimEnvironment(dt=0.5, end=10.0)
    sink = sw.Queue(maxlen=1000, name="sink")
    svc = sw.Service(
        capacity=2, buffer_size=10, next_q=sink,
        default_service_time=0.5,
        rng=np.random.default_rng(0), name="svc",
    )
    gen = sw.ArrivalGenerator(
        interarrival=lambda r: 0.4,
        factory=lambda env: sw.Entity(),
        target=svc, rng=np.random.default_rng(1), name="gen",
    )
    rec = sw.ServiceUtilisationRecorder(svc)
    env.register(gen)
    env.register(svc)
    env.register(sink)
    env.register(rec)
    env.run()
    n = len(rec.times)
    assert n > 1
    busy = np.asarray(rec.busy_time)
    assert busy.shape == (n, 2)
    util = np.asarray(rec.utilisation)
    assert util.shape == (n,)
    # Utilisation must lie in [0, 1] (within numerical slack at t=0).
    assert util.min() >= -1e-9
    assert util.max() <= 1.0 + 1e-9
    # Cumulative busy time is monotone non-decreasing per channel.
    for ch in range(2):
        diffs = np.diff(busy[:, ch])
        assert (diffs >= -1e-9).all()


def test_warehouse_stock_recorder_captures_changes():
    inv = sw.InventoryItems(
        part_names=["a", "b"],
        unit_cost=[1.0, 1.0],
        stock_level=[10.0, 5.0],
        batchsize=[10.0, 5.0],
        reorder_points=[2.0, 1.0],
        repairable_prc=[0.0, 0.0],
        repair_times=[1.0, 1.0],
        newbuy_leadtimes=[1.0, 1.0],
    )
    wh = sw.Warehouse(inventory=inv, name="wh")
    rec = sw.WarehouseStockRecorder(wh)
    env = sw.SimEnvironment(dt=1.0, end=5.0)
    env.register(wh)
    env.register(rec)
    env.run(until=2.0)
    wh.decrement_vector(np.array([1.0, 0.5]))
    env.run(until=5.0)
    stock = np.asarray(rec.stock)
    assert stock.shape[0] == len(rec.times)
    assert stock.shape[1] == 2
    assert rec.sku_names == ("a", "b")
    np.testing.assert_allclose(rec.reorder_points, [2.0, 1.0])
    # Some sample after the decrement should be lower than the initial.
    assert (stock[-1] <= stock[0]).any()


# --------------------------------------------------------------------------- #
# Plot helpers -- skipped when plotly absent                                  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def plotly_go() -> Any:
    return pytest.importorskip("plotly.graph_objects")


def _is_figure(obj: Any) -> bool:
    return obj.__class__.__name__ == "Figure"


def _round_trip_json(fig: Any) -> dict:
    blob = fig.to_json()
    parsed = json.loads(blob)
    assert "data" in parsed and "layout" in parsed
    return parsed


def test_plot_helpers_raise_without_plotly(monkeypatch):
    """Force the lazy import to fail and confirm the error is friendly."""
    if have_plotly():
        # Simulate plotly missing by sabotaging the lazy importer.
        import simweave.viz._plotly as p
        monkeypatch.setattr(
            p,
            "require_go",
            lambda: (_ for _ in ()).throw(ImportError("simweave.viz requires plotly.")),
        )
    with pytest.raises(ImportError) as excinfo:
        sw.plot_state_trajectories(_make_dummy_result())
    assert "plotly" in str(excinfo.value).lower()


def _make_dummy_result():
    """Build a minimal SimulationResult-like object for trajectory plots."""
    t = np.linspace(0, 1, 11)
    state = np.column_stack([np.sin(t), np.cos(t)])
    return sw.SimulationResult(
        time=t, state=state,
        state_labels=("x", "v"),
        system_name="dummy", method="rk4",
    )


def test_plot_state_trajectories_returns_figure(plotly_go):
    fig = sw.plot_state_trajectories(_make_dummy_result())
    assert _is_figure(fig)
    parsed = _round_trip_json(fig)
    assert len(parsed["data"]) == 2  # one trace per state channel


def test_plot_phase_portrait_returns_figure(plotly_go):
    fig = sw.plot_phase_portrait(_make_dummy_result(), x_idx=0, y_idx=1)
    assert _is_figure(fig)
    parsed = _round_trip_json(fig)
    # trajectory + start + end
    assert len(parsed["data"]) == 3


def test_plot_phase_portrait_rejects_missing_channel(plotly_go):
    res = _make_dummy_result()
    with pytest.raises(IndexError):
        sw.plot_phase_portrait(res, x_idx=0, y_idx=5)


def test_plot_queue_length_returns_figure(plotly_go):
    env = sw.SimEnvironment(dt=1.0, end=5.0)
    q = sw.Queue(maxlen=5, name="q")
    rec = sw.QueueLengthRecorder(q)
    env.register(q)
    env.register(rec)
    q.enqueue(sw.Entity())
    env.run()
    fig = sw.plot_queue_length(rec)
    assert _is_figure(fig)
    _round_trip_json(fig)


def test_plot_service_utilisation_returns_figure(plotly_go):
    env = sw.SimEnvironment(dt=0.5, end=4.0)
    sink = sw.Queue(maxlen=100, name="sink")
    svc = sw.Service(capacity=2, buffer_size=5, next_q=sink,
                     default_service_time=0.5, name="svc")
    rec = sw.ServiceUtilisationRecorder(svc)
    env.register(svc)
    env.register(sink)
    env.register(rec)
    svc.enqueue(sw.Entity())
    env.run()
    fig = sw.plot_service_utilisation(rec)
    assert _is_figure(fig)
    _round_trip_json(fig)


def test_plot_warehouse_stock_returns_figure(plotly_go):
    inv = sw.InventoryItems(
        part_names=["a"],
        unit_cost=[1.0],
        stock_level=[10.0],
        batchsize=[5.0],
        reorder_points=[3.0],
        repairable_prc=[0.0],
        repair_times=[1.0],
        newbuy_leadtimes=[1.0],
    )
    wh = sw.Warehouse(inventory=inv, name="wh")
    rec = sw.WarehouseStockRecorder(wh)
    env = sw.SimEnvironment(dt=1.0, end=3.0)
    env.register(wh)
    env.register(rec)
    env.run()
    fig = sw.plot_warehouse_stock(rec)
    assert _is_figure(fig)
    _round_trip_json(fig)


def test_plot_mc_fan_with_array(plotly_go):
    rng = np.random.default_rng(0)
    samples = np.cumsum(rng.normal(0, 0.1, size=(50, 30)), axis=1)
    fig = sw.plot_mc_fan(samples)
    assert _is_figure(fig)
    _round_trip_json(fig)


def test_plot_mc_fan_with_mc_result_and_times(plotly_go):
    rng = np.random.default_rng(0)
    samples = np.cumsum(rng.normal(0, 0.1, size=(20, 40)), axis=1)
    times = np.linspace(0, 4, 40)
    mc = sw.MCResult(n_runs=20, seeds=list(range(20)), samples=samples,
                     scenario_name="walk")
    fig = sw.plot_mc_fan(mc, times=times)
    assert _is_figure(fig)
    parsed = _round_trip_json(fig)
    assert any("walk" in str(layout) for layout in [parsed.get("layout", {})])


def test_plot_mc_fan_with_tuple_input(plotly_go):
    times = np.linspace(0, 1, 10)
    samples = np.random.default_rng(0).normal(size=(15, 10))
    fig = sw.plot_mc_fan((times, samples))
    assert _is_figure(fig)


def test_plot_mc_fan_rejects_1d(plotly_go):
    with pytest.raises(ValueError):
        sw.plot_mc_fan(np.array([1.0, 2.0, 3.0]))


def test_plot_mc_fan_rejects_times_mismatch(plotly_go):
    samples = np.zeros((5, 10))
    with pytest.raises(ValueError):
        sw.plot_mc_fan(samples, times=np.arange(7))


def test_plot_agent_path_returns_figure(plotly_go):
    g = sw.grid_graph(4, 4, diagonal=False)
    agent = sw.Agent(graph=g, start_node=(0, 0), tasks=[(3, 3)],
                     speed=1.0, heuristic=sw.manhattan, name="bot")
    env = sw.SimEnvironment(dt=0.5, end=20.0)
    env.register(agent)
    env.run()
    fig = sw.plot_agent_path(agent, graph=g)
    assert _is_figure(fig)
    _round_trip_json(fig)


def test_theme_application_changes_template(plotly_go):
    res = _make_dummy_result()
    fig_light = sw.plot_state_trajectories(res, theme="light")
    fig_dark = sw.plot_state_trajectories(res, theme="dark")
    # Templates differ -> serialised JSON differs.
    assert fig_light.layout.template != fig_dark.layout.template
