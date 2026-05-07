"""Tests for simweave.faults — fault injection and dataset generation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import simweave as sw
from simweave.continuous.solver import simulate
from simweave.continuous.systems import MassSpringDamper, ThermalRC
from simweave.faults import (
    FaultDataset,
    FaultInjector,
    FaultProfile,
    ParameterFault,
)
from simweave.faults.recorder import FaultRecorder


# ---------------------------------------------------------------------------
# FaultProfile
# ---------------------------------------------------------------------------


class TestFaultProfile:
    def test_health_before_onset_is_one(self):
        p = FaultProfile(onset_time=100.0, failure_time=200.0)
        assert p.health_index(0.0) == pytest.approx(1.0)
        assert p.health_index(99.9) == pytest.approx(1.0)

    def test_health_at_failure_is_zero(self):
        p = FaultProfile(onset_time=100.0, failure_time=200.0)
        assert p.health_index(200.0) == pytest.approx(0.0)
        assert p.health_index(500.0) == pytest.approx(0.0)

    def test_linear_midpoint(self):
        p = FaultProfile(onset_time=0.0, failure_time=100.0, shape="linear")
        assert p.health_index(50.0) == pytest.approx(0.5)

    def test_exponential_bounds(self):
        p = FaultProfile(onset_time=0.0, failure_time=100.0, shape="exponential")
        hi_start = p.health_index(0.01)
        hi_end = p.health_index(99.99)
        assert hi_start > 0.9, "exponential shape should start near healthy"
        assert hi_end < 0.1, "exponential shape should end near failed"

    def test_abrupt_after_onset_is_zero(self):
        p = FaultProfile(onset_time=50.0, failure_time=100.0, shape="abrupt")
        assert p.health_index(50.0) == pytest.approx(0.0)
        assert p.health_index(75.0) == pytest.approx(0.0)

    def test_callable_shape(self):
        # Custom S-curve: health = cos(pi * progress / 2)^2
        def s_curve(prog):
            return math.cos(math.pi * prog / 2) ** 2

        p = FaultProfile(onset_time=0.0, failure_time=10.0, shape=s_curve)
        assert p.health_index(0.0) == pytest.approx(1.0, abs=1e-6)
        assert p.health_index(10.0) == pytest.approx(0.0, abs=1e-6)

    def test_invalid_time_order_raises(self):
        with pytest.raises(ValueError):
            FaultProfile(onset_time=200.0, failure_time=100.0)

    def test_equal_times_raises(self):
        with pytest.raises(ValueError):
            FaultProfile(onset_time=100.0, failure_time=100.0)

    def test_rul_before_failure(self):
        p = FaultProfile(onset_time=0.0, failure_time=100.0)
        assert p.rul(0.0) == pytest.approx(100.0)
        assert p.rul(60.0) == pytest.approx(40.0)

    def test_rul_after_failure_is_zero(self):
        p = FaultProfile(onset_time=0.0, failure_time=100.0)
        assert p.rul(150.0) == pytest.approx(0.0)

    def test_unknown_shape_raises(self):
        p = FaultProfile(onset_time=0.0, failure_time=10.0, shape="bogus")
        with pytest.raises(ValueError):
            p.health_index(5.0)

    def test_monotone_decreasing_linear(self):
        p = FaultProfile(onset_time=10.0, failure_time=110.0, shape="linear")
        times = np.linspace(10.0, 110.0, 20)
        values = [p.health_index(t) for t in times]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_monotone_decreasing_exponential(self):
        p = FaultProfile(onset_time=10.0, failure_time=110.0, shape="exponential")
        times = np.linspace(10.0, 110.0, 20)
        values = [p.health_index(t) for t in times]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


# ---------------------------------------------------------------------------
# ParameterFault
# ---------------------------------------------------------------------------


class TestParameterFault:
    def test_fields(self):
        p = FaultProfile(onset_time=0.0, failure_time=100.0)
        f = ParameterFault(param="k", profile=p, max_delta=1.5)
        assert f.param == "k"
        assert f.relative is True

    def test_relative_false(self):
        p = FaultProfile(onset_time=0.0, failure_time=100.0)
        f = ParameterFault(param="c", profile=p, max_delta=5.0, relative=False)
        assert f.relative is False


# ---------------------------------------------------------------------------
# FaultInjector
# ---------------------------------------------------------------------------


class TestFaultInjector:
    def _make_injector(self, shape="linear"):
        sys = MassSpringDamper(mass=1.0, stiffness=4.0, damping=0.4)
        profile = FaultProfile(onset_time=1.0, failure_time=5.0, mode="spring_loss", shape=shape)
        fault = ParameterFault(param="stiffness", profile=profile, max_delta=0.75, relative=True)
        return FaultInjector(system=sys, faults=[fault])

    def test_wraps_state(self):
        inj = self._make_injector()
        x0 = inj.initial_state()
        assert x0.shape == (2,)

    def test_state_labels_forwarded(self):
        inj = self._make_injector()
        labels = inj.state_labels()
        assert "position" in labels or len(labels) == 2

    def test_nominal_restored_after_derivatives(self):
        sys = MassSpringDamper(mass=1.0, stiffness=4.0, damping=0.4)
        profile = FaultProfile(onset_time=0.0, failure_time=10.0)
        fault = ParameterFault(param="stiffness", profile=profile, max_delta=2.0, relative=True)
        inj = FaultInjector(system=sys, faults=[fault])
        k_nominal = sys.stiffness
        _ = inj.derivatives(5.0, inj.initial_state())
        assert sys.stiffness == pytest.approx(k_nominal)

    def test_no_faults_health_is_one(self):
        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        inj = FaultInjector(system=sys, faults=[])
        assert inj.overall_health(0.0) == pytest.approx(1.0)
        assert inj.active_mode(0.0) == "healthy"

    def test_overall_health_equals_min(self):
        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        p1 = FaultProfile(onset_time=0.0, failure_time=100.0, mode="A")
        p2 = FaultProfile(onset_time=0.0, failure_time=50.0, mode="B")
        f1 = ParameterFault(param="R_th", profile=p1, max_delta=1.0)
        f2 = ParameterFault(param="C_th", profile=p2, max_delta=0.5)
        inj = FaultInjector(system=sys, faults=[f1, f2])
        # At t=25: p1.health=0.75, p2.health=0.5 → min=0.5
        assert inj.overall_health(25.0) == pytest.approx(0.5)

    def test_active_mode_worst_fault(self):
        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        p1 = FaultProfile(onset_time=0.0, failure_time=100.0, mode="A")
        p2 = FaultProfile(onset_time=0.0, failure_time=50.0, mode="B")
        f1 = ParameterFault(param="R_th", profile=p1, max_delta=1.0)
        f2 = ParameterFault(param="C_th", profile=p2, max_delta=0.5)
        inj = FaultInjector(system=sys, faults=[f1, f2])
        assert inj.active_mode(25.0) == "B"

    def test_active_mode_healthy_before_onset(self):
        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        p = FaultProfile(onset_time=100.0, failure_time=200.0, mode="X")
        inj = FaultInjector(sys, [ParameterFault("R_th", p, 1.0)])
        assert inj.active_mode(0.0) == "healthy"

    def test_simulate_integrates_without_error(self):
        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        profile = FaultProfile(onset_time=10.0, failure_time=80.0)
        fault = ParameterFault(param="R_th", profile=profile, max_delta=2.0)
        inj = FaultInjector(system=sys, faults=[fault])
        result = simulate(inj, t_span=(0.0, 100.0), dt=1.0)
        assert result.state.shape[0] > 0
        assert not np.any(np.isnan(result.state))

    def test_faulted_temperature_higher_than_healthy(self):
        """Increasing R_th should raise steady-state temperature."""
        def heat(t):
            return 20.0

        healthy = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        r_healthy = simulate(healthy, t_span=(0.0, 200.0), dt=1.0, inputs=heat)

        faulted_sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        profile = FaultProfile(onset_time=0.0, failure_time=200.0)
        fault = ParameterFault(param="R_th", profile=profile, max_delta=3.0)
        inj = FaultInjector(faulted_sys, [fault])
        r_faulted = simulate(inj, t_span=(0.0, 200.0), dt=1.0, inputs=heat)

        # Final temperature should be higher in the faulted run
        assert r_faulted.state[-1, 0] > r_healthy.state[-1, 0]

    def test_rul_zero_after_failure(self):
        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        profile = FaultProfile(onset_time=0.0, failure_time=50.0)
        inj = FaultInjector(sys, [ParameterFault("R_th", profile, 1.0)])
        assert inj.overall_rul(100.0) == pytest.approx(0.0)

    def test_name_property(self):
        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        inj = FaultInjector(sys, [])
        assert "ThermalRC" in inj.name


# ---------------------------------------------------------------------------
# FaultDataset
# ---------------------------------------------------------------------------


class TestFaultDataset:
    def _make_dataset(self, shape="linear", noise_std=None) -> FaultDataset:
        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        profile = FaultProfile(onset_time=20.0, failure_time=80.0, mode="test_fault", shape=shape)
        fault = ParameterFault(param="R_th", profile=profile, max_delta=2.0)
        inj = FaultInjector(sys, [fault])
        result = simulate(inj, t_span=(0.0, 100.0), dt=1.0)
        return FaultDataset.from_result(result, inj, noise_std=noise_std,
                                        rng=np.random.default_rng(0))

    def test_lengths_consistent(self):
        ds = self._make_dataset()
        n = len(ds.time)
        assert ds.features.shape[0] == n
        assert ds.health_index.shape[0] == n
        assert ds.rul.shape[0] == n
        assert ds.is_failed.shape[0] == n
        assert ds.failure_mode.shape[0] == n

    def test_health_index_range(self):
        ds = self._make_dataset()
        assert np.all(ds.health_index >= 0.0)
        assert np.all(ds.health_index <= 1.0)

    def test_is_failed_consistent_with_health(self):
        ds = self._make_dataset()
        assert np.all(ds.is_failed == (ds.health_index <= 0.0))

    def test_rul_non_negative(self):
        ds = self._make_dataset()
        assert np.all(ds.rul >= 0.0)

    def test_failure_mode_labels(self):
        ds = self._make_dataset()
        modes = set(ds.failure_mode.tolist())
        assert "healthy" in modes
        assert "test_fault" in modes

    def test_noise_changes_features(self):
        ds_clean = self._make_dataset(noise_std=None)
        ds_noisy = self._make_dataset(noise_std=1.0)
        # Features should differ due to noise
        assert not np.allclose(ds_clean.features, ds_noisy.features)

    def test_feature_names_present(self):
        ds = self._make_dataset()
        assert len(ds.feature_names) >= 1
        assert "temperature" in ds.feature_names

    def test_repr(self):
        ds = self._make_dataset()
        r = repr(ds)
        assert "FaultDataset" in r
        assert "test_fault" in r

    def test_len(self):
        ds = self._make_dataset()
        assert len(ds) == len(ds.time)

    def test_train_test_split_sizes(self):
        ds = self._make_dataset()
        train, test = ds.train_test_split(test_frac=0.2)
        assert len(train) + len(test) == len(ds)
        assert len(test) == pytest.approx(len(ds) * 0.2, abs=2)

    def test_train_test_split_shuffle(self):
        ds = self._make_dataset()
        train_s, test_s = ds.train_test_split(
            test_frac=0.2, shuffle=True, rng=np.random.default_rng(7)
        )
        assert len(train_s) + len(test_s) == len(ds)

    def test_concat(self):
        ds1 = self._make_dataset("linear")
        ds2 = self._make_dataset("exponential")
        combined = FaultDataset.concat([ds1, ds2])
        assert len(combined) == len(ds1) + len(ds2)

    def test_concat_mismatched_names_raises(self):
        ds1 = self._make_dataset()
        sys2 = MassSpringDamper(mass=1.0, stiffness=4.0, damping=0.4)
        profile2 = FaultProfile(onset_time=0.0, failure_time=50.0)
        fault2 = ParameterFault(param="stiffness", profile=profile2, max_delta=1.0)
        inj2 = FaultInjector(sys2, [fault2])
        result2 = simulate(inj2, t_span=(0.0, 100.0), dt=1.0)
        ds2 = FaultDataset.from_result(result2, inj2)
        with pytest.raises(ValueError, match="feature_names"):
            FaultDataset.concat([ds1, ds2])

    def test_concat_empty_raises(self):
        with pytest.raises(ValueError):
            FaultDataset.concat([])

    def test_to_dataframe_requires_pandas(self):
        """to_dataframe should work if pandas is installed, or give a clear error."""
        ds = self._make_dataset()
        try:
            import pandas  # noqa: F401
            df = ds.to_dataframe()
            assert "time" in df.columns
            assert "health_index" in df.columns
            assert "rul" in df.columns
            assert "is_failed" in df.columns
            assert "failure_mode" in df.columns
            assert "temperature" in df.columns
            assert len(df) == len(ds)
        except ImportError:
            with pytest.raises(ImportError, match="pandas"):
                ds.to_dataframe()

    def test_from_recorder(self):
        """FaultRecorder path produces same labels as from_result."""
        from simweave.continuous.solver import ContinuousProcess
        from simweave.core.environment import SimEnvironment

        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        profile = FaultProfile(onset_time=10.0, failure_time=50.0, mode="rc_fault")
        fault = ParameterFault(param="R_th", profile=profile, max_delta=1.0)
        inj = FaultInjector(sys, [fault])

        proc = ContinuousProcess(inj, method="rk4", n_substeps=1)
        recorder = FaultRecorder(inj)

        env = SimEnvironment(dt=1.0, end=60.0)
        env.register(proc)
        env.register(recorder)
        env.run(until=60.0)

        result = proc.result()
        ds = FaultDataset.from_recorder(recorder, result)
        assert len(ds) == len(result.time)
        assert "rc_fault" in set(ds.failure_mode.tolist())


# ---------------------------------------------------------------------------
# FaultRecorder
# ---------------------------------------------------------------------------


class TestFaultRecorder:
    def test_samples_each_tick(self):
        from simweave.continuous.solver import ContinuousProcess
        from simweave.core.environment import SimEnvironment

        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        profile = FaultProfile(onset_time=5.0, failure_time=15.0, mode="m")
        fault = ParameterFault(param="R_th", profile=profile, max_delta=1.0)
        inj = FaultInjector(sys, [fault])

        proc = ContinuousProcess(inj)
        rec = FaultRecorder(inj)

        env = SimEnvironment(dt=1.0, end=20.0)
        env.register(proc)
        env.register(rec)
        env.run(until=20.0)

        assert len(rec.times) > 0
        assert len(rec.health_index) == len(rec.times)
        assert len(rec.rul) == len(rec.times)
        assert len(rec.is_failed) == len(rec.times)
        assert len(rec.failure_mode) == len(rec.times)

    def test_recorder_health_in_range(self):
        from simweave.continuous.solver import ContinuousProcess
        from simweave.core.environment import SimEnvironment

        sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=500.0)
        profile = FaultProfile(onset_time=5.0, failure_time=15.0)
        inj = FaultInjector(sys, [ParameterFault("R_th", profile, 1.0)])

        proc = ContinuousProcess(inj)
        rec = FaultRecorder(inj)
        env = SimEnvironment(dt=1.0, end=20.0)
        env.register(proc)
        env.register(rec)
        env.run(until=20.0)

        assert all(0.0 <= h <= 1.0 for h in rec.health_index)


# ---------------------------------------------------------------------------
# Top-level simweave imports
# ---------------------------------------------------------------------------


def test_top_level_imports():
    assert sw.FaultProfile is FaultProfile
    assert sw.ParameterFault is ParameterFault
    assert sw.FaultInjector is FaultInjector
    assert sw.FaultRecorder is FaultRecorder
    assert sw.FaultDataset is FaultDataset


def test_toml_init_version_match():
    """pyproject.toml version field and simweave.__version__ must agree."""
    toml_path = Path(__file__).parent.parent / "pyproject.toml"
    text = toml_path.read_text(encoding="utf-8")
    # Minimal parse: find 'version = "x.y.z"' under [project]
    import re
    m = re.search(r'^\[project\].*?^version\s*=\s*"([^"]+)"', text, re.M | re.S)
    assert m is not None, "Could not find version in pyproject.toml [project]"
    assert m.group(1) == sw.__version__
