import numpy as np
import pytest

from simweave.mc.runner import MCResult, run_monte_carlo, run_batched_mc


def _scalar_scenario(seed: int) -> float:
    rng = np.random.default_rng(seed)
    return float(rng.normal(0.0, 1.0))


def _vector_scenario(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=3)


def test_mc_result_stats():
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    r = MCResult(n_runs=4, seeds=[0, 1, 2, 3], samples=samples)
    assert r.mean() == pytest.approx(2.5)
    assert r.std() == pytest.approx(np.std(samples))
    assert r.quantile(0.5) == pytest.approx(2.5)


def test_run_serial_returns_expected_count():
    r = run_monte_carlo(_scalar_scenario, n_runs=10, executor="serial")
    assert r.n_runs == 10
    assert len(r.samples) == 10
    assert r.seeds == list(range(10))


def test_run_serial_is_deterministic_by_seed():
    r1 = run_monte_carlo(_scalar_scenario, n_runs=5, executor="serial")
    r2 = run_monte_carlo(_scalar_scenario, n_runs=5, executor="serial")
    assert np.allclose(r1.samples, r2.samples)


def test_run_serial_with_custom_seeds():
    seeds = [100, 200, 300]
    r = run_monte_carlo(_scalar_scenario, n_runs=3, seeds=seeds, executor="serial")
    assert r.seeds == seeds


def test_bad_executor_raises():
    with pytest.raises(ValueError):
        run_monte_carlo(_scalar_scenario, n_runs=3, executor="magic")


def test_seeds_length_mismatch_raises():
    with pytest.raises(ValueError):
        run_monte_carlo(_scalar_scenario, n_runs=3, seeds=[1, 2], executor="serial")


def test_vector_samples_stack_to_array():
    r = run_monte_carlo(_vector_scenario, n_runs=8, executor="serial")
    arr = np.asarray(r.samples)
    assert arr.shape == (8, 3)
    assert np.allclose(r.mean(), arr.mean(axis=0))


def test_threads_executor_runs():
    r = run_monte_carlo(_scalar_scenario, n_runs=5, executor="threads", n_workers=2)
    assert r.n_runs == 5


def test_batched_mc_vectorised():
    def batched(rng, n):
        return rng.normal(0.0, 1.0, size=(n, 4))

    r = run_batched_mc(batched, n_runs=100, seed=42)
    assert r.samples.shape == (100, 4)
    # Sample mean should be near zero for N=100.
    assert abs(float(r.mean().mean())) < 0.5


def test_batched_mc_shape_mismatch_raises():
    def bad(rng, n):
        return rng.normal(0.0, 1.0, size=(n + 1, 2))

    with pytest.raises(ValueError):
        run_batched_mc(bad, n_runs=5, seed=0)


def test_scenario_name_preserved():
    r = run_monte_carlo(
        _scalar_scenario, n_runs=3, executor="serial", scenario_name="demo"
    )
    assert r.scenario_name == "d