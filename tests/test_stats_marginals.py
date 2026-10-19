"""Marginals, special functions and the ``ppf`` attributes on discrete helpers."""
from __future__ import annotations

import numpy as np
import pytest

import simweave as sw
from simweave.stats import (
    Deterministic, Empirical, Exponential, LogNormal, Normal, Uniform, Weibull, as_marginal,
)
from simweave.stats import special as sp


def test_numpy_normal_fallbacks_match_scipy():
    pytest.importorskip("scipy")
    p = np.concatenate([np.logspace(-10, -1, 50), np.linspace(0.1, 0.9, 50), 1 - np.logspace(-10, -1, 50)])
    assert np.allclose(sp.ndtri_numpy(p), sp.ndtri(p), rtol=1e-8, atol=1e-9)
    x = np.linspace(-8, 8, 201)
    assert np.allclose(sp.ndtr_numpy(x), sp.ndtr(x), rtol=1e-10, atol=1e-14)


def test_ndtr_ndtri_round_trip():
    p = np.linspace(1e-6, 1 - 1e-6, 101)
    assert np.allclose(sp.ndtr_numpy(sp.ndtri_numpy(p)), p, atol=1e-12)


@pytest.mark.parametrize(
    "marg, mean",
    [
        (Exponential(2.0), 2.0),
        (Uniform(1.0, 5.0), 3.0),
        (LogNormal.from_mean_cv(3.0, 0.8), 3.0),
        (Weibull(1.5, 2.0), None),
        (Normal(10.0, 1.0), 10.0),
    ],
)
def test_ppf_cdf_inverse_and_moments(marg, mean):
    u = np.linspace(0.01, 0.99, 99)
    assert np.allclose(marg.cdf(marg.ppf(u)), u, atol=1e-7)
    x = marg.rvs(np.random.default_rng(1), 200_000)
    m = mean if mean is not None else marg.mean
    assert x.mean() == pytest.approx(m, rel=0.02)


def test_marginal_is_callable_distribution():
    rng = np.random.default_rng(0)
    assert Exponential(1.0)(rng) > 0
    assert sw.EntityProperties(service_time=Exponential(1.0)).draw_service_time(rng) > 0


def test_deterministic_and_empirical():
    assert Deterministic(2.5).ppf(0.3) == 2.5
    data = np.random.default_rng(2).lognormal(0, 1, 5000)
    e = Empirical(data)
    q = e.ppf([0.1, 0.5, 0.9])
    assert np.allclose(q, np.quantile(data, [0.1, 0.5, 0.9]), rtol=0.02)
    assert e.cdf(np.median(data)) == pytest.approx(0.5, abs=0.01)


def test_as_marginal_accepts_scipy_arrays_and_helper_closures():
    stats = pytest.importorskip("scipy.stats")
    m = as_marginal(stats.gamma(2.0))
    assert m.ppf(0.5) == pytest.approx(stats.gamma(2.0).ppf(0.5))
    assert isinstance(as_marginal([1.0, 2.0, 3.0]), Empirical)
    ex = as_marginal(sw.exponential(2.0))
    assert ex.ppf(0.5) == pytest.approx(2.0 * np.log(2.0))
    with pytest.raises(TypeError):
        as_marginal(object())


def test_existing_helpers_expose_ppf_without_changing_draws():
    for dist in (sw.exponential(2.0), sw.uniform(1, 3), sw.normal(5, 1), sw.deterministic(4)):
        assert callable(dist) and callable(dist.ppf) and callable(dist.cdf)
    a = sw.exponential(2.0)(np.random.default_rng(7))
    b = float(np.random.default_rng(7).exponential(2.0))
    assert a == b
    assert sw.normal(0, 1, clip_nonnegative=True).ppf(0.1) == 0.0
    assert sw.normal(0, 1, clip_nonnegative=False).ppf(0.5) == pytest.approx(0.0, abs=1e-9)


def test_validation():
    with pytest.raises(ValueError):
        Exponential(0)
    with pytest.raises(ValueError):
        Weibull(0, 1)
    with pytest.raises(ValueError):
        Empirical([1.0])


def test_helper_closure_ppf_cdf_values():
    e, u, n, d = sw.exponential(2.0), sw.uniform(1, 3), sw.normal(5, 1), sw.deterministic(4)
    assert e.cdf(e.ppf(0.3)) == pytest.approx(0.3) and e.cdf(-1.0) == 0.0 and e.mean == 2.0
    assert u.ppf(0.5) == 2.0 and u.cdf(2.0) == 0.5 and u.mean == 2.0
    assert n.cdf(5.0) == pytest.approx(0.5) and n.cdf(-1.0) == 0.0
    assert d.ppf(0.2) == 4.0 and d.ppf(np.array([0.1, 0.9])).tolist() == [4.0, 4.0]
    assert d.cdf(3.0) == 0.0 and d.cdf(4.0) == 1.0 and d.mean == 4.0
    free = sw.normal(0, 1, clip_nonnegative=False)
    assert free.cdf(0.0) == pytest.approx(0.5) and free.ppf(0.975) == pytest.approx(1.96, abs=0.01)
