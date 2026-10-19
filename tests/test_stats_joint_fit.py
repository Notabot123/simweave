"""JointSampler, MarkovCopulaProcess and copula fitting."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

import simweave as sw
from simweave.stats import (
    ClaytonCopula, Exponential, FrankCopula, GaussianCopula, GumbelCopula, JointSampler,
    LogNormal, MarkovCopulaProcess, fit_copula, kendall_tau, pseudo_obs,
)


def _js(tau=0.6):
    return JointSampler(
        GaussianCopula.from_kendall_tau(tau),
        {"gap": Exponential(1.0), "size": LogNormal.from_mean_cv(2.0, 0.5)},
    )


def test_joint_preserves_marginals_and_dependence():
    x = _js().sample(np.random.default_rng(1), 100_000)
    assert x["gap"].mean() == pytest.approx(1.0, rel=0.02)
    assert x["size"].mean() == pytest.approx(2.0, rel=0.02)
    assert kendall_tau(x["gap"][:20_000], x["size"][:20_000]) == pytest.approx(0.6, abs=0.02)


def test_draw_matches_block_sample_and_is_deterministic():
    js = _js()
    js.block = 64
    first = [js.draw(np.random.default_rng(3))["gap"]]
    rng = np.random.default_rng(3)
    draws = [js.draw(rng) for _ in range(64)]
    ref = _js().sample(np.random.default_rng(3), 64)
    assert [d["gap"] for d in draws] == list(ref["gap"])
    assert first[0] == draws[0]["gap"]
    # a different generator flushes the buffer
    other = js.draw(np.random.default_rng(99))
    assert other["gap"] != draws[1]["gap"]


def test_joint_pickles():
    js = _js()
    js.draw(np.random.default_rng(0))
    js2 = pickle.loads(pickle.dumps(js))
    assert js2.names == js.names


def test_joint_validation():
    with pytest.raises(ValueError):
        JointSampler(GaussianCopula(0.5), {"a": Exponential(1.0)})
    with pytest.raises(ValueError):
        JointSampler(GaussianCopula(0.5), {})


def test_markov_process_preserves_marginal_and_lag1_dependence():
    for cop in (GumbelCopula.from_kendall_tau(0.6), ClaytonCopula.from_kendall_tau(0.6),
                FrankCopula.from_kendall_tau(0.6), GaussianCopula.from_kendall_tau(0.6)):
        proc = MarkovCopulaProcess(cop, Exponential(2.0))
        x = proc.sample(np.random.default_rng(4), 20_000)
        assert x.mean() == pytest.approx(2.0, rel=0.05)
        assert kendall_tau(x[:-1], x[1:]) == pytest.approx(0.6, abs=0.03)
        assert kendall_tau(x[:-2], x[2:]) < 0.6  # dependence decays with lag


def test_markov_process_reset_and_independence():
    proc = MarkovCopulaProcess(GumbelCopula(1.0), Exponential(1.0))
    x = proc.sample(np.random.default_rng(5), 10_000)
    assert abs(kendall_tau(x[:-1], x[1:])) < 0.03
    a = proc(np.random.default_rng(1))
    proc.reset()
    assert proc(np.random.default_rng(1)) == a
    pickle.loads(pickle.dumps(proc))
    with pytest.raises(ValueError):
        MarkovCopulaProcess(GaussianCopula.from_kendall_tau(0.3, dim=3), Exponential(1.0))


def test_pseudo_obs_open_interval_and_ties():
    u = pseudo_obs(np.array([[1, 5], [2, 5], [3, 5], [4, 5]], dtype=float))
    assert u.min() > 0 and u.max() < 1
    assert np.allclose(u[:, 1], 0.5)  # all tied -> average rank
    assert kendall_tau(np.arange(10.0), np.arange(10.0) ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize("make, expect", [
    (lambda: GumbelCopula.from_kendall_tau(0.6), "gumbel"),
    (lambda: ClaytonCopula.from_kendall_tau(0.6), "clayton"),
    (lambda: GaussianCopula.from_kendall_tau(0.6), "gaussian"),
])
def test_fit_auto_recovers_family_and_tau(make, expect):
    wins = 0
    for seed in range(5):
        u = make().sample_u(4_000, np.random.default_rng(seed))
        fit = fit_copula(u)
        assert fit.tau == pytest.approx(0.6, abs=0.04)
        wins += fit.family == expect
    assert wins >= 4


def test_fit_negative_dependence_and_survival_variants():
    u = GumbelCopula.from_kendall_tau(0.6).flipped((1,)).sample_u(4_000, np.random.default_rng(1))
    fit = fit_copula(u)
    assert fit.tau == pytest.approx(-0.6, abs=0.04)
    assert "gumbel" in fit.family and "flipped" in fit.family
    assert fit.copula.kendall_tau() < 0
    assert fit.table[0]["aic"] <= fit.table[-1]["aic"]
    u2 = GumbelCopula.from_kendall_tau(0.6).survival().sample_u(4_000, np.random.default_rng(2))
    assert fit_copula(u2).family in ("gumbel_survival", "clayton")  # lower-tail shapes


def test_fit_explicit_family_and_mle():
    pytest.importorskip("scipy")
    u = ClaytonCopula.from_kendall_tau(0.5).sample_u(5_000, np.random.default_rng(7))
    f_tau = fit_copula(u, "clayton")
    f_mle = fit_copula(u, "clayton", method="mle")
    assert f_mle.loglik >= f_tau.loglik - 1.0
    assert f_mle.copula.theta == pytest.approx(2.0, rel=0.1)
    with pytest.raises(ValueError):
        fit_copula(u, "nonsense")


def test_fit_multivariate_gaussian():
    c = GaussianCopula.from_kendall_tau(0.5, dim=3)
    fit = fit_copula(c.sample_u(5_000, np.random.default_rng(8)))
    assert fit.copula.dim == 3 and fit.tau == pytest.approx(0.5, abs=0.04)


def test_joint_from_data_round_trip():
    rng = np.random.default_rng(9)
    src = JointSampler(
        GumbelCopula.from_kendall_tau(0.6),
        {"gap": Exponential(1.0), "size": LogNormal.from_mean_cv(2.0, 0.5)},
    ).sample_array(rng, 5_000)
    js = JointSampler.from_data(src, ["gap", "size"])
    x = js.sample(np.random.default_rng(1), 20_000)
    assert x["gap"].mean() == pytest.approx(1.0, rel=0.06)
    assert kendall_tau(x["gap"][:10_000], x["size"][:10_000]) == pytest.approx(0.6, abs=0.04)


def test_stats_importable_from_top_level():
    assert sw.JointSampler is JointSampler and sw.fit_copula is fit_copula
