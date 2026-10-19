"""Edge cases, error paths and numpy fallbacks for ``simweave.stats``."""
from __future__ import annotations

import numpy as np
import pytest

from simweave.stats import (
    ClaytonCopula, Deterministic, Empirical, Exponential, Flipped, FrankCopula, GaussianCopula,
    GumbelCopula, JointSampler, LogNormal, MarkovCopulaProcess, Normal, ScipyMarginal,
    StudentTCopula, Uniform, Weibull, as_marginal, fit_copula, kendall_tau, spearman_rho,
    tail_dependence_empirical,
)
from simweave.stats import fit as fit_mod
from simweave.stats import special as sp


def test_kendall_tau_numpy_fallback_matches_and_subsamples(monkeypatch):
    rng = np.random.default_rng(0)
    u = GaussianCopula.from_kendall_tau(0.5).sample_u(1200, rng)
    ref = kendall_tau(u[:, 0], u[:, 1])
    monkeypatch.setattr(sp, "have_scipy", lambda: False)
    assert kendall_tau(u[:, 0], u[:, 1]) == pytest.approx(ref, abs=1e-6)
    with pytest.warns(UserWarning, match="subsampling"):
        t = kendall_tau(u[:, 0], u[:, 1], max_n=500)
    assert t == pytest.approx(0.5, abs=0.12)
    assert kendall_tau(np.ones(10), np.arange(10.0)) == 0.0
    with pytest.raises(ValueError):
        kendall_tau([1.0], [1.0])


def test_spearman_and_pseudo_obs_validation():
    x = np.arange(50.0)
    assert spearman_rho(x, x**3) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fit_mod.pseudo_obs([1.0, np.nan])
    assert fit_mod.pseudo_obs(np.arange(5.0)).shape == (5, 1)


def test_fit_errors_and_options():
    u = GumbelCopula.from_kendall_tau(0.5).sample_u(800, np.random.default_rng(1))
    with pytest.raises(ValueError):
        fit_copula(u, method="bogus")
    with pytest.raises(ValueError):
        fit_copula(u[:, 0])
    with pytest.raises(ValueError):
        fit_copula(np.hstack([u, u[:, :1]]), "clayton")
    for fam in ("frank", "gumbel_survival", "gaussian"):
        assert fit_copula(u, fam).family.startswith(fam)
    r = fit_copula(np.hstack([u, u[:, :1]]), "t")
    assert r.copula.dim == 3


def test_fit_mle_all_families():
    pytest.importorskip("scipy")
    u = FrankCopula.from_kendall_tau(0.5).sample_u(2000, np.random.default_rng(2))
    for fam in ("gaussian", "t", "clayton", "gumbel", "frank", "clayton_survival"):
        r = fit_copula(u, fam, method="mle")
        assert np.isfinite(r.loglik)
    auto = fit_copula(u, method="mle")
    assert auto.table[0]["aic"] == auto.aic


def test_fit_mle_requires_scipy(monkeypatch):
    monkeypatch.setattr(sp, "_scipy_special", lambda: None)
    with pytest.raises(ImportError, match="simweave\\[optim\\]"):
        fit_copula(np.random.default_rng(0).random((50, 2)), "clayton", method="mle")


def test_student_t_copula_extras():
    pytest.importorskip("scipy")
    t = StudentTCopula.from_kendall_tau(-0.4, df=5.0)
    assert t.tail_dependence()["lu"] > 0 and t.tail_dependence()["ll"] == 0
    assert t.rho == pytest.approx(t.corr[0, 1])
    assert t.kendall_tau() == pytest.approx(-0.4, abs=1e-6)
    t3 = StudentTCopula.from_kendall_tau(0.5, dim=3)
    assert t3.sample_u(50, np.random.default_rng(0)).shape == (50, 3)
    assert np.asarray(t3.kendall_tau()).shape == (3, 3)
    with pytest.raises(ValueError):
        t3.rho
    with pytest.raises(ValueError):
        StudentTCopula(0.3, df=0)
    with pytest.raises(ValueError):
        StudentTCopula.from_kendall_tau(1.5)
    with pytest.raises(ValueError):
        t3.log_pdf(np.full((2, 2), 0.5))


def test_copula_validation_and_unsupported_ops():
    g3 = GaussianCopula.from_kendall_tau(0.3, dim=3)
    assert np.asarray(g3.kendall_tau()).shape == (3, 3)
    with pytest.raises(ValueError):
        g3.rho
    with pytest.raises(ValueError):
        GaussianCopula.from_kendall_tau(-0.9, dim=3)
    with pytest.raises(ValueError):
        GaussianCopula.from_kendall_tau(1.0)
    with pytest.raises(ValueError):
        GaussianCopula([[1.0, 0.5, 0.1], [0.5, 1.0]])
    with pytest.raises(ValueError):
        GaussianCopula([[1.0, 0.99, 0.0], [0.99, 1.0, 0.99], [0.0, 0.99, 1.0]])
    with pytest.raises(ValueError):
        GaussianCopula(0.5).log_pdf(np.full((3, 3), 0.5))
    with pytest.raises(ValueError):
        ClaytonCopula(1.0, dim=1)
    with pytest.raises(ValueError):
        GumbelCopula.from_kendall_tau(1.0)
    c3 = ClaytonCopula(2.0, dim=3)
    assert np.isfinite(c3.log_pdf(np.full((2, 3), 0.4))).all()
    assert c3.kendall_tau() == pytest.approx(0.5)
    with pytest.raises(NotImplementedError):
        GumbelCopula(2.0, dim=3).log_pdf(np.full((2, 3), 0.4))
    with pytest.raises(NotImplementedError):
        FrankCopula(2.0, dim=3).log_pdf(np.full((2, 3), 0.4))
    f3 = Flipped(c3, (0,))
    with pytest.raises(NotImplementedError):
        f3.kendall_tau()
    with pytest.raises(NotImplementedError):
        f3.tail_dependence()
    with pytest.raises(NotImplementedError):
        f3.conditional_ppf(0.5, 0.5)
    assert f3.flipped((1,)).axes == (0, 1)
    assert np.isfinite(f3.log_pdf(np.full((2, 3), 0.4))).all()


def test_flipped_conditional_all_axis_combinations():
    base = GumbelCopula.from_kendall_tau(0.5)
    rng = np.random.default_rng(3)
    u, w = rng.random(15_000), rng.random(15_000)
    for axes, sign in (((0,), -1), ((1,), -1), ((0, 1), 1)):
        c = Flipped(base, axes)
        assert kendall_tau(u, c.conditional_ppf(w, u)) == pytest.approx(sign * 0.5, abs=0.03)


def test_frank_and_debye_edges():
    assert sp.frank_tau(0.0) == 0.0
    assert sp.frank_tau(-3.0) == pytest.approx(-sp.frank_tau(3.0))
    assert sp.debye1(60.0) == pytest.approx(np.pi**2 / 360.0)
    with pytest.raises(ValueError):
        sp.debye1(0.0)
    with pytest.raises(ValueError):
        sp.frank_theta_from_tau(0.9999999999)
    with pytest.raises(ValueError):
        FrankCopula.from_kendall_tau(-0.2)


def test_special_scalar_and_scipy_gate(monkeypatch):
    assert float(sp.ndtri_numpy(0.5)) == pytest.approx(0.0, abs=1e-12)
    assert sp.ndtri_numpy(np.array([0.001, 0.999])).shape == (2,)
    monkeypatch.setattr(sp, "_scipy_special", lambda: None)
    assert not sp.have_scipy()
    assert sp.ndtr(0.0) == pytest.approx(0.5)
    assert float(sp.ndtri(0.975)) == pytest.approx(1.95996, abs=1e-4)
    with pytest.raises(ImportError, match="simweave\\[optim\\]"):
        sp.t_cdf(0.0, 3.0)
    with pytest.raises(ImportError):
        sp.t_ppf(0.5, 3.0)
    with pytest.raises(ImportError):
        StudentTCopula(0.3)


def test_marginal_edges():
    scipy_stats = pytest.importorskip("scipy.stats")
    m = ScipyMarginal(scipy_stats.expon(scale=2.0))
    assert m.mean == pytest.approx(2.0)
    assert m.cdf(2.0) == pytest.approx(1 - np.exp(-1))
    assert m.rvs(np.random.default_rng(0), 5).shape == (5,)
    for bad in (lambda: Uniform(2, 1), lambda: Normal(0, 0), lambda: LogNormal(0, 0),
                lambda: LogNormal.from_mean_cv(-1, 1), lambda: Deterministic(-1)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        Empirical([1.0, np.inf])
    assert LogNormal(0, 1).cdf(0.0) == 0.0 and LogNormal(0, 1).cdf(1.0) == pytest.approx(0.5)
    assert Normal(0, 1).cdf(-1.0) == 0.0
    assert Normal(0, 1, clip_nonnegative=False).cdf(0.0) == pytest.approx(0.5)
    assert Weibull(2, 1).cdf(-1.0) == 0.0
    assert Exponential(1.0).cdf(-1.0) == 0.0
    assert Uniform(0, 2).cdf(1.0) == 0.5 and Uniform(0, 2).ppf(0.5) == 1.0
    d = Deterministic(3.0)
    assert d.cdf(2.9) == 0.0 and d.cdf(3.0) == 1.0 and d.ppf([0.1, 0.9]).tolist() == [3.0, 3.0]

    class Duck:
        def ppf(self, u):
            return np.asarray(u) * 2

    dm = as_marginal(Duck())
    assert dm.ppf(0.25) == 0.5 and np.isnan(dm.mean)
    with pytest.raises(AttributeError):
        dm.cdf(1.0)


def test_joint_edges():
    js = JointSampler(GaussianCopula(0.4), {"a": Exponential(1.0), "b": Uniform(0, 1)}, block=4)
    rng = np.random.default_rng(0)
    assert [js.draw(rng)["a"] for _ in range(9)]  # crosses block refills
    with pytest.raises(ValueError):
        JointSampler(GaussianCopula(0.4), {"a": Exponential(1.0), "b": Exponential(1.0)}, block=0)
    data = np.random.default_rng(1).random((300, 2))
    with pytest.raises(ValueError):
        JointSampler.from_data(data, ["a"])
    with pytest.raises(ValueError):
        JointSampler.from_data(data, ["a", "b"], marginals="bogus")
    out = JointSampler.from_data(
        data, ["a", "b"], marginals={"a": Exponential(1.0), "b": Uniform(0, 1)}, family="frank"
    )
    assert out.names == ("a", "b")


def test_measures_and_serial_edges():
    with pytest.raises(ValueError):
        tail_dependence_empirical(np.ones((5, 3)))
    with pytest.raises(ValueError):
        tail_dependence_empirical(np.ones((5, 2)), q=0.9)
    assert np.isnan(tail_dependence_empirical(np.full((5, 2), 0.5), 0.1)["ll"])
    p = MarkovCopulaProcess(GaussianCopula(0.5), Deterministic(2.0))
    assert p(np.random.default_rng(0)) == 2.0
    assert 0 < p.next_u(np.random.default_rng(1)) < 1
