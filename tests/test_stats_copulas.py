"""Copula sampling: rank correlation, tail dependence, densities, conditionals."""
from __future__ import annotations

import numpy as np
import pytest

from simweave.stats import (
    ClaytonCopula, Flipped, FrankCopula, GaussianCopula, GumbelCopula, StudentTCopula,
    kendall_tau, tail_dependence_empirical,
)
from simweave.stats import special as sp

BASE = {
    "gaussian": GaussianCopula.from_kendall_tau,
    "clayton": ClaytonCopula.from_kendall_tau,
    "gumbel": GumbelCopula.from_kendall_tau,
    "frank": FrankCopula.from_kendall_tau,
}


def _make(name, tau):
    if name == "t":
        pytest.importorskip("scipy")
        return StudentTCopula.from_kendall_tau(tau, df=4.0)
    return BASE[name](tau)


FAMILIES = ["gaussian", "t", "clayton", "gumbel", "frank"]


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
def test_kendall_tau_recovered(name, tau):
    c = _make(name, tau)
    u = c.sample_u(20_000, np.random.default_rng(3))
    assert c.kendall_tau() == pytest.approx(tau, abs=1e-6)
    assert kendall_tau(u[:, 0], u[:, 1]) == pytest.approx(tau, abs=0.02)


@pytest.mark.parametrize("name", FAMILIES)
def test_uniform_margins_and_range(name):
    u = _make(name, 0.6).sample_u(50_000, np.random.default_rng(4))
    assert u.min() > 0 and u.max() < 1
    for j in range(2):
        assert np.sort(u[:, j])[[5_000, 25_000, 45_000]] == pytest.approx([0.1, 0.5, 0.9], abs=0.015)


@pytest.mark.parametrize("name", FAMILIES)
def test_density_integrates_to_one(name):
    c = _make(name, 0.5)
    g = (np.arange(300) + 0.5) / 300
    pts = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    assert np.exp(c.log_pdf(pts)).mean() == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("name", FAMILIES)
def test_conditional_ppf_reproduces_dependence(name):
    c = _make(name, 0.6)
    rng = np.random.default_rng(5)
    u, w = rng.random(20_000), rng.random(20_000)
    v = c.conditional_ppf(w, u)
    assert kendall_tau(u, v) == pytest.approx(0.6, abs=0.02)
    assert np.sort(v)[[2_000, 10_000, 18_000]] == pytest.approx([0.1, 0.5, 0.9], abs=0.02)


def test_tail_dependence_matches_theory():
    q = 0.01
    rng = np.random.default_rng(6)
    cl = ClaytonCopula.from_kendall_tau(0.5)
    emp = tail_dependence_empirical(cl.sample_u(400_000, rng), q)
    assert emp["ll"] == pytest.approx(cl.tail_dependence()["ll"], abs=0.08)
    assert emp["uu"] < 0.25
    gu = GumbelCopula.from_kendall_tau(0.5)
    emp = tail_dependence_empirical(gu.sample_u(400_000, rng), q)
    assert emp["uu"] == pytest.approx(gu.tail_dependence()["uu"], abs=0.08)
    assert emp["ll"] < 0.2
    ga = GaussianCopula.from_kendall_tau(0.5)
    assert tail_dependence_empirical(ga.sample_u(400_000, rng), q)["uu"] < 0.35
    assert ga.tail_dependence()["uu"] == 0.0


def test_student_t_tail_dependence_positive():
    pytest.importorskip("scipy")
    t = StudentTCopula.from_kendall_tau(0.5, df=3.0)
    td = t.tail_dependence()
    assert td["ll"] == td["uu"] > 0.3
    emp = tail_dependence_empirical(t.sample_u(400_000, np.random.default_rng(8)), 0.01)
    assert emp["uu"] == pytest.approx(td["uu"], abs=0.08)


def test_flipped_negates_tau_and_moves_tail():
    g = GumbelCopula.from_kendall_tau(0.5)
    f = g.flipped((1,))
    assert isinstance(f, Flipped)
    assert f.kendall_tau() == pytest.approx(-0.5)
    assert f.tail_dependence()["ul"] == pytest.approx(g.tail_dependence()["uu"])
    u = f.sample_u(20_000, np.random.default_rng(9))
    assert kendall_tau(u[:, 0], u[:, 1]) == pytest.approx(-0.5, abs=0.02)
    # the joint extreme is "u1 large, u2 small"
    emp = tail_dependence_empirical(f.sample_u(400_000, np.random.default_rng(9)), 0.01)
    assert emp["ul"] > 0.3 and emp["uu"] < 0.1
    assert f.flipped((1,)) is g  # double flip cancels
    s = g.survival()
    assert s.tail_dependence()["ll"] == pytest.approx(g.tail_dependence()["uu"])
    assert s.kendall_tau() == pytest.approx(0.5)


def test_flipped_conditional_ppf_matches_sampling():
    f = GumbelCopula.from_kendall_tau(0.6).flipped((1,))
    rng = np.random.default_rng(10)
    u, w = rng.random(20_000), rng.random(20_000)
    assert kendall_tau(u, f.conditional_ppf(w, u)) == pytest.approx(-0.6, abs=0.02)


def test_multivariate_gaussian_and_archimedean():
    g = GaussianCopula.from_kendall_tau(0.5, dim=4)
    u = g.sample_u(20_000, np.random.default_rng(11))
    assert u.shape == (20_000, 4)
    assert kendall_tau(u[:, 0], u[:, 3]) == pytest.approx(0.5, abs=0.03)
    c = ClaytonCopula(2.0, dim=3)
    u = c.sample_u(20_000, np.random.default_rng(12))
    assert kendall_tau(u[:, 0], u[:, 2]) == pytest.approx(0.5, abs=0.03)


def test_seeded_reproducibility():
    c = GumbelCopula.from_kendall_tau(0.5)
    a = c.sample_u(100, np.random.default_rng(1))
    b = c.sample_u(100, np.random.default_rng(1))
    d = c.sample_u(100, np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, d)


def test_gumbel_theta_one_is_independence():
    u = GumbelCopula(1.0).sample_u(20_000, np.random.default_rng(1))
    assert abs(kendall_tau(u[:, 0], u[:, 1])) < 0.03


def test_frank_tau_inversion_round_trip():
    for tau in (0.05, 0.3, 0.7, 0.9):
        th = sp.frank_theta_from_tau(tau)
        assert sp.frank_tau(th) == pytest.approx(tau, abs=1e-6)


def test_validation():
    with pytest.raises(ValueError):
        ClaytonCopula.from_kendall_tau(-0.3)
    with pytest.raises(ValueError):
        GumbelCopula(0.5)
    with pytest.raises(ValueError):
        GaussianCopula([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):
        GaussianCopula(1.0)
    with pytest.raises(ValueError):
        FrankCopula(-1.0)
    with pytest.raises(ValueError):
        GumbelCopula(2.0).flipped((2,))
