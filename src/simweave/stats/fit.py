"""Rank statistics and copula fitting from data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from simweave.stats import special as sp
from simweave.stats.copulas import (
    ClaytonCopula,
    Copula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
)


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged."""
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    hi = np.cumsum(counts)
    return (hi - (counts - 1) / 2.0)[inv.ravel()]


def pseudo_obs(data) -> np.ndarray:
    """Rank-transform each column to ``rank / (n + 1)`` (open unit interval)."""
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if not np.all(np.isfinite(x)):
        raise ValueError("data must be finite.")
    n = x.shape[0]
    return np.column_stack([_avg_ranks(x[:, j]) for j in range(x.shape[1])]) / (n + 1.0)


def kendall_tau(x, y, *, max_n: int = 10_000) -> float:
    """Kendall's tau-b.

    Uses scipy when present; otherwise an O(n^2) numpy pass (subsampled with a
    warning above ``max_n`` points).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("x and y must have equal length >= 2.")
    if sp.have_scipy():
        from scipy.stats import kendalltau

        return float(kendalltau(x, y).statistic)
    if x.size > max_n:
        warnings.warn(
            f"kendall_tau without scipy is O(n^2); subsampling {max_n} of {x.size} points.",
            stacklevel=2,
        )
        idx = np.random.default_rng(0).choice(x.size, max_n, replace=False)
        x, y = x[idx], y[idx]
    num = dx2 = dy2 = 0.0
    for i in range(0, x.size, 500):
        sx = np.sign(x[i : i + 500, None] - x[None, :])
        sy = np.sign(y[i : i + 500, None] - y[None, :])
        num += float((sx * sy).sum())
        dx2 += float((sx * sx).sum())
        dy2 += float((sy * sy).sum())
    return num / np.sqrt(dx2 * dy2) if dx2 > 0 and dy2 > 0 else 0.0


def spearman_rho(x, y) -> float:
    r = pseudo_obs(np.column_stack([x, y]))
    return float(np.corrcoef(r[:, 0], r[:, 1])[0, 1])


@dataclass
class FitResult:
    """Outcome of :func:`fit_copula`."""

    copula: Copula
    family: str
    tau: float
    loglik: float
    aic: float
    n: int
    table: list[dict] = field(default_factory=list)  # all candidates, best first

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FitResult(family={self.family!r}, tau={self.tau:.3f}, aic={self.aic:.1f}, n={self.n})"


_FAMILIES = {
    "gaussian": GaussianCopula,
    "t": StudentTCopula,
    "clayton": ClaytonCopula,
    "gumbel": GumbelCopula,
    "frank": FrankCopula,
}
_OFFERED_AUTO = ("gaussian", "t", "clayton", "clayton_survival", "gumbel", "gumbel_survival", "frank")


def _build(name: str, tau: float, df: float) -> Copula:
    tau = float(np.clip(tau, 1e-3, 0.97))
    surv = name.endswith("_survival")
    base = name.removesuffix("_survival")
    if base == "gaussian":
        cop = GaussianCopula.from_kendall_tau(tau)
    elif base == "t":
        cop = StudentTCopula.from_kendall_tau(tau, df=df)
    else:
        cop = _FAMILIES[base].from_kendall_tau(tau)
    return cop.survival() if surv else cop


def _refine_mle(name: str, cop: Copula, u: np.ndarray, df: float) -> Copula:
    from scipy.optimize import minimize_scalar

    base = name.removesuffix("_survival")
    surv = name.endswith("_survival")

    def make(x: float) -> Copula:
        if base == "gaussian":
            return GaussianCopula(x)
        if base == "t":
            return StudentTCopula(x, df)
        return _FAMILIES[base](x)

    inner = cop.base if surv else cop
    x0 = inner.rho if base in ("gaussian", "t") else inner.theta
    lo, hi = {
        "gaussian": (-0.999, 0.999),
        "t": (-0.999, 0.999),
        "clayton": (1e-3, max(x0 * 4, 1.0)),
        "gumbel": (1.0, max(x0 * 3, 1.5)),
        "frank": (1e-3, max(x0 * 4, 1.0)),
    }[base]
    res = minimize_scalar(
        lambda x: -float(make(x).log_pdf(u).sum()), bounds=(lo, hi), method="bounded"
    )
    best = make(float(res.x))
    return best.survival() if surv else best


def fit_copula(
    data,
    family: str = "auto",
    *,
    method: str = "tau",
    df: float = 5.0,
) -> FitResult:
    """Fit a bivariate copula (or a Gaussian/t copula of higher dimension).

    Parameters
    ----------
    data:
        ``(n, d)`` observations. Only ranks are used, so marginals do not matter.
    family:
        ``"auto"`` (rank candidates by AIC) or one of ``"gaussian"``, ``"t"``,
        ``"clayton"``, ``"gumbel"``, ``"frank"``, optionally with a
        ``"_survival"`` suffix (``"clayton_survival"``) for upper-tail shapes.
    method:
        ``"tau"`` (default; Kendall-tau inversion, numpy only) or ``"mle"``
        (refine the parameter by maximum likelihood on pseudo-observations;
        needs scipy).
    df:
        Degrees of freedom for the t copula (held fixed).

    Negative dependence is handled by reflecting the second axis; the returned
    copula is then a flipped copula.
    """
    if method not in ("tau", "mle"):
        raise ValueError("method must be 'tau' or 'mle'.")
    if method == "mle":
        sp.require_scipy("method='mle'")
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("data must be (n, d) with d >= 2.")
    n, d = x.shape
    u = pseudo_obs(x)

    if d > 2:
        if family not in ("auto", "gaussian", "t"):
            raise ValueError("For d > 2 only 'gaussian' and 't' (or 'auto') are supported.")
        fam = "gaussian" if family == "auto" else family
        tau_m = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                tau_m[i, j] = tau_m[j, i] = kendall_tau(x[:, i], x[:, j])
        r = np.sin(np.pi * tau_m / 2.0)
        w, v = np.linalg.eigh(r)  # project to a valid correlation matrix
        r = (v * np.clip(w, 1e-6, None)) @ v.T
        dg = np.sqrt(np.diag(r))
        r = r / np.outer(dg, dg)
        cop = GaussianCopula(r) if fam == "gaussian" else StudentTCopula(r, df)
        ll = float(cop.log_pdf(u).sum())
        k = d * (d - 1) // 2
        mean_tau = float(np.mean(tau_m[np.triu_indices(d, 1)]))
        return FitResult(cop, fam, mean_tau, ll, 2 * k - 2 * ll, n)

    tau = kendall_tau(x[:, 0], x[:, 1])
    flip = tau < 0
    uf = u.copy()
    if flip:
        uf[:, 1] = 1.0 - uf[:, 1]
    atau = abs(tau)

    if family == "auto":
        names = [f for f in _OFFERED_AUTO if f != "t" or sp.have_scipy()]
    else:
        valid = set(_FAMILIES) | {f"{k}_survival" for k in _FAMILIES}
        if family not in valid:
            raise ValueError(f"unknown family {family!r}; choose from {sorted(valid)} or 'auto'.")
        names = [family]

    rows = []
    for name in names:
        try:
            cop = _build(name, atau, df)
            if method == "mle":
                cop = _refine_mle(name, cop, uf, df)
            ll = float(cop.log_pdf(uf).sum())
        except (ValueError, ImportError) as e:
            rows.append({"family": name, "error": str(e)})
            continue
        if not np.isfinite(ll):
            continue
        rows.append({"family": name, "copula": cop, "loglik": ll, "aic": 2.0 - 2.0 * ll})
    ok = sorted((r for r in rows if "aic" in r), key=lambda r: r["aic"])
    if not ok:
        raise ValueError(
            "No candidate copula could be fitted; " + "; ".join(r.get("error", "") for r in rows)
        )
    best = ok[0]
    cop = best["copula"].flipped((1,)) if flip else best["copula"]
    fam = best["family"] + (" (axis 1 flipped)" if flip else "")
    table = [{"family": r["family"], "loglik": r["loglik"], "aic": r["aic"]} for r in ok]
    return FitResult(cop, fam, float(tau), best["loglik"], best["aic"], n, table)
