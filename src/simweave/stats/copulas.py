"""Copulas: dependence structures on the unit hypercube.

``sample_u(n, rng)`` returns an ``(n, dim)`` array of uniforms whose joint law
is the copula. Pair it with marginals (see :class:`~simweave.stats.JointSampler`)
to get correlated draws with any marginal distributions.

Gaussian, Clayton, Gumbel and Frank work on numpy alone; the Student-t copula
needs scipy (``pip install simweave[optim]``).

Negative dependence and the "other" tails are obtained by flipping axes:
``GumbelCopula.from_kendall_tau(0.5).flipped((1,))`` is a Gumbel copula with
the second axis reflected (Kendall tau -0.5, dependence concentrated where the
first variable is large and the second small).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from simweave.stats import special as sp
from simweave.stats.special import EPS, clip_unit

_TAIL_KEYS = ("ll", "uu", "lu", "ul")


def _tails(**kw) -> dict:
    out = {k: 0.0 for k in _TAIL_KEYS}
    out.update(kw)
    return out


def _as_u(u, dim):
    u = np.atleast_2d(np.asarray(u, dtype=float))
    if u.shape[1] != dim:
        raise ValueError(f"expected {dim} columns, got {u.shape[1]}.")
    return clip_unit(u)


class Copula(ABC):
    """Base class. ``dim`` is the number of coupled variables."""

    dim: int

    @abstractmethod
    def sample_u(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points, shape ``(n, dim)``."""

    @abstractmethod
    def log_pdf(self, u) -> np.ndarray:
        """Log copula density at the rows of ``u``."""

    def kendall_tau(self):
        raise NotImplementedError

    def tail_dependence(self) -> dict:
        """Corner tail-dependence coefficients (bivariate).

        Keys: ``ll`` = P(U2<q | U1<q), ``uu`` = P(U2>1-q | U1>1-q),
        ``lu`` = P(U2>1-q | U1<q), ``ul`` = P(U2<q | U1>1-q), as ``q -> 0``.
        """
        raise NotImplementedError

    def conditional_ppf(self, w, u):
        """Quantile of ``U2 | U1 = u`` at probability ``w`` (bivariate)."""
        raise NotImplementedError(f"{type(self).__name__} has no conditional quantile.")

    def flipped(self, axes=(1,)) -> "Copula":
        """Reflect the listed axes: ``u_k -> 1 - u_k``."""
        return Flipped(self, axes)

    def survival(self) -> "Copula":
        """Survival copula (all axes reflected); swaps lower and upper tails."""
        return Flipped(self, tuple(range(self.dim)))

    def _check_n(self, n):
        if n < 0:
            raise ValueError("n must be non-negative.")


# ---------------------------------------------------------------------------
# Elliptical
# ---------------------------------------------------------------------------


def _corr_matrix(corr, dim=None) -> np.ndarray:
    a = np.asarray(corr, dtype=float)
    if a.ndim == 0:
        if not -1.0 < float(a) < 1.0:
            raise ValueError("correlation must lie in (-1, 1).")
        a = np.array([[1.0, float(a)], [float(a), 1.0]])
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("corr must be a square matrix or a scalar.")
    if not np.allclose(a, a.T) or not np.allclose(np.diag(a), 1.0):
        raise ValueError("corr must be symmetric with unit diagonal.")
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as e:
        raise ValueError("corr must be positive definite.") from e
    return a


def _exchangeable(tau: float, dim: int) -> np.ndarray:
    rho = math.sin(math.pi * tau / 2.0)
    a = np.full((dim, dim), rho)
    np.fill_diagonal(a, 1.0)
    return a


class GaussianCopula(Copula):
    """Gaussian copula with correlation matrix (or scalar rho for two variables)."""

    def __init__(self, corr) -> None:
        self.corr = _corr_matrix(corr)
        self.dim = self.corr.shape[0]
        self._chol = np.linalg.cholesky(self.corr)

    @classmethod
    def from_kendall_tau(cls, tau: float, dim: int = 2) -> "GaussianCopula":
        if not -1.0 < tau < 1.0:
            raise ValueError("tau must lie in (-1, 1).")
        if dim > 2 and tau < -1.0 / (dim - 1):
            raise ValueError("tau too negative for an exchangeable matrix of this dimension.")
        return cls(_exchangeable(tau, dim))

    @property
    def rho(self) -> float:
        if self.dim != 2:
            raise ValueError("rho is defined for bivariate copulas only.")
        return float(self.corr[0, 1])

    def sample_u(self, n, rng):
        z = rng.standard_normal((n, self.dim)) @ self._chol.T
        return clip_unit(sp.ndtr(z))

    def log_pdf(self, u):
        x = sp.ndtri(_as_u(u, self.dim))
        inv = np.linalg.inv(self.corr)
        _, logdet = np.linalg.slogdet(self.corr)
        quad = np.einsum("ni,ij,nj->n", x, inv - np.eye(self.dim), x)
        return -0.5 * logdet - 0.5 * quad

    def kendall_tau(self):
        t = 2.0 / math.pi * np.arcsin(self.corr)
        return float(t[0, 1]) if self.dim == 2 else t

    def tail_dependence(self):
        return _tails()

    def conditional_ppf(self, w, u):
        r = self.rho
        zu = sp.ndtri(clip_unit(u))
        zw = sp.ndtri(clip_unit(w))
        return clip_unit(sp.ndtr(r * zu + math.sqrt(1.0 - r * r) * zw))


class StudentTCopula(Copula):
    """Student-t copula (requires scipy). Symmetric tail dependence."""

    def __init__(self, corr, df: float = 4.0) -> None:
        sp.require_scipy("The Student-t copula")
        if df <= 0:
            raise ValueError("df must be positive.")
        self.corr = _corr_matrix(corr)
        self.dim = self.corr.shape[0]
        self.df = float(df)
        self._chol = np.linalg.cholesky(self.corr)

    @classmethod
    def from_kendall_tau(cls, tau: float, dim: int = 2, df: float = 4.0) -> "StudentTCopula":
        if not -1.0 < tau < 1.0:
            raise ValueError("tau must lie in (-1, 1).")
        return cls(_exchangeable(tau, dim), df)

    @property
    def rho(self) -> float:
        if self.dim != 2:
            raise ValueError("rho is defined for bivariate copulas only.")
        return float(self.corr[0, 1])

    def sample_u(self, n, rng):
        z = rng.standard_normal((n, self.dim)) @ self._chol.T
        w = rng.chisquare(self.df, size=n) / self.df
        return clip_unit(sp.t_cdf(z / np.sqrt(w)[:, None], self.df))

    def log_pdf(self, u):
        from scipy.special import gammaln

        u = _as_u(u, self.dim)
        d, nu = self.dim, self.df
        x = sp.t_ppf(u, nu)
        inv = np.linalg.inv(self.corr)
        _, logdet = np.linalg.slogdet(self.corr)
        quad = np.einsum("ni,ij,nj->n", x, inv, x)
        log_joint = (
            gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * math.log(nu * math.pi)
            - 0.5 * logdet - 0.5 * (nu + d) * np.log1p(quad / nu)
        )
        log_marg = (
            gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
            - 0.5 * (nu + 1) * np.log1p(x * x / nu)
        ).sum(axis=1)
        return log_joint - log_marg

    def kendall_tau(self):
        t = 2.0 / math.pi * np.arcsin(self.corr)
        return float(t[0, 1]) if self.dim == 2 else t

    def tail_dependence(self):
        r, nu = self.rho, self.df
        a = abs(r)
        lam = 2.0 * float(sp.t_cdf(-math.sqrt((nu + 1.0) * (1.0 - a) / (1.0 + a)), nu + 1.0))
        # Negative rho puts the coefficient on the anti-diagonal corners.
        return _tails(ll=lam, uu=lam) if r >= 0 else _tails(lu=lam, ul=lam)

    def conditional_ppf(self, w, u):
        r, nu = self.rho, self.df
        xu = sp.t_ppf(clip_unit(u), nu)
        tw = sp.t_ppf(clip_unit(w), nu + 1.0)
        x = r * xu + np.sqrt((nu + xu * xu) * (1.0 - r * r) / (nu + 1.0)) * tw
        return clip_unit(sp.t_cdf(x, nu))


# ---------------------------------------------------------------------------
# Archimedean (exchangeable, positive dependence; use .flipped() for negative)
# ---------------------------------------------------------------------------


class _Archimedean(Copula):
    def __init__(self, theta: float, dim: int = 2) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2.")
        self.theta = float(theta)
        self.dim = int(dim)
        self._check_theta()

    @abstractmethod
    def _check_theta(self) -> None: ...

    @abstractmethod
    def _frailty(self, n, rng) -> np.ndarray: ...

    @abstractmethod
    def _psi(self, t, v): ...  # Laplace transform of the frailty, given frailty draw v

    def sample_u(self, n, rng):
        v = self._frailty(n, rng)
        e = rng.exponential(1.0, size=(n, self.dim))
        return clip_unit(self._psi(e, v[:, None]))


class ClaytonCopula(_Archimedean):
    """Clayton copula: lower-tail dependence ``2**(-1/theta)``."""

    def _check_theta(self):
        if self.theta <= 0:
            raise ValueError("Clayton theta must be > 0 (use .flipped() for negative dependence).")

    @classmethod
    def from_kendall_tau(cls, tau: float, dim: int = 2) -> "ClaytonCopula":
        if not 0.0 < tau < 1.0:
            raise ValueError("Clayton tau must lie in (0, 1); use .flipped() for negative tau.")
        return cls(2.0 * tau / (1.0 - tau), dim)

    def _frailty(self, n, rng):
        return rng.gamma(1.0 / self.theta, 1.0, size=n)

    def _psi(self, t, v):
        return (1.0 + t / v) ** (-1.0 / self.theta)

    def log_pdf(self, u):
        u = _as_u(u, self.dim)
        d, th = self.dim, self.theta
        coef = sum(math.log1p(k * th) for k in range(d))
        s = (u ** (-th)).sum(axis=1) - d + 1.0
        return coef - (1.0 + th) * np.log(u).sum(axis=1) - (d + 1.0 / th) * np.log(s)

    def kendall_tau(self):
        return self.theta / (self.theta + 2.0)

    def tail_dependence(self):
        return _tails(ll=2.0 ** (-1.0 / self.theta))

    def conditional_ppf(self, w, u):
        th = self.theta
        u, w = clip_unit(u), clip_unit(w)
        return clip_unit(((w ** (-th / (1.0 + th)) - 1.0) * u ** (-th) + 1.0) ** (-1.0 / th))


class GumbelCopula(_Archimedean):
    """Gumbel copula: upper-tail dependence ``2 - 2**(1/theta)``."""

    def _check_theta(self):
        if self.theta < 1.0:
            raise ValueError("Gumbel theta must be >= 1.")

    @classmethod
    def from_kendall_tau(cls, tau: float, dim: int = 2) -> "GumbelCopula":
        if not 0.0 <= tau < 1.0:
            raise ValueError("Gumbel tau must lie in [0, 1); use .flipped() for negative tau.")
        return cls(1.0 / (1.0 - tau), dim)

    def _frailty(self, n, rng):
        a = 1.0 / self.theta
        if a >= 1.0:
            return np.ones(n)
        th = rng.uniform(0.0, math.pi, size=n)
        e = rng.exponential(1.0, size=n)
        return (
            np.sin(a * th) / np.sin(th) ** (1.0 / a)
            * (np.sin((1.0 - a) * th) / e) ** ((1.0 - a) / a)
        )

    def _psi(self, t, v):
        return np.exp(-((t / v) ** (1.0 / self.theta)))

    def _cdf2(self, u, v):
        th = self.theta
        s = (-np.log(u)) ** th + (-np.log(v)) ** th
        return np.exp(-(s ** (1.0 / th)))

    def log_pdf(self, u):
        if self.dim != 2:
            raise NotImplementedError("Gumbel log_pdf is implemented for dim=2.")
        u = _as_u(u, 2)
        th = self.theta
        a, b = -np.log(u[:, 0]), -np.log(u[:, 1])
        s = a**th + b**th
        s1 = s ** (1.0 / th)
        return (
            -s1 - np.log(u[:, 0]) - np.log(u[:, 1])
            + (th - 1.0) * (np.log(a) + np.log(b))
            + (1.0 / th - 2.0) * np.log(s)
            + np.log(s1 + th - 1.0)
        )

    def kendall_tau(self):
        return 1.0 - 1.0 / self.theta

    def tail_dependence(self):
        return _tails(uu=2.0 - 2.0 ** (1.0 / self.theta))

    def conditional_ppf(self, w, u):
        """Invert ``h(v|u) = dC/du`` by vectorised bisection."""
        th = self.theta
        u = clip_unit(u)
        w = clip_unit(w)
        u, w = np.broadcast_arrays(u, w)
        a = -np.log(u)
        lo = np.full(u.shape, EPS)
        hi = np.full(u.shape, 1.0 - EPS)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            s = a**th + (-np.log(mid)) ** th
            h = np.exp(-(s ** (1.0 / th))) * s ** (1.0 / th - 1.0) * a ** (th - 1.0) / u
            too_low = h < w  # h increases with v
            lo = np.where(too_low, mid, lo)
            hi = np.where(too_low, hi, mid)
        return clip_unit(0.5 * (lo + hi))


class FrankCopula(_Archimedean):
    """Frank copula (theta > 0): symmetric, no tail dependence."""

    def _check_theta(self):
        if self.theta <= 0:
            raise ValueError("Frank theta must be > 0 (use .flipped() for negative dependence).")

    @classmethod
    def from_kendall_tau(cls, tau: float, dim: int = 2) -> "FrankCopula":
        return cls(sp.frank_theta_from_tau(tau), dim)

    def _frailty(self, n, rng):
        return rng.logseries(-math.expm1(-self.theta), size=n).astype(float)

    def _psi(self, t, v):
        th = self.theta
        return -np.log1p(-(-math.expm1(-th)) * np.exp(-t / v)) / th

    def log_pdf(self, u):
        if self.dim != 2:
            raise NotImplementedError("Frank log_pdf is implemented for dim=2.")
        u = _as_u(u, 2)
        th = self.theta
        a = -np.expm1(-th)
        num = th * a * np.exp(-th * (u[:, 0] + u[:, 1]))
        den = a - np.expm1(-th * u[:, 0]) * np.expm1(-th * u[:, 1])
        return np.log(num) - 2.0 * np.log(den)

    def kendall_tau(self):
        return sp.frank_tau(self.theta)

    def tail_dependence(self):
        return _tails()

    def conditional_ppf(self, w, u):
        th = self.theta
        u, w = clip_unit(u), clip_unit(w)
        v = -np.log1p(w * math.expm1(-th) / (w + (1.0 - w) * np.exp(-th * u))) / th
        return clip_unit(v)


# ---------------------------------------------------------------------------
# Axis flips
# ---------------------------------------------------------------------------


class Flipped(Copula):
    """A copula with some axes reflected (``u_k -> 1 - u_k``)."""

    def __init__(self, base: Copula, axes=(1,)) -> None:
        axes = tuple(sorted(set(int(a) for a in axes)))
        if not axes or any(a < 0 or a >= base.dim for a in axes):
            raise ValueError(f"axes must be a non-empty subset of range({base.dim}).")
        self.base = base
        self.axes = axes
        self.dim = base.dim

    def _flip(self, u):
        u = np.array(u, dtype=float, copy=True)
        u[:, list(self.axes)] = 1.0 - u[:, list(self.axes)]
        return u

    def sample_u(self, n, rng):
        return clip_unit(self._flip(self.base.sample_u(n, rng)))

    def log_pdf(self, u):
        return self.base.log_pdf(self._flip(_as_u(u, self.dim)))

    def _parity(self) -> int:
        return len(self.axes) % 2

    def kendall_tau(self):
        if self.dim != 2:
            raise NotImplementedError("kendall_tau of a flipped copula is bivariate-only.")
        t = self.base.kendall_tau()
        return -t if self._parity() else t

    def tail_dependence(self):
        if self.dim != 2:
            raise NotImplementedError("tail_dependence is bivariate-only.")
        b = self.base.tail_dependence()
        f0, f1 = 0 in self.axes, 1 in self.axes
        out = {}
        # corner (a, b) of the flipped copula = corner (a', b') of the base.
        corners = {"ll": (0, 0), "uu": (1, 1), "lu": (0, 1), "ul": (1, 0)}
        inv = {v: k for k, v in corners.items()}
        for key, (a, bb) in corners.items():
            a2 = 1 - a if f0 else a
            b2 = 1 - bb if f1 else bb
            out[key] = b[inv[(a2, b2)]]
        return out

    def conditional_ppf(self, w, u):
        if self.dim != 2:
            raise NotImplementedError("conditional_ppf is bivariate-only.")
        f0, f1 = 0 in self.axes, 1 in self.axes
        u_in = 1.0 - np.asarray(u, dtype=float) if f0 else u
        w_in = 1.0 - np.asarray(w, dtype=float) if f1 else w
        v = self.base.conditional_ppf(w_in, u_in)
        return clip_unit(1.0 - v if f1 else v)

    def flipped(self, axes=(1,)) -> Copula:
        merged = set(self.axes) ^ set(int(a) for a in axes)  # double flip cancels
        return self.base if not merged else Flipped(self.base, tuple(merged))
