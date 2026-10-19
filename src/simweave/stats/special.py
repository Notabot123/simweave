"""Special functions used by copulas and marginals.

numpy has no ``erf``/``ndtri``, so the normal CDF and quantile use
``scipy.special`` when it is importable and a small numpy fallback otherwise.
Student-t functions need scipy and raise a friendly ``ImportError`` without it.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12

_SPS = None  # scipy.special, resolved lazily so ``import simweave`` never imports scipy
_SPS_TRIED = False

_erfc_vec = np.frompyfunc(math.erfc, 1, 1)


def _scipy_special():
    global _SPS, _SPS_TRIED
    if not _SPS_TRIED:
        _SPS_TRIED = True
        try:
            from scipy import special

            _SPS = special
        except ImportError:
            _SPS = None
    return _SPS


def have_scipy() -> bool:
    return _scipy_special() is not None


def require_scipy(feature: str) -> None:
    if _scipy_special() is None:
        raise ImportError(
            f"{feature} requires scipy. "
            "Install with `pip install simweave[optim]` or `pip install scipy`."
        )


def clip_unit(u):
    """Clamp to the open unit interval so ``ppf`` never returns ``inf``."""
    return np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)


def ndtr_numpy(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * np.asarray(_erfc_vec(-x / math.sqrt(2.0)), dtype=float)


# Acklam's rational approximation, refined by two Halley steps against ndtr.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)


def ndtri_numpy(p):
    p = np.asarray(p, dtype=float)
    scalar = p.ndim == 0
    p = np.atleast_1d(p)
    x = np.empty_like(p)
    plow = 0.02425
    lo = p < plow
    hi = p > 1 - plow
    mid = ~(lo | hi)
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
        den = ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        x[mid] = num / den
    for mask, sign in ((lo, 1.0), (hi, -1.0)):
        if mask.any():
            pp = p[mask] if sign > 0 else 1.0 - p[mask]
            q = np.sqrt(-2.0 * np.log(pp))
            num = ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
            den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
            x[mask] = sign * num / den
    for _ in range(2):
        e = ndtr_numpy(x) - p
        u = e * math.sqrt(2 * math.pi) * np.exp(0.5 * x * x)
        x = x - u / (1.0 + 0.5 * x * u)
    return float(x[0]) if scalar else x


def ndtr(x):
    """Standard normal CDF."""
    sps = _scipy_special()
    if sps is not None:
        return sps.ndtr(np.asarray(x, dtype=float))
    return ndtr_numpy(x)


def ndtri(p):
    """Standard normal quantile function."""
    sps = _scipy_special()
    if sps is not None:
        return sps.ndtri(np.asarray(p, dtype=float))
    return ndtri_numpy(p)


def t_cdf(x, df):
    require_scipy("The Student-t copula")
    return _scipy_special().stdtr(df, np.asarray(x, dtype=float))


def t_ppf(p, df):
    require_scipy("The Student-t copula")
    return _scipy_special().stdtrit(df, np.asarray(p, dtype=float))


# ---------------------------------------------------------------------------
# Frank copula: Debye function and tau inversion
# ---------------------------------------------------------------------------

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def debye1(theta: float) -> float:
    """First Debye function ``D1(theta) = (1/theta) * int_0^theta t/(e^t-1) dt`` (theta > 0)."""
    theta = float(theta)
    if theta <= 0:
        raise ValueError("debye1 requires theta > 0.")
    if theta > 40.0:
        return math.pi**2 / (6.0 * theta)  # tail integral is negligible
    t = 0.5 * theta * (_GL_X + 1.0)
    f = np.where(t < 1e-12, 1.0, t / np.expm1(np.maximum(t, 1e-12)))
    return float(0.5 * np.sum(_GL_W * f))


def frank_tau(theta: float) -> float:
    theta = float(theta)
    if theta == 0.0:
        return 0.0
    sign = 1.0 if theta > 0 else -1.0
    a = abs(theta)
    return sign * (1.0 - 4.0 / a * (1.0 - debye1(a)))


def frank_theta_from_tau(tau: float) -> float:
    """Invert :func:`frank_tau` for ``0 < tau < 1`` by bisection."""
    if not 0.0 < tau < 1.0:
        raise ValueError("Frank tau must lie in (0, 1); use .flipped() for negative tau.")
    lo, hi = 1e-8, 1000.0
    if tau >= frank_tau(hi):
        raise ValueError("tau too close to 1 for the Frank copula.")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if frank_tau(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
