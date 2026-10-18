"""Empirical dependence measures."""

from __future__ import annotations

import numpy as np


def tail_dependence_empirical(u, q: float = 0.01) -> dict:
    """Empirical corner tail-dependence at level ``q`` for bivariate uniforms.

    Keys match :meth:`Copula.tail_dependence`: ``ll``, ``uu``, ``lu``, ``ul``.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must be (n, 2).")
    if not 0.0 < q < 0.5:
        raise ValueError("q must lie in (0, 0.5).")
    a, b = u[:, 0], u[:, 1]
    lo_a, hi_a = a < q, a > 1 - q

    def cond(mask, other):
        return float(other[mask].mean()) if mask.any() else float("nan")

    return {
        "ll": cond(lo_a, b < q),
        "uu": cond(hi_a, b > 1 - q),
        "lu": cond(lo_a, b > 1 - q),
        "ul": cond(hi_a, b < q),
    }
