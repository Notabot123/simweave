"""Serially dependent sequences via a lag-1 copula Markov chain."""

from __future__ import annotations

import numpy as np

from simweave.stats.copulas import Copula
from simweave.stats.marginals import as_marginal
from simweave.stats.special import clip_unit


class MarkovCopulaProcess:
    """Stationary sequence with a chosen marginal and lag-1 dependence ``copula``.

    ``U_t = h^{-1}(W | U_{t-1})`` with ``W ~ U(0, 1)``, so each value has exactly
    the requested marginal and consecutive values have exactly the copula's
    dependence. It is a callable ``(rng) -> float``, so it plugs straight into
    ``ArrivalGenerator(interarrival=...)``.

    The process carries state, so build one **per replicate** inside the
    scenario function; do not share it across Monte Carlo runs.
    """

    def __init__(self, copula: Copula, marginal) -> None:
        if copula.dim != 2:
            raise ValueError("MarkovCopulaProcess needs a bivariate copula.")
        copula.conditional_ppf(0.5, 0.5)  # fail fast if unsupported
        self.copula = copula
        self.marginal = as_marginal(marginal)
        self._u: float | None = None

    def reset(self) -> None:
        self._u = None

    def next_u(self, rng: np.random.Generator) -> float:
        if self._u is None:
            u = float(rng.random())
        else:
            u = float(self.copula.conditional_ppf(rng.random(), self._u))
        self._u = float(clip_unit(u))
        return self._u

    def __call__(self, rng: np.random.Generator) -> float:
        return float(self.marginal.ppf(self.next_u(rng)))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Convenience: ``n`` consecutive values (continues the chain)."""
        return np.array([self(rng) for _ in range(n)])
