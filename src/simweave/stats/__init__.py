"""Dependence modelling: marginals, copulas, joint and serial sampling, fitting.

Pure numpy at import time. Gaussian, Clayton, Gumbel and Frank copulas work on a
base install; the Student-t copula and ``method="mle"`` fitting need scipy
(``pip install simweave[optim]``).
"""

from simweave.stats.copulas import (
    ClaytonCopula,
    Copula,
    Flipped,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
)
from simweave.stats.fit import FitResult, fit_copula, kendall_tau, pseudo_obs, spearman_rho
from simweave.stats.joint import JointSampler
from simweave.stats.marginals import (
    Deterministic,
    Empirical,
    Exponential,
    LogNormal,
    Marginal,
    Normal,
    ScipyMarginal,
    Uniform,
    Weibull,
    as_marginal,
)
from simweave.stats.measures import tail_dependence_empirical
from simweave.stats.serial import MarkovCopulaProcess

__all__ = [
    "Copula", "GaussianCopula", "StudentTCopula", "ClaytonCopula", "GumbelCopula",
    "FrankCopula", "Flipped",
    "Marginal", "Exponential", "Uniform", "Normal", "LogNormal", "Weibull",
    "Deterministic", "Empirical", "ScipyMarginal", "as_marginal",
    "JointSampler", "MarkovCopulaProcess",
    "fit_copula", "FitResult", "pseudo_obs", "kendall_tau", "spearman_rho",
    "tail_dependence_empirical",
]
