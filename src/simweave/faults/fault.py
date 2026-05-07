"""Fault profiles and parameter-fault descriptors for predictive-maintenance datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

ProfileShape = Literal["linear", "exponential", "abrupt"] | Callable[[float], float]


@dataclass
class FaultProfile:
    """Temporal evolution of a single fault from healthy to fully failed.

    Parameters
    ----------
    onset_time : float
        Simulation time at which degradation begins. The system is fully
        healthy before this point (health index = 1).
    failure_time : float
        Simulation time at which the system reaches the fully-failed state
        (health index = 0). Must be strictly greater than *onset_time*.
    mode : str
        Failure mode label written into dataset labels, e.g.
        ``"insulation_loss"`` or ``"bearing_wear"``.  Defaults to
        ``"fault"``.
    shape : {"linear", "exponential", "abrupt"} or callable
        Degradation curve shape over the window [onset_time, failure_time]:

        ``"linear"``
            Constant degradation rate; health falls from 1 to 0 uniformly.
        ``"exponential"``
            Convex profile: slow initial degradation that accelerates sharply
            near *failure_time*.  Characteristic of fatigue and wear-out
            mechanisms.
        ``"abrupt"``
            System is fully healthy until *onset_time*, then fails
            instantaneously.  Useful for modelling sudden fracture or
            electrical short-circuit.
        callable
            Receives *progress* ∈ [0, 1] (0 = onset, 1 = failure) and must
            return a health value in [0, 1].  Use this for custom profiles
            such as S-curve, two-phase, or empirically fitted curves.

    Notes
    -----
    All shapes are monotonically decreasing.  Stochastic variation is
    intentionally absent at the profile level; add sensor noise through
    :meth:`~simweave.faults.dataset.FaultDataset.from_result` instead.
    """

    onset_time: float
    failure_time: float
    mode: str = "fault"
    shape: ProfileShape = "linear"

    def __post_init__(self) -> None:
        if self.failure_time <= self.onset_time:
            raise ValueError(
                f"failure_time ({self.failure_time}) must be strictly greater "
                f"than onset_time ({self.onset_time})."
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def health_index(self, t: float) -> float:
        """Health index in [0, 1] at simulation time *t*.

        Returns 1.0 before *onset_time* and 0.0 from *failure_time* onward.
        """
        if t < self.onset_time:
            return 1.0
        if t >= self.failure_time:
            return 0.0
        progress = (t - self.onset_time) / (self.failure_time - self.onset_time)
        return float(np.clip(self._shape_fn(progress), 0.0, 1.0))

    def rul(self, t: float) -> float:
        """Remaining useful life in simulation time units at time *t*."""
        return max(0.0, self.failure_time - t)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _shape_fn(self, progress: float) -> float:
        """Evaluate the degradation shape at *progress* ∈ [0, 1]."""
        if callable(self.shape):
            return float(self.shape(progress))
        if self.shape == "linear":
            return 1.0 - progress
        if self.shape == "exponential":
            # Convex: stays near healthy initially, drops sharply near failure.
            # h(p) = (exp(k*(1-p)) - 1) / (exp(k) - 1), k=5
            k = 5.0
            return (math.exp(k * (1.0 - progress)) - 1.0) / (math.exp(k) - 1.0)
        if self.shape == "abrupt":
            # Once onset is passed, treat as immediately failed.
            return 0.0
        raise ValueError(
            f"Unknown shape {self.shape!r}. "
            "Expected 'linear', 'exponential', 'abrupt', or a callable."
        )


@dataclass
class ParameterFault:
    """A fault expressed as a perturbation of a named attribute on a DynamicSystem.

    Parameters
    ----------
    param : str
        Name of the attribute to perturb on the wrapped system, e.g.
        ``"R_th"`` on a :class:`~simweave.continuous.systems.ThermalRC` or
        ``"c"`` on a :class:`~simweave.continuous.systems.MassSpringDamper`.
        The attribute must exist and be a plain ``float``.
    profile : FaultProfile
        Temporal evolution of this fault.
    max_delta : float
        Magnitude of the perturbation at full failure (health index = 0).
        When *relative* is ``True`` this is a fractional multiplier
        (e.g. ``2.0`` doubles the nominal value at full failure).
        When *relative* is ``False`` it is an additive offset.
    relative : bool
        If ``True`` (default):
            ``perturbed = nominal × (1 + max_delta × fault_fraction)``
        If ``False``:
            ``perturbed = nominal + max_delta × fault_fraction``

        where ``fault_fraction = 1 − health_index ∈ [0, 1]``.
    """

    param: str
    profile: FaultProfile
    max_delta: float
    relative: bool = True


__all__ = ["FaultProfile", "ParameterFault", "ProfileShape"]
