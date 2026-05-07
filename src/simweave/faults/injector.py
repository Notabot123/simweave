"""FaultInjector: wraps a DynamicSystem and injects ParameterFaults at runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from simweave.continuous.solver import DynamicSystem
from simweave.faults.fault import ParameterFault

if TYPE_CHECKING:
    pass


class FaultInjector(DynamicSystem):
    """Wraps a :class:`~simweave.continuous.solver.DynamicSystem` and applies
    one or more :class:`~simweave.faults.fault.ParameterFault` objects.

    The injector satisfies the
    :class:`~simweave.continuous.solver.SupportsDynamics` protocol so it can be
    passed directly to :func:`~simweave.continuous.solver.simulate` or wrapped
    in a :class:`~simweave.continuous.solver.ContinuousProcess` without any
    changes to the solver.

    On each :meth:`derivatives` call the injector:

    1. Computes the fault fraction (``1 − health_index``) for every fault.
    2. Perturbs the named parameter(s) on the wrapped system proportionally.
    3. Delegates to the underlying system's :meth:`derivatives`.
    4. Restores nominal parameter values (safe with RK4's multi-eval per step).

    Parameters
    ----------
    system : DynamicSystem
        The underlying physical model to wrap.
    faults : list[ParameterFault]
        Fault definitions to inject simultaneously.  Multiple faults on
        different parameters are all active at once; the system-level health
        index is the *minimum* across all individual fault health indices and
        the active failure mode is that of the most-degraded fault.

    Examples
    --------
    >>> from simweave.continuous.systems import ThermalRC
    >>> from simweave.faults import FaultProfile, ParameterFault, FaultInjector
    >>> profile = FaultProfile(onset_time=200, failure_time=800,
    ...                        mode="insulation_loss", shape="exponential")
    >>> fault = ParameterFault(param="R_th", profile=profile, max_delta=2.0)
    >>> injector = FaultInjector(ThermalRC(R_th=1.0, C_th=500.0), [fault])
    >>> from simweave.continuous.solver import simulate
    >>> result = simulate(injector, t_span=(0, 1000), dt=1.0)
    """

    def __init__(
        self,
        system: DynamicSystem,
        faults: list[ParameterFault],
    ) -> None:
        self.system = system
        self.faults: list[ParameterFault] = list(faults)
        # Snapshot nominal parameter values at construction time.
        self._nominal: dict[str, float] = {}
        for f in self.faults:
            if f.param not in self._nominal:
                val = getattr(system, f.param)
                self._nominal[f.param] = float(val)

    # ------------------------------------------------------------------
    # SupportsDynamics interface
    # ------------------------------------------------------------------

    def initial_state(self) -> np.ndarray:
        return self.system.initial_state()

    def state_labels(self) -> tuple[str, ...]:
        if hasattr(self.system, "state_labels"):
            return tuple(self.system.state_labels())
        n = self.system.initial_state().size
        return tuple(f"x{i}" for i in range(n))

    def inputs(self, t: float):
        if hasattr(self.system, "inputs"):
            return self.system.inputs(t)
        return None

    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        inputs=None,
    ) -> np.ndarray:
        self._apply_faults(t)
        try:
            return self.system.derivatives(t, state, inputs)
        finally:
            self._restore_nominal()

    # ------------------------------------------------------------------
    # Health / label helpers
    # ------------------------------------------------------------------

    def overall_health(self, t: float) -> float:
        """Minimum health index across all faults (1 = healthy, 0 = failed)."""
        if not self.faults:
            return 1.0
        return min(f.profile.health_index(t) for f in self.faults)

    def active_mode(self, t: float) -> str:
        """Label of the most-degraded fault at time *t*, or ``"healthy"``."""
        if not self.faults:
            return "healthy"
        worst = min(self.faults, key=lambda f: f.profile.health_index(t))
        if worst.profile.health_index(t) >= 1.0:
            return "healthy"
        return worst.profile.mode

    def overall_rul(self, t: float) -> float:
        """Minimum remaining useful life across all faults (simulation time units)."""
        if not self.faults:
            return float("inf")
        return min(f.profile.rul(t) for f in self.faults)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_faults(self, t: float) -> None:
        for f in self.faults:
            nominal = self._nominal[f.param]
            fault_fraction = 1.0 - f.profile.health_index(t)
            if f.relative:
                perturbed = nominal * (1.0 + f.max_delta * fault_fraction)
            else:
                perturbed = nominal + f.max_delta * fault_fraction
            setattr(self.system, f.param, perturbed)

    def _restore_nominal(self) -> None:
        for param, val in self._nominal.items():
            setattr(self.system, param, val)

    @property
    def name(self) -> str:
        return f"FaultInjector({self.system.name})"


__all__ = ["FaultInjector"]
