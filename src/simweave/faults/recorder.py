"""FaultRecorder: time-series capture of health labels for a FaultInjector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from simweave.viz.recorders import _Recorder
from simweave.faults.injector import FaultInjector

if TYPE_CHECKING:
    from simweave.core.environment import SimEnvironment


class FaultRecorder(_Recorder):
    """Snapshot health index, RUL, and failure-mode label each environment tick.

    Register this *after* the
    :class:`~simweave.continuous.solver.ContinuousProcess` that wraps the
    injector so the labels align with the post-tick system state.

    Parameters
    ----------
    injector : FaultInjector
        The injector whose fault profiles are sampled each tick.
    name : str, optional
        Display name for this recorder.

    Attributes
    ----------
    times : list[float]
        Simulation times of each sample.
    health_index : list[float]
        Overall (minimum) health index in [0, 1] at each sample.
    rul : list[float]
        Minimum remaining useful life (simulation time units) across faults.
    is_failed : list[bool]
        ``True`` when health index has reached 0.
    failure_mode : list[str]
        Label of the most-degraded fault (``"healthy"`` when all faults are
        before their onset time).
    """

    def __init__(
        self,
        injector: FaultInjector,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"faults({injector.name})")
        self.injector = injector
        self.health_index: list[float] = []
        self.rul: list[float] = []
        self.is_failed: list[bool] = []
        self.failure_mode: list[str] = []

    def _sample(self, env: "SimEnvironment", t: float) -> None:
        hi = self.injector.overall_health(t)
        self.times.append(t)
        self.health_index.append(hi)
        self.rul.append(self.injector.overall_rul(t))
        self.is_failed.append(hi <= 0.0)
        self.failure_mode.append(self.injector.active_mode(t))


__all__ = ["FaultRecorder"]
