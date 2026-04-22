"""First-order state-space ODE solver and plug-in system protocol.

Two use modes:

1. Standalone: call :func:`simulate` with a system, time span, and ``dt``.
   Returns a :class:`SimulationResult` with full time/state arrays.
2. Hybrid: wrap a system in :class:`ContinuousProcess` and register with a
   :class:`~simweave.core.environment.SimEnvironment`. The environment clock
   then drives the integration. Useful when discrete events and continuous
   dynamics share a timeline (e.g. a vehicle queued at a junction whose
   suspension is still integrating over bumps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, runtime_checkable

import numpy as np

from simweave.core.entity import Entity


ArrayLike = np.ndarray
InputFunc = Callable[[float], np.ndarray | float | int | None]


@dataclass(slots=True)
class SimulationResult:
    time: np.ndarray
    state: np.ndarray
    state_labels: tuple[str, ...]
    system_name: str
    method: str

    def final_state(self) -> np.ndarray:
        return self.state[-1].copy()


@runtime_checkable
class SupportsDynamics(Protocol):
    """Protocol for plug-in dynamic systems."""

    def initial_state(self) -> np.ndarray: ...

    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        inputs: np.ndarray | float | int | None = None,
    ) -> np.ndarray: ...


class DynamicSystem:
    """Base class for first-order state-space dynamic systems."""

    def initial_state(self) -> np.ndarray:
        raise NotImplementedError

    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        inputs: np.ndarray | float | int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def inputs(self, t: float) -> np.ndarray | float | int | None:
        return None

    def state_labels(self) -> tuple[str, ...]:
        x0 = self.initial_state()
        return tuple(f"x{i}" for i in range(x0.size))

    def validate_state(self, state: np.ndarray) -> None:
        if state.ndim != 1:
            raise ValueError("State must be a 1D array.")

    @property
    def name(self) -> str:
        return self.__class__.__name__


def _as_state_vector(state: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=float)
    if arr.ndim != 1:
        raise ValueError("State must be coercible to a 1D float array.")
    return arr


def _resolve_inputs(
    system: SupportsDynamics,
    external_inputs: Optional[InputFunc],
    t: float,
) -> np.ndarray | float | int | None:
    if external_inputs is not None:
        return external_inputs(t)
    if hasattr(system, "inputs"):
        return system.inputs(t)  # type: ignore[attr-defined]
    return None


def _step_euler(system, t, x, dt, external_inputs):
    u = _resolve_inputs(system, external_inputs, t)
    dx = np.asarray(system.derivatives(t, x, u), dtype=float)
    return x + dt * dx


def _step_rk4(system, t, x, dt, external_inputs):
    u1 = _resolve_inputs(system, external_inputs, t)
    k1 = np.asarray(system.derivatives(t, x, u1), dtype=float)
    u2 = _resolve_inputs(system, external_inputs, t + 0.5 * dt)
    k2 = np.asarray(
        system.derivatives(t + 0.5 * dt, x + 0.5 * dt * k1, u2), dtype=float
    )
    k3 = np.asarray(
        system.derivatives(t + 0.5 * dt, x + 0.5 * dt * k2, u2), dtype=float
    )
    u4 = _resolve_inputs(system, external_inputs, t + dt)
    k4 = np.asarray(system.derivatives(t + dt, x + dt * k3, u4), dtype=float)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_STEPPERS = {"euler": _step_euler, "rk4": _step_rk4}


def simulate(
    system: SupportsDynamics,
    t_span: tuple[float, float],
    dt: float,
    x0: Iterable[float] | np.ndarray | None = None,
    method: str = "rk4",
    inputs: Optional[InputFunc] = None,
) -> SimulationResult:
    """Integrate a system over ``t_span`` with fixed step ``dt``."""
    t0, tf = t_span
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0.")
    if method not in _STEPPERS:
        raise ValueError(f"Unknown method '{method}'. Use one of {tuple(_STEPPERS)}")

    x_init = _as_state_vector(system.initial_state() if x0 is None else x0)
    if hasattr(system, "validate_state"):
        system.validate_state(x_init)  # type: ignore[attr-defined]

    n_steps = int(np.floor((tf - t0) / dt)) + 1
    time = t0 + np.arange(n_steps + 1, dtype=float) * dt
    time = time[time <= tf + 1e-12]

    state = np.zeros((time.size, x_init.size), dtype=float)
    state[0] = x_init

    stepper = _STEPPERS[method]
    for i in range(1, time.size):
        state[i] = stepper(system, time[i - 1], state[i - 1], dt, inputs)

    labels = (
        tuple(system.state_labels())
        if hasattr(system, "state_labels")
        else tuple(f"x{i}" for i in range(x_init.size))
    )
    return SimulationResult(
        time=time,
        state=state,
        state_labels=labels,
        system_name=getattr(system, "name", system.__class__.__name__),
        method=method,
    )


# ---------------------------------------------------------------------------
# Hybrid wrapper: turn a DynamicSystem into a SimEnvironment process.
# ---------------------------------------------------------------------------


class ContinuousProcess(Entity):
    """Drive a DynamicSystem from a SimEnvironment's tick loop.

    Each tick, integrates ``n_substeps`` sub-intervals of length ``dt / n_substeps``.
    The previous state is appended to ``history_state`` with the corresponding
    time in ``history_time``.
    """

    def __init__(
        self,
        system: SupportsDynamics,
        method: str = "rk4",
        n_substeps: int = 1,
        inputs: InputFunc | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            name=name or getattr(system, "name", system.__class__.__name__)
        )
        if method not in _STEPPERS:
            raise ValueError(f"Unknown method '{method}'")
        if n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")
        self.system = system
        self.method = method
        self.n_substeps = n_substeps
        self.inputs = inputs
        self.state = _as_state_vector(system.initial_state())
        self._stepper = _STEPPERS[method]
        self.history_time: list[float] = []
        self.history_state: list[np.ndarray] = []

    def on_register(self, env) -> None:
        super().on_register(env)
        self.history_time.append(env.clock.t)
        self.history_state.append(self.state.copy())

    def tick(self, dt: float, env) -> None:
        super().tick(dt, env)
        sub_dt = dt / self.n_substeps
        t = env.clock.t
        for _ in range(self.n_substeps):
            self.state = self._stepper(self.system, t, self.state, sub_dt, self.inputs)
            t += sub_dt
        self.history_time.append(env.clock.t + dt)
        self.history_state.append(self.state.copy())

    def has_work(self, env) -> bool:
        return True  # continuous dynamics always "have work"

    def result(self) -> SimulationResult:
        return SimulationResult(
            time=np.asarray(self.history_time),
            state=np.asarray(self.history_state),
            state_labels=tuple(self.system.state_labels())
            if hasattr(self.system, "state_labels")
            else tuple(f"x{i}" for i in range(self.state.size)),
            system_name=getattr(self.system, "name", self.system.__class__.__name__),
            method=self.method,
        )
