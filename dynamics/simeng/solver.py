from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol

import numpy as np

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


class SupportsDynamics(Protocol):
    """Protocol for plug-in systems.

    Systems supply:
    - initial_state()
    - derivatives(t, state, inputs)
    Optional:
    - inputs(t)
    - state_labels()
    - validate_state(state)
    """

    def initial_state(self) -> np.ndarray: ...

    def derivatives(
        self,
        t: float,
        state: np.ndarray,
        inputs: np.ndarray | float | int | None = None,
    ) -> np.ndarray: ...


class DynamicSystem:
    """Base class for dynamic systems using first-order state equations.

    Subclasses should override:
        - initial_state
        - derivatives

    Optionally override:
        - inputs
        - state_labels
        - validate_state
    """

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
    if hasattr(system, 'inputs'):
        return system.inputs(t)  # type: ignore[attr-defined]
    return None


def _step_euler(
    system: SupportsDynamics,
    t: float,
    x: np.ndarray,
    dt: float,
    external_inputs: Optional[InputFunc],
) -> np.ndarray:
    u = _resolve_inputs(system, external_inputs, t)
    dx = np.asarray(system.derivatives(t, x, u), dtype=float)
    return x + dt * dx


def _step_rk4(
    system: SupportsDynamics,
    t: float,
    x: np.ndarray,
    dt: float,
    external_inputs: Optional[InputFunc],
) -> np.ndarray:
    u1 = _resolve_inputs(system, external_inputs, t)
    k1 = np.asarray(system.derivatives(t, x, u1), dtype=float)

    u2 = _resolve_inputs(system, external_inputs, t + 0.5 * dt)
    k2 = np.asarray(system.derivatives(t + 0.5 * dt, x + 0.5 * dt * k1, u2), dtype=float)
    k3 = np.asarray(system.derivatives(t + 0.5 * dt, x + 0.5 * dt * k2, u2), dtype=float)

    u4 = _resolve_inputs(system, external_inputs, t + dt)
    k4 = np.asarray(system.derivatives(t + dt, x + dt * k3, u4), dtype=float)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


_STEPPERS = {
    'euler': _step_euler,
    'rk4': _step_rk4,
}


def simulate(
    system: SupportsDynamics,
    t_span: tuple[float, float],
    dt: float,
    x0: Iterable[float] | np.ndarray | None = None,
    method: str = 'rk4',
    inputs: Optional[InputFunc] = None,
) -> SimulationResult:
    """Simulate a system over a time span.

    Parameters
    ----------
    system:
        Any object implementing the dynamic system protocol.
    t_span:
        (start_time, end_time)
    dt:
        Fixed integration step.
    x0:
        Optional initial state override.
    method:
        'euler' or 'rk4'
    inputs:
        Optional external input function u(t). If omitted, uses system.inputs(t).
    """
    t0, tf = t_span
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0.")
    if method not in _STEPPERS:
        raise ValueError(f"Unknown method '{method}'. Use one of {tuple(_STEPPERS)}")

    x_init = _as_state_vector(system.initial_state() if x0 is None else x0)
    if hasattr(system, 'validate_state'):
        system.validate_state(x_init)  # type: ignore[attr-defined]

    n_steps = int(np.floor((tf - t0) / dt)) + 1
    time = t0 + np.arange(n_steps + 1, dtype=float) * dt
    time = time[time <= tf + 1e-12]

    state = np.zeros((time.size, x_init.size), dtype=float)
    state[0] = x_init

    stepper = _STEPPERS[method]
    for i in range(1, time.size):
        state[i] = stepper(system, time[i - 1], state[i - 1], dt, inputs)

    labels = tuple(system.state_labels()) if hasattr(system, 'state_labels') else tuple(f"x{i}" for i in range(x_init.size))
    return SimulationResult(time=time, state=state, state_labels=labels, system_name=getattr(system, 'name', system.__class__.__name__), method=method)
