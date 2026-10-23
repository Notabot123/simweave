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
from simweave.units.si import SIUnit

ArrayLike = np.ndarray
InputFunc = Callable[[float], np.ndarray | float | int | None]


@dataclass(slots=True)
class SimulationResult:
    time: np.ndarray
    state: np.ndarray
    state_labels: tuple[str, ...]
    system_name: str
    method: str
    inputs: np.ndarray | None = None

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


@runtime_checkable
class Controller(Protocol):
    """Protocol for sampled-data (digital) controllers.

    ``update`` is called once per control period with the current time and
    state and returns the control input, which the solver holds constant
    (zero-order hold) until the next control tick. ``reset`` is optional and,
    when present, is called at the start of every run.
    """

    def update(
        self, t: float, state: np.ndarray, dt: float
    ) -> np.ndarray | float | int | None: ...


class WrappedResult:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class DynamicSystem:
    """Base class for first-order state-space dynamic systems."""

    STATE_UNITS: dict[str, type] = {}

    @staticmethod
    def _val(x):
        """ Allow SIUnits to be supplied despite float for internal calcs """

        if isinstance(x, SIUnit):
            return x.value
        if isinstance(x, (int, float, np.ndarray)):
            return x
        raise TypeError(f"Unsupported type for numeric value: {type(x)}")

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
        
    def get_state_unit(self, label: str):
        return self.STATE_UNITS.get(label)
    
    def wrap_states(self, result):
        wrapped = {}

        labels = result.state_labels
        state = result.state

        for i, label in enumerate(labels):
            unit_cls = self.STATE_UNITS.get(label)

            if unit_cls is not None:
                wrapped[label] = unit_cls(state[:, i])
            else:
                wrapped[label] = state[:, i]

        return WrappedResult(wrapped)

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


def _control_steps(control_dt: float | None, dt: float) -> int:
    """Number of integration steps per control period."""
    if control_dt is None:
        return 1
    if control_dt <= 0:
        raise ValueError("control_dt must be positive.")
    k = int(round(control_dt / dt))
    if k < 1 or abs(k * dt - control_dt) > 1e-9 * max(1.0, abs(control_dt)):
        raise ValueError(
            f"control_dt ({control_dt}) must be a whole multiple of dt ({dt})."
        )
    return k


def _hold(value) -> InputFunc:
    """Wrap a held control value as a time-independent input function."""
    return lambda t: value


def _post_step(system, state: np.ndarray) -> np.ndarray:
    hook = getattr(system, "post_step", None)
    if hook is None:
        return state
    return np.asarray(hook(state), dtype=float)


def _reset_controls(system) -> None:
    """Call the optional ``reset_controls()`` hook at the start of a run."""
    hook = getattr(system, "reset_controls", None)
    if hook is not None:
        hook()


def simulate(
    system: SupportsDynamics,
    t_span: tuple[float, float],
    dt: float,
    x0: Iterable[float] | np.ndarray | None = None,
    method: str = "rk4",
    inputs: Optional[InputFunc] = None,
    controller: Optional[Controller] = None,
    control_dt: float | None = None,
) -> SimulationResult:
    """Integrate a system over ``t_span`` with fixed step ``dt``.

    Parameters
    ----------
    controller:
        Optional sampled-data controller (see :class:`Controller`). Every
        ``control_dt`` seconds it is called as ``controller.update(t, state,
        control_dt)`` and its output is held constant across the integration
        steps of that period. It replaces ``inputs`` and the system's own
        ``inputs(t)``, so passing both ``controller`` and ``inputs`` is an
        error.
    control_dt:
        Control period; must be a whole multiple of ``dt``. Defaults to
        ``dt`` (the controller runs every step). Ignored without a controller.

    Notes
    -----
    If the system defines ``post_step(state) -> state`` it is applied after
    every integration step (for example to clamp a ground contact or
    renormalise a quaternion).

    If the system defines ``control_update(t, state, dt)`` it is called once
    before every integration step with the true step size ``dt`` (and
    ``reset_controls()``, if present, is called at the start of the run). A
    system that carries its own feedback law should use this hook instead of
    evaluating a stateful controller inside ``derivatives``, which is called
    several times per step and so cannot advance an integrator correctly.
    """
    t0, tf = t_span
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0.")
    if method not in _STEPPERS:
        raise ValueError(f"Unknown method '{method}'. Use one of {tuple(_STEPPERS)}")
    if controller is not None and inputs is not None:
        raise ValueError("Pass either `controller` or `inputs`, not both.")
    k_ctrl = _control_steps(control_dt, dt) if controller is not None else 1

    x_init = _as_state_vector(system.initial_state() if x0 is None else x0)
    if hasattr(system, "validate_state"):
        system.validate_state(x_init)  # type: ignore[attr-defined]
    _reset_controls(system)
    control_update = getattr(system, "control_update", None)

    n_steps = int(np.floor((tf - t0) / dt)) + 1
    time = t0 + np.arange(n_steps + 1, dtype=float) * dt
    time = time[time <= tf + 1e-12]

    state = np.zeros((time.size, x_init.size), dtype=float)
    state[0] = x_init

    held = None
    if controller is not None:
        if hasattr(controller, "reset"):
            controller.reset()
        held = controller.update(time[0], state[0].copy(), k_ctrl * dt)
        t0_input = held
    else:
        t0_input = _resolve_inputs(system, inputs, time[0])
    inputs_log = [np.atleast_1d(t0_input).astype(float) if t0_input is not None else np.zeros(1)]

    stepper = _STEPPERS[method]
    for i in range(1, time.size):
        t_prev = time[i - 1]

        # --- resolve inputs ---
        if controller is not None:
            if (i - 1) % k_ctrl == 0 and i > 1:
                held = controller.update(t_prev, state[i - 1].copy(), k_ctrl * dt)
            inputs_fn = _hold(held)
            u = held
        elif inputs is not None:
            inputs_fn = inputs
            u = inputs(t_prev)
        else:
            inputs_fn = inputs
            u = system.inputs(t_prev)

        # normalise to array for storage
        if u is None:
            u_arr = np.zeros(1)
        else:
            u_arr = np.atleast_1d(u).astype(float)

        inputs_log.append(u_arr)

        # --- the system's own feedback law, once per step with the true dt ---
        if control_update is not None:
            control_update(t_prev, state[i - 1].copy(), dt)

        # --- step ---
        state[i] = _post_step(
            system, stepper(system, time[i - 1], state[i - 1], dt, inputs_fn)
        )

    # gather inputs
    inputs_array = np.vstack(inputs_log)

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
        inputs=inputs_array,
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
        controller: Controller | None = None,
        control_dt: float | None = None,
    ) -> None:
        super().__init__(
            name=name or getattr(system, "name", system.__class__.__name__)
        )
        if method not in _STEPPERS:
            raise ValueError(f"Unknown method '{method}'")
        if n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")
        if controller is not None and inputs is not None:
            raise ValueError("Pass either `controller` or `inputs`, not both.")
        if control_dt is not None and control_dt <= 0:
            raise ValueError("control_dt must be positive.")
        self.system = system
        self.method = method
        self.n_substeps = n_substeps
        self.inputs = inputs
        self.controller = controller
        self.control_dt = control_dt
        self._held = None
        self._substep_index = 0
        self.state = _as_state_vector(system.initial_state())
        self._stepper = _STEPPERS[method]
        self.history_time: list[float] = []
        self.history_state: list[np.ndarray] = []

    def on_register(self, env) -> None:
        super().on_register(env)
        self.history_time.append(env.clock.t)
        self.history_state.append(self.state.copy())
        if self.controller is not None and hasattr(self.controller, "reset"):
            self.controller.reset()
        _reset_controls(self.system)
        self._substep_index = 0

    def tick(self, dt: float, env) -> None:
        super().tick(dt, env)
        sub_dt = dt / self.n_substeps
        t = env.clock.t
        k_ctrl = (
            _control_steps(self.control_dt, sub_dt)
            if self.controller is not None
            else 1
        )
        control_update = getattr(self.system, "control_update", None)
        for _ in range(self.n_substeps):
            inputs_fn = self.inputs
            if self.controller is not None:
                if self._substep_index % k_ctrl == 0:
                    self._held = self.controller.update(
                        t, self.state.copy(), k_ctrl * sub_dt
                    )
                inputs_fn = _hold(self._held)
            if control_update is not None:
                control_update(t, self.state.copy(), sub_dt)
            self.state = _post_step(
                self.system,
                self._stepper(self.system, t, self.state, sub_dt, inputs_fn),
            )
            self._substep_index += 1
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