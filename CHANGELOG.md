# Changelog

## Unreleased

### Fixed

- `ThermalRC(controller=pid)` gave a closed-loop result that depended on the solver
  step size. The PID was called inside `derivatives()` (several times per step) with a
  hard-coded `dt = 1e-3`, so its integral term did not advance by real time: the same
  gains ended at 343.5 K for `dt=0.05` and 350.6 K for `dt=0.01`. The embedded controller
  now runs once per solver step with the true `dt` and its output is held across the step,
  so the result is independent of `dt` (349.7 K final in both) and matches
  `simulate(plant, ..., controller=pid, control_dt=dt)`. Controller state is also reset at
  the start of every run (it used to carry over between runs).

### Added

- Optional system hooks `control_update(t, state, dt)` and `reset_controls()`: when a
  system defines them, `simulate` and `ContinuousProcess` call `control_update` once before
  every integration step with the true step size, and `reset_controls` at the start of a run.
  They sit alongside the existing `post_step` hook.

### Deprecated

- `ThermalRC(controller=...)` now emits a `DeprecationWarning`. Pass the controller to
  `simulate(plant, ..., controller=pid, control_dt=...)` (or to `ContinuousProcess`) instead.
  Demo 20 and the docs use that route.

## 0.9.0 - 2026-10-19

Two additions: flight dynamics (a quadcopter you can fly in Python, plus the control hooks it
needed) and dependence modelling (copulas for correlated draws in discrete-event simulations).

### Added

- `simulate(..., controller=, control_dt=)` and the matching
  `ContinuousProcess` arguments: a sampled-data controller protocol
  (`simweave.continuous.Controller`) whose output is held across the
  integration steps of each control period. Existing calls are unchanged.
- Optional `post_step(state) -> state` hook on dynamic systems, applied after
  every integration step.
- `PIDController` options: `output_limits`, `anti_windup`,
  `derivative_on_measurement`, `derivative_tau`, plus an `update()` adapter.
  Defaults reproduce the previous output; `dt <= 0` now raises `ValueError`.
- Flight dynamics design plan (`FLIGHT_DESIGN.md`).
- `simweave.flight` (numpy only): `Quadrotor` (16-state rigid body with
  first-order rotor response, drag, ground contact and an energy model),
  `QuadrotorParams`, `Mixer` (X-layout control allocation with attitude-first
  desaturation), wind models (`ConstantWind`, `GustWind`, `TurbulentWind`) and
  rotation/quaternion helpers. Rotor effectiveness is exposed as
  `motor_eff_1..4` for `FaultInjector`.
- `simweave.flight` control and missions: `CascadedController` (position,
  velocity, attitude and rate loops with tilt limit and yaw-rate feed-forward),
  `ControllerGains`, trajectories `Hold`, `Quintic` and `Figure8`,
  `Mission.figure8()` (take off, figure-8, land with continuous transitions),
  and `fly()` returning a `FlightResult` with `metrics()`, `phase_rms()`,
  `to_dict()`, `to_frame()` and `to_viewer()` (Y-up viewer columns).
- Demo 27: quadcopter figure-8 mission in wind.
- Flight plots: `plot_trajectory_3d`, `plot_flight_summary` and
  `plot_landing_dispersion` (plotly imported lazily, like the other helpers).
- `fly()` accepts a wrapper exposing `.system` (such as `FaultInjector`), and
  `FlightResult.touchdown()` also reports the touchdown `x` and `y`.
- Demo 28 (wind and payload Monte Carlo of landing dispersion) and demo 29
  (progressive rotor degradation).
- Blog post "Flying a Quadcopter in Simulation", a flight module docs page, and
  README, API reference and skill updates (the skill's PID example now uses the
  real `PIDController(Kp=...)` signature).

- `simweave.stats` (numpy only): `GaussianCopula`, `ClaytonCopula`, `GumbelCopula`,
  `FrankCopula` and `StudentTCopula` (needs scipy), each with `from_kendall_tau`,
  `.flipped(axes)`, `.survival()` and `tail_dependence()`; marginals with `ppf`/`cdf`
  (`Exponential`, `Uniform`, `Normal`, `LogNormal`, `Weibull`, `Deterministic`,
  `Empirical`, `ScipyMarginal`, `as_marginal`).
- `JointSampler` for named dependent draws, and `MarkovCopulaProcess` for serially
  dependent sequences (bursty arrivals with unchanged marginals) that plugs into
  `ArrivalGenerator(interarrival=...)`.
- `ArrivalGenerator(joint=, interarrival_key=, gap_alignment=)` and
  `EntityProperties.from_draw()` to carry a jointly drawn attribute (service time,
  size) onto an entity. Without `joint` behaviour is unchanged.
- `fit_copula` / `JointSampler.from_data`: fit from observations by Kendall-tau inversion
  and rank families by AIC; `method="mle"` needs scipy.
- `sw.exponential`, `sw.uniform`, `sw.normal`, `sw.deterministic` now also expose
  `ppf`, `cdf` and (where defined) `mean`, so they can be used as copula marginals.
  Their draws are unchanged.
- Demos 30 (serial dependence) and 31 (copula families), and the copula design note
  (`COPULA_DESIGN.md`).
- Blog post "When Arrivals Aren't Independent: Copulas in SimWeave".

### Fixed

- `FaultInjector` did not forward the wrapped system's `post_step` hook, so
  wrapping a system that relies on it (the quadrotor's ground contact) let it
  fall through the floor. It now forwards the hook.

### Notes

- scipy stays optional; only the Student-t copula and `method="mle"` need it and raise
  the usual "pip install simweave[optim]" `ImportError`. `import simweave` still does not
  import scipy.

## 0.8.2 - 2026-10-18

### Fixed

- `import simweave` failed on a base install (no extras) with
  `ModuleNotFoundError: No module named 'plotly'`. `simweave.viz.vehicle_dynamics`
  imported `plotly.subplots` at module level, and `simweave.roads` pulls in
  `simweave.viz`. plotly is now imported lazily inside `plot_vehicle_metrics`,
  which raises the usual "pip install simweave[viz]" `ImportError` when plotly
  is missing. 0.8.1 on PyPI is affected; upgrade to 0.8.2.
- `Service.average_residence()` double counted time spent waiting in the buffer
  (an item's `age` already includes it), so it read about twice the true wait for
  a single stage (about 4.3 against a theoretical 2.5 for M/M/1 at
  `lam=0.6, mu=1.0`). It now reports age at completion minus age at stage entry,
  i.e. waiting plus service for that stage, so chained services are per-stage.
- Docstrings for `Queue.average_wait()` and `Queue.average_length()` said
  "residence time"; they are the queue-only `Wq` and `Lq`. Use
  `Service.average_residence()` for time in the stage. (#27)

### Added

- Regression test (`tests/test_base_install_imports.py`) that blocks plotly via an
  import hook and checks `import simweave` works and plot helpers raise the
  friendly `ImportError`.
- CI job `base-install` that installs the built wheel with base dependencies
  only in a clean venv and runs an import + smoke test.
