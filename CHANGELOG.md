# Changelog

## 0.9.0 - 2026-10-18

Flight dynamics: a quadcopter you can fly in Python, plus the control hooks it needed.

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

### Fixed

- `FaultInjector` did not forward the wrapped system's `post_step` hook, so
  wrapping a system that relies on it (the quadrotor's ground contact) let it
  fall through the floor. It now forwards the hook.

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
