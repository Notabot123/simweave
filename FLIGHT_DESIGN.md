# Flight dynamics design plan (`simweave.flight`)

Status: **plan only, nothing implemented.** Written against `main` at 0.8.2.
Target release: 0.9.0 (new public subpackage, additive, no breaking changes).

## 1. Goal

Simulate a quadcopter flying a mission end to end: take off, hold, fly a
figure-8, hold, land. The result is a time series that can replace the
hard-coded path in the EdgeWeave 3D drone demo
(`workflow_app_learn/.../user_blocks/drone_flight_nodes.py`), which already
emits `time, x, y, z, roll, pitch, yaw` "so it can be swapped for a true
simulated quadrotor later". Keep the design extensible to general flight
dynamics, but do not build the general framework before a second vehicle
exists (section 9).

## 2. Feasibility summary

High. The existing pieces cover most of the plumbing:

| Need | Already in SimWeave | Gap |
|---|---|---|
| ODE integration | `DynamicSystem`, fixed-step `simulate()` (euler/rk4) | No state-feedback controller hook, no post-step hook |
| Plant model | pattern in `continuous/systems/*` | Write `Quadrotor` (new) |
| Control | `PIDController` (scalar) | No output limits, anti-windup, or derivative-on-measurement |
| Faults | `FaultInjector` perturbs named scalar params | Needs per-motor scalar params to hit one rotor |
| Monte Carlo | `run_monte_carlo` (process pool, seeded) | None; scenario callable must be picklable |
| Hybrid DES | `ContinuousProcess`, `Service`, `ArrivalGenerator` | None for a first demo |
| Plotting | `simweave.viz`, lazy plotly import | New 3D trajectory plot |

Only numpy is needed for the core. scipy stays optional (splines only).

## 3. Findings that shape the design

1. **`simulate()` cannot do closed-loop control.** It resolves inputs with
   `inputs(t)`, time only (`continuous/solver.py`, `_resolve_inputs`). A
   controller needs the state.
2. **The existing workaround is unsound for stateful controllers.**
   `ThermalRC.derivatives` calls `self.controller(T, dt)` with a hard-coded
   `dt = 1e-3` (`continuous/systems/thermal.py:78`), and `PIDController`
   mutates its integral and previous error on every call. RK4 evaluates
   `derivatives` four times per step, so the integral advances four times per
   step by the wrong dt. Do not copy this pattern. (Side finding: demo 20 and
   any user of `ThermalRC(controller=...)` are affected; fixing it is a
   separate small PR, see PR 1.)
3. **Real flight controllers are sampled-data.** The controller runs at a fixed
   rate (100-500 Hz) and holds its output (zero-order hold) between ticks while
   the plant integrates continuously. The solver should model that directly.
4. **EdgeWeave already solved the geometry.** The drone node derives roll,
   pitch and yaw from the acceleration needed (differential flatness: thrust
   vector `T = a + g*up`). The same math drives the attitude setpoint in the
   cascaded controller, and doubles as a consistency check.

## 4. Package layout

```
src/simweave/flight/
  __init__.py        # public re-exports, numpy-only import
  frames.py          # rotation matrices, euler<->quaternion, wrap_angle
  params.py          # QuadrotorParams dataclass (+ unit-aware construction)
  quadrotor.py       # Quadrotor(DynamicSystem)
  mixer.py           # X-config allocation matrix: (T, tau) <-> 4 rotor thrusts
  wind.py            # Wind models: constant, gust (1-cos), seeded turbulence
  control.py         # CascadedController (position->velocity->attitude->rate)
  trajectory.py      # Figure8, quintic segments, Mission (takeoff/hover/.../land)
  results.py         # FlightResult: frame/dict export, metrics, viewer adapter
src/simweave/viz/flight.py   # plot_trajectory_3d, plot_flight_summary (lazy plotly)
```

`import simweave.flight` must work on a base install (numpy only), guarded by
a test in the style of `tests/test_base_install_imports.py`. Plotly is imported
lazily inside the plot functions (see the 0.8.2 changelog).

## 5. Plant model: `Quadrotor`

- **World frame:** Z up (ENU-like). **Body frame:** x forward, y left, z up.
  ZYX Euler angles (yaw, pitch, roll). The EdgeWeave viewer is Y-up, so the
  viewer mapping is an adapter on `FlightResult`, not baked into the physics.
- **State (16):** `x y z vx vy vz roll pitch yaw p q r f1 f2 f3 f4`, where
  `f1..f4` are the actual rotor thrusts (first-order motor lag with time
  constant `motor_tau`). `STATE_UNITS` maps the labels to the existing unit
  classes so `wrap_states` works.
- **Input (4):** commanded rotor thrusts in newtons (what a mixer outputs).
  Optional external wind through a `Wind` model, not through `inputs`.
- **Equations:**
  - translation: `m*a = R(roll,pitch,yaw) @ [0,0,sum(f)] - m*g*z_hat - drag`
    with linear (optionally plus quadratic) drag on airspeed `v - wind`.
  - rotation: `I*omega_dot = tau - omega x (I*omega)`, where `tau` comes from
    the allocation matrix (arm length, rotor torque-to-thrust ratio) and
    Euler rates follow from the standard kinematic map.
  - motors: `f_dot = (f_cmd - f) / motor_tau`, saturated to `[0, f_max]` at the
    command.
- **Ground contact:** a `post_step` rule clamps `z >= 0` and kills downward
  velocity and body rates when landed/disarmed (section 6).
- **Euler limitation:** the kinematic map is singular at pitch = +/-90 degrees.
  That is fine for a tilt-limited mission (limit about 35 degrees) and is
  documented. Aerobatics (flips) would need quaternions; that arrives with the
  phase 2 6-DOF base (section 9) and is out of scope now.
- **Default parameters** (a roughly 1.2 kg sport quad; to be validated in the
  hover test, not tuned to look good): `m=1.2`, `Ixx=Iyy~0.012`, `Izz~0.022`
  kg m^2, arm `0.17 m`, `f_max~7 N` per rotor (thrust-to-weight about 2.4),
  torque/thrust ratio about `0.016 m`, `motor_tau=0.03 s`, linear drag
  about `0.1 N s/m`.
- **Fault hooks:** expose scalars `motor_eff_1..4` (default 1.0) scaling each
  rotor's thrust, so `FaultInjector` with a `ParameterFault` can degrade one
  motor over time. This needs no change to `faults/`.

## 6. Solver changes (the one core edit)

Two small, backward-compatible additions to `continuous/solver.py`:

1. **Sampled-data controller.**
   `simulate(system, t_span, dt, ..., controller=None, control_dt=None)`.
   A `Controller` protocol: `reset()` and `update(t, state, dt_c) -> ndarray`.
   Every `control_dt` (a whole multiple of `dt`, validated) the loop calls
   `update` once and holds the result across the RK4 sub-evaluations by passing
   a held-value function as `inputs`. Held values are logged in
   `result.inputs`. `ContinuousProcess` gets the same option so hybrid DES runs
   work. With `controller=None` behaviour is byte-identical to today.
2. **`post_step(state) -> state` hook** (optional method on the system),
   called after each accepted step. Used for ground contact now and for
   quaternion renormalisation later. Absent method means no change.

`PIDController` gets backward-compatible keyword options: `output_limits`,
anti-windup (clamp or back-calculation), `derivative_on_measurement`, optional
derivative low-pass `tau_d`. Default arguments reproduce current output.

## 7. Control and trajectories

**Cascaded controller** (`flight/control.py`), single `control_dt` (default
250 Hz) with an optional divider so outer loops run slower:

```
reference (pos, vel, acc, yaw)
  -> position P  -> velocity setpoint (speed-limited)
  -> velocity PID (+ reference acceleration feedforward) -> desired accel a
  -> thrust vector T = m(a + g*z_hat): collective = |T|, body-z target = T/|T|
     (tilt-limited; same flatness geometry as the EdgeWeave node)
  -> attitude P -> body-rate setpoint
  -> rate PID -> body torques
  -> mixer -> 4 rotor thrusts (desaturate: preserve attitude torques over
     collective thrust when clipping)
```

Gains come from sensible defaults scaled by mass and inertia, overridable by
the user. All loops use the upgraded PID so anti-windup works through
saturation.

**Trajectories** (`flight/trajectory.py`) share one protocol:
`ref(t) -> (pos, vel, acc, yaw)`.

- `Figure8(a, b, z0, period)`: `x = a sin(wt)`, `y = (b/2) sin(2wt)`, optional
  altitude oscillation, analytic derivatives (same shape as the EdgeWeave
  default path). Yaw modes: fixed, or face velocity (hold the last yaw when
  speed is near zero).
- `quintic(state0, state1, T)`: minimum-jerk segment with position, velocity
  and acceleration boundary conditions on each axis. This is the glue that lets
  hover (zero velocity) blend into the figure-8 (non-zero velocity) smoothly.
- `Mission([...])`: sequence of `Takeoff(height)`, `Hover(T)`, `Fly(Figure8,
  laps)`, `Land(rate)`. Landing descends with a limited final rate and then
  disarms, so the ground-contact rule triggers. Segment joins are
  continuous in position, velocity and acceleration.
- Stretch: `WaypointTrajectory` (C2 spline, scipy optional) as a port of the
  EdgeWeave node, so arbitrary waypoint lists also fly under physics.

## 8. Results, metrics, integrations

- `FlightResult` wraps `SimulationResult` plus the reference. Exports a dict of
  arrays (and a pandas DataFrame when pandas is installed; pandas is not a
  base dependency) with columns `time, x, y, z, roll, pitch, yaw` first, then
  velocities, rates, rotor thrusts. `to_viewer(y_up=True)` does the axis and
  angle-convention mapping for EdgeWeave.
- Metrics: RMS and max position error vs reference, max tilt, fraction of time
  any rotor is saturated, ideal-rotor energy (momentum theory, with figure of
  merit) in Wh, touchdown speed and lateral landing error.
- **Plots:** `plot_trajectory_3d(result, reference=None)` and
  `plot_flight_summary(result)` (error, tilt, rotor thrusts), themed like the
  other viz helpers.
- **Faults:** one-rotor degradation demo with `FaultInjector`; the controller
  compensates until a rotor saturates, then yaw/altitude drift. This is a
  realistic predictive-maintenance story linking to the existing blog.
- **Monte Carlo:** `run_monte_carlo` over random wind and payload mass,
  returning tracking RMS and landing dispersion. The scenario must be a
  module-level function so the process pool can pickle it.
- **Hybrid DES (stretch):** drone delivery. Orders arrive via
  `ArrivalGenerator`; a `Service` stage uses mission duration and battery
  energy from the flight sim (looked up from a pre-run payload/distance sweep)
  as service and recharge time.

## 9. Broader flight dynamics (phase 2, not now)

Recommended order: ship the quadrotor first, then generalise from real code.

- `RigidBody6DOF` base with quaternion attitude (using the `post_step`
  renormalisation hook), and a `ForceModel` protocol (gravity, propulsion,
  aero) summed into force and moment.
- ISA atmosphere (`density(h)`), wind/turbulence (Dryden, seeded).
- Fixed-wing small-UAV model with linear aerodynamic coefficients.
- `trim()` and `linearize()` (numerical Jacobian, eigenvalues) as generic
  continuous-systems tools; useful well beyond flight.

Why defer: an abstraction drawn from a single vehicle is usually wrong. The
`frames.py` helpers (rotations, euler/quaternion) are written in phase 1 so
phase 2 reuses them.

## 10. Test strategy

- **frames/mixer:** rotation matrices orthonormal, euler/quaternion round trip,
  mixer matrix invertible, `thrusts -> (T, tau) -> thrusts` identity.
- **Plant physics (no controller):** free fall matches `z = -g t^2/2` (RK4 is
  exact for this); hover thrust `m g / 4` per rotor holds `z` constant to
  numerical tolerance; yaw torque gives `r_dot = tau / Izz`; constant drag
  reaches the analytic terminal velocity.
- **Solver hook:** a P controller on a first-order plant under ZOH matches the
  analytic discrete-time solution; `controller=None` leaves all existing tests
  untouched; `post_step` is called once per step; `control_dt` not a multiple
  of `dt` raises.
- **PID:** limits clamp, anti-windup stops integrator growth during
  saturation, defaults reproduce the old output.
- **Closed loop:** hover from a 1 m offset converges with bounded overshoot;
  figure-8 tracking RMS below a threshold fixed after the first measured run;
  constant 5 m/s wind steady-state error near zero (integral action);
  takeoff reaches height; landing touches down below a speed limit; saturation
  never produces NaN.
- **Determinism:** seeded wind gives identical results across runs.
- **Faults:** one-rotor degradation increases tracking error monotonically
  with severity.
- **Base-install import guard** for `simweave.flight`.
- **Performance spike first:** a pure-Python RK4 loop at `dt = 2-5 ms` for a
  60 s mission is tens of thousands of small numpy calls. Measure before
  committing to defaults, and keep `derivatives` allocation-light. Monte Carlo
  of hundreds of missions depends on this number, and I have not measured it.
- **EdgeWeave consistency (manual / demo, not CI):** fly the node's default
  figure-8 and compare simulated roll/pitch against the node's flatness
  attitudes.

## 11. Delivery plan (one PR each, in order)

1. **Solver and PID:** `controller=`/`control_dt` and `post_step` in `simulate`
   and `ContinuousProcess`; PID upgrades; tests. Optionally move `ThermalRC`
   onto the new hook (fixes finding 2) or file that fix separately.
2. **Flight core:** `frames`, `params`, `Quadrotor`, `mixer`, `wind`, plant
   physics tests, base-install guard. Includes the performance spike.
3. **Control and missions:** `CascadedController`, trajectories, `Mission`,
   `FlightResult`, closed-loop tests, demo 27 (take off, figure-8, land).
4. **Viz and integrations:** 3D plots, wind Monte Carlo demo 28, motor fault
   demo 29, packaging/extras check.
5. **Docs and release:** `docs/modules/flight.md` and `mkdocs.yml` nav,
   `SIMWEAVE_API.md`, `claude_skill/SKILL.md`, a blog post
   (`blog-writer` skill) with a companion notebook (`companion-notebooks`
   skill), CHANGELOG, release 0.9.0.
6. **Later:** EdgeWeave "Quadrotor Simulation" node in the EdgeWeave repo
   (same output columns as the Drone Flight Path node, run in the worker
   subprocess); then phase 2.

## 12. Decisions (agreed with the user)

1. **Coordinates:** Z-up world; `FlightResult.to_viewer()` adapts to the Y-up
   EdgeWeave viewer.
2. **Solver edit:** `controller=`/`control_dt` and `post_step` go into the
   public `simulate()` and `ContinuousProcess`.
3. **Attitude:** Euler angles with a tilt limit in v1; quaternions arrive with
   phase 2.
4. **Phase 2:** fixed-wing / 6-DOF base after 0.9.0, in a follow-up release.
5. **Realism:** first-order motor lag and an ideal-rotor energy model are in v1.
6. **ThermalRC PID fix:** a separate PR after PR 1, with a changelog entry
   (demo 20 output will change).
7. **Demo numbers:** flight takes 27-29; the copula plan starts at 30.
