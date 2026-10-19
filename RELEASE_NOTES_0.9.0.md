## What's new in v0.9.0

Two headline additions: **flight dynamics** (a quadcopter you can fly in Python) and **dependence modelling** (copulas for correlated draws in discrete-event simulations). Both need only numpy.

```bash
pip install -U simweave
```

### New module: `simweave.flight`: multirotor flight dynamics

| Component | Description |
|---|---|
| `Quadrotor` | 16-state rigid-body quadcopter with first-order rotor response, aerodynamic drag, ground contact and an energy model. Rotor effectiveness is exposed as `motor_eff_1..4`, so `FaultInjector` can weaken a motor |
| `Mixer` | X-layout control allocation with attitude-first desaturation |
| `CascadedController` | Position, velocity, thrust-vector/attitude and body-rate loops with tilt limit and yaw-rate feed-forward; `ControllerGains` for tuning |
| `Mission` / `Hold` / `Quintic` / `Figure8` | Reference trajectories. `Mission.figure8()` chains take-off, hover, figure-8 laps and a soft landing with continuous position, velocity and acceleration at every join |
| `ConstantWind` / `GustWind` / `TurbulentWind` | Wind models acting on the air-relative velocity |
| `fly()` / `FlightResult` | Flies a mission and returns `metrics()`, `phase_rms()`, `touchdown()`, `to_dict()`, `to_frame()` and `to_viewer()` (Y-up columns for a 3D viewer). Also accepts a wrapper such as `FaultInjector` |

### New module: `simweave.stats`: copulas and dependent sampling

| Component | Description |
|---|---|
| `GaussianCopula`, `ClaytonCopula`, `GumbelCopula`, `FrankCopula` | Copulas that work on a numpy-only install. `from_kendall_tau(tau)`, `.flipped(axes)` and `.survival()` for negative dependence and the opposite tails, and `tail_dependence()` for the tail coefficients |
| `StudentTCopula` | Symmetric tail dependence. Needs scipy |
| `Exponential`, `Uniform`, `Normal`, `LogNormal`, `Weibull`, `Deterministic`, `Empirical`, `ScipyMarginal` | Marginals with `ppf`/`cdf`. `sw.exponential`, `sw.uniform`, `sw.normal` and `sw.deterministic` also expose `ppf`/`cdf`, so they work as marginals too (their draws are unchanged) |
| `JointSampler` | Named dependent draws: `joint.draw(rng)` gives `{"interarrival": ..., "service_time": ...}` |
| `MarkovCopulaProcess` | Serially dependent sequences (bursty arrivals) with an unchanged marginal. A plain callable, so it drops into `ArrivalGenerator(interarrival=...)` |
| `ArrivalGenerator(joint=, interarrival_key=, gap_alignment=)` | Opt-in joint draw of the gap and entity attributes. `gap_alignment="before"` (default) or `"after"` chooses which entity receives the attributes |
| `EntityProperties.from_draw()` | Turns a joint draw into per-entity properties: the drawn service time becomes the entity's service time, the rest go to `extras` |
| `fit_copula` / `JointSampler.from_data` | Fit from observations by Kendall-tau inversion and rank families by AIC; `method="mle"` needs scipy |

### Control and solver hooks

| Addition | Description |
|---|---|
| `simulate(..., controller=, control_dt=)` | Sampled-data controller protocol (`simweave.continuous.Controller`); the output is held across the integration steps of each control period. `ContinuousProcess` takes the same arguments |
| `post_step(state) -> state` | Optional hook on dynamic systems, applied after every integration step |
| `PIDController` options | `output_limits`, `anti_windup`, `derivative_on_measurement`, `derivative_tau` and an `update()` adapter. Defaults reproduce the previous output |

### New visualisation helpers

| Helper | Description |
|---|---|
| `plot_trajectory_3d` | The flown path in 3D, coloured by speed, with the reference dashed |
| `plot_flight_summary` | Position error, tilt, rotor thrusts and power over time with the mission phases shaded |
| `plot_landing_dispersion` | Landing points scattered around the target with 50% and 95% circles |

### New demos

| Demo | Description |
|---|---|
| `27_quadcopter_figure8.py` | Take off, fly a figure-8 in wind, land; tracking metrics per phase |
| `28_quadcopter_wind_montecarlo.py` | Landing dispersion over random wind and unknown payload |
| `29_quadcopter_motor_fault.py` | Degrade one rotor until the controller runs out of thrust |
| `30_serial_dependence_queue.py` | Same exponential gaps, but bursty: about 4.4 against about 20 time units in the system |
| `31_copula_families_queue.py` | Same marginals and Kendall tau under Gaussian, Clayton, Gumbel and Student-t copulas in one queue |

### Docs

Two new blog posts ("Flying a Quadcopter in Simulation" and "When Arrivals Aren't Independent: Copulas in SimWeave"), Flight and Stats module pages, and the design notes. Companion notebooks `06_quadcopter_flight.ipynb` and `07_copulas_dependence.ipynb` are in [simweave-notebooks](https://github.com/Notabot123/simweave-notebooks).

## Upgrade notes

- **Nothing breaks.** Existing `simulate(...)` calls, `ArrivalGenerator(...)` calls and the `sw.exponential`/`uniform`/`normal`/`deterministic` helpers behave as before. `ArrivalGenerator`'s `joint=` and the controller arguments are opt-in.
- **scipy stays optional.** Only `StudentTCopula` and `fit_copula(..., method="mle")` need it; they raise the usual "pip install simweave[optim]" `ImportError` without it. `import simweave` does not import scipy.
- A `MarkovCopulaProcess` carries state. Build one per Monte Carlo replicate, inside the scenario function.
- v0.9.0 is the first release since v0.8.2; the flight and copula work were not published separately.

## Fixed

- `FaultInjector` did not forward the wrapped system's `post_step` hook, so wrapping a system that relies on it (the quadrotor's ground contact) let it fall through the floor. It now forwards the hook. (#33)
- The numpy-only normal CDF failed on a single scalar input. Fixed before release. (#35)

## What's Changed
* feat: sampled-data controller hook, post_step and PID upgrades by @Notabot123 in https://github.com/Notabot123/simweave/pull/30
* feat: add simweave.flight quadrotor plant, mixer and wind models by @Notabot123 in https://github.com/Notabot123/simweave/pull/31
* feat: quadrotor cascaded controller, trajectories, missions and FlightResult by @Notabot123 in https://github.com/Notabot123/simweave/pull/32
* feat: flight plots, wind Monte Carlo and motor fault demos, FaultInjector post_step fix by @Notabot123 in https://github.com/Notabot123/simweave/pull/33
* docs: quadcopter flight blog post, API and skill updates; release 0.9.0 by @Notabot123 in https://github.com/Notabot123/simweave/pull/34
* feat: simweave.stats copulas, joint and serial sampling, fitting by @Notabot123 in https://github.com/Notabot123/simweave/pull/35

**Full Changelog**: https://github.com/Notabot123/simweave/compare/v0.8.2...v0.9.0
