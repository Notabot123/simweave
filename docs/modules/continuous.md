# Continuous

Fixed-step ODE integration plus a small library of canonical example
systems used throughout the demos and tests.

## Quick example

```python
import numpy as np
import simweave as sw

msd = sw.MassSpringDamper(mass=1.0, damping=0.4, stiffness=4.0)
res = sw.simulate(msd, t_span=(0.0, 12.0), dt=0.01, x0=np.array([1.0, 0.0]))
```

<iframe src="../../embeds/msd_states.html"
        width="100%" height="480" frameborder="0"
        loading="lazy"
        title="Damped MSD trajectories"></iframe>

<iframe src="../../embeds/msd_phase.html"
        width="100%" height="480" frameborder="0"
        loading="lazy"
        title="Damped MSD phase portrait"></iframe>

## Available systems

- `MassSpringDamper`
- `SimplePendulum`
- `QuarterCarModel`
- `HalfCarModel`
- `RollCarModel`
- `SeriesRLC`
- `ThermalRC`
- `TwoMassThermal`

Each implements the `DynamicSystem` / `SupportsDynamics` protocol so
custom systems plug into `simulate()` the same way.

## Hybrid: continuous + discrete

Wrap a system in `ContinuousProcess` to register it on a
`SimEnvironment`. The integrator advances one tick per `env.tick()`
call, so continuous physics share the clock with queues and agents.

# Vehicle dynamics examples
## Quarter car
```python
d = sw.QuarterCarModel(250, 40, 15000, 1500, 200000)
r = sw.simulate(d, (0.0, 2.0), dt=0.001, inputs=lambda t: 0.01)
```
See full demo: `demos/10_quarter_car.py`

## Half car (pitch)
```python
model = sw.HalfCarModel(
    1200, 2500, 60, 60,
    20000, 20000,
    1500, 1500,
    150000, 150000,
    1.2, 1.6
)

r = sw.simulate(model, (0.0, 2.0), dt=0.001,
                inputs=lambda t: (0.01, 0.01))
```
See full demo: `demos/17_half_car.py`

## Roll model (left/right)

```python
model = sw.RollCarModel(
    1200, 2200, 60, 60,
    20000, 20000,
    1500, 1500,
    150000, 150000,
    1.6
)

r = sw.simulate(model, (0.0, 2.0), dt=0.001,
                inputs=lambda t: (0.01, 0.0))
```
See full demo: `demos/18_roll_car_model.py`

## API

::: simweave.continuous
    options:
      show_root_heading: false
      show_source: true
