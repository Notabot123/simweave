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
- `SeriesRLC`
- `ThermalRC`
- `TwoMassThermal`

Each implements the `DynamicSystem` / `SupportsDynamics` protocol so
custom systems plug into `simulate()` the same way.

## Hybrid: continuous + discrete

Wrap a system in `ContinuousProcess` to register it on a
`SimEnvironment`. The integrator advances one tick per `env.tick()`
call, so continuous physics share the clock with queues and agents.

## API

::: simweave.continuous
    options:
      show_root_he