# simeng framework example

This package provides a small generic solver framework for first-order state-space systems,
along with several example models:

- Mass-spring-damper
- Quarter-car suspension
- Simple pendulum
- Series RLC circuit

## Install test dependencies

```bash
pip install pytest numpy
```

## Run tests

```bash
pytest -q
```

## Example

```python
from simeng.solver import simulate
from simeng.systems import QuarterCarModel

model = QuarterCarModel(
    sprung_mass=250,
    unsprung_mass=40,
    suspension_stiffness=15000,
    damping=1500,
    tyre_stiffness=200000,
)

road = lambda t: 0.05 if t >= 0.1 else 0.0
result = simulate(model, t_span=(0.0, 1.0), dt=0.001, inputs=road)
```

## Design notes

Each system is a plug-in implementing:

- `initial_state()`
- `derivatives(t, state, inputs)`
- optionally `state_labels()`

The shared solver then integrates using fixed-step Euler or RK4.
