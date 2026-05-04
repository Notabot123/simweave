# Continuous

Fixed-step ODE integration plus a small library of canonical example
systems used throughout the demos and tests.

All continuous models use SI units:
- Displacement: metres (m)
- Velocity: m/s
- Angles: radians (rad)

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
- `FullCarModel`
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
A 2 DOF (Degrees of Freedom) model with 4 variables (position + velocity).

- z_s → sprung mass (body)
- z_u → unsprung mass (wheel)

```python
d = sw.QuarterCarModel(250, 40, 15000, 1500, 200000)
r = sw.simulate(d, (0.0, 2.0), dt=0.001, inputs=lambda t: 0.01)
```
See full demo: `demos/10_quarter_car.py`

## Half car (pitch)
A 4 DOF (Degrees of Freedom) model for simulating pitch movement between front and back.

- z_s → vertical body motion
- theta → pitch
- z_uf → front wheel
- z_ur → rear wheel

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
See full demo: `demos/17_half_car_pitch.py`

## Roll model (left/right)
A 4 DOF (Degrees of Freedom) model for simulating roll movement between left and right.

- z_s → vertical body motion
- phi → roll
- z_ul → left wheel
- z_ur → right wheel

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
See full demo: `demos/18_half_car_roll_model.py`

## Full Car Model
A 7 DOF (Degrees of Freedom) model is included, extending prior examples.

- z_s → vertical body motion
- theta → pitch
- phi → roll
- z_ufl → front-left
- z_ufr → front-right
- z_url → rear-left
- z_urr → rear-right

```python
model = sw.FullCarModel(
    sprung_mass=1200,
    pitch_inertia=1500,
    roll_inertia=800,
    unsprung_mass=40,
    k_s=20000,
    c_s=1500,
    k_t=200000,
    a=1.2,
    b=1.3,
    track_width=1.6,
)

result = simulate(
    model, (0.0, 3.0),
    dt=0.001, 
    inputs=lambda t: np.array([0.01, 0.01, 0.0, 0.0]) if t > 1 else np.zeros(4),
)

fig = sw.plot_vehicle_metrics(
    res,
    title="Full car response (bump input)",
    model=model,
)
```
See full demo: `demos/19_full_car_dynamics.py`

<iframe src="../../embeds/full_car_metrics.html"
        width="100%" height="900" frameborder="0"
        loading="lazy">
</iframe>

## Suspension Controllers
From v0.6.0, SimWeave includes controllers to help optimise dynamics according to prefered performance criteria:
- Comfort (Acceleration of Vertical Vibrations)
- Grip (Dynamic Tyre Loading)
- Hybrid

These are realised by Skyhook and Groundhook control.
See:
- simweave.continuous.control.suspension.SkyhookDamper
- simweave.continuous.control.suspension.GroundhookDamper
- simweave.continuous.control.suspension.HybridActiveDamper

The former imagines one end of a damper 'connected to the sky' (an inertial reference point).
The latter images on end affixed to the ground. This ensure dissipative force applies with respect to body movement, or wheel movement only as opposed to the releative velocities.

Example controller usage:
```python
passive = QuarterCarModel(...)
controlled = QuarterCarModel(..., controller=SkyhookDamper(1500))

r_passive = simulate(passive, (0, 2), dt=0.001, inputs=lambda t: 0.01)
r_control = simulate(controlled, (0, 2), dt=0.001, inputs=lambda t: 0.01)

z_passive = r_passive.state[:, 0]
z_control = r_control.state[:, 0]

print(f"Standard deviation controlled system: {z_control.std()}")
print(f"Standard deviation passive system: {z_passive.std()}")
```

A difference between an idealised behaviour, and the passive suspension can be used as a control signal.
This could be used by an actuator in place of a damper, capable of adding energy into the system as a 'Fully Active' system.
Alternatively, it could be a variable damping coefficient e.g. magnetorheological (MR) dampers.

To ensure fidelity with a real-world suspension that is only dissipative in nature, a wrapper allows for controller force to only apply when Force * relative velocity is negative.

See: simweave.continuous.control.suspension.SemiActiveWrapper

```python
# enforce dissipative constraint
if F * v_rel > 0:
    return 0.0
```
Example wrapper usage:
```python
controller = SemiActiveWrapper(SkyhookDamper(1500))
```

## API

::: simweave.continuous
    options:
      show_root_heading: false
      show_source: true
