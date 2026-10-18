# Flight

`simweave.flight` simulates multirotor flight dynamics: a quadrotor model, a
cascaded position/attitude controller, reference trajectories and missions
(take off, figure-8, land), wind and fault injection. It needs only numpy. See
`FLIGHT_DESIGN.md` in the repository for the full plan.

## Fly a mission

```python
from simweave.flight import Mission, Quadrotor, ConstantWind, fly

mission = Mission.figure8(height=3.0, a=4.0, b=2.0, period=20.0)
flight = fly(Quadrotor(wind=ConstantWind((4.0, 0.0, 0.0))), mission)

print(flight.metrics())       # rms_error, max_error, energy_wh, touchdown_speed, ...
print(flight.phase_rms())     # tracking error per phase
data = flight.to_dict()       # time, x, y, z, roll, pitch, yaw, ... (Z up, radians)
viewer = flight.to_viewer()   # same, mapped for a Y-up 3D viewer
```

`Mission.figure8` chains take off, hover, a quintic blend into the figure-8,
the figure-8 laps, a blend out, hover and a soft landing. Position, velocity
and acceleration are continuous through every join. Build your own from
`Hold`, `Quintic` (minimum-jerk move between two `RefPoint` states) and
`Figure8`, or pass any callable `ref(t) -> RefPoint`.

`fly` uses a `CascadedController`: position P, velocity PI, thrust-vector and
attitude (SO(3) error), body-rate PI, then the mixer. It models the vehicle
with `vehicle.params`, so changing `vehicle.mass` afterwards (a payload) or a
`motor_eff_*` value tests it against model mismatch; the integral terms absorb
it. Tune with `ControllerGains`. The motors switch off once the reference and
the vehicle are on the ground.

Run `demos/27_quadcopter_figure8.py` for a complete example.

## Conventions

- World frame is **Z up**; body frame is x forward, y left, z up.
- Euler angles are ZYX. Positive roll banks right, positive pitch is nose down,
  positive yaw is counter-clockwise seen from above.
- Rotors are numbered 1 front-right, 2 front-left, 3 rear-left, 4 rear-right.

## Quick example

```python
import numpy as np
from simweave.continuous import simulate
from simweave.flight import Quadrotor, Mixer

quad = Quadrotor(hover=True, position=(0, 0, 5))
mixer = Mixer(quad.params)


class Hover:
    def update(self, t, state, dt):
        return mixer.allocate(quad.params.weight, (0.0, 0.0, 0.0))


result = simulate(quad, (0, 5), dt=0.005, controller=Hover(), control_dt=0.01)
print(result.state[-1, :3])   # still at (0, 0, 5)
```

The quadrotor has 16 states (`x y z vx vy vz roll pitch yaw p q r f1 f2 f3 f4`,
where `f1`..`f4` are the actual rotor thrusts, which follow the commanded
thrusts with time constant `motor_tau`). Its input is the four commanded rotor
thrusts. `Mixer.allocate(collective, torques)` converts a collective thrust and
body torques into rotor thrusts within limits, giving attitude torques priority
over collective thrust when the request does not fit.

## Disturbances and faults

- `wind=` takes `ConstantWind`, `GustWind` (smooth "1 - cos" gust) or
  `TurbulentWind` (seeded Gauss-Markov turbulence). Drag acts on the velocity
  relative to the air.
- `motor_eff_1` .. `motor_eff_4` scale each rotor's thrust, so a
  `FaultInjector` with a `ParameterFault` can degrade a single rotor.
- `Quadrotor.power(thrusts)` gives rotor power by momentum theory, for energy
  estimates.

## Ground contact

The floor at `z = 0` stops downward motion. While the rotors cannot lift the
vehicle it stays upright and still, so a landed vehicle does not tumble.

## API

::: simweave.flight
    options:
      show_root_heading: false
      show_source: true
