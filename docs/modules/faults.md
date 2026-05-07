# Fault Injection

`simweave.faults` injects physically-meaningful parameter faults into
continuous dynamic systems to produce labelled time-series data for
**predictive-maintenance (PdM)** model training.

A faulted simulation produces a :class:`~simweave.faults.FaultDataset` whose
columns include:

| Column | Description |
|---|---|
| **features** | State / sensor observations (optionally noise-corrupted) |
| **health_index** | Float in [0, 1] — 1 = healthy, 0 = fully failed |
| **rul** | Remaining useful life in simulation time units |
| **is_failed** | Boolean, True from failure time onward |
| **failure_mode** | String label identifying which failure mode is active |

These labels support three common PdM tasks simultaneously:

| Task | Target column |
|---|---|
| Binary fault detection | `is_failed` |
| RUL regression (LSTM, TCN) | `rul` |
| Multi-class mode identification | `failure_mode` |

---

## Concepts

### FaultProfile

A :class:`~simweave.faults.FaultProfile` defines *when* and *how fast* a fault
evolves.  It maps a simulation time `t` to a **health index** in [0, 1]:

```
health index
    1 ─────────────┐
                   │  degradation window
                   └──────────── 0
    onset_time      failure_time
```

Three built-in degradation shapes are available:

| Shape | Description | Typical use case |
|---|---|---|
| `"linear"` | Constant degradation rate | Uniform surface wear, corrosion |
| `"exponential"` | Slow start, rapid end (convex) | Fatigue, crack propagation |
| `"abrupt"` | Healthy then instant failure | Electrical short, fracture |
| callable | Custom curve from *progress* ∈ [0,1] | Any empirically-fitted profile |

All profiles are **monotonically decreasing** by design.  Stochastic variation
is modelled at the *sensor* level (see `noise_std` in
:meth:`~simweave.faults.FaultDataset.from_result`) rather than at the profile
level, keeping the degradation trajectory deterministic and reproducible across
seeds.

### ParameterFault

A :class:`~simweave.faults.ParameterFault` maps a `FaultProfile` to a named
attribute on a :class:`~simweave.continuous.solver.DynamicSystem`:

```python
ParameterFault(
    param="R_th",        # attribute on the DynamicSystem
    profile=profile,
    max_delta=3.0,       # 3× increase at full failure (relative=True)
    relative=True,
)
```

At each ODE step the injector computes:

```
fault_fraction = 1 − health_index(t)        # 0 = healthy, 1 = failed
perturbed = nominal × (1 + max_delta × fault_fraction)   # relative mode
```

Common parameter–system pairings:

| System | Parameter | Physical meaning |
|---|---|---|
| `ThermalRC` | `R_th` | Insulation loss / thermal resistance increase |
| `ThermalRC` | `C_th` | Phase-change material degradation |
| `MassSpringDamper` | `stiffness` | Spring fatigue / crack |
| `MassSpringDamper` | `damping` | Damper seal wear |
| `SeriesRLC` | `R` | Conductor corrosion |
| `SeriesRLC` | `C` | Capacitor electrolyte degradation |
| `QuarterCarModel` | `k_s` | Suspension spring rate change |

### FaultInjector

:class:`~simweave.faults.FaultInjector` wraps any
:class:`~simweave.continuous.solver.DynamicSystem` and satisfies the
:class:`~simweave.continuous.solver.SupportsDynamics` protocol, so it plugs
directly into :func:`~simweave.continuous.solver.simulate` or a
:class:`~simweave.continuous.solver.ContinuousProcess` without any changes to
the solver.

**Multiple simultaneous faults** are fully supported — attach several
`ParameterFault` objects to one injector.  The system-level health index is the
*minimum* across all faults, and the active failure mode is labelled from the
most-degraded fault.  This enables multi-class classification datasets covering
several independent failure modes.

### FaultDataset

:class:`~simweave.faults.FaultDataset` assembles all labels into a numpy-backed
dataclass.  Labels are computed **analytically** from the fault profiles at each
time step, so no separate recorder is needed for standalone
:func:`~simweave.continuous.solver.simulate` runs.

Key methods:

| Method | Description |
|---|---|
| `from_result(result, injector)` | Build from a `SimulationResult` (standalone path) |
| `from_recorder(recorder, result)` | Build from a `FaultRecorder` (hybrid-env path) |
| `concat([ds1, ds2, ...])` | Stack multiple runs into one training corpus |
| `train_test_split(test_frac)` | Sequential or shuffled split |
| `to_dataframe()` | Export to `pandas.DataFrame` (requires `pip install pandas`) |

---

## Quick start

### Standalone simulate path (most common)

```python
import numpy as np
import simweave as sw
from simweave.faults import FaultProfile, ParameterFault, FaultInjector, FaultDataset

# 1. Describe the degradation
profile = FaultProfile(
    onset_time=200,          # seconds — degradation begins
    failure_time=800,        # seconds — fully failed
    mode="insulation_loss",
    shape="exponential",     # slow start, rapid end
)

# 2. Map to a system parameter
fault = ParameterFault(
    param="R_th",            # thermal resistance on ThermalRC
    profile=profile,
    max_delta=3.0,           # rises to 4× nominal at full failure
    relative=True,
)

# 3. Wrap the system
system = sw.ThermalRC(thermal_resistance=1.0, thermal_capacitance=800.0)
injector = FaultInjector(system=system, faults=[fault])

# 4. Simulate (same API as always)
def heat_input(t):
    return 50.0   # constant 50 W

result = sw.simulate(injector, t_span=(0, 1000), dt=0.5, inputs=heat_input)

# 5. Build labelled dataset with sensor noise
rng = np.random.default_rng(42)
ds = FaultDataset.from_result(result, injector, noise_std=0.5, rng=rng)

print(ds)
# FaultDataset(n=2001, features=['temperature'], modes=['healthy', 'insulation_loss'], failed=401)

# 6. Export to pandas
df = ds.to_dataframe()
print(df.head())

# 7. Split for training
train, test = ds.train_test_split(test_frac=0.2)
print(f"Train: {len(train)}  Test: {len(test)}")
```

### Multiple failure modes

```python
from simweave.faults import FaultProfile, ParameterFault, FaultInjector, FaultDataset
import simweave as sw

system = sw.ThermalRC(thermal_resistance=1.0, thermal_capacitance=800.0)

faults = [
    ParameterFault(
        param="R_th",
        profile=FaultProfile(onset_time=300, failure_time=700,
                             mode="insulation_loss", shape="linear"),
        max_delta=3.0,
    ),
    ParameterFault(
        param="C_th",
        profile=FaultProfile(onset_time=500, failure_time=900,
                             mode="capacitance_loss", shape="exponential"),
        max_delta=0.6,    # C_th drops to 40% of nominal
        relative=True,
    ),
]

injector = FaultInjector(system=system, faults=faults)
result = sw.simulate(injector, t_span=(0, 1000), dt=0.5)
ds = FaultDataset.from_result(result, injector, noise_std=0.3)

print(set(ds.failure_mode))
# {'healthy', 'insulation_loss', 'capacitance_loss'}
```

### Assembling a training corpus

Combine healthy and faulted runs from the same system into a single dataset:

```python
import numpy as np
from simweave.faults import FaultDataset, FaultInjector, FaultProfile, ParameterFault
from simweave.continuous.solver import simulate
from simweave.continuous.systems import ThermalRC

def run(onset, failure, shape, seed):
    sys = ThermalRC(thermal_resistance=1.0, thermal_capacitance=800.0)
    if onset is None:
        # Healthy run
        inj = FaultInjector(sys, faults=[])
    else:
        profile = FaultProfile(onset_time=onset, failure_time=failure,
                               mode="fault", shape=shape)
        fault = ParameterFault(param="R_th", profile=profile, max_delta=3.0)
        inj = FaultInjector(sys, faults=[fault])
    result = simulate(inj, t_span=(0, 1000), dt=1.0)
    return FaultDataset.from_result(result, inj, noise_std=0.5,
                                    rng=np.random.default_rng(seed))

datasets = [
    run(None, None, None, seed=0),        # healthy
    run(200, 700, "linear", seed=1),
    run(400, 900, "exponential", seed=2),
    run(300, 600, "linear", seed=3),
]

corpus = FaultDataset.concat(datasets)
train, test = corpus.train_test_split(test_frac=0.2)

# Feed directly to PyTorch / TensorFlow / scikit-learn
X_train = train.features          # (N_train, n_features)
y_rul   = train.rul               # RUL regression
y_mode  = train.failure_mode      # multi-class classification
```

### Hybrid environment path

Use :class:`~simweave.faults.FaultRecorder` when combining a faulted system
with discrete events in a :class:`~simweave.core.environment.SimEnvironment`:

```python
from simweave.continuous.solver import ContinuousProcess
from simweave.faults import FaultRecorder, FaultDataset
import simweave as sw

proc = ContinuousProcess(injector)
recorder = FaultRecorder(injector)

env = sw.SimEnvironment(dt=1.0, end=1000.0)
env.register(proc)
env.register(recorder)   # register after proc for post-tick alignment
env.run(until=1000.0)

result = proc.result()
ds = FaultDataset.from_recorder(recorder, result)
```

---

## Visualisation

Two plot helpers ship with `simweave.viz`:

```python
import simweave as sw

# Sensor signals with shaded fault window
fig = sw.plot_fault_signals(result, injector)
fig.show()

# Health index + RUL on a dual y-axis
fig = sw.plot_health_index(ds, show_rul=True)
fig.show()
```

---

## API reference

::: simweave.faults.FaultProfile

::: simweave.faults.ParameterFault

::: simweave.faults.FaultInjector

::: simweave.faults.FaultRecorder

::: simweave.faults.FaultDataset
