# simeng — Public API Reference

> Drop-in reference for apps that consume simeng (e.g. EdgeWeave).
> Copy this file into your app's repo so future coding-assistant sessions
> (and humans) don't have to crawl simeng's source to learn the surface.

Version targeted: **0.1.x**

---

## Install

```bash
pip install simeng            # stable PyPI
pip install -i https://test.pypi.org/simple/ simeng==0.1.0rc1   # pre-release testing
```

For integration into EdgeWeave, pin with a compatible-release constraint:

```toml
# pyproject.toml / requirements.txt
simeng>=0.1,<0.2
```

Pre-1.0 the minor version is the API-break boundary (SemVer-ish, cautious).

---

## Top-level imports

Everything frequently needed is re-exported from `simeng` itself. Anything
niche lives in the submodules (`simeng.continuous`, `simeng.discrete`,
`simeng.agents`, `simeng.spatial`, `simeng.supplychain`, `simeng.mc`,
`simeng.units`).

```python
import simeng

simeng.__version__            # "0.1.0"
```

---

## Core runtime (`simeng.core`)

| Name              | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `SimEnvironment`  | Top-level clock + entity host. Runs the simulation loop.  |
| `Clock`           | Monotonic time source (start, t, tick).                   |
| `EventQueue`      | Min-heap of scheduled events.                             |
| `ScheduledEvent`  | Event dataclass (time, callback, payload).                |
| `Entity`          | Base class for anything that ticks. Override `tick`.      |
| `configure_logging` / `get_logger` | stdlib-logging wiring.                    |

Typical skeleton:

```python
from simeng import SimEnvironment, Entity

class Pinger(Entity):
    def tick(self, dt, env):
        super().tick(dt, env)
        print(f"tick at t={env.clock.t:.3f}")

env = SimEnvironment(dt=0.1)
env.register(Pinger(name="pinger"))
env.run(until=1.0)
```

---

## Continuous dynamics (`simeng.continuous`)

### The integrator

```python
from simeng.continuous import simulate, SimulationResult

result: SimulationResult = simulate(
    system,                    # anything satisfying SupportsDynamics
    t_span=(0.0, 10.0),
    dt=0.01,
    method="rk4",              # or "euler"
    inputs=lambda t: 0.0,      # optional; system.inputs used otherwise
)
result.time                    # ndarray shape (N,)
result.state                   # ndarray shape (N, n_states)
result.state_labels            # tuple of strings, from system
```

### Writing your own system

Subclass `DynamicSystem` or duck-type `SupportsDynamics`:

```python
from simeng.continuous import DynamicSystem
import numpy as np

class Exponential(DynamicSystem):
    def __init__(self, k, x0=1.0):
        self.k = k
        self._x0 = np.array([x0])

    def initial_state(self):
        return self._x0.copy()

    def state_labels(self):
        return ("x",)

    def derivatives(self, t, state, inputs=None):
        return -self.k * state
```

### Built-in systems

All imported from `simeng` directly or `simeng.continuous`.

| Class                 | State                         | Inputs      |
|-----------------------|-------------------------------|-------------|
| `MassSpringDamper`    | `[x, x_dot]`                  | force F(t)  |
| `SimplePendulum`      | `[theta, theta_dot]`          | torque      |
| `QuarterCarModel`     | `[z_s, z_s_dot, z_u, z_u_dot]`| road z(t)   |
| `SeriesRLC`           | `[q, i]`                      | voltage V(t)|
| `ThermalRC`           | `[T]`                         | heat Q(t)   |
| `TwoMassThermal`      | `[T_core, T_sink]`            | heat Q(t) on core |

Construct with keyword arguments that map to the engineering quantities — see
each class's docstring and the `demos/` folder for worked examples.

### Hybrid: continuous inside a discrete environment

```python
from simeng import SimEnvironment
from simeng.continuous import ContinuousProcess, MassSpringDamper

env = SimEnvironment(dt=0.01)
proc = ContinuousProcess(MassSpringDamper(mass=1, damping=0.2, stiffness=4.0))
env.register(proc)
env.run(until=10.0)

res = proc.result()            # SimulationResult as if run by simulate(...)
```

---

## Discrete-event (`simeng.discrete`)

Primitives for queueing-theory-style simulations.

| Name                | Summary                                                        |
|---------------------|----------------------------------------------------------------|
| `EntityProperties`  | Named dict of sampled properties attached to an Entity.        |
| `exponential`       | Rate-parameterised IID sampler. Accepts `rate` (λ) or `mean`.  |
| `uniform`           | Uniform on `[a, b]`.                                           |
| `normal`            | Gaussian; clipped to >= 0 where required.                      |
| `deterministic`     | Always returns the same value (for testing).                   |
| `set_default_seed`  | Seed the library-wide default RNG.                             |
| `Queue`             | FIFO buffer (bounded or unbounded).                            |
| `PriorityQueue`     | Heap keyed by any comparable priority.                         |
| `Resource`          | Single-capacity lockable resource.                             |
| `ResourcePool`      | N-capacity pool with wait-queue.                               |
| `Service`           | Consumes from a queue, holds a resource, emits after T.        |
| `ArrivalGenerator`  | Feeds newly created entities into the system via a sampler.    |

Minimal pattern:

```python
from simeng import (
    SimEnvironment, ArrivalGenerator, Queue, Service, Resource, exponential,
)

env    = SimEnvironment(dt=1.0)
q      = Queue(name="wait_queue")
server = Resource(name="server")
svc    = Service(queue=q, resource=server, service_time=exponential(rate=0.4))
gen    = ArrivalGenerator(interarrival=exponential(rate=0.3), sink=q)

env.register(gen, q, server, svc)
env.run(until=500.0)
```

---

## Agents & pathfinding (`simeng.agents`, `simeng.spatial`)

```python
from simeng import Graph, grid_graph, Agent, Compass
from simeng import a_star, dijkstra, manhattan, euclidean, chebyshev

g = grid_graph(width=10, height=10)
path = a_star(g, start=(0, 0), goal=(9, 9), heuristic=manhattan)
```

`Agent` and `Compass` provide orientation-aware movement along a path; they
slot into `SimEnvironment` via `register(...)` like any entity.

---

## Monte Carlo (`simeng.mc`)

```python
from simeng.mc import run_monte_carlo, MCResult

def trial(rng, config):
    # build env, run, return scalar or dict of scalars
    ...

result: MCResult = run_monte_carlo(trial, n_trials=1_000, seed=42)
result.mean, result.std, result.samples
```

Also available: `run_batched_mc` for chunked parallel execution.

---

## Supply chain (`simeng.supplychain`)

```python
from simeng import InventoryItems, Warehouse
```

Registered as `Entity` subclasses — drop into any `SimEnvironment`.

---

## Units (`simeng.units`)

SI-exponent-tracked dimensional quantities. Enforces addition/subtraction
compatibility and auto-types multiplication/division results.

```python
from simeng import Distance, TimeUnit, Velocity

d = Distance(100)          # [m]
t = TimeUnit(10, "s")
v = d / t                  # -> Velocity, unit "m/s"

Distance(1) + Distance(2)  # OK
Distance(1) + Velocity(2)  # TypeError: different dimensions
```

Available concrete classes: `Distance`, `Velocity`, `Acceleration`, `Mass`,
`Force`, `Area`, `Volume`, `TimeUnit`. Use `SIUnit` directly for anything
else by supplying the 7-tuple of exponents
`(m, kg, A, K, mol, cd, s)`.

---

## Currency (`simeng.currency`)

Decimal-backed, currency-tagged monetary amounts with strict same-currency
arithmetic, explicit FX conversion, and ASCII-default formatting. Designed
so financial quantities inside a simulation behave like units — you cannot
silently add pounds to dollars, you cannot float-drift your way into
phantom pennies, and cross-currency work always goes through a user-owned
FX converter.

```python
from simeng import Money, StaticFXConverter

rent  = Money(1500, "GBP")
bonus = Money("123.45", "USD")

fx = StaticFXConverter({("GBP", "USD"): "1.27"})

total_gbp = (rent + bonus.to("GBP", fx)).round_to_currency()
print(total_gbp)                     # "GBP 1,597.21"
```

### `Money`

Frozen, hashable dataclass. Amount is stored as `decimal.Decimal`; int /
float / str inputs are coerced via `Decimal(str(value))` to avoid
binary-float drift. Currency code is an ISO 4217 three-letter identifier
(case-insensitive; normalised to uppercase).

| Operation                     | Result                                           |
|-------------------------------|--------------------------------------------------|
| `Money + Money` (same ccy)    | `Money` — sums amounts                           |
| `Money + Money` (diff ccy)    | `CurrencyMismatchError` (subclass of `TypeError`)|
| `Money * scalar`              | `Money`                                          |
| `Money * Money`               | `TypeError` (dimensionally ill-defined)          |
| `Money / scalar`              | `Money`                                          |
| `Money / Money` (same ccy)    | `float` (ratio)                                  |
| `Money // Money` (same ccy)   | `int` (integer ratio)                            |
| `Money == Money`              | `True` iff both currency and amount match        |
| `<`, `<=`, `>`, `>=`          | Same-currency only; cross-ccy raises             |
| `sum([...])` without start    | `TypeError` — use `sum([...], start=Money(0, cc))` |
| `m.round_to_currency()`       | `Money` quantised to the currency's minor units, banker's rounding |
| `m.to(target, fx)`            | `Money` in `target` currency via `fx.rate(...)`  |
| `m.decimals`                  | `int` — minor-unit places for that currency      |
| `m.is_negative()`, `is_zero()`| `bool`                                           |

Negatives are legitimate (debts, refunds, signed cashflows). Rounding is
only applied when you ask for it — `round_to_currency` or format time.

### FX conversion

simeng deliberately ships **no live rates**. You supply a converter:

```python
from simeng import StaticFXConverter, CallableFXConverter

# In-memory table. Inverse lookups happen automatically:
fx = StaticFXConverter({("GBP", "USD"): "1.27"})
fx.rate("GBP", "USD")  # Decimal('1.27')
fx.rate("USD", "GBP")  # Decimal('1') / Decimal('1.27')

# Wrap an arbitrary callable (e.g. your existing rate-lookup function):
my_fx = CallableFXConverter(lambda src, tgt, at=None: my_lookup(src, tgt, at))
```

Both satisfy the `FXConverter` runtime-checkable `Protocol`, so you can
type-annotate against the protocol and swap implementations.

### Formatting

```python
from simeng import Money, format_money

m = Money("1234.567", "GBP")

str(m)                      # "GBP 1,234.57"    (default, banker's-rounded)
f"{m}"                      # "GBP 1,234.57"
f"{m:r}"                    # "GBP 1,234.57"    (explicit round)
f"{m:raw}"                  # "1234.567"        (full-precision, no tag)
format_money(m, "en_GB")    # "£1,234.57"       (needs simeng[intl])
```

The locale-aware path requires `babel`. Install with:

```bash
pip install "simeng[intl]"
```

### Non-ISO currencies (crypto, fixtures)

```python
from simeng import register_custom, unregister_custom, Money

register_custom("BTC", decimals=8)
Money("0.00000001", "BTC")

unregister_custom("BTC")
```

`register_custom` refuses to shadow a real ISO 4217 code. Custom
registrations are **process-local** — they do not persist across
interpreter invocations.

---

## Code-generation contract for EdgeWeave

When the app generates Python for a user, prefer the **top-level import** so
generated code stays version-tolerant:

```python
# GOOD — generated code
import simeng
from simeng import SimEnvironment, MassSpringDamper, simulate

# ACCEPTABLE — when a submodule is clearly the right level
from simeng.continuous import ContinuousProcess
```

Avoid importing from private submodules (anything with a leading underscore
or anything not listed in `simeng.__all__`). These are not covered by the
SemVer pin and may move between 0.x releases.

---

## Minimal end-to-end template for EdgeWeave-generated scripts

```python
"""Auto-generated by EdgeWeave."""
from simeng import SimEnvironment, MassSpringDamper
from simeng.continuous import ContinuousProcess

def build_env():
    env = SimEnvironment(dt=0.01)
    sys = MassSpringDamper(mass=1.0, damping=0.5, stiffness=4.0, x0=(1.0, 0.0))
    env.register(ContinuousProcess(sys, method="rk4", name="plant"))
    return env

def main():
    env = build_env()
    env.run(until=10.0)
    plant = env.get("plant")
    print(plant.result().state[-1])

if __name__ == "__main__":
    main()
```

---

## Changelog pointer

`CHANGELOG.md` at the simeng repo root tracks breaking changes with explicit
migration notes per minor version. When EdgeWeave bumps the simeng pin, the
first check should be that file.
