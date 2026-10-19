# Stats: copulas and dependence

`simweave.stats` lets a simulation draw **dependent** random quantities: an
inter-arrival gap together with a job's service time, or successive gaps that
cluster into bursts. A *copula* describes the dependence; *marginals* describe
each quantity's own distribution; you choose them separately.

It needs only numpy. Gaussian, Clayton, Gumbel and Frank copulas work on a base
install; the Student-t copula and `method="mle"` fitting need scipy
(`pip install simweave[optim]`) and say so if it is missing. The design is in
[Copula sampling](../design/copulas.md).

## Joint draws: gap and service time

```python
import numpy as np
import simweave as sw
from simweave.stats import Exponential, JointSampler

joint = JointSampler(
    sw.GumbelCopula.from_kendall_tau(0.5).flipped((0,)),   # negative dependence
    {"interarrival": Exponential(1.0), "service_time": Exponential(0.8)},
)

def factory(env, draw):                       # draw = {"interarrival": ..., "service_time": ...}
    job = sw.Entity()
    job.sim_properties = sw.EntityProperties.from_draw(draw)   # fixed service time + extras
    return job

gen = sw.ArrivalGenerator(None, factory, service, rng=rng, joint=joint, gap_alignment="after")
```

`ArrivalGenerator(joint=...)` is opt-in; without it nothing changes. The
generator draws from a child generator seeded from `rng`, so adding a copula
does not disturb your other random draws.

**Which entity gets the attribute?** `gap_alignment="before"` (default) attaches
the attributes to the entity that arrives after the drawn gap (an attribute vs
the time since the last arrival). `"after"` attaches them to the entity just
created, with the gap being the time to the *next* arrival. They change queue
behaviour differently, so pick the one that matches your story.

## Bursty arrivals with identical marginals

```python
from simweave.stats import MarkovCopulaProcess

def scenario(seed):
    gaps = MarkovCopulaProcess(sw.GumbelCopula.from_kendall_tau(0.6), Exponential(1.0))
    gen = sw.ArrivalGenerator(gaps, factory, service, rng=np.random.default_rng(seed))
    ...
```

Each gap keeps exactly the `Exponential(1.0)` marginal, but consecutive gaps
have the copula's dependence. The process is stateful: build it **inside** the
scenario so each Monte Carlo replicate gets its own. See demo 30.

## Copulas

| Copula | Tail dependence | Notes |
| --- | --- | --- |
| `GaussianCopula` | none | any dimension |
| `StudentTCopula` | symmetric | scipy |
| `ClaytonCopula` | lower | exchangeable, any dimension |
| `GumbelCopula` | upper | exchangeable, any dimension |
| `FrankCopula` | none | exchangeable, any dimension |

All have `from_kendall_tau(tau)`. Negative dependence and the opposite tails use
`.flipped(axes)` and `.survival()`:
`GumbelCopula.from_kendall_tau(0.5).flipped((1,))` is large on the first axis with
small on the second. `copula.tail_dependence()` returns the corner coefficients
(`ll`, `uu`, `lu`, `ul`).

## Marginals

`Exponential`, `Uniform`, `Normal`, `LogNormal` (`from_mean_cv`), `Weibull`,
`Deterministic`, `Empirical(data)`, and `ScipyMarginal` / `as_marginal` for any
frozen `scipy.stats` distribution. The existing `sw.exponential`, `sw.uniform`,
`sw.normal` and `sw.deterministic` helpers also carry `ppf`/`cdf` and can be
used directly as marginals.

## Fitting from data

```python
fit = sw.fit_copula(observations)            # (n, 2) array; ranks only
print(fit.family, fit.tau, fit.table)        # candidates ranked by AIC
joint = sw.JointSampler.from_data(observations, ["interarrival", "size"])
```

`fit_copula` inverts Kendall's tau (numpy only) and ranks the families, with
survival variants for upper-tail shapes and a flip for negative dependence.
`method="mle"` refines the parameter by maximum likelihood (scipy).

## API

::: simweave.stats
