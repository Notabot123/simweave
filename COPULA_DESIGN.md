# Copula-driven sampling for discrete-event simulation

**Status:** Phases 0 to 3 implemented (released in 0.9.0); Phases 4 and 5 are future extensions. It ships in 0.9.0 together with the flight work.

## Decisions so far (user)

- Package name: `simweave.stats`.
- scipy stays **optional** (existing `[optim]` extra); the base install stays numpy-only and lightweight. Student-t copula and MLE fitting need scipy and raise a friendly `ImportError`; docs and demos that use them say so.
- `gap_alignment` default: `"before"` (conventional: the attribute belongs to the entity that arrived after the gap).
- First release: Phases 0 to 3. Phases 4 and 5 are recorded as future extensions.
- Demo numbers: implemented as demos 30 (serial dependence) and 31 (copula families); the reliability demo (32 in the plan below) remains future work.


## Motivation

Every random quantity in `simweave.discrete` is drawn on its own: an
`ArrivalGenerator` draws an inter-arrival gap, the `factory` then builds an
entity, and a `Service` later draws a service time from
`EntityProperties.service_time`. Nothing lets those draws be *dependent*.
Real systems are full of dependence: big jobs arrive in clusters, long repairs
follow hard use, demand spikes coincide with supplier delays, one storm takes
out several vehicles at once.

A **copula** separates the *dependence structure* from the *marginals*: a
copula `C` on `[0,1]^d` has uniform margins, and
`(F1^-1(U1), ..., Fd^-1(Ud))` with `U ~ C` has the marginals `F1..Fd` and the
dependence of `C`. So users keep the marginals they already know (exponential
gaps, lognormal sizes) and choose the dependence separately: its strength
(Kendall's tau) *and* its shape (tail dependence).

## Findings from the current code

| Area | Today | Consequence |
| --- | --- | --- |
| `discrete/properties.py` | `exponential/uniform/normal/deterministic` return closures `Callable[[Generator], float]`. No CDF, no inverse CDF. `exponential(mean)` takes a **mean** (the docstring of `VehicleArrivalProcess` says `rate=`; unrelated doc bug, see Open questions). | Copulas need `ppf(u)`. A new marginal protocol is required; the closures must keep working. |
| `discrete/services.py` `ArrivalGenerator` | `__init__` draws the first `_countdown` from `interarrival(rng)`; `tick` loops `while remaining >= _countdown`, calls `factory(env)`, enqueues, redraws. | The gap and the entity are drawn separately; a joint draw needs a hook, and the pairing (which entity gets which gap) must be defined. |
| `Service._WorkChannel._try_start` | Pulls `item.sim_properties.draw_service_time(rng)` when the item has `sim_properties`. | An attribute can reach Service with **no Service change** by giving the entity a per-entity `EntityProperties` whose `service_time` is `deterministic(value)`. |
| `supplychain` | `pareto_sweep` -> `poisson_reorder_points` / `cost_optimise_stock` are **analytic**: `lam = demand_rate * logistics_delay`, then `scipy.stats.poisson.ppf`. `Warehouse` lead times are deterministic arrays (`newbuy_leadtimes`). | `pareto_sweep` has no random draws to correlate. Showcase 3 needs a simulation-based path (see ranking). |
| `reliability.ReliableEntity._check_failure` | Each subsystem fails by an independent `rng.random() < 1-exp(-lambda*dt)` per tick. | Common-cause failures need a shared hazard modifier, not just a copula; see the dedicated section. |
| `mc.run_monte_carlo` | `f(seed) -> Any`; users build `np.random.default_rng(seed)` inside. `set_default_seed` only seeds the module-level generator used by `EntityProperties.draw_service_time(rng=None)`. | All copula objects must be stateless w.r.t. RNG (take `rng` per call) and picklable. Stateful processes (serial dependence) must be built per replicate. |
| Dependencies | Base install is numpy only; `scipy` is the `[optim]` extra; `tests/test_base_install_imports.py` enforces clean imports on a base install. | Everything in the new package must import without scipy; scipy-needing features raise the usual friendly `ImportError`. |

## Goals

1. Joint sampling of correlated quantities in DES (inter-arrival time with an
   entity attribute such as service time or size; successive inter-arrival
   gaps; demand with lead time; component lifetimes in a fleet).
2. **Base install works** for the common cases (Gaussian, Clayton, Gumbel,
   Frank copulas with exponential/uniform/lognormal/Weibull marginals).
3. **Backwards compatible**: no change to the behaviour or signatures of
   existing public calls; new behaviour is opt-in via new keyword arguments.
4. Reproducible: all randomness flows through an explicit
   `numpy.random.Generator`; works under `run_monte_carlo` and with
   `set_default_seed`.
5. Fit from data (pseudo-observations, Kendall's tau) so users can calibrate
   from logs.

## Non-goals (first release)

- Vine copulas / arbitrary-dimension non-exchangeable Archimedean copulas
  (document as a follow-up; pair-copula Markov chain for serial dependence is
  in scope, general D-/C-vines are not).
- Time-varying copula parameters, regime switching.
- Vectorised "all replicates at once" copula batches beyond what `sample_u(n)`
  gives for free.
- Copula-based optimisation inside `cost_optimise_stock` (analytic path stays
  as is).

## Package layout

New subpackage `simweave.stats` (pure numpy at import; scipy lazily). Name
chosen over `simweave.copulas` because marginals, rank statistics and special
functions belong beside it and may later host other distribution tooling.

```
src/simweave/stats/
    __init__.py          # public re-exports
    marginals.py         # Marginal protocol, Exponential/Uniform/LogNormal/
                         #   Weibull/Normal/Deterministic classes, Empirical,
                         #   ScipyMarginal, as_marginal()
    special.py           # ndtr/ndtri (scipy if present, else numpy fallback),
                         #   debye_1 for Frank, t cdf/ppf (scipy only)
    copulas.py           # Copula ABC; GaussianCopula, StudentTCopula,
                         #   ClaytonCopula, GumbelCopula, FrankCopula;
                         #   Flipped (axis flips / survival)
    joint.py             # JointSampler(copula, marginals, names)
    serial.py            # MarkovCopulaProcess (lag-1 copula chain)
    fit.py               # pseudo_obs, kendall_tau, spearman_rho,
                         #   fit_copula (tau-inversion, optional MLE), AIC rank
    measures.py          # tail_dependence_empirical, kendall_tau_from_params
tests/test_stats_*.py
```

`simweave/__init__.py` re-exports the headline names (`JointSampler`,
`GaussianCopula`, `ClaytonCopula`, `GumbelCopula`, `FrankCopula`,
`StudentTCopula`, `fit_copula`, `MarkovCopulaProcess`, `Empirical`); the rest
via `simweave.stats`.

## Phase 0: marginal protocol (`stats/marginals.py`)

```python
class Marginal(Protocol):
    def ppf(self, u: np.ndarray) -> np.ndarray: ...   # required
    def cdf(self, x: np.ndarray) -> np.ndarray: ...   # needed for fitting/density
    def rvs(self, rng: Generator, size=None): ...     # independent draw
    mean: float                                       # optional, for diagnostics
```

Concrete classes (all closed form, numpy only except `LogNormal`'s normal
quantile): `Exponential(mean)` (same `mean` convention as
`sw.exponential`), `Uniform(low, high)`, `LogNormal(mu, sigma)` (plus
`LogNormal.from_mean_cv(mean, cv)`), `Weibull(shape, scale)`,
`Normal(mean, std, clip_nonnegative=True)` (clipping applied after `ppf`,
mirroring `sw.normal`), `Deterministic(value)`. Plus:

- `Empirical(data)`: `ppf` is a linear-interpolated quantile function,
  `cdf` the ECDF; the route for calibrating from logs.
- `ScipyMarginal(frozen)`: wraps any `scipy.stats` frozen distribution;
  `as_marginal()` also auto-wraps anything with `.ppf` (duck typing), so a
  raw `scipy.stats.gamma(2.0)` can be passed directly.

**Compatibility with existing closures.** `sw.exponential(2.0)` etc. keep
returning closures, but each factory also sets function attributes
(`draw.ppf`, `draw.cdf`) so the existing helpers are accepted as marginals
directly (`as_marginal` checks for `ppf`). `uniform`, `exponential` get both;
`normal` gets them via `ndtri`; `deterministic` gets a constant `ppf`. No
signature or return-type change, which keeps `EntityProperties.service_time`
and every demo untouched. The closures built by `normal(...,
clip_nonnegative=False)` and with clipping both carry the matching `ppf`.

**scipy-optional decision.** `special.ndtr` / `ndtri` use `scipy.special`
when importable and otherwise a numpy implementation: Wichura AS241 (PPND16,
~1e-16 relative error) for `ndtri`, and `math.erfc` through a cached
`np.frompyfunc` (or a Cody rational approximation, vectorised) for `ndtr`.
That is enough for Gaussian copulas and lognormal/normal marginals on a base
install. Students-t needs the t CDF/quantile (regularised incomplete beta):
`StudentTCopula` and tail-coefficient helpers for it **require scipy** and
raise `ImportError("... pip install simweave[optim]")`. A test pins the
numpy fallback against scipy to ~1e-9 across `(1e-12, 1-1e-12)` in the dev
environment. Rationale: a small, auditable fallback keeps "numpy only" true
for the 80% case without shipping a hand-rolled incomplete beta.

## Phase 1: copulas (`stats/copulas.py`)

```python
class Copula(ABC):
    dim: int
    def sample_u(self, n: int, rng: Generator) -> np.ndarray: ...   # (n, dim)
    def log_pdf(self, u) -> np.ndarray: ...        # for MLE / AIC (Phase 3)
    def kendall_tau(self) -> float | np.ndarray: ...    # bivariate / matrix
    def tail_dependence(self) -> tuple[float, float]: ...   # (lower, upper), bivariate
    @classmethod
    def from_kendall_tau(cls, tau, **kw) -> "Copula": ...
    def flipped(self, axes=(1,)) -> "Copula": ...   # u_k -> 1-u_k for listed axes
    def survival(self) -> "Copula": ...             # flip all axes
```

| Copula | Sampler (numpy-only) | tau <-> param | Tail dependence (lower, upper) |
| --- | --- | --- | --- |
| `GaussianCopula(corr)` | Cholesky, `ndtr(z @ L.T)` | `rho = sin(pi tau / 2)` | (0, 0) |
| `StudentTCopula(corr, df)` | normal / `sqrt(chi2/df)`, then t CDF (scipy) | `rho = sin(pi tau / 2)` | symmetric, `2 t_{df+1}(-sqrt((df+1)(1-rho)/(1+rho)))` |
| `ClaytonCopula(theta, dim)` | Marshall-Olkin: `V ~ Gamma(1/theta)`, `U = (1+E/V)^(-1/theta)` | `theta = 2 tau / (1 - tau)` | (`2^(-1/theta)`, 0) |
| `GumbelCopula(theta, dim)` | Marshall-Olkin with positive-stable `V` (Chambers-Mallows-Stuck, in numpy), `U = exp(-(E/V)^(1/theta))` | `theta = 1 / (1 - tau)` | (0, `2 - 2^(1/theta)`) |
| `FrankCopula(theta, dim)` | theta>0: Marshall-Olkin with `rng.logseries`; theta<0 bivariate: conditional inversion | tau via Debye `D1` (numpy quadrature, bracketed root) | (0, 0) |

Notes:

- Archimedean copulas are **exchangeable** and (for the Marshall-Olkin
  samplers) positive-dependence only. Negative dependence is obtained by
  `.flipped((1,))`, which also puts the tail dependence where the user wants it
  (e.g. "large service time with *short* following gap" = upper tail of S
  against lower tail of A = Gumbel with the gap axis flipped). The `Flipped`
  wrapper delegates `sample_u`, `log_pdf` (pdf is invariant up to the
  reflection) and re-maps `tail_dependence`.
- Gaussian/t take a correlation matrix, so `dim >= 2` works; the Archimedean
  ones take `dim`.
- A prototype of Gaussian/Clayton/Gumbel samplers with Marshall-Olkin was run
  against scipy during planning; they sample correctly in pure numpy.
- All `sample_u` clamp to `[eps, 1-eps]` (`eps = 1e-12`) so that `ppf`
  never returns `inf`.
- Validation: dimension, positive-definite correlation, parameter ranges, and
  the tau ranges (Clayton `tau in (0,1)`, etc.) raise `ValueError` with a
  message naming the allowed range and the `.flipped()` escape hatch.

## Phase 2: joint sampling and DES integration

### `JointSampler` (`stats/joint.py`)

```python
js = sw.JointSampler(
    copula=sw.GumbelCopula.from_kendall_tau(0.5).flipped((1,)),
    marginals={"interarrival": sw.Exponential(mean=1.0),
               "service_time": sw.Exponential(mean=0.8)},
)
draw = js.draw(rng)        # -> dict {"interarrival": 0.83, "service_time": 1.9}
block = js.sample(rng, n)  # -> dict of arrays, for analysis and tests
```

`draw` serves from a small internal block buffer (default 256) filled with one
vectorised `sample_u` call, because per-event scalar copula sampling is the
dominant cost otherwise. The buffer lives on the `JointSampler`
instance, so reproducibility is per `(seed, call order)`; the cost is that a
`rng` shared with other consumers sees its stream advance in blocks. Document
this, and give `ArrivalGenerator` a *dedicated child generator* by default
(`rng.spawn(1)[0]`) so adding a copula never perturbs the other draws.

### `ArrivalGenerator` hook (backwards compatible)

Add two keyword-only parameters; `interarrival` becomes `Optional`.

```python
ArrivalGenerator(
    interarrival=None,           # required unless joint= is given
    factory=...,                 # factory(env) -> Entity   (joint is None)
                                 # factory(env, draw) -> Entity   (joint given)
    target=queue, rng=rng,
    joint=js,                    # JointSampler or None
    interarrival_key="interarrival",
    gap_alignment="before",      # "before" | "after"
)
```

Behaviour rules:

- `joint is None`: byte-for-byte the current code path. Existing tests/demos
  are the regression suite.
- `joint` given: the gap comes from `draw[interarrival_key]`, the *whole draw
  dict* is passed to `factory(env, draw)`. Passing the arity explicitly (rather
  than inspecting the factory's signature) keeps the contract obvious.
- **Pairing semantics matter and must be a documented, tested choice.**
  `gap_alignment="before"` (default): the pair `(gap, attrs)` is drawn when the
  *next* arrival is scheduled and `attrs` go to the entity that arrives after that
  gap, i.e. the attribute correlates with "time since the previous arrival".
  `"after"`: a pair is drawn when an entity is created; its attrs go to that
  entity and its gap schedules the *following* arrival, i.e. the attribute
  correlates with "time until the next arrival". The first countdown is then
  drawn from the gap marginal alone. Prototype: both alignments change waits
  materially, "after" acts through the Lindley term `S_n - A_{n+1}` directly,
  "before" only indirectly (through the previous wait); users picking a
  physical story must pick the alignment that matches it.
- Draw order inside the existing `while remaining >= self._countdown` loop is
  preserved so multi-arrival-per-tick handling stays correct; the pending
  `attrs` are stored next to `_countdown`.
- `dt` quantisation caveat for docs: arrival times are rounded up to ticks, so
  very short correlated gaps are truncated at the tick (already true today;
  a tick `dt << min gap` keeps the copula's tail visible).

Same optional pair of keywords for `roads.VehicleArrivalProcess`
(Phase 5, to enable the traffic showcase), left out of the first merge to keep
the PR small.

### Getting an attribute to `Service` / `EntityProperties`

Two supported routes, no change to `Service`:

1. **Helper** (recommended):
   ```python
   props = sw.EntityProperties.from_draw(draw, service_key="service_time",
                                         entity_type="job")
   ```
   Builds a per-entity `EntityProperties` with
   `service_time=deterministic(draw[service_key])` and `extras = remaining
   draw items` (e.g. `size`, `payload`). Per-entity allocation is cheap (a
   small dataclass); `balk_on_length`/`renege_after` pass through as kwargs.
   The service time is therefore fixed at arrival, same distribution as drawing at
   start-of-service, now carrying its dependence on the gap. Negative or NaN
   draws fail loudly via `deterministic`'s existing check.
2. **Manual**: the factory writes whatever it wants onto
   `entity.extras`/attributes; `Service` already honours any
   `sim_properties` exposing `draw_service_time(rng)`.

Optional follow-up: `EntityProperties.service_time` could accept a
`Marginal` (it only needs `__call__(rng)`); not required for this work.

### Serial dependence (`stats/serial.py`)

```python
gaps = sw.MarkovCopulaProcess(
    copula=sw.GumbelCopula.from_kendall_tau(0.6),   # bivariate (U_{t-1}, U_t)
    marginal=sw.Exponential(mean=1.0),
)
ArrivalGenerator(interarrival=gaps, ...)    # callable (rng) -> float: no hook needed
```

A lag-1 copula Markov chain: `U_t = h^{-1}(w | U_{t-1})` with `w ~ U(0,1)`, where `h`
is the conditional CDF of the copula (closed form for Gaussian, t, Clayton,
Frank; Gumbel via a vectorised bisection on `h`, numpy only). Marginal is
preserved exactly (stationary uniform margin), dependence is exactly the
copula. It is a *stateful callable*, so:

- it plugs into the unchanged `interarrival` slot;
- it exposes `reset(rng)` and must be constructed **inside** the scenario
  builder per replicate (state is not shared across `run_monte_carlo`
  workers); the docs and a test (`pickle` round trip, two replicates
  independent) cover this;
- the first state is drawn from `U(0,1)` (stationary start), so no warm-up bias.

## Phase 3: fitting (`stats/fit.py`)

```python
u = sw.stats.pseudo_obs(data)                  # rank/(n+1), shape (n, d)
tau = sw.stats.kendall_tau(x, y)               # O(n log n) via scipy if present,
                                               #  else O(n^2) chunked, n capped w/ warning
fit = sw.fit_copula(data, family="auto")       # -> FitResult
fit.copula; fit.family; fit.params; fit.tau; fit.aic; fit.table   # ranked families
```

- Method of moments by Kendall-tau inversion (Gaussian, t at fixed `df`,
  Clayton, Gumbel, Frank): closed form, numpy only, robust, and the default.
- Optional canonical MLE on pseudo-observations (`method="mle"`, needs scipy
  `minimize_scalar`) using `log_pdf`; `family="auto"` ranks families by AIC
  (log-likelihood from `log_pdf`), with the flip variants considered for
  negative tau.
- `JointSampler.from_data(df_or_array, names, marginal="empirical"|<family>)`
  convenience: empirical or parametric marginals plus the fitted copula.
- Pseudo-observation scaling `rank/(n+1)` keeps values in the open interval.

## Phase 4: reliability and supply chain hooks

- **Reliability (common cause).** A copula over per-tick failure uniforms is
  the wrong abstraction: failures there are Bernoulli per tick, so tiny
  per-tick probabilities make the copula nearly invisible. Instead model
  *lifetimes*: `Fleet(..., common_cause=CommonCause(copula, ...))` draws a joint
  vector of time-to-failure uniforms per subsystem class across entities at
  commissioning/repair, converted through the Weibull/exponential marginal,
  and `ReliableEntity` uses a precomputed `next_failure_time` instead of the
  per-tick coin flip **only when** a `lifetime_source` is supplied. Default
  behaviour untouched. This is the largest integration and therefore a
  later phase.
- **Supply chain.** Add a simulation path where `Warehouse` accepts an optional
  `lead_time_sampler` (callable) replacing the deterministic
  `newbuy_leadtimes[need_reorder]` when a reorder is placed, and a demand
  sampler hook, so demand and lead time can come from one `JointSampler`.
  `pareto_sweep` itself stays analytic; the showcase compares the analytic
  Poisson answer (independence) with simulated service levels under dependence
  and with a **copula-aware safety-stock** helper
  `reorder_point_from_joint(js, availability)` (Monte Carlo quantile of
  demand-during-lead-time `D * L` per unit time) that can feed `pareto_sweep`'s
  reorder points. Needs a decision, see open questions.

## Showcases: evaluated and ranked

All numbers below are from throwaway Lindley-recursion prototypes
(exponential marginals, utilisation 0.8, Kendall tau magnitude 0.3 to 0.6,
Gaussian vs Clayton vs Gumbel, one axis flipped; 10 replicates x 100k
customers for the replicated rows). They are indicative, not results.

Finding that shapes the ranking: **dependence strength changes waits a lot**
(mean wait about 3.2 independent vs 4.3 to 5.3 under negative gap/service
dependence), but **the copula *family* at equal tau changes the 99th and
99.9th percentiles only slightly** (e.g. tau 0.6: P99.9 of 41.6 +/- 2.3
Gaussian, 39.3 +/- 1.1 Clayton, 44.4 +/- 3.3 Gumbel). The three families are
statistically hard to separate with exponential marginals at this load. The
headline of showcase 1 should therefore be "dependence matters", with the
tail-shape comparison reported honestly with confidence intervals and, if
needed, strengthened with heavier-tailed (lognormal, cv > 1) marginals and/or
higher utilisation or by plotting overflow probabilities of a finite buffer.

| Rank | Showcase | Why | Risks / needs |
| --- | --- | --- | --- |
| 1 | **(2) Serial dependence -> bursty arrivals, identical marginals** | Cleanest story: the marginal histogram is identical to the Poisson-like baseline, yet queue waits explode (autocorrelated arrivals are the classic hidden-risk case). Works through the unchanged `interarrival` slot, easy to explain, large effect. | Needs `MarkovCopulaProcess` (Gumbel `h^{-1}` by bisection). Show lag-1 scatter + ACF + wait CCDF. |
| 2 | **(1) Same marginals and tau, Gaussian vs Clayton vs Gumbel (+ independence)** | The most direct answer to "draw from copulas": one `JointSampler`, three copulas, MC with CIs. | Family differences are small at the tail (see above); frame as "dependence vs none" first, "tail shape" second; use lognormal service times and a finite buffer to bring out differences. |
| 3 | **(4) Correlated common-cause failures in fleets** | Visible, intuitive result (fleet-wide availability variance and probability of the worst-case simultaneous outage rise with tail dependence while mean availability stays equal). Real value for `simweave.reliability` users. | Largest integration (lifetime-based failure path). Phase 4. |
| 4 | **(3) Demand vs lead-time in supply chain** | Practical (safety stock moves). | `pareto_sweep` is analytic and cannot see it; needs the sim-path hook and a MC-quantile safety-stock helper, so it is more design than showcase. Phase 4. |
| 5 | **(5) Drone delivery capstone** | Ties to the flight thread; payload, flight time, interarrival. | Depends on the quadcopter work landing and a stable interface; keep it a later, separate effort. Only the *interface contract* (a `JointSampler` plus a factory that reads `draw`) needs agreeing now. |

Recommended demo set (numbers are placeholders; use the next free ones): `30` = showcase 2 (serial dependence), `31` = showcase 1
(copula families), `32` = common-cause failures; supply-chain and drone
capstone as later numbers. The existing demo style applies
(`_bootstrap.py` import, module docstring with scenario and `Run::` line,
parameters block, MC replicates with `run_monte_carlo`, plotly figure via
`simweave.viz`).

## Reproducibility and Monte Carlo

- No global state: copulas and marginals are immutable value objects taking
  `rng` per call; `JointSampler` owns only a draw buffer (documented), and
  `MarkovCopulaProcess` owns only the chain state. Both pickle cleanly.
- Recommended pattern, in docs:
  ```python
  def scenario(seed):
      rng = np.random.default_rng(seed)
      js = sw.JointSampler(...)                 # build inside, per replicate
      gen = sw.ArrivalGenerator(None, factory, queue, rng=rng, joint=js)
      ...
  res = sw.run_monte_carlo(scenario, n_runs=50)
  ```
- `set_default_seed` only affects calls that fall back to the module generator
  (`EntityProperties.draw_service_time(rng=None)`); `JointSampler.draw` requires
  an explicit `rng` (or the `ArrivalGenerator`'s), so copula draws are always
  governed by the scenario seed. A `default_rng()` fallback is exposed for
  notebooks only.
- Tests assert identical output for identical seeds, different output for
  different seeds, and equality across `executor="serial"` and `"processes"`.

## Test strategy

Unit (`tests/test_stats_*.py`, base-install-safe unless marked `scipy`):

1. **Rank correlation recovered.** For each family and tau in
   `{0.2, 0.5, 0.8}`, sample `n = 200_000`, assert empirical Kendall tau
   within 0.01 of target (`from_kendall_tau` round trip); for Gaussian/t also
   Spearman against `(6/pi) asin(rho/2)`. Flipped copulas give `-tau`.
2. **Marginals preserved.** After `ppf`, KS distance (own numpy KS stat; scipy
   `kstest` when available) < threshold against the closed-form CDF; moments
   (mean, variance) within MC error; `Empirical` reproduces data quantiles.
3. **Tail dependence coefficients.** Empirical
   `lambda_L(q) = P(U2<q | U1<q)` at `q = 1e-2, 5e-3` against the analytic
   value, with tolerances scaled to sampling error (Clayton lower, Gumbel upper,
   Gaussian -> 0 as `q` shrinks, t symmetric positive). Flipped variants map
   tails to the right corner.
4. **Special functions.** numpy `ndtri`/`ndtr` vs scipy to 1e-9 in
   `(1e-12, 1-1e-12)`; `ndtr(ndtri(p)) == p`; Frank `D1`/tau inversion round
   trip; Gumbel bisection `h(h^{-1}(w|u)|u) == w`.
5. **Joint sampler.** Block buffering gives the same sequence as unbuffered
   draws for a fixed seed (determinism contract); `from_data` recovers tau.
6. **Fitting.** Simulate from a known copula, `fit_copula(family="auto")`
   picks the right family at n=5k in at least ~90% of seeds and tau within
   0.03; pseudo-observations lie in the open unit interval.
7. **Serial process.** Marginal histogram unchanged; lag-1 Kendall tau of the
   output matches the copula; lag-2 dependence is the chain's implied value;
   `reset` and pickling behave; independence copula reproduces iid statistics.

Integration:

8. `ArrivalGenerator` default path unchanged (existing `test_discrete_*`,
   `test_integration_littles_law.py`); with `joint` and an independence
   copula the queue statistics match the independent run within CI;
   `gap_alignment` before/after hand-checked on a deterministic copula
   (comonotone, fixed marginals) where the pairing can be asserted exactly.
9. `EntityProperties.from_draw` -> `Service` -> measured service times equal
   the drawn `service_time`s; `extras` carries the other attributes.
10. Little's law still holds with correlated gaps/services (`L = lambda W`).
11. A `run_monte_carlo` smoke test with `executor="processes"` (picklability).
12. Base-install import test extended to `simweave.stats` (blocks scipy through
    the existing import hook pattern; Gaussian/Clayton/Gumbel/Frank work, t
    raises the friendly `ImportError`).

CI: the dev extra already has scipy; add one job leg that runs the stats tests
without scipy (like the existing `base-install` job).

## Documentation, blog, changelog

- `docs/modules/stats.md` (mkdocstrings, same style as the other module pages)
  and entries in `mkdocs.yml` nav; API pages are auto-generated.
- `docs/design/copulas.md` including this file with the same
  `include-markdown` pattern as `docs/design/sweeps.md`; add a row to
  `docs/design/index.md` ("Copula sampling | Proposal for v0.9") and a nav
  item under "Design notes".
- `docs/concepts.md`: short "Dependence" section; `docs/modules/discrete.md`:
  `ArrivalGenerator(joint=...)`, `EntityProperties.from_draw`.
- Blog post following the `blog-writer` skill (opening TOC, closing "Try it out
  yourself" Binder link): "When arrivals aren't independent: copulas in
  SimWeave", built around demos 30/31 with the lag-1 scatter and wait CCDF
  figures; a companion notebook in the notebooks repo via the
  `companion-notebooks` skill. Add it to `docs/blog/index.md`.
- `CHANGELOG.md` `## 0.9.0` with **Added** (`simweave.stats`, `JointSampler`,
  `ArrivalGenerator(joint=, gap_alignment=)`, `EntityProperties.from_draw`,
  marginal `ppf` attributes on existing helpers) and **Notes** (default
  behaviour unchanged; scipy needed for t-copula only). Follow the repo
  commit style (`feat:` / `docs:` conventional prefixes, PR per phase, no
  attribution trailers).
- Update `SIMWEAVE_API.md`, `claude_skill/` (the simweave skill) and the
  EdgeWeave notes with the new surface once it ships.
- Versioning: minor bump 0.9.0 once Phases 0 to 3 land; Phase 4 may ship as
  0.10.

## Phasing and PR plan

| Phase | Scope | Ships |
| --- | --- | --- |
| 0 | `stats/marginals.py`, `special.py`, `ppf` attributes on existing helpers, tests 2 and 4 | marginals usable standalone |
| 1 | Copula classes, `from_kendall_tau`, flips, tail coefficients, tests 1 and 3 | `sample_u` |
| 2 | `JointSampler`, `ArrivalGenerator(joint=)`, `EntityProperties.from_draw`, `MarkovCopulaProcess`, tests 5, 7 to 11; demos 30 and 31 | the feature users asked for |
| 3 | Fitting (`pseudo_obs`, `fit_copula`, `from_data`), test 6 | calibration from logs |
| 4 | Reliability lifetime-based common cause; `Warehouse` samplers + MC safety stock; demo 32 and supply-chain demo | showcase 3 and 4 |
| 5 | `VehicleArrivalProcess(joint=)`; drone capstone once the flight thread lands | showcase 5 |

Each phase is an independent PR to `main` with its own tests and CHANGELOG
entry under the next unreleased heading; Phases 0 to 2 are the minimum for a
release.

## Risks

- **Pairing semantics** (before/after): easy to get wrong silently. Mitigated
  by an explicit parameter, an exact deterministic test, and docs that show
  the Lindley term.
- **Per-event cost**: scalar copula draws in the tick loop; mitigated by
  block buffering. A benchmark target (>= 200k draws/s Gaussian on a laptop)
  is part of Phase 2.
- **Over-interpreting tail effects**: dependence families look similar in
  moderate-load queues; docs and demos must show CIs.
- **Tick quantisation** truncating short gaps (see above).
- **Stateful processes under MC**: the serial process must be built per
  replicate; enforced by docs and a test, not by the type system.
- Numpy `ndtri` fallback accuracy in far tails: bounded by clamping `u` to
  `[1e-12, 1 - 1e-12]`.

## Open questions for the user

1. **Package name**: `simweave.stats` (proposed) vs `simweave.copulas` /
   `simweave.dependence`?
2. **scipy policy**: OK that Student-t (and MLE fitting) require the `[optim]`
   extra while Gaussian/Clayton/Gumbel/Frank work on a base install? The
   alternative is to ship a numpy incomplete-beta and have no scipy
   requirement at all.
3. **Default `gap_alignment`**: `"before"` (attribute describes the entity
   that just arrived after the gap) or `"after"` (attribute predicts the next
   gap)? It changes how strongly queues feel the dependence.
4. **Scope of first release**: stop after Phase 2 (+ Phase 3 fitting), or
   include reliability common-cause in the same release?
5. **`ppf` attributes on the existing closures** (`sw.exponential(2.0).ppf`):
   acceptable, or prefer to leave `discrete.properties` untouched and only
   use the new `Exponential(...)` classes?
6. **Supply-chain**: should `pareto_sweep` itself gain a simulated /
   copula-aware option, or stay analytic with a separate helper
   (`reorder_point_from_joint`) as proposed?
7. **Drone capstone interface** with the flight session: agree now that the
   contract is "`JointSampler` draw dict -> factory", and who owns the
   scenario?
8. **Unrelated doc nit** found on the way: `VehicleArrivalProcess` docstring
   says `exponential(rate=lambda)` but `sw.exponential` takes `mean`; fix in
   a separate small PR?
9. **Dimension > 2 for the Archimedean family**: exchangeable only is fine for
   fleets (identical vehicles)? Heterogeneous fleets would need vines.
