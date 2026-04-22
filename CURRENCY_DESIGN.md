# Design proposal: `simeng.currency`

Status: **proposal** (not implemented).
Purpose: support monetary quantities in simulations for finance
practitioners, mirroring the dimensional-analysis discipline of
`simeng.units` without falsely implying that currency is a physical
dimension.

This document is the argument and the shape. Whether to build it is a
follow-on decision.

---

## Why it's worth considering

1. **There's real demand.** Monte Carlo over portfolios, queueing
   models where each served customer produces revenue, supply-chain
   models with holding and stock-out costs — all of these want a
   first-class "money" type, both for correctness and for pretty
   printing.
2. **The SI machinery already gives us the mistake-prevention story.**
   Adding USD to GBP without explicit conversion is exactly the kind
   of error `Distance + Velocity` is meant to catch. Currency is not
   dimensional, but the *enforcement pattern* transfers cleanly.
3. **The upside of getting it wrong is non-trivial.** A finance
   simulation that silently sums mixed-currency flows will give a
   confident, completely meaningless answer. Typed money makes that a
   loud `TypeError`.

## Why it's worth being cautious

1. **FX rates are live data.** Hard-coding them ages badly; pulling
   them from an API couples simeng to network, keys, and rate-limits.
2. **Currencies aren't truly fungible.** $1 today ≠ $1 next year;
   adding a time dimension (discount rates) is a second conversation
   that an over-eager API can pretend doesn't exist.
3. **Operators get awkward.** `price = £ + %VAT` works; `£ + $` should
   not. But `£ * scalar` and `£ / Time` (for a rate) should. Getting
   this right means more operator overloading, which tends to invite
   subtle bugs.
4. **Formatting carries culture.** `£1,234.56` vs `1.234,56 €` vs
   `¥1235` (no decimals). Locale-aware output via `babel` is
   out of scope for a zero-dep core library; a sensible fallback with
   an optional dep extra is the pragmatic compromise.

## Recommendation

**Ship it, scoped tightly.** Do the three things simeng is uniquely
positioned to do well:
- tag values with an immutable currency code,
- refuse cross-currency arithmetic unless a converter is explicitly
  supplied,
- format cleanly without opinions about timezone or inflation.

Leave these **out** of the core:
- live FX data,
- discount rates / NPV,
- tax treatment.

Users who need those things bring them as a strategy object.

---

## Proposed surface (`simeng.currency`)

```python
from simeng.currency import Money, FXConverter, format_money
```

### `Money`

```python
@dataclass(frozen=True, slots=True)
class Money:
    amount: Decimal | float       # stored as Decimal internally
    currency: str                 # ISO 4217, e.g. "GBP", "USD", "JPY"

    # arithmetic
    def __add__(self, other: Money) -> Money                # same-currency only
    def __sub__(self, other: Money) -> Money
    def __mul__(self, scalar: int | float | Decimal) -> Money
    def __rmul__(self, scalar) -> Money
    def __truediv__(self, scalar_or_money) -> Money | float # money/money -> ratio

    def to(self, target: str, converter: FXConverter) -> Money
    def __format__(self, spec: str) -> str                  # locale-aware
```

Invariants:
- Same-currency `+`/`-` succeed; cross-currency raise
  `CurrencyMismatchError` (subclass of `TypeError` for compatibility).
- `Money(10, "GBP") * 3` returns `Money(30, "GBP")` — scalar only, no
  integer/float on the right that means a different currency.
- `Money(10, "GBP") / Money(2, "GBP")` returns `5.0` (dimensionless
  ratio); `Money(10, "GBP") / Money(2, "USD")` raises unless a
  converter is bound.
- Stored as `decimal.Decimal` internally with a default precision
  appropriate to each currency (JPY → 0 decimals, most → 2, BTC/ETH →
  8 if we ever include crypto).

### `FXConverter`

Protocol — users provide the implementation.

```python
@runtime_checkable
class FXConverter(Protocol):
    def rate(self, source: str, target: str, at: datetime | None = None) -> Decimal: ...

class StaticFXConverter:
    """Developer-supplied fixed-rate table. Never hits the network."""
    def __init__(self, rates: dict[tuple[str, str], Decimal]): ...

class CallableFXConverter:
    """Wrap any callable of signature (src, tgt, at) -> rate."""
```

The point: simeng **never ships live rates**. If the user wants real
rates, they build a converter that calls their broker/API and inject it.

### `format_money`

Pluggable formatter. Default is ASCII-safe:

```python
format_money(Money(1234.5, "GBP"))   # "GBP 1,234.50"
format_money(Money(1234.5, "USD"))   # "USD 1,234.50"
format_money(Money(1235, "JPY"))     # "JPY 1,235"
```

With optional extra:

```bash
pip install simeng[intl]             # pulls in babel
```

```python
format_money(Money(1234.5, "GBP"), locale="en_GB")  # "£1,234.50"
format_money(Money(1234.5, "GBP"), locale="de_DE")  # "1.234,50 £"
```

---

## Why not overload the SI machinery?

Tempting, since the shape is similar. **Don't.**

- Currency is not a physical dimension. Bolting it onto `SIUnit`
  muddies the meaning of "dimensional analysis."
- SI's `_KNOWN_BY_EXP` cleverness (auto-retyping `m/s` to `Velocity`)
  makes no sense for currency — we don't have derived currency types.
- Keeping `simeng.currency` as a parallel, narrower module means we
  can evolve it (discount curves, time-value-of-money) without
  breaking `simeng.units`.

There's one exception worth flagging: **rates** like "£/hour" arise
naturally (billing). A later extension can compose `Money` with
`TimeUnit`:

```python
rate = Money(50, "GBP") / TimeUnit(1, "hrs")   # -> Rate(50, "GBP/h")
rate * TimeUnit(2, "hrs")                       # -> Money(100, "GBP")
```

This would be a `Rate` class in `simeng.currency` that stores a `Money`
and a `TimeUnit` and exposes `__mul__(TimeUnit) -> Money`. Worth
scoping but not in the first cut.

---

## Minimal first-cut scope

If we do only one thing, do this:

1. `Money(amount, currency)` frozen dataclass, Decimal-backed.
2. Same-currency `+`, `-`, scalar `*`, scalar `/`, money/money → float.
3. `CurrencyMismatchError`.
4. `format_money` ASCII default.
5. `to(target, converter)` with `StaticFXConverter`.
6. No live-rate integrations; no NPV; no babel dep.

That's a tight ~200 lines of code + thorough tests. It solves 90% of
finance-simulation use cases and leaves the hard, opinionated bits to
user code.

---

## Open questions for Stuart

1. Do you want **Decimal-everywhere** semantics by default (banker's
   rounding, slower), or let users opt in via a module-level flag?
   Recommendation: Decimal default. Speed hit is negligible compared
   to the integrator inner loop.
2. Do you care about **negative money** (debts)? Recommendation: yes,
   let it pass through — finance code has signed flows constantly.
3. **Currency code validation** — strict ISO 4217 (~180 codes, list
   baked in) or permissive (any uppercase string)? Recommendation:
   strict list, with an escape hatch `Money.register_custom("XYZ",
   decimals=2)` for crypto / in-game currencies / test fixtures.
4. Does EdgeWeave's code-generation path need this early (i.e. is it
   on the 0.2 train) or can it land later (0.3)? That drives how
   aggressively I pursue it.
