# Units

Lightweight SI quantity helpers with **dimensional analysis built in**.
Every quantity carries a 7-tuple of SI exponents — `(m, kg, A, K, mol,
cd, s)` — and arithmetic operations compose those exponents and re-type
the result automatically.

```python
import simweave as sw

d = sw.Distance(120.0)              # 120 m
t = sw.TimeUnit(2.0, unit="mins")    # 2 minutes  (stored as 120.0 s)
m = sw.Mass(1500.0)                  # 1500 kg

v = d / t                            # -> Velocity(1.0 m/s)   (auto-typed)
a = v / sw.TimeUnit(10.0)            # -> Acceleration(0.1 m/s^2)
f = m * a                            # -> Force(150.0 N)
```

The current concrete classes:

- `SIUnit` (base; carries the value, unit string, and exponents)
- `Distance`, `Velocity`, `Acceleration`
- `Mass`, `Force`
- `Area`, `Volume`
- `TimeUnit`

## What it does

- **Composes dimensions on `*` and `/`.** Multiplying or dividing two
  quantities adds or subtracts their SI exponent tuples. If the result
  matches a registered shape (e.g. `(1, 0, 0, 0, 0, 0, -1)` is metres
  per second) the result is returned as the corresponding concrete class
  — `Velocity` in that case — so isinstance checks Just Work.
- **Enforces dimensional consistency on `+` and `-`.** Adding a
  `Distance` to a `TimeUnit` raises `TypeError`. Adding two `Distance`
  values returns a `Distance`.
- **Time conversion at construction time.** `TimeUnit` accepts
  `unit="s" | "ms" | "min" | "h" | "day"` and stores the value in
  seconds canonically.

## What it does *not* do

- **Convert between unit *systems*.** Other than `TimeUnit`, the classes
  do not parse imperial input or non-canonical SI prefixes — there is no
  `Distance(1.0, "ft")` or `Mass(2.2, "lb")`. If you need
  feet-to-metres or pounds-to-kg conversions, either pre-convert at the
  boundary of your code or reach for [`pint`][pint], which solves a
  superset of the problem at the cost of a heavier dependency.
- **Format as imperial.** All `__str__` output is in the canonical SI
  unit string (e.g. `"120.0 [m]"`, `"3600.0 [s]"`).

[pint]: https://pint.readthedocs.io/

For a runnable walkthrough see
[`demos/15_units_dimensional.py`](https://github.com/Notabot123/simweave/blob/main/demos/15_units_dimensional.py).

## API

::: simweave.units
    options:
      show_root_heading: false
      show_source: true
