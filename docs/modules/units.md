# Units

Lightweight SI quantity helpers. Useful for keeping engineering models
self-documenting without taking on a full unit-algebra dependency like
`pint`.

```python
import simweave as sw

speed = sw.Velocity(15.0, "m/s")
mass  = sw.Mass(2.0, "kg")
```

The current set:

- `SIUnit` (base)
- `Distance`, `Velocity`, `Acceleration`
- `Mass`, `Force`
- `Area`, `Volume`
- `TimeUnit`

These are deliberately thin — they wrap a value plus a label. They do
not perform algebraic unit conversion. Use `pint` if you need that.

## API

::: simweave.units
    options:
      show_root_heading: false
      show_source: true
