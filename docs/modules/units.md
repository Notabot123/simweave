# Units

Simple helpers for working with physical quantities like distance, time, and force—with **dimensional analysis built in**.

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

The base class:

- `SIUnit` (base; carries the value, unit string, and exponents)

# Available Units

## Basic
- `Distance`
- `TimeUnit`
- `Mass`

## Derived
- `Velocity`
- `Acceleration`
- `Force`
- `Energy`
- `Power`
- `Pressure`
- `Area`
- `Volume`
- `Frequency`
- `Temperature` (absolute, e.g. Kelvin or Celsius)
- `TemperatureDelta` (differences)

## Unit Conversion

You can construct values using common units:

```python
d = sw.Distance(10, "ft")      # feet → stored internally as metres
v = sw.Velocity(60, "mph")     # miles per hour
e = sw.Energy(1, "kWh")        # kilowatt-hour                    # -> Force(150.0 N)
```
Convert explicitly:

```python
d.to("m")      # → 3.048
d.to("ft")     # → 10.0
```
## Displaying Values

```python
d = sw.Distance(10, "ft")

d.format()          # "3.048 [m]"
d.format("ft")      # "10.0 [ft]"
```

For nicer output:

```python
e = sw.Energy(1500)

e.auto_format()     # "1.5 [kJ]"
```

## Temperature (special case)

Temperature supports both Kelvin and Celsius:

```python
t = sw.Temperature(0, "C")

t.value        # 273.15 (stored in Kelvin)
t.to("C")      # 0.0
```

Differences are handled explicitly:

```python
t1 = sw.Temperature(30, "C")
t2 = sw.Temperature(20, "C")

delta = t1 - t2    # TemperatureDelta
t3 = t2 + delta    # Temperature
```

Adding two absolute temperatures is not allowed:

```python
t1 + t2   # ! raises TypeError
```

## Physical Constants

Convenience constants are available, for all SIunits and several physical constants:

```python
from simweave.units.constants import kg, m, s, g, c

force = 10 * kg * m / s**2
weight = 80 * kg * g
energy = sw.Mass(1) * c**2
```

Full set of physical constants at current release (v0.5.0):
```python
# Acceleration due to gravity
g = 9.80665 * m / s**2

# Speed of light
c = 299_792_458 * m / s

# Planck constant
h = 6.62607015e-34 * J * s

# Boltzmann constant
k_B = 1.380649e-23 * J / K
```

## What it does
- Combines units correctly when multiplying/dividing
- Prevents invalid operations (e.g. adding distance and time)
- Converts units at construction time
- Keeps everything internally in standard SI units

## What it does *not* do (yet)
- Full unit parsing (e.g. "kg*m/s^2" strings)
- Complex unit systems beyond SI
- NumPy array support (planned)


For a runnable walkthrough see
[`demos/15_units_dimensional.py`](https://github.com/Notabot123/simweave/blob/main/demos/15_units_dimensional.py).

## API

::: simweave.units
    options:
      show_root_heading: false
      show_source: true
