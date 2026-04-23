"""SI-units dimensional algebra.

``simweave.units`` tracks the seven SI exponents on every quantity. When
you multiply or divide two quantities, SimWeave composes the exponents
and re-types the result to the matching concrete class if one is
registered. This means you can write physics expressions that *read like
physics* and have the type system catch dimension errors for you.

Run::

    python demos/15_units_dimensional.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path when not pip-installed)

from simweave.units.si import (
    Acceleration,
    Distance,
    Force,
    Mass,
    TimeUnit,
    Velocity,
)


def _show(label: str, value) -> None:
    print(f"{label:<28} {type(value).__name__:<14} {value}")


def main() -> None:
    print("--- Constructing quantities ----------------------------------")
    d = Distance(120.0)            # 120 m
    t = TimeUnit(2.0, unit="mins")  # 2 minutes -> stored as 120.0 s
    m = Mass(1500.0)                # 1.5 t car
    _show("d = Distance(120 m)", d)
    _show("t = TimeUnit(2 mins)", t)
    _show("m = Mass(1500 kg)", m)

    print()
    print("--- Auto-typed division and multiplication -------------------")
    # Distance / Time -> Velocity (the result is auto-typed).
    v = d / t
    _show("d / t  (m / s)", v)
    assert isinstance(v, Velocity), "Expected Velocity from m/s"

    # Velocity / Time -> Acceleration.
    a = v / TimeUnit(10.0)
    _show("v / TimeUnit(10 s)", a)
    assert isinstance(a, Acceleration)

    # Mass * Acceleration -> Force.
    f = m * a
    _show("m * a  (kg * m/s^2)", f)
    assert isinstance(f, Force)

    print()
    print("--- Same dimension, same class: addition is permitted --------")
    d2 = Distance(80.0)
    _show("d + d2", d + d2)

    print()
    print("--- Mixed dimensions: addition raises ------------------------")
    try:
        _ = d + t
    except TypeError as e:
        print(f"TypeError as expected: {e}")

    print()
    print("--- Time conversions (only TimeUnit has a scale map) ---------")
    for unit in ("s", "ms", "min", "h", "day"):
        _show(f"TimeUnit(1, '{unit}')", TimeUnit(1.0, unit=unit))


if __name__ == "__main__":
    main()
