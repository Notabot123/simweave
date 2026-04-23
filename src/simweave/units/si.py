"""SI unit classes with exponent-tracked dimensional analysis.

Every unit carries a 7-tuple of SI exponents in the order
(metre, kilogram, ampere, kelvin, mole, candela, second). Multiplication and
division compose exponents; the result is re-typed to a known concrete class
when the exponents match one of the registered shapes (e.g. m * m -> Area).
Addition and subtraction are only permitted between operands of the same
concrete type.

This module is pure-Python and has no numpy dependency so it can be used in
tight inner loops without allocating ndarrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

_SI_SYMBOLS = ("m", "kg", "A", "K", "mol", "cd", "s")


def _unit_string(exponents: tuple[int, ...]) -> str:
    parts: list[str] = []
    for symbol, exp in zip(_SI_SYMBOLS, exponents):
        if exp == 0:
            continue
        parts.append(symbol if exp == 1 else f"{symbol}^{exp}")
    return "*".join(parts) if parts else "dimensionless"


def _known_unit(exponents: tuple[int, ...]):
    return _KNOWN_BY_EXP.get(exponents)


@dataclass(slots=True)
class SIUnit:
    """Generic SI quantity with exponent-tracked dimensional analysis."""

    value: float
    unit: str = "dimensionless"
    exponents: list[int] = field(default_factory=lambda: [0] * 7)

    def __post_init__(self) -> None:
        if isinstance(self.value, SIUnit):
            self.value = self.value.value
        self.value = float(self.value)
        if len(self.exponents) != 7:
            raise ValueError("SI exponents must have length 7.")

    # -- representation ------------------------------------------------
    def __str__(self) -> str:
        return f"{self.value} [{self.unit}]"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.value}, unit={self.unit!r})"

    # -- additive ------------------------------------------------------
    def _check_same_dim(self, other: "SIUnit") -> None:
        if tuple(self.exponents) != tuple(other.exponents):
            raise TypeError(
                f"Cannot add/subtract {self.unit} and {other.unit}: different dimensions."
            )

    def __add__(self, other: object) -> "SIUnit":
        if not isinstance(other, SIUnit):
            raise TypeError(
                f"Cannot add {type(self).__name__} and {type(other).__name__}."
            )
        self._check_same_dim(other)
        cls = type(self) if type(self) is type(other) else SIUnit
        if cls is SIUnit:
            return SIUnit(self.value + other.value, self.unit, list(self.exponents))
        return cls(self.value + other.value)  # type: ignore[arg-type]

    def __sub__(self, other: object) -> "SIUnit":
        if not isinstance(other, SIUnit):
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from {type(self).__name__}."
            )
        self._check_same_dim(other)
        cls = type(self) if type(self) is type(other) else SIUnit
        if cls is SIUnit:
            return SIUnit(self.value - other.value, self.unit, list(self.exponents))
        return cls(self.value - other.value)  # type: ignore[arg-type]

    # -- multiplicative -----------------------------------------------
    def __mul__(self, other: object) -> "SIUnit":
        if isinstance(other, (int, float)):
            return (
                type(self)(self.value * float(other))
                if type(self) is not SIUnit
                else SIUnit(self.value * float(other), self.unit, list(self.exponents))
            )
        if isinstance(other, SIUnit):
            new_exp = tuple(a + b for a, b in zip(self.exponents, other.exponents))
            cls = _known_unit(new_exp)
            if cls is None:
                return SIUnit(
                    self.value * other.value, _unit_string(new_exp), list(new_exp)
                )
            return cls(self.value * other.value)
        raise TypeError(f"Unsupported multiplication with {type(other).__name__}.")

    def __rmul__(self, other: object) -> "SIUnit":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "SIUnit":
        if isinstance(other, (int, float)):
            return (
                type(self)(self.value / float(other))
                if type(self) is not SIUnit
                else SIUnit(self.value / float(other), self.unit, list(self.exponents))
            )
        if isinstance(other, SIUnit):
            new_exp = tuple(a - b for a, b in zip(self.exponents, other.exponents))
            cls = _known_unit(new_exp)
            if cls is None:
                return SIUnit(
                    self.value / other.value, _unit_string(new_exp), list(new_exp)
                )
            return cls(self.value / other.value)
        raise TypeError(f"Unsupported division with {type(other).__name__}.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SIUnit):
            return NotImplemented
        return (
            tuple(self.exponents) == tuple(other.exponents)
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((type(self), self.value, tuple(self.exponents)))


# ---------------------------------------------------------------------------
# Concrete dimensioned units. The existing public names are preserved so that
# the quarter-car / pendulum / RLC examples remain compatible.
# ---------------------------------------------------------------------------


class Distance(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit="m", exponents=[1, 0, 0, 0, 0, 0, 0])


class Velocity(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit="m/s", exponents=[1, 0, 0, 0, 0, 0, -1])


class Acceleration(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit="m/s^2", exponents=[1, 0, 0, 0, 0, 0, -2])


class Mass(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit="kg", exponents=[0, 1, 0, 0, 0, 0, 0])


class Force(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit="N", exponents=[1, 1, 0, 0, 0, 0, -2])


class Area(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit="m^2", exponents=[2, 0, 0, 0, 0, 0, 0])


class Volume(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit="m^3", exponents=[3, 0, 0, 0, 0, 0, 0])


class TimeUnit(SIUnit):
    """Time, stored canonically in seconds regardless of input unit."""

    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "s": 1.0,
        "sec": 1.0,
        "ms": 1e-3,
        "mins": 60.0,
        "min": 60.0,
        "hrs": 3600.0,
        "h": 3600.0,
        "days": 86400.0,
        "day": 86400.0,
    }

    def __init__(self, value: float | SIUnit, unit: str = "s"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported time unit: {unit}")
        super().__init__(
            value=float(value) * self._SCALE_MAP[unit],
            unit="s",
            exponents=[0, 0, 0, 0, 0, 0, 1],
        )


_KNOWN_BY_EXP: dict[tuple[int, ...], type[SIUnit]] = {
    (1, 0, 0, 0, 0, 0, 0): Distance,
    (1, 0, 0, 0, 0, 0, -1): Velocity,
    (1, 0, 0, 0, 0, 0, -2): Acceleration,
    (0, 1, 0, 0, 0, 0, 0): Mass,
    (2, 0, 0, 0, 0, 0, 0): Area,
    (3, 0, 0, 0, 0, 0, 0): Volume,
    (1, 1, 0, 0, 0, 0, -2): Force,
    (0, 0, 0, 0, 0, 0, 1): TimeUnit,
}
