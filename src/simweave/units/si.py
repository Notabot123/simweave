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
import numpy as np
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

def _retype(value: float, exponents: tuple[int, ...]) -> "SIUnit":
    cls = _known_unit(exponents)
    if cls is not None:
        return cls(value)
    return SIUnit(value, _unit_string(exponents), list(exponents))


@dataclass(slots=True)
class SIUnit:
    """Generic SI quantity with exponent-tracked dimensional analysis."""

    value: float
    unit: str = "dimensionless"
    exponents: list[int] = field(default_factory=lambda: [0] * 7)

    def __post_init__(self) -> None:
        if isinstance(self.value, SIUnit):
            self.value = self.value.value
        
        if isinstance(self.value, np.ndarray):
            pass  # keep as-is
        elif isinstance(self.value, (int, float)):
            self.value = float(self.value)
        else:
            raise TypeError("Value must be numeric or numpy array.")
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
            return _retype(self.value * other.value, new_exp)
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
            return _retype(self.value / other.value, new_exp)
        raise TypeError(f"Unsupported division with {type(other).__name__}.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SIUnit):
            return NotImplemented

        if tuple(self.exponents) != tuple(other.exponents):
            return False

        if np is not None and isinstance(self.value, np.ndarray):
            return np.array_equal(self.value, other.value)

        return self.value == other.value

    def __hash__(self) -> int:
        if np is not None and isinstance(self.value, np.ndarray):
            raise TypeError("SIUnit with numpy array is not hashable")
        return hash((type(self), self.value, tuple(self.exponents)))   
    
    
    # -- exponentiation ---------------------------------------------------
    def __pow__(self, power: float) -> "SIUnit":
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be numeric.")

        new_exp = tuple(e * power for e in self.exponents)

        # ensure resulting exponents are integers
        if not all(abs(e - round(e)) < 1e-12 for e in new_exp):
            raise TypeError(
                f"Power {power} produces non-integer exponents: {new_exp}"
            )

        new_exp = tuple(int(round(e)) for e in new_exp)

        return _retype(self.value ** power, new_exp)


    # -- convenience methods -----------------------------------------------
    def to(self, unit: str) -> float:
        scale_map = getattr(type(self), "_SCALE_MAP", None)
        if scale_map is None:
            raise TypeError(f"{type(self).__name__} does not support unit conversion.")
        if unit not in scale_map:
            raise ValueError(f"Unsupported unit for {type(self).__name__}: {unit}")
        return self.value / scale_map[unit]


    def to_unit(self, unit: str) -> "SIUnit":
        return type(self)(self.to(unit), unit)


    def format(self, unit: str | None = None, precision: int | None = None) -> str:
        if unit is None:
            val = self.value
            unit_str = self.unit
        else:
            val = self.to(unit)
            unit_str = unit

        if precision is not None:
            np.round(val, precision)

        return f"{val} [{unit_str}]"


    def auto_format(self, precision: int = 3) -> str:
        scale_map = getattr(type(self), "_SCALE_MAP", None)
        display_units = getattr(type(self), "_DISPLAY_UNITS", None)

        if not scale_map or not display_units:
            return self.format(precision=precision)

        for unit in reversed(display_units):
            val = self.to(unit)
            if abs(val) >= 1:
                return f"{np.round(val, precision)} [{unit}]"

        unit = display_units[0]
        return f"{np.round(self.to(unit), precision)} [{unit}]"
    
    def sqrt(self):
        return self ** 0.5
    
    def cbrt(self):
        return self ** (1/3)


# ---------------------------------------------------------------------------
# Concrete dimensioned units. The existing public names are preserved so that
# the quarter-car / pendulum / RLC examples remain compatible.
# ---------------------------------------------------------------------------


class Distance(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 1e-3,
        "km": 1000.0,
        "miles": 1609.344,
        "ft": 0.3048,
        "in": 0.0254,
    }
    _UNIT_ALIASES: ClassVar[dict[str, str]] = {
        "foot": "ft",
        "feet": "ft",
        "inches": "in",
    }

    def __init__(self, value: float | SIUnit, unit: str = "m"):
        if isinstance(value, SIUnit):
            value = value.value
        unit = self._UNIT_ALIASES.get(unit, unit)
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported distance unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="m",
            exponents=[1, 0, 0, 0, 0, 0, 0],
        )


class Velocity(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "m/s": 1.0,
        "mph": 0.44704,
        "kph": 0.27777778,
        "ft/s": 0.3048,
        "fps": 0.3048,
        "knot": 0.51444444,
        "mach": 343,

    }

    def __init__(self, value: float | SIUnit, unit: str = "m/s"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported velocity unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="m/s",
            exponents=[1, 0, 0, 0, 0, 0, -1],
        )


class Acceleration(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "m/s^2": 1.0,
        "g": 9.80665,  # standard gravity
    }

    def __init__(self, value: float | SIUnit, unit: str = "m/s^2"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported acceleration unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="m/s^2",
            exponents=[1, 0, 0, 0, 0, 0, -2],
        )


class Mass(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "kg": 1.0,
        "g": 1e-3,
        "mg": 1e-6,
        "tonne": 1000.0,
        "lb": 0.45359237,
        "lbs": 0.45359237,
    }

    _UNIT_ALIASES: ClassVar[dict[str, str]] = {
        "pound": "lb",
        "pounds": "lb",
    }

    def __init__(self, value: float | SIUnit, unit: str = "kg"):
        if isinstance(value, SIUnit):
            value = value.value
        unit = self._UNIT_ALIASES.get(unit, unit)
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported mass unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="kg",
            exponents=[0, 1, 0, 0, 0, 0, 0],
        )


class Force(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "N": 1.0,
        "kN": 1e3,
        "lbf": 4.4482216152605,
    }

    def __init__(self, value: float | SIUnit, unit: str = "N"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported force unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="N",
            exponents=[1, 1, 0, 0, 0, 0, -2],
        )


class Area(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "m^2": 1.0,
        "cm^2": 1e-4,
        "mm^2": 1e-6,
        "km^2": 1e6,
        "ft^2": 0.092903,
        "in^2": 0.00064516,
        "acre": 4046.8564224,
    }

    def __init__(self, value: float | SIUnit, unit: str = "m^2"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported area unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="m^2",
            exponents=[2, 0, 0, 0, 0, 0, 0],
        )   


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
            value=value * self._SCALE_MAP[unit],
            unit="s",
            exponents=[0, 0, 0, 0, 0, 0, 1],
        )

    
class Pressure(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "Pa": 1.0,
        "kPa": 1e3,
        "MPa": 1e6,
        "bar": 1e5,
        "atm": 101325.0,
        "psi": 6894.757,
    }
    _DISPLAY_UNITS: ClassVar[list[str]] = ["Pa", "kPa", "MPa", "bar"]

    def __init__(self, value: float | SIUnit, unit: str = "Pa"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported pressure unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="Pa",
            exponents=[-1, 1, 0, 0, 0, 0, -2],
        )

class Energy(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "J": 1.0,
        "kJ": 1e3,
        "MJ": 1e6,
        "Wh": 3600.0,
        "kWh": 3.6e6,
        "cal": 4.184,
        "kcal": 4184.0,
    }
    _DISPLAY_UNITS: ClassVar[list[str]] = ["J", "kJ", "MJ"]

    def __init__(self, value: float | SIUnit, unit: str = "J"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported energy unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="J",
            exponents=[2, 1, 0, 0, 0, 0, -2],
        )

class Power(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "W": 1.0,
        "kW": 1e3,
        "MW": 1e6,
        "hp": 745.7,
    }
    _DISPLAY_UNITS: ClassVar[list[str]] = ["W", "kW", "MW"]

    def __init__(self, value: float | SIUnit, unit: str = "W"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported power unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="W",
            exponents=[2, 1, 0, 0, 0, 0, -3],
        )

class Frequency(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "Hz": 1.0,
        "kHz": 1e3,
        "MHz": 1e6,
        "rpm": 1.0 / 60.0,
    }

    def __init__(self, value: float | SIUnit, unit: str = "Hz"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported frequency unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="Hz",
            exponents=[0, 0, 0, 0, 0, 0, -1],
        )


class Temperature(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "K": 1.0,
        "C": 1.0,  # handled via offset
    }

    _OFFSET_MAP: ClassVar[dict[str, float]] = {
        "K": 0.0,
        "C": 273.15,
    }

    def __init__(self, value: float | SIUnit, unit: str = "K"):
        if isinstance(value, SIUnit):
            value = value.value

        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported temperature unit: {unit}")

        # Convert to Kelvin
        kelvin_value = float(value) * self._SCALE_MAP[unit] + self._OFFSET_MAP[unit]

        super().__init__(
            value=kelvin_value,
            unit="K",
            exponents=[0, 0, 0, 1, 0, 0, 0],
        )

    def to(self, unit: str) -> float:
        """ Convert temperature units. For temperatures, note an offset as well as scaling """
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported temperature unit: {unit}")

        # Convert from Kelvin
        return (self.value - self._OFFSET_MAP[unit]) / self._SCALE_MAP[unit]
    
    def __sub__(self, other: object) -> "TemperatureDelta":
        if not isinstance(other, Temperature):
            raise TypeError("Can only subtract Temperature from Temperature.")
        return TemperatureDelta(self.value - other.value)
    
    def __add__(self, other: object) -> "Temperature":
        if isinstance(other, TemperatureDelta):
            return Temperature(self.value + other.value)
        raise TypeError("Can only add TemperatureDelta to Temperature.")
    
    def __radd__(self, other: object) -> "Temperature":
        return self.__add__(other)

class TemperatureDelta(SIUnit):
    """ Delta temperature to allow relative and absolute """
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        "K": 1.0,
        "C": 1.0,  # same scaling for differences
    }

    def __init__(self, value: float | SIUnit, unit: str = "K"):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f"Unsupported temperature difference unit: {unit}")
        super().__init__(
            value=value * self._SCALE_MAP[unit],
            unit="K",
            exponents=[0, 0, 0, 1, 0, 0, 0],
        )

# electric relevant units
class Current(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit="A", exponents=[0, 0, 1, 0, 0, 0, 0])

class Voltage(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(
            value=value,
            unit="V",
            exponents=[2, 1, -1, 0, 0, 0, -3],
        )

class Resistance(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(
            value=value,
            unit="Ω",
            exponents=[2, 1, -2, 0, 0, 0, -3],
        )

class Capacitance(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(
            value=value,
            unit="F",
            exponents=[-2, -1, 2, 0, 0, 0, 4],
        )

class Resistivity(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(
            value=value,
            unit="Ω·m",
            exponents=[3, 1, -2, 0, 0, 0, -3],
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
    (-1, 1, 0, 0, 0, 0, -2): Pressure,
    (2, 1, 0, 0, 0, 0, -2): Energy,
    (2, 1, 0, 0, 0, 0, -3): Power,
    (0, 0, 0, 0, 0, 0, -1): Frequency,
    (0, 0, 0, 1, 0, 0, 0): Temperature,
    (0, 0, 1, 0, 0, 0, 0): Current,
    (2, 1, -1, 0, 0, 0, -3): Voltage,
    (2, 1, -2, 0, 0, 0, -3): Resistance,
    (-2, -1, 2, 0, 0, 0, 4): Capacitance,
    (3, 1, -2, 0, 0, 0, -3): Resistivity,
}
