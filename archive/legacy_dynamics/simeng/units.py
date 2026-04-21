from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


def _known_unit_from_exponents(exponents: tuple[int, ...]):
    return {
        (1, 0, 0, 0, 0, 0, 0): Distance,
        (1, 0, 0, 0, 0, 0, -1): Velocity,
        (1, 0, 0, 0, 0, 0, -2): Acceleration,
        (0, 1, 0, 0, 0, 0, 0): Mass,
        (2, 0, 0, 0, 0, 0, 0): Area,
        (3, 0, 0, 0, 0, 0, 0): Volume,
        (1, 1, 0, 0, 0, 0, -2): Force,
        (0, 0, 0, 0, 0, 0, 1): TimeUnit,
    }.get(exponents)


def _unit_string_from_exponents(exponents: tuple[int, ...]) -> str:
    base = ['m', 'kg', 'A', 'K', 'mol', 'cd', 's']
    parts: list[str] = []
    for symbol, exp in zip(base, exponents):
        if exp == 0:
            continue
        if exp == 1:
            parts.append(symbol)
        else:
            parts.append(f'{symbol}^{exp}')
    return '*'.join(parts) if parts else 'dimensionless'


@dataclass(slots=True)
class SIUnit:
    value: float
    unit: str
    exponents: list[int] = field(default_factory=lambda: [0] * 7)

    def __post_init__(self) -> None:
        if isinstance(self.value, SIUnit):
            self.value = self.value.value
        self.value = float(self.value)
        if len(self.exponents) != 7:
            raise ValueError('SI exponents must have length 7.')

    def __str__(self) -> str:
        return f'{self.value} [{self.unit}]'

    def __add__(self, other: object):
        if type(self) is not type(other):
            raise TypeError(f'Cannot add {type(self).__name__} and {type(other).__name__}.')
        return type(self)(self.value + other.value)  # type: ignore[arg-type]

    def __sub__(self, other: object):
        if type(self) is not type(other):
            raise TypeError(f'Cannot subtract {type(other).__name__} from {type(self).__name__}.')
        return type(self)(self.value - other.value)  # type: ignore[arg-type]

    def __mul__(self, other: object):
        if isinstance(other, (int, float)):
            return type(self)(self.value * other)
        if isinstance(other, SIUnit):
            new_exp = tuple(a + b for a, b in zip(self.exponents, other.exponents))
            cls = _known_unit_from_exponents(new_exp)
            if cls is None:
                return SIUnit(self.value * other.value, _unit_string_from_exponents(new_exp), list(new_exp))
            return cls(self.value * other.value)
        raise TypeError(f'Unsupported multiplication between {type(self).__name__} and {type(other).__name__}.')

    def __rmul__(self, other: object):
        return self.__mul__(other)

    def __truediv__(self, other: object):
        if isinstance(other, (int, float)):
            return type(self)(self.value / other)
        if isinstance(other, SIUnit):
            new_exp = tuple(a - b for a, b in zip(self.exponents, other.exponents))
            cls = _known_unit_from_exponents(new_exp)
            if cls is None:
                return SIUnit(self.value / other.value, _unit_string_from_exponents(new_exp), list(new_exp))
            return cls(self.value / other.value)
        raise TypeError(f'Unsupported division between {type(self).__name__} and {type(other).__name__}.')


class Distance(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit='m', exponents=[1, 0, 0, 0, 0, 0, 0])


class Velocity(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit='m/s', exponents=[1, 0, 0, 0, 0, 0, -1])


class Acceleration(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit='m/s^2', exponents=[1, 0, 0, 0, 0, 0, -2])


class Mass(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit='kg', exponents=[0, 1, 0, 0, 0, 0, 0])


class Force(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit='N', exponents=[1, 1, 0, 0, 0, 0, -2])


class Area(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit='m^2', exponents=[2, 0, 0, 0, 0, 0, 0])


class Volume(SIUnit):
    def __init__(self, value: float | SIUnit):
        super().__init__(value=value, unit='m^3', exponents=[3, 0, 0, 0, 0, 0, 0])


class TimeUnit(SIUnit):
    _SCALE_MAP: ClassVar[dict[str, float]] = {
        's': 1.0,
        'sec': 1.0,
        'ms': 1e-3,
        'mins': 60.0,
        'min': 60.0,
        'hrs': 3600.0,
        'h': 3600.0,
    }

    def __init__(self, value: float | SIUnit, unit: str = 's'):
        if isinstance(value, SIUnit):
            value = value.value
        if unit not in self._SCALE_MAP:
            raise ValueError(f'Unsupported time unit: {unit}')
        super().__init__(value=float(value) * self._SCALE_MAP[unit], unit='s', exponents=[0, 0, 0, 0, 0, 0, 1])
