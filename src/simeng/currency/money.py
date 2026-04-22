"""The :class:`Money` type — a currency-tagged Decimal value.

Design notes
------------
* **Decimal everywhere, no float drift.** Amounts are stored as
  :class:`decimal.Decimal`. Integer, float, and string inputs are
  coerced through ``Decimal(str(value))`` to avoid the classic
  ``Decimal(0.1) -> Decimal('0.1000000000000000055511151231257827021181583404541015625')``
  surprise.
* **Banker's rounding at display/quantisation time.** Computation keeps
  full precision; rounding is only applied when the caller asks for it
  (``round_to_currency`` or ``format_money``). ``ROUND_HALF_EVEN`` is
  the default — configurable via ``rounding=`` where supported.
* **Cross-currency arithmetic refuses.** ``GBP + USD`` raises
  :class:`CurrencyMismatchError` rather than silently coercing. Users
  must call ``money.to(target, converter)`` with an explicit FX
  converter to change currencies.
* **Scalar-only multiplication.** ``Money * int/float/Decimal`` returns
  a ``Money`` in the same currency. ``Money * Money`` is ill-defined
  dimensionally (square-pounds?) and raises.
* **Ratios are dimensionless.** ``Money / Money`` (same currency)
  returns a plain ``float``. Different currencies raise.
* **Negatives allowed.** Debts, refunds, and signed cashflows are
  legitimate — no ``ValueError`` on negative amounts.
* **Frozen, hashable.** Two ``Money`` values are equal iff both
  currency and amount match exactly. Hash is consistent with equality.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation
from typing import TYPE_CHECKING, Any, Union

from simeng.currency.codes import get_decimals, is_valid_currency

if TYPE_CHECKING:
    from datetime import datetime

    from simeng.currency.fx import FXConverter


Numeric = Union[int, float, Decimal, str]


class CurrencyMismatchError(TypeError):
    """Raised when an operation requires two Money values to share a currency
    but they don't (e.g. ``GBP + USD``).

    Subclasses :class:`TypeError` so that ``except TypeError`` catches it in
    legacy code, while new code can ``except CurrencyMismatchError`` for
    precise handling.
    """


def _coerce_amount(value: Any) -> Decimal:
    """Coerce ``value`` to Decimal without float-binary drift."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        # bool is a subclass of int — refuse it explicitly; "True pounds"
        # is almost certainly a bug.
        raise TypeError("Money amount cannot be a bool.")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        # Route through str so 0.1 becomes Decimal('0.1'), not the binary
        # expansion.
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(f"Cannot parse {value!r} as a decimal amount.") from exc
    raise TypeError(
        f"Money amount must be numeric (int/float/Decimal/str); got {type(value).__name__}."
    )


def _normalise_code(code: Any) -> str:
    if not isinstance(code, str):
        raise TypeError(f"Currency code must be a string; got {type(code).__name__}.")
    up = code.strip().upper()
    if not up:
        raise ValueError("Currency code must not be empty.")
    return up


@dataclass(frozen=True, slots=True)
class Money:
    """A currency-tagged monetary amount.

    Parameters
    ----------
    amount:
        The numeric amount. Accepts ``int``, ``float``, ``Decimal``, or
        ``str``. Float inputs are routed via :class:`str` to avoid
        binary-float drift.
    currency:
        ISO 4217 three-letter code (case-insensitive; normalised to
        uppercase). Must be either a known ISO 4217 code or a code
        previously registered via
        :func:`simeng.currency.codes.register_custom`.

    Examples
    --------
    >>> from simeng.currency import Money
    >>> Money(100, "GBP") + Money("50.25", "GBP")
    Money(Decimal('150.25'), 'GBP')
    >>> Money(100, "GBP") * 3
    Money(Decimal('300'), 'GBP')
    """

    amount: Decimal
    currency: str

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Coerce and validate. Frozen dataclass requires object.__setattr__.
        object.__setattr__(self, "amount", _coerce_amount(self.amount))
        code = _normalise_code(self.currency)
        if not is_valid_currency(code):
            raise ValueError(
                f"Unknown currency code: {code!r}. "
                f"Use simeng.currency.register_custom({code!r}, decimals=...) "
                f"if intentional."
            )
        object.__setattr__(self, "currency", code)

    # ------------------------------------------------------------------
    # Rounding / quantisation
    # ------------------------------------------------------------------
    @property
    def decimals(self) -> int:
        """Canonical minor-unit decimal places for this currency."""
        return get_decimals(self.currency)

    def round_to_currency(self, rounding: str = ROUND_HALF_EVEN) -> "Money":
        """Return a new ``Money`` rounded to the currency's minor-unit
        precision. Defaults to banker's rounding (``ROUND_HALF_EVEN``)."""
        quant = Decimal(10) ** -self.decimals
        return Money(self.amount.quantize(quant, rounding=rounding), self.currency)

    # ------------------------------------------------------------------
    # Sign / absolute
    # ------------------------------------------------------------------
    def __neg__(self) -> "Money":
        return Money(-self.amount, self.currency)

    def __pos__(self) -> "Money":
        return self

    def __abs__(self) -> "Money":
        return Money(abs(self.amount), self.currency)

    def is_negative(self) -> bool:
        return self.amount < 0

    def is_zero(self) -> bool:
        return self.amount == 0

    # ------------------------------------------------------------------
    # Additive operators
    # ------------------------------------------------------------------
    def _check_same_currency(self, other: "Money", op: str) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(
                f"Cannot {op} {self.currency} and {other.currency}: "
                f"different currencies. Use .to(target, converter) to convert."
            )

    def __add__(self, other: object) -> "Money":
        if not isinstance(other, Money):
            return NotImplemented
        self._check_same_currency(other, "+")
        return Money(self.amount + other.amount, self.currency)

    def __radd__(self, other: object) -> "Money":
        # Permit sum([...], start=Money(0, "GBP")) without weird interactions;
        # but refuse bare int 0 which is what ``sum()`` defaults to, because
        # mixing currencies via sum() with default start is a common bug.
        if other == 0:
            raise TypeError(
                "Cannot add Money to bare 0; pass start=Money(0, currency) "
                "to sum() to accumulate Money values."
            )
        return NotImplemented

    def __sub__(self, other: object) -> "Money":
        if not isinstance(other, Money):
            return NotImplemented
        self._check_same_currency(other, "-")
        return Money(self.amount - other.amount, self.currency)

    # ------------------------------------------------------------------
    # Multiplicative operators
    # ------------------------------------------------------------------
    def __mul__(self, other: object) -> "Money":
        if isinstance(other, Money):
            raise TypeError(
                "Money * Money is not dimensionally meaningful. "
                "Multiply Money by a scalar (int/float/Decimal) instead."
            )
        if isinstance(other, bool):
            raise TypeError("Cannot multiply Money by bool.")
        if isinstance(other, (int, float, Decimal, str)):
            return Money(self.amount * _coerce_amount(other), self.currency)
        return NotImplemented

    def __rmul__(self, other: object) -> "Money":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Union["Money", float]:
        if isinstance(other, Money):
            self._check_same_currency(other, "/")
            if other.amount == 0:
                raise ZeroDivisionError("Division by zero Money.")
            return float(self.amount / other.amount)
        if isinstance(other, bool):
            raise TypeError("Cannot divide Money by bool.")
        if isinstance(other, (int, float, Decimal, str)):
            divisor = _coerce_amount(other)
            if divisor == 0:
                raise ZeroDivisionError("Division by zero.")
            return Money(self.amount / divisor, self.currency)
        return NotImplemented

    def __floordiv__(self, other: object) -> Union["Money", int]:
        if isinstance(other, Money):
            self._check_same_currency(other, "//")
            if other.amount == 0:
                raise ZeroDivisionError("Floor-division by zero Money.")
            return int(self.amount // other.amount)
        if isinstance(other, (int, float, Decimal, str)) and not isinstance(
            other, bool
        ):
            divisor = _coerce_amount(other)
            if divisor == 0:
                raise ZeroDivisionError("Floor-division by zero.")
            return Money(self.amount // divisor, self.currency)
        return NotImplemented

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.currency == other.currency and self.amount == other.amount

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.currency, self.amount))

    def _check_comparable(self, other: object, op: str) -> "Money":
        if not isinstance(other, Money):
            raise TypeError(
                f"Cannot compare Money with {type(other).__name__} using {op!r}."
            )
        self._check_same_currency(other, op)
        return other

    def __lt__(self, other: object) -> bool:
        other_m = self._check_comparable(other, "<")
        return self.amount < other_m.amount

    def __le__(self, other: object) -> bool:
        other_m = self._check_comparable(other, "<=")
        return self.amount <= other_m.amount

    def __gt__(self, other: object) -> bool:
        other_m = self._check_comparable(other, ">")
        return self.amount > other_m.amount

    def __ge__(self, other: object) -> bool:
        other_m = self._check_comparable(other, ">=")
        return self.amount >= other_m.amount

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def to(
        self,
        target: str,
        converter: "FXConverter",
        at: "datetime | None" = None,
    ) -> "Money":
        """Convert to ``target`` currency using ``converter.rate(...)``.

        Short-circuits to ``self`` (no-op) when the target currency
        already matches. Preserves full Decimal precision; call
        :meth:`round_to_currency` afterwards if you want to quantise
        to the target's minor-unit precision.

        Parameters
        ----------
        target:
            ISO 4217 code to convert to.
        converter:
            Any object satisfying :class:`simeng.currency.FXConverter`.
        at:
            Optional datetime passed through to the converter (for
            historical-rate lookups). Converters may ignore it.
        """
        target_code = _normalise_code(target)
        if target_code == self.currency:
            return self
        rate = converter.rate(self.currency, target_code, at=at)
        rate_d = _coerce_amount(rate)
        return Money(self.amount * rate_d, target_code)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Money({self.amount!r}, {self.currency!r})"

    def __str__(self) -> str:
        # Delayed import avoids a circular dependency at module load time.
        from simeng.currency.format import format_money

        return format_money(self)

    def __format__(self, spec: str) -> str:
        from simeng.currency.format import format_money

        return format_money(self, spec=spec)
