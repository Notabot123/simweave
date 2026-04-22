"""Tests for :mod:`simeng.currency`.

Covers the full Money / FX / format contract: construction, rounding,
arithmetic, comparison, conversion, registry, formatting, and the
edge cases documented in ``CURRENCY_DESIGN.md``.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from simeng.currency import (
    CallableFXConverter,
    CurrencyMismatchError,
    FXConverter,
    Money,
    StaticFXConverter,
    format_money,
    get_decimals,
    is_valid_currency,
    list_codes,
    register_custom,
    unregister_custom,
)


# =====================================================================
# Construction & coercion
# =====================================================================


def test_construct_from_int():
    m = Money(100, "GBP")
    assert m.amount == Decimal("100")
    assert m.currency == "GBP"


def test_construct_from_str():
    m = Money("100.50", "USD")
    assert m.amount == Decimal("100.50")


def test_construct_from_decimal_preserves_precision():
    # Use a 4-decimal ISO currency so we don't need register_custom here.
    m = Money(Decimal("1.2345"), "CLF")
    assert m.amount == Decimal("1.2345")


def test_construct_from_decimal_high_precision_via_custom():
    register_custom("XBTC", decimals=8)
    try:
        m = Money(Decimal("1.23456789"), "XBTC")
        assert m.amount == Decimal("1.23456789")
    finally:
        unregister_custom("XBTC")


def test_construct_from_float_avoids_binary_drift():
    # 0.1 in binary is not exact; routing via str() avoids the horror show.
    m = Money(0.1, "GBP")
    assert m.amount == Decimal("0.1")


def test_construct_refuses_bool():
    # bool is an int subclass — explicit refusal prevents "True pounds" bugs.
    with pytest.raises(TypeError):
        Money(True, "GBP")


def test_construct_refuses_unknown_currency():
    with pytest.raises(ValueError):
        Money(100, "ZZZ")


def test_construct_refuses_non_numeric_amount():
    with pytest.raises(TypeError):
        Money(object(), "GBP")


def test_construct_refuses_unparseable_string():
    with pytest.raises(ValueError):
        Money("not a number", "GBP")


def test_currency_normalised_to_uppercase():
    m = Money(10, "gbp")
    assert m.currency == "GBP"


def test_currency_whitespace_stripped():
    m = Money(10, "  usd  ")
    assert m.currency == "USD"


def test_currency_code_must_be_string():
    with pytest.raises(TypeError):
        Money(10, 123)  # type: ignore[arg-type]


def test_currency_code_empty_rejected():
    with pytest.raises(ValueError):
        Money(10, "")


# =====================================================================
# Registry
# =====================================================================


def test_is_valid_currency_iso():
    assert is_valid_currency("GBP")
    assert is_valid_currency("jpy")  # case-insensitive
    assert not is_valid_currency("ZZZ")


def test_get_decimals_iso():
    assert get_decimals("GBP") == 2
    assert get_decimals("JPY") == 0
    assert get_decimals("KWD") == 3
    assert get_decimals("CLF") == 4


def test_get_decimals_unknown_raises():
    with pytest.raises(KeyError):
        get_decimals("ZZZ")


def test_register_custom_allows_crypto_codes():
    register_custom("DOGE", decimals=8)
    try:
        assert is_valid_currency("DOGE")
        assert get_decimals("DOGE") == 8
        m = Money("0.00000001", "DOGE")
        assert m.amount == Decimal("0.00000001")
    finally:
        unregister_custom("DOGE")


def test_register_custom_refuses_to_shadow_iso():
    with pytest.raises(ValueError):
        register_custom("USD", decimals=2)


def test_register_custom_rejects_negative_decimals():
    with pytest.raises(ValueError):
        register_custom("XXX", decimals=-1)


def test_register_custom_rejects_empty_code():
    with pytest.raises(ValueError):
        register_custom("", decimals=2)


def test_unregister_custom_is_noop_for_missing():
    unregister_custom("NEVER_REGISTERED")  # should not raise


def test_list_codes_returns_sorted_tuple_including_iso():
    codes = list_codes()
    assert "GBP" in codes
    assert "USD" in codes
    assert codes == tuple(sorted(codes))


def test_list_codes_exclude_custom():
    register_custom("TESTX", decimals=2)
    try:
        with_custom = list_codes(include_custom=True)
        without_custom = list_codes(include_custom=False)
        assert "TESTX" in with_custom
        assert "TESTX" not in without_custom
    finally:
        unregister_custom("TESTX")


# =====================================================================
# Sign / absolute
# =====================================================================


def test_negation():
    m = Money(100, "GBP")
    assert (-m).amount == Decimal("-100")


def test_positive_is_identity():
    m = Money(100, "GBP")
    assert +m is m


def test_abs_for_negative():
    m = Money(-50, "GBP")
    assert abs(m).amount == Decimal("50")


def test_is_negative_and_is_zero():
    assert Money(-1, "GBP").is_negative()
    assert not Money(1, "GBP").is_negative()
    assert Money(0, "GBP").is_zero()


def test_negative_money_allowed():
    # Refunds, debts, signed cashflows are legitimate.
    m = Money("-42.50", "USD")
    assert m.amount == Decimal("-42.50")


# =====================================================================
# Addition / subtraction
# =====================================================================


def test_add_same_currency():
    result = Money(100, "GBP") + Money("50.25", "GBP")
    assert result.amount == Decimal("150.25")
    assert result.currency == "GBP"


def test_add_cross_currency_raises():
    with pytest.raises(CurrencyMismatchError):
        _ = Money(100, "GBP") + Money(50, "USD")


def test_currency_mismatch_is_type_error():
    # Subclass relationship keeps legacy `except TypeError` working.
    with pytest.raises(TypeError):
        _ = Money(100, "GBP") + Money(50, "USD")


def test_subtract_same_currency():
    result = Money(100, "GBP") - Money("25.50", "GBP")
    assert result.amount == Decimal("74.50")


def test_subtract_cross_currency_raises():
    with pytest.raises(CurrencyMismatchError):
        _ = Money(100, "GBP") - Money(50, "USD")


def test_add_to_non_money_returns_not_implemented():
    # Python converts NotImplemented to TypeError at the operator level.
    with pytest.raises(TypeError):
        _ = Money(100, "GBP") + 100


def test_sum_with_explicit_start_works():
    values = [Money(10, "GBP"), Money(20, "GBP"), Money(30, "GBP")]
    total = sum(values, start=Money(0, "GBP"))
    assert total.amount == Decimal("60")


def test_sum_without_start_raises_explicit_error():
    # The default start=0 would silently break — we refuse it loudly.
    values = [Money(10, "GBP"), Money(20, "GBP")]
    with pytest.raises(TypeError, match="bare 0"):
        sum(values)  # type: ignore[arg-type]


# =====================================================================
# Multiplication / division
# =====================================================================


def test_multiply_by_scalar_int():
    assert (Money(100, "GBP") * 3).amount == Decimal("300")


def test_multiply_by_scalar_float_via_str_routing():
    # 0.1 * 100 in float is 10.000000000000002 — we should get clean 10.
    m = Money(100, "GBP") * 0.1
    assert m.amount == Decimal("10.0")


def test_multiply_by_scalar_decimal():
    assert (Money(100, "GBP") * Decimal("1.5")).amount == Decimal("150.0")


def test_rmul_scalar_on_left():
    assert (3 * Money(100, "GBP")).amount == Decimal("300")


def test_multiply_money_by_money_refused():
    # Square-pounds are not a thing.
    with pytest.raises(TypeError):
        _ = Money(100, "GBP") * Money(2, "GBP")


def test_multiply_by_bool_refused():
    with pytest.raises(TypeError):
        _ = Money(100, "GBP") * True


def test_divide_by_scalar():
    assert (Money(100, "GBP") / 4).amount == Decimal("25")


def test_divide_by_zero_scalar_raises():
    with pytest.raises(ZeroDivisionError):
        _ = Money(100, "GBP") / 0


def test_divide_money_by_money_same_currency_returns_float():
    ratio = Money(100, "GBP") / Money(50, "GBP")
    assert isinstance(ratio, float)
    assert ratio == 2.0


def test_divide_money_by_money_cross_currency_raises():
    with pytest.raises(CurrencyMismatchError):
        _ = Money(100, "GBP") / Money(50, "USD")


def test_divide_money_by_zero_money_raises():
    with pytest.raises(ZeroDivisionError):
        _ = Money(100, "GBP") / Money(0, "GBP")


def test_divide_by_bool_refused():
    with pytest.raises(TypeError):
        _ = Money(100, "GBP") / True


def test_floordiv_money_by_scalar():
    m = Money(100, "GBP") // 3
    assert isinstance(m, Money)
    assert m.amount == Decimal("33")


def test_floordiv_money_by_money_same_currency_returns_int():
    q = Money(100, "GBP") // Money(30, "GBP")
    assert isinstance(q, int)
    assert q == 3


def test_floordiv_by_zero_money_raises():
    with pytest.raises(ZeroDivisionError):
        _ = Money(100, "GBP") // Money(0, "GBP")


# =====================================================================
# Equality, hash, comparison
# =====================================================================


def test_equality_requires_same_currency():
    assert Money(100, "GBP") == Money(100, "GBP")
    assert Money(100, "GBP") != Money(100, "USD")


def test_equality_with_non_money_is_not_equal():
    assert Money(100, "GBP") != 100
    assert Money(100, "GBP") != "100 GBP"


def test_hash_matches_equality():
    a = Money(100, "GBP")
    b = Money(100, "GBP")
    assert hash(a) == hash(b)
    # Different currencies differ.
    assert hash(a) != hash(Money(100, "USD"))


def test_hash_usable_in_set():
    s = {Money(100, "GBP"), Money(100, "GBP"), Money(100, "USD")}
    assert len(s) == 2


def test_comparison_same_currency():
    assert Money(100, "GBP") < Money(200, "GBP")
    assert Money(100, "GBP") <= Money(100, "GBP")
    assert Money(200, "GBP") > Money(100, "GBP")
    assert Money(200, "GBP") >= Money(200, "GBP")


def test_comparison_cross_currency_raises():
    with pytest.raises(CurrencyMismatchError):
        _ = Money(100, "GBP") < Money(200, "USD")


def test_comparison_with_non_money_raises():
    with pytest.raises(TypeError):
        _ = Money(100, "GBP") < 200


# =====================================================================
# Rounding / quantisation
# =====================================================================


def test_round_to_currency_default_bankers():
    # ROUND_HALF_EVEN: 2.5 -> 2, 3.5 -> 4 at whole-pence boundaries.
    assert Money("1.005", "GBP").round_to_currency().amount == Decimal("1.00")
    assert Money("1.015", "GBP").round_to_currency().amount == Decimal("1.02")


def test_round_to_currency_zero_decimals():
    # JPY has no minor units.
    assert Money("1234.5", "JPY").round_to_currency().amount == Decimal("1234")


def test_round_to_currency_three_decimals():
    # KWD is a 3-decimal currency.
    assert Money("1.2345", "KWD").round_to_currency().amount == Decimal("1.234")


def test_decimals_property():
    assert Money(1, "GBP").decimals == 2
    assert Money(1, "JPY").decimals == 0
    assert Money(1, "KWD").decimals == 3


# =====================================================================
# Formatting
# =====================================================================


def test_format_default_ascii():
    assert format_money(Money("1234.5", "GBP")) == "GBP 1,234.50"


def test_format_default_rounds_to_currency_precision():
    # 1234.567 with GBP 2dp -> rounded by banker's rounding.
    assert format_money(Money("1234.567", "GBP")) == "GBP 1,234.57"


def test_format_zero_decimal_currency():
    assert format_money(Money("1234", "JPY")) == "JPY 1,234"


def test_format_three_decimal_currency():
    assert format_money(Money("1.234", "KWD")) == "KWD 1.234"


def test_format_negative_amount():
    assert format_money(Money("-42.50", "GBP")) == "-GBP 42.50"


def test_format_spec_r_rounds():
    m = Money("1234.567", "GBP")
    assert f"{m:r}" == "GBP 1,234.57"


def test_format_spec_raw():
    m = Money("1234.56789", "GBP")
    # "raw" means full-precision decimal text, no grouping, no tag.
    assert f"{m:raw}" == "1234.56789"


def test_format_unknown_spec_raises():
    with pytest.raises(ValueError):
        format(Money(1, "GBP"), "weird")


def test_str_uses_default_format():
    assert str(Money("100", "GBP")) == "GBP 100.00"


# =====================================================================
# FX conversion — StaticFXConverter
# =====================================================================


def test_static_fx_forward_rate():
    fx = StaticFXConverter({("GBP", "USD"): "1.27"})
    assert fx.rate("GBP", "USD") == Decimal("1.27")


def test_static_fx_inverse_rate_auto():
    # Registering only GBP->USD should let us look up USD->GBP as 1/1.27.
    fx = StaticFXConverter({("GBP", "USD"): "1.27"})
    inv = fx.rate("USD", "GBP")
    assert inv == Decimal("1") / Decimal("1.27")


def test_static_fx_identity_rate_is_one():
    fx = StaticFXConverter({("GBP", "USD"): "1.27"})
    assert fx.rate("GBP", "GBP") == Decimal("1")


def test_static_fx_missing_rate_raises():
    fx = StaticFXConverter({("GBP", "USD"): "1.27"})
    with pytest.raises(KeyError):
        fx.rate("GBP", "EUR")


def test_static_fx_case_insensitive_keys():
    fx = StaticFXConverter({("gbp", "usd"): "1.27"})
    assert fx.rate("GBP", "USD") == Decimal("1.27")


def test_static_fx_ignores_explicit_identity_entry():
    # Identity rates are always 1 — explicit entry is silently ignored.
    fx = StaticFXConverter({("GBP", "GBP"): "5", ("GBP", "USD"): "1.27"})
    assert fx.rate("GBP", "GBP") == Decimal("1")


def test_static_fx_float_rate_coerced_via_str():
    fx = StaticFXConverter({("GBP", "USD"): 1.27})
    # 1.27 as float -> str -> Decimal('1.27'), no binary drift.
    assert fx.rate("GBP", "USD") == Decimal("1.27")


def test_static_fx_protocol_runtime_check():
    fx = StaticFXConverter({("GBP", "USD"): "1.27"})
    assert isinstance(fx, FXConverter)


# =====================================================================
# FX conversion — CallableFXConverter
# =====================================================================


def test_callable_fx_wraps_function():
    def lookup(src, tgt, at=None):
        return 1.27 if (src, tgt) == ("GBP", "USD") else 1.0

    fx = CallableFXConverter(lookup)
    assert fx.rate("GBP", "USD") == Decimal("1.27")


def test_callable_fx_identity_shortcircuit():
    sentinel = []

    def lookup(src, tgt, at=None):
        sentinel.append((src, tgt))
        return 1.0

    fx = CallableFXConverter(lookup)
    assert fx.rate("GBP", "GBP") == Decimal("1")
    # Short-circuit: the wrapped function should never be called.
    assert sentinel == []


def test_callable_fx_satisfies_protocol():
    fx = CallableFXConverter(lambda s, t, at=None: 1.0)
    assert isinstance(fx, FXConverter)


# =====================================================================
# Money.to() conversion
# =====================================================================


def test_money_to_same_currency_is_shortcircuit():
    fx = StaticFXConverter({})
    m = Money(100, "GBP")
    # Should not call fx.rate at all.
    assert m.to("GBP", fx) is m


def test_money_to_cross_currency():
    fx = StaticFXConverter({("GBP", "USD"): "1.27"})
    m = Money(100, "GBP").to("USD", fx)
    assert m.currency == "USD"
    assert m.amount == Decimal("127.00")


def test_money_to_uses_inverse_rate_automatically():
    fx = StaticFXConverter({("GBP", "USD"): "1.25"})
    m = Money(125, "USD").to("GBP", fx)
    # 125 * (1 / 1.25) = 100
    assert m.amount == Decimal("100")


def test_money_to_preserves_full_precision():
    # Conversion does NOT auto-round; caller must ask for it.
    fx = StaticFXConverter({("GBP", "USD"): "1.2345"})
    m = Money(100, "GBP").to("USD", fx)
    # 100 * 1.2345 = 123.4500
    assert m.amount == Decimal("123.4500")


def test_money_to_then_round_to_currency():
    fx = StaticFXConverter({("GBP", "USD"): "1.2345"})
    m = Money(100, "GBP").to("USD", fx).round_to_currency()
    # Rounded to 2dp with banker's rounding.
    assert m.amount == Decimal("123.45")


def test_money_to_unknown_currency_rejected():
    fx = StaticFXConverter({("GBP", "ZZZ"): "1.0"})
    with pytest.raises(ValueError):
        # Decoded from the target code validation path.
        _ = Money(100, "GBP").to("ZZZ", fx)


# =====================================================================
# Immutability / hashability sanity
# =====================================================================


def test_money_is_frozen():
    m = Money(100, "GBP")
    with pytest.raises(Exception):
        m.amount = Decimal("999")  # type: ignore[misc]


def test_money_repr_roundtrip_hint():
    # repr is cosmetic but should include both amount and currency.
    r = repr(Money(100, "GBP"))
    assert "100" in r
    assert "GBP" in r
