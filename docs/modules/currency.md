# Currency

`Money`, FX conversion, and locale-aware formatting.

## `Money`

```python
import simweave as sw

price = sw.Money(199, "GBP")
tax   = sw.Money(40, "GBP")
total = price + tax            # Money(239, "GBP")
```

`Money` enforces currency consistency. Adding `Money(1, "GBP")` and
`Money(1, "USD")` raises `CurrencyMismatchError`.

## FX conversion

```python
fx = sw.StaticFXConverter({("GBP", "USD"): 1.27})
usd = fx.convert(sw.Money(100, "GBP"), to="USD")     # Money(127, "USD")
```

For dynamic rates, use `CallableFXConverter(fn)` where `fn(base, quote)`
returns the rate.

## Custom currencies

```python
sw.register_custom("ZED", decimals=2)
sw.is_valid_currency("ZED")        # True
sw.unregister_custom("ZED")
```

## Locale-aware formatting (with `[intl]`)

```python
sw.format_money(sw.Money(1234.5, "EUR"), locale="de_DE")
# '1.234,50 €'
```

## API

::: simweave.currency
    options:
      show_root_heading: false
      show_source: true
