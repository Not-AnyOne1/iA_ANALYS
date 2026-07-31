"""Unit tests for indicators.py.

Every indicator is checked against a hand-computed expectation or a
mathematical property that must hold, not against whatever the code happens
to return — an indicator test that just records current output catches
nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import indicators
from market_data import Candle


def _ts(i: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)


def _c(i: int, high: float, low: float, close: float, open_: float | None = None) -> Candle:
    return Candle(timestamp=_ts(i), open=open_ if open_ is not None else close,
                  high=high, low=low, close=close, volume=1.0)


def _series(closes, spread: float = 1.0):
    """Candles whose highs/lows straddle each close by ``spread``."""
    return [_c(i, close + spread, close - spread, close) for i, close in enumerate(closes)]


# --------------------------------------------------------------------------- EMA

def test_ema_of_a_flat_series_is_that_value():
    assert indicators.ema([5.0] * 50, 10) == pytest.approx(5.0)


def test_ema_seeds_with_sma_when_series_is_exactly_the_period():
    # With len == period there is nothing to smooth, so EMA == SMA.
    assert indicators.ema([1.0, 2.0, 3.0, 4.0], 4) == pytest.approx(2.5)


def test_ema_matches_a_hand_computed_step():
    # SMA seed over [1,2,3] = 2.0; alpha = 2/4 = 0.5; next value 10:
    #   2.0 * 0.5 + 10 * 0.5 = 6.0
    assert indicators.ema([1.0, 2.0, 3.0, 10.0], 3) == pytest.approx(6.0)


def test_ema_returns_none_when_series_is_too_short():
    assert indicators.ema([1.0, 2.0], 5) is None


def test_ema_rejects_a_nonsense_period():
    with pytest.raises(ValueError):
        indicators.ema([1.0], 0)


def test_ema_reacts_faster_than_a_longer_ema():
    rising = [float(i) for i in range(100)]
    fast, slow = indicators.ema(rising, 10), indicators.ema(rising, 50)
    assert fast > slow  # shorter EMA tracks a rising series more closely


def test_ema_set_alignment_bullish():
    rising = _series([float(i) for i in range(1, 301)])
    assert indicators.ema_set(rising).alignment == "bullish"


def test_ema_set_alignment_bearish():
    falling = _series([float(i) for i in range(300, 0, -1)])
    assert indicators.ema_set(falling).alignment == "bearish"


def test_ema_set_alignment_unknown_when_series_too_short():
    short = _series([1.0] * 30)      # enough for EMA20, not EMA200
    result = indicators.ema_set(short)
    assert result.ema_20 is not None
    assert result.ema_200 is None
    assert result.alignment == "unknown"


# --------------------------------------------------------------------------- RSI

def test_rsi_is_100_for_an_unbroken_advance():
    assert indicators.rsi(_series([float(i) for i in range(1, 40)])) == pytest.approx(100.0)


def test_rsi_is_0_for_an_unbroken_decline():
    assert indicators.rsi(_series([float(i) for i in range(40, 1, -1)])) == pytest.approx(0.0)


def test_rsi_of_a_flat_series_is_neutral():
    # No gains and no losses: neither overbought nor oversold.
    assert indicators.rsi(_series([10.0] * 40)) == pytest.approx(50.0)


def test_rsi_stays_within_bounds():
    import random
    random.seed(7)
    closes = [100.0]
    for _ in range(200):
        closes.append(max(1.0, closes[-1] + random.uniform(-3, 3)))
    value = indicators.rsi(_series(closes))
    assert 0.0 <= value <= 100.0


def test_rsi_returns_none_when_too_short():
    assert indicators.rsi(_series([1.0] * 5), period=14) is None


# --------------------------------------------------------------------------- ATR

def test_true_range_uses_the_widest_of_the_three_measures():
    candles = [_c(0, 10, 9, 9.5), _c(1, 12, 11, 11.5)]
    # high-low = 1; |high - prev_close| = |12-9.5| = 2.5 (widest); |low-prev| = 1.5
    assert indicators.true_ranges(candles) == [pytest.approx(2.5)]


def test_atr_of_constant_range_candles_equals_that_range():
    # Every candle spans exactly 2.0 and closes at the same price, so every
    # true range is 2.0 and any smoothing of it is still 2.0.
    candles = [_c(i, 11, 9, 10) for i in range(40)]
    assert indicators.atr(candles) == pytest.approx(2.0)


def test_atr_is_never_negative():
    import random
    random.seed(3)
    candles = []
    price = 100.0
    for i in range(100):
        price += random.uniform(-2, 2)
        candles.append(_c(i, price + 1, price - 1, price))
    assert indicators.atr(candles) >= 0


def test_atr_returns_none_when_too_short():
    assert indicators.atr(_series([1.0] * 5), period=14) is None


def test_atr_percent_is_relative_to_price():
    candles = [_c(i, 101, 99, 100) for i in range(40)]
    # ATR is 2.0 on a 100 close -> 2%
    assert indicators.atr_percent(candles) == pytest.approx(2.0)


# -------------------------------------------------------------------- volatility

def test_volatility_unknown_without_enough_data():
    assert indicators.volatility_state(_series([1.0] * 5)) == "unknown"


def test_volatility_normal_for_a_steady_range():
    candles = [_c(i, 11, 9, 10) for i in range(200)]
    assert indicators.volatility_state(candles) == "normal"


def test_volatility_high_after_a_range_expansion():
    calm = [_c(i, 100.5, 99.5, 100) for i in range(150)]
    wild = [_c(150 + i, 110, 90, 100) for i in range(30)]
    assert indicators.volatility_state(calm + wild) == "high"


def test_volatility_low_after_a_range_contraction():
    wild = [_c(i, 110, 90, 100) for i in range(150)]
    calm = [_c(150 + i, 100.2, 99.8, 100) for i in range(30)]
    assert indicators.volatility_state(wild + calm) == "low"


# --------------------------------------------------------------------------- ADX

def test_adx_is_none_without_enough_candles():
    result = indicators.adx(_series([1.0] * 10))
    assert result.adx is None


def test_adx_is_high_and_plus_di_dominant_in_a_clean_uptrend():
    candles = [_c(i, 100 + i * 2 + 1, 100 + i * 2 - 1, 100 + i * 2) for i in range(60)]
    result = indicators.adx(candles)
    assert result.adx is not None
    assert result.plus_di > result.minus_di
    assert result.adx > 50          # a straight line is maximally trending
    assert result.strength == "very_strong"


def test_adx_minus_di_dominant_in_a_clean_downtrend():
    candles = [_c(i, 200 - i * 2 + 1, 200 - i * 2 - 1, 200 - i * 2) for i in range(60)]
    result = indicators.adx(candles)
    assert result.minus_di > result.plus_di


def test_adx_is_low_in_a_flat_market():
    candles = [_c(i, 101, 99, 100) for i in range(60)]
    result = indicators.adx(candles)
    assert result.adx is not None
    assert result.adx < 25
    assert result.strength in ("absent", "weak")


def test_adx_strength_bands():
    assert indicators.ADXResult(15, 1, 1).strength == "absent"
    assert indicators.ADXResult(22, 1, 1).strength == "weak"
    assert indicators.ADXResult(30, 1, 1).strength == "strong"
    assert indicators.ADXResult(60, 1, 1).strength == "very_strong"
    assert indicators.ADXResult(None, None, None).strength == "unknown"


# --------------------------------------------------------------------- determinism

def test_every_indicator_is_deterministic():
    candles = _series([100 + (i % 7) for i in range(300)])
    assert indicators.ema_set(candles) == indicators.ema_set(candles)
    assert indicators.rsi(candles) == indicators.rsi(candles)
    assert indicators.atr(candles) == indicators.atr(candles)
    assert indicators.adx(candles) == indicators.adx(candles)
    assert indicators.volatility_state(candles) == indicators.volatility_state(candles)


def test_indicators_never_fabricate_on_empty_input():
    assert indicators.ema_set([]).ema_20 is None
    assert indicators.rsi([]) is None
    assert indicators.atr([]) is None
    assert indicators.atr_percent([]) is None
    assert indicators.adx([]).adx is None
    assert indicators.volatility_state([]) == "unknown"
