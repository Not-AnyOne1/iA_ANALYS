"""Entry location and extension: is this a good trade from THIS price?"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

import entry_quality as eq
import indicators
from market_data import Candle


def _ts(i: int) -> datetime:
    return datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(hours=i)


def _candle(i: int, close: float, half_range: float = 0.5) -> Candle:
    return Candle(timestamp=_ts(i), open=close, high=close + half_range,
                  low=close - half_range, close=close, volume=1.0)


def _history(n: int = 260):
    """A ranging market: oscillation with no trend, so 'normal' moves are
    the oscillation's own swings."""
    return [_candle(i, 100.0 + 2.0 * math.sin(i / 5.0)) for i in range(n)]


def _then(candles, steps):
    """Append a run of closes moving by ``step`` each bar."""
    out = list(candles)
    price = out[-1].close
    for step in steps:
        price += step
        out.append(_candle(len(out), price))
    return out


# ------------------------------------------------------------------ location

@pytest.mark.parametrize("entry,factor,label", [
    (100.0, 1.0, "favourable"),     # at the range low
    (105.0, 1.0, "neutral"),        # equilibrium
    (110.0, 0.5, "unfavourable"),   # at the range high
])
def test_long_location_is_graded_through_the_range(entry, factor, label):
    loc = eq.assess_location("long", entry, 100.0, 110.0)
    assert loc.factor == factor
    assert loc.label == label


def test_range_high_long_is_penalised():
    """Phase 10 #6. Buying the top of the range costs the most a location can."""
    top = eq.assess_location("long", 110.0, 100.0, 110.0)
    mid = eq.assess_location("long", 105.0, 100.0, 110.0)
    assert top.factor < mid.factor
    assert top.factor == 0.5
    assert top.zone == "upper"


def test_range_low_short_is_penalised():
    """Phase 10 #7. The mirror image: selling the bottom of the range."""
    bottom = eq.assess_location("short", 100.0, 100.0, 110.0)
    top = eq.assess_location("short", 110.0, 100.0, 110.0)
    assert bottom.factor == 0.5 and bottom.label == "unfavourable"
    assert top.factor == 1.0 and top.label == "favourable"


@pytest.mark.parametrize("position", [0.0, 0.2, 0.5, 0.6, 0.75, 0.9, 1.0])
def test_long_and_short_are_exact_mirrors(position):
    long_ = eq.assess_location("long", 100.0 + 10.0 * position, 100.0, 110.0)
    short = eq.assess_location("short", 100.0 + 10.0 * (1 - position), 100.0, 110.0)
    assert long_.factor == short.factor
    assert long_.label == short.label


def test_location_is_graded_not_a_ban():
    """No "never buy the high" rule: the factor falls smoothly, one hundredth
    of the range at a time, and never below half."""
    factors = [eq.assess_location("long", 100.0 + i / 10.0, 100.0, 110.0).factor
               for i in range(0, 101)]
    assert factors == sorted(factors, reverse=True)
    assert min(factors) == 0.5 and max(factors) == 1.0
    assert max(a - b for a, b in zip(factors, factors[1:])) <= 0.011


def test_beyond_the_range_is_no_better_than_its_edge():
    above = eq.assess_location("long", 125.0, 100.0, 110.0)
    assert above.factor == 0.5
    assert above.zone == "above range"
    assert above.position > 1                # the true position is still reported


@pytest.mark.parametrize("args", [
    (None, 105.0, 100.0, 110.0),                 # no direction
    ("long", None, 100.0, 110.0),                # no entry
    ("long", 105.0, None, None),                 # no dealing range
    ("long", 105.0, 110.0, 110.0),               # zero-width range
    ("long", float("nan"), 100.0, 110.0),
    ("long", 105.0, 100.0, float("inf")),
])
def test_unknown_location_fails_safe(args):
    loc = eq.assess_location(*args)
    assert loc.label == "unknown"
    assert loc.factor == 1.0
    assert "unknown" in loc.detail


# ----------------------------------------------------------------- extension

def test_a_normal_move_costs_nothing():
    flat = _then(_history(), [0.0] * 14)
    ext = eq.assess_extension("long", flat)
    assert ext.label == "normal"
    assert ext.factor == 1.0


def test_strongly_extended_long_is_penalised():
    """Phase 10 #8. A long after the biggest run in the instrument's recent
    history is a late entry."""
    run_up = _then(_history(), [1.0] * 14)
    ext = eq.assess_extension("long", run_up)
    assert ext.label == "strong"
    assert ext.percentile > 0.9
    assert ext.factor < 0.6
    assert ext.move_atr > 0


def test_strongly_extended_short_is_penalised():
    """Phase 10 #9. The mirror image."""
    sell_off = _then(_history(), [-1.0] * 14)
    ext = eq.assess_extension("short", sell_off)
    assert ext.label == "strong"
    assert ext.factor < 0.6


def test_extension_is_not_a_reversal_signal():
    """An extended run penalises trades in ITS direction only. A short after
    a run-up is not late — and nothing here ever proposes the short."""
    run_up = _then(_history(), [1.0] * 14)
    assert eq.assess_extension("short", run_up).factor == 1.0
    assert eq.assess_extension("short", run_up).percentile == 0.0


def test_a_bigger_run_is_never_a_better_entry():
    factors = [eq.assess_extension("long", _then(_history(), [step] * 14)).factor
               for step in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)]
    assert factors == sorted(factors, reverse=True)


def test_extension_uses_no_future_candles():
    """The ATR at each bar depends only on bars up to it, so the assessment
    of a prefix is the same however much data exists after it."""
    series = _then(_history(), [1.0] * 14 + [-0.5] * 20)
    for k in (150, 200, 274):
        assert indicators.atr_series(series[:k]) == indicators.atr_series(series)[:k]
    # What happened after the entry cannot change how late the entry was.
    after_crash = series[:274] + [_candle(274 + i, 50.0) for i in range(20)]
    assert eq.assess_extension("long", series[:274]) == \
        eq.assess_extension("long", after_crash[:274])


def test_atr_series_ends_at_atr():
    series = _history()
    assert indicators.atr_series(series)[-1] == pytest.approx(indicators.atr(series))
    assert indicators.atr_series(series)[:14] == [None] * 14


@pytest.mark.parametrize("candles,why", [
    ([], "no candles"),
    (_history(20), "too little history to define normal"),
])
def test_unknown_extension_fails_safe(candles, why):
    ext = eq.assess_extension("long", candles)
    assert ext.label == "unknown", why
    assert ext.factor == 1.0


def test_malformed_candles_do_not_produce_nan():
    series = _then(_history(), [1.0] * 14)
    series[100] = Candle(timestamp=_ts(100), open=float("nan"), high=float("nan"),
                         low=float("nan"), close=float("nan"), volume=1.0)
    ext = eq.assess_extension("long", series)
    assert math.isfinite(ext.factor)
    assert 0.5 <= ext.factor <= 1.0


# ------------------------------------------------------------------- both

def test_entry_factor_is_the_product_and_bounded():
    worst = eq.assess("long", 110.0, 100.0, 110.0, _then(_history(), [1.0] * 14))
    best = eq.assess("long", 100.0, 100.0, 110.0, _then(_history(), [0.0] * 14))
    assert worst.factor == pytest.approx(worst.location.factor * worst.extension.factor)
    assert 0.25 <= worst.factor < 0.5
    assert best.factor == 1.0


@pytest.mark.parametrize("direction", ["long", "short"])
def test_every_factor_is_bounded_and_finite(direction):
    series = _history()
    for steps in ([0.0] * 14, [1.0] * 14, [-1.0] * 14, [3.0] * 14):
        for entry in (90.0, 100.0, 105.0, 110.0, 130.0):
            q = eq.assess(direction, entry, 100.0, 110.0, _then(series, steps))
            for value in (q.factor, q.location.factor, q.extension.factor):
                assert math.isfinite(value)
                assert 0.25 <= q.factor <= 1.0
                assert 0.5 <= q.location.factor <= 1.0
                assert 0.5 <= q.extension.factor <= 1.0
