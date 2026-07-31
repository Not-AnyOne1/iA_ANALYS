"""Unit tests for structure_engine.py (RFC-005).

Entirely offline and deterministic: every test hand-builds a CandleSeries
and asserts on the exact SwingPoint/StructureEvent/StructureAnalysis
returned — no network, no stubbing, no randomness anywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_data import Candle, CandleSeries, Timeframe
from structure_engine import (
    StructureEngine,
    StructureEventType,
    SwingType,
    TrendDirection,
)


# --------------------------------------------------------------------------- helpers

def _ts(i: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)


def _candles(rows) -> CandleSeries:
    """rows: list of (high, low, close) tuples, one per candle index."""
    candles = tuple(
        Candle(timestamp=_ts(i), open=close, high=high, low=low, close=close, volume=1.0)
        for i, (high, low, close) in enumerate(rows)
    )
    return CandleSeries(symbol="XAUUSD", timeframe=Timeframe.M1, provider="test", candles=candles)


# --------------------------------------------------------------------------- swing detection

def test_swing_high_confirmed_with_lookback_one():
    series = _candles([(1, 0, 0.5), (5, 0, 2.5), (1, 0, 0.5)])
    result = StructureEngine(swing_lookback=1).analyze(series)

    highs = [s for s in result.swing_points if s.type == SwingType.HIGH]
    assert len(highs) == 1
    assert highs[0].index == 1
    assert highs[0].price == 5
    assert highs[0].strength == 1


def test_swing_low_confirmed_with_lookback_one():
    series = _candles([(10, 5, 7.5), (10, 1, 5.5), (10, 5, 7.5)])
    result = StructureEngine(swing_lookback=1).analyze(series)

    lows = [s for s in result.swing_points if s.type == SwingType.LOW]
    assert len(lows) == 1
    assert lows[0].index == 1
    assert lows[0].price == 1
    assert lows[0].strength == 1


def test_swing_at_series_edge_is_never_confirmed():
    # A local extreme at index 0 or the last index can never be confirmed —
    # the neighbors needed to confirm it don't exist in the given data.
    series = _candles([(5, 0, 2.5), (1, 0, 0.5), (5, 0, 2.5)])
    result = StructureEngine(swing_lookback=1).analyze(series)

    assert [s for s in result.swing_points if s.type == SwingType.HIGH] == []


def test_swing_strength_extends_beyond_minimum_lookback():
    # A 5-candle spike: qualifies as a 1-bar AND a 2-bar fractal.
    series = _candles([(2, 0, 1), (2, 0, 1), (10, 0, 5), (2, 0, 1), (2, 0, 1)])
    result = StructureEngine(swing_lookback=1).analyze(series)

    highs = [s for s in result.swing_points if s.type == SwingType.HIGH]
    assert len(highs) == 1
    assert highs[0].index == 2
    assert highs[0].strength == 2


def test_default_swing_lookback_is_two():
    # Under lookback=1 this confirms; under the default (2) it can't, since
    # only 3 candles exist and radius is capped by series length.
    series = _candles([(1, 0, 0.5), (5, 0, 2.5), (1, 0, 0.5)])

    assert StructureEngine(swing_lookback=1).analyze(series).swing_points != ()
    assert StructureEngine().analyze(series).swing_points == ()


def test_invalid_swing_lookback_raises():
    try:
        StructureEngine(swing_lookback=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for swing_lookback=0")


# --------------------------------------------------------------------------- trend classification

# lookback=1 fixtures below share the same shape: swing highs confirmed at
# indices 1,3,5 and swing lows confirmed at indices 2,4 — only the actual
# price values differ per scenario.

def test_trend_bullish_on_higher_highs_and_higher_lows():
    rows = [
        (10, 8, 9), (12, 11, 11.5), (9, 5, 6), (14, 10, 11),
        (9, 7, 8), (16, 13, 13.5), (10, 6, 8),
    ]
    result = StructureEngine(swing_lookback=1).analyze(_candles(rows))
    assert result.trend == TrendDirection.BULLISH


def test_trend_bearish_on_lower_highs_and_lower_lows():
    rows = [
        (9, 8, 8.5), (16, 11, 13.5), (5, 9, 7), (14, 10, 12),
        (5, 6, 5.5), (12, 13, 12.5), (9, 7, 8),
    ]
    result = StructureEngine(swing_lookback=1).analyze(_candles(rows))
    assert result.trend == TrendDirection.BEARISH


def test_trend_ranging_on_mixed_signals():
    # Higher highs (12,14,16) but lower lows (9,6) — mixed, so RANGING.
    rows = [
        (10, 8, 9), (12, 11, 11.5), (9, 9, 9), (14, 10, 11),
        (9, 6, 7), (16, 13, 13.5), (10, 7, 8),
    ]
    result = StructureEngine(swing_lookback=1).analyze(_candles(rows))
    assert result.trend == TrendDirection.RANGING


def test_trend_unknown_with_too_few_swings():
    series = _candles([(1, 0, 0.5), (5, 0, 2.5), (1, 0, 0.5)])
    result = StructureEngine(swing_lookback=1).analyze(series)
    assert result.trend == TrendDirection.UNKNOWN


def test_trend_unknown_on_flat_market():
    rows = [(10, 8, 9)] * 10
    result = StructureEngine(swing_lookback=1).analyze(_candles(rows))
    assert result.trend == TrendDirection.UNKNOWN
    assert result.swing_points == ()
    assert result.events == ()


# --------------------------------------------------------------------------- BOS / CHoCH

def _established_bullish_rows():
    """8 candles (idx0-7) establishing a BULLISH trend via swings at
    idx1(12), idx3(14), idx5(16) [highs] and idx2(5), idx4(7) [lows],
    with idx6 deliberately shaped to avoid becoming a swing itself.
    lookback=1 throughout.
    """
    return [
        (10, 8, 9),        # idx0
        (12, 11, 11.5),    # idx1 -> swing high 12
        (9, 5, 6),         # idx2 -> swing low 5
        (14, 10, 11),      # idx3 -> swing high 14
        (9, 7, 8),         # idx4 -> swing low 7
        (16, 13, 13.5),    # idx5 -> swing high 16
        (15, 13.5, 14),    # idx6 -> not a swing (by construction)
    ]


def test_bos_on_bullish_continuation_break():
    rows = _established_bullish_rows() + [(20, 15, 17)]  # idx7: close 17 > 16
    result = StructureEngine(swing_lookback=1).analyze(_candles(rows))

    assert len(result.events) == 1
    event = result.events[0]
    assert event.type == StructureEventType.BOS
    assert event.direction == TrendDirection.BULLISH
    assert event.candle_index == 7
    assert event.break_price == 17
    assert event.broken_swing.price == 16
    assert event.broken_swing.index == 5
    assert result.last_event is event


def test_choch_on_bearish_reversal_break():
    rows = _established_bullish_rows() + [(8, 3, 4)]  # idx7: close 4 < 7
    result = StructureEngine(swing_lookback=1).analyze(_candles(rows))

    assert len(result.events) == 1
    event = result.events[0]
    assert event.type == StructureEventType.CHOCH
    assert event.direction == TrendDirection.BEARISH
    assert event.candle_index == 7
    assert event.break_price == 4
    assert event.broken_swing.price == 7
    assert event.broken_swing.index == 4
    assert result.last_event is event


def test_events_are_chronologically_ordered_and_last_event_is_the_latest():
    rows = (
        _established_bullish_rows()
        + [(20, 15, 17)]  # idx7: BOS (bullish break of 16)
        + [(18, 5, 6)]    # idx8: bearish break of active_low(7) -> CHoCH
    )
    result = StructureEngine(swing_lookback=1).analyze(_candles(rows))

    assert len(result.events) == 2
    assert result.events[0].type == StructureEventType.BOS
    assert result.events[0].candle_index == 7
    assert result.events[1].type == StructureEventType.CHOCH
    assert result.events[1].candle_index == 8
    assert result.last_event is result.events[1]
    # Chronological ordering invariant, not just this fixture's shape.
    assert [e.candle_index for e in result.events] == sorted(e.candle_index for e in result.events)


def test_broken_swing_is_retired_and_does_not_retrigger():
    # After idx7's BOS retires the swing at 16, later candles staying above
    # 16 must not produce a second event against the same swing.
    rows = _established_bullish_rows() + [(20, 15, 17), (19, 16, 17.5)]
    result = StructureEngine(swing_lookback=1).analyze(_candles(rows))

    assert len(result.events) == 1  # only the original BOS, not a repeat


# --------------------------------------------------------------------------- edge cases

def test_empty_series_returns_unknown_and_no_swings_or_events():
    series = CandleSeries(symbol="XAUUSD", timeframe=Timeframe.M1, provider="test", candles=())
    result = StructureEngine().analyze(series)

    assert result.trend == TrendDirection.UNKNOWN
    assert result.swing_points == ()
    assert result.events == ()
    assert result.last_event is None


def test_insufficient_candles_for_any_swing_returns_unknown():
    series = _candles([(10, 8, 9), (11, 9, 10)])  # only 2 candles
    result = StructureEngine().analyze(series)

    assert result.trend == TrendDirection.UNKNOWN
    assert result.swing_points == ()
    assert result.events == ()


def test_result_carries_through_symbol_and_timeframe():
    series = CandleSeries(
        symbol="EURUSD", timeframe=Timeframe.H4, provider="test",
        candles=_candles([(10, 8, 9), (11, 9, 10)]).candles,
    )
    result = StructureEngine().analyze(series)

    assert result.symbol == "EURUSD"
    assert result.timeframe == Timeframe.H4
