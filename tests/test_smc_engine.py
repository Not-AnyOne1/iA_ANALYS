"""Unit tests for smc_engine.py (RFC-006).

Entirely offline and deterministic: every test hand-builds candles/swings/
events (or, for a couple of integration tests, feeds a hand-built candle
series through the real StructureEngine first) and asserts on the exact
SMC objects returned. Most tests call the engine's detector methods
directly (each detector is exercised in isolation, the same style used in
test_structure_engine.py) so each test only needs to construct the minimal
input that specific rule cares about.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from market_data import Candle, CandleSeries, Timeframe
from smc_engine import LiquiditySide, OrderBlock, SMCEngine
from structure_engine import (
    StructureAnalysis,
    StructureEvent,
    StructureEventType,
    StructureEngine,
    SwingPoint,
    SwingType,
    TrendDirection,
)


# --------------------------------------------------------------------------- helpers

def _ts(i: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)


def _swing(index: int, price: float, type_: SwingType, strength: int = 1) -> SwingPoint:
    return SwingPoint(index=index, timestamp=_ts(index), price=price, type=type_, strength=strength)


def _candle(high: float, low: float, close: float, open_: float = None, i: int = 0) -> Candle:
    if open_ is None:
        open_ = close
    return Candle(timestamp=_ts(i), open=open_, high=high, low=low, close=close, volume=1.0)


def _series(candles, symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1) -> CandleSeries:
    return CandleSeries(symbol=symbol, timeframe=timeframe, provider="test", candles=tuple(candles))


def _structure(
    swings=(), events=(), trend=TrendDirection.UNKNOWN,
    symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1,
) -> StructureAnalysis:
    events = tuple(events)
    return StructureAnalysis(
        symbol=symbol, timeframe=timeframe, trend=trend,
        swing_points=tuple(swings), events=events,
        last_event=events[-1] if events else None,
    )


def _event(candle_index, direction, break_price, broken_swing, type_=StructureEventType.BOS) -> StructureEvent:
    return StructureEvent(
        type=type_, direction=direction, timestamp=_ts(candle_index),
        candle_index=candle_index, break_price=break_price, broken_swing=broken_swing,
    )


def _order_block(index: int, high: float, low: float, direction: TrendDirection, strength: float = 1.0) -> OrderBlock:
    fake_broken = _swing(max(index - 1, 0), low if direction == TrendDirection.BULLISH else high,
                          SwingType.LOW if direction == TrendDirection.BULLISH else SwingType.HIGH)
    fake_event = _event(index + 1, direction, high if direction == TrendDirection.BULLISH else low, fake_broken)
    return OrderBlock(
        timeframe=Timeframe.M1, candle_index=index, timestamp=_ts(index),
        price_high=high, price_low=low, strength=strength,
        direction=direction, source_event=fake_event,
    )


# --------------------------------------------------------------------------- equal highs/lows

def test_equal_highs_groups_prices_within_tolerance():
    swings = (
        _swing(1, 100.0, SwingType.HIGH),
        _swing(3, 100.04, SwingType.HIGH),
        _swing(5, 150.0, SwingType.HIGH),
    )
    levels = SMCEngine()._detect_equal_levels(swings, SwingType.HIGH, Timeframe.M1)

    assert len(levels) == 1
    level = levels[0]
    assert level.side == LiquiditySide.BUY_SIDE
    assert level.strength == 2.0
    assert level.candle_index == 3
    assert abs(level.price_high - 100.02) < 1e-9
    assert level.price_high == level.price_low


def test_equal_lows_groups_prices_within_tolerance():
    swings = (_swing(2, 50.0, SwingType.LOW), _swing(4, 50.02, SwingType.LOW))
    levels = SMCEngine()._detect_equal_levels(swings, SwingType.LOW, Timeframe.M1)

    assert len(levels) == 1
    assert levels[0].side == LiquiditySide.SELL_SIDE
    assert levels[0].strength == 2.0


def test_equal_highs_ignores_prices_outside_tolerance():
    swings = (_swing(1, 100.0, SwingType.HIGH), _swing(3, 105.0, SwingType.HIGH))
    assert SMCEngine()._detect_equal_levels(swings, SwingType.HIGH, Timeframe.M1) == ()


# --------------------------------------------------------------------------- liquidity pools

def test_liquidity_pools_include_clustered_and_singleton_swings():
    grouped = (_swing(1, 100.0, SwingType.HIGH), _swing(3, 100.02, SwingType.HIGH))
    lone_high = _swing(6, 150.0, SwingType.HIGH)
    lone_low = _swing(2, 50.0, SwingType.LOW)
    swings = grouped + (lone_high, lone_low)

    engine = SMCEngine()
    equal_highs = engine._detect_equal_levels(swings, SwingType.HIGH, Timeframe.M1)
    equal_lows = engine._detect_equal_levels(swings, SwingType.LOW, Timeframe.M1)
    pools = engine._detect_liquidity_pools(swings, equal_highs, equal_lows, Timeframe.M1)

    assert len(pools) == 3
    assert sorted(p.strength for p in pools) == [1.0, 1.0, 2.0]
    sides = {p.side for p in pools}
    assert sides == {LiquiditySide.BUY_SIDE, LiquiditySide.SELL_SIDE}


# --------------------------------------------------------------------------- liquidity sweeps

def test_liquidity_sweep_on_swing_high_wick_and_close_reject():
    swing = _swing(1, 100.0, SwingType.HIGH)
    candles = (
        _candle(95, 90, 92, i=0),
        _candle(100, 95, 97, i=1),
        _candle(105, 96, 98, open_=99, i=2),
    )
    sweeps = SMCEngine()._detect_liquidity_sweeps(candles, (swing,), Timeframe.M1)

    assert len(sweeps) == 1
    sweep = sweeps[0]
    assert sweep.candle_index == 2
    assert sweep.side == LiquiditySide.BUY_SIDE
    assert sweep.direction == TrendDirection.BEARISH
    assert sweep.price_high == 105
    assert sweep.price_low == 100
    assert sweep.strength == 5
    assert sweep.swept_level is swing


def test_liquidity_sweep_on_swing_low_wick_and_close_reject():
    swing = _swing(1, 50.0, SwingType.LOW)
    candles = (
        _candle(60, 55, 57, i=0),
        _candle(55, 50, 52, i=1),
        _candle(54, 45, 53, open_=48, i=2),
    )
    sweeps = SMCEngine()._detect_liquidity_sweeps(candles, (swing,), Timeframe.M1)

    assert len(sweeps) == 1
    sweep = sweeps[0]
    assert sweep.side == LiquiditySide.SELL_SIDE
    assert sweep.direction == TrendDirection.BULLISH
    assert sweep.price_low == 45
    assert sweep.price_high == 50
    assert sweep.strength == 5


def test_no_sweep_on_clean_close_through():
    swing = _swing(1, 100.0, SwingType.HIGH)
    candles = (
        _candle(95, 90, 92, i=0),
        _candle(100, 95, 97, i=1),
        _candle(105, 96, 102, open_=99, i=2),
    )
    assert SMCEngine()._detect_liquidity_sweeps(candles, (swing,), Timeframe.M1) == ()


def test_at_most_one_sweep_per_swing():
    swing = _swing(1, 100.0, SwingType.HIGH)
    candles = (
        _candle(95, 90, 92, i=0),
        _candle(100, 95, 97, i=1),
        _candle(105, 96, 98, open_=99, i=2),  # first sweep
        _candle(106, 96, 98, open_=99, i=3),  # would also qualify, must not double-count
    )
    sweeps = SMCEngine()._detect_liquidity_sweeps(candles, (swing,), Timeframe.M1)
    assert len(sweeps) == 1
    assert sweeps[0].candle_index == 2


# --------------------------------------------------------------------------- FVG / IFVG

def test_bullish_fvg_detected():
    candles = (_candle(100, 95, 97, i=0), _candle(103, 101, 102, i=1), _candle(110, 105, 107, i=2))
    fvgs = SMCEngine()._detect_fvgs(candles, Timeframe.M1)

    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.direction == TrendDirection.BULLISH
    assert fvg.candle_index == 1
    assert fvg.price_low == 100
    assert fvg.price_high == 105
    assert fvg.strength == 5


def test_bearish_fvg_detected():
    candles = (_candle(115, 110, 112, i=0), _candle(108, 106, 107, i=1), _candle(100, 95, 97, i=2))
    fvgs = SMCEngine()._detect_fvgs(candles, Timeframe.M1)

    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.direction == TrendDirection.BEARISH
    assert fvg.price_high == 110
    assert fvg.price_low == 100


def test_no_fvg_on_overlapping_candles():
    candles = (_candle(100, 95, 97, i=0), _candle(103, 101, 102, i=1), _candle(99, 94, 96, i=2))
    assert SMCEngine()._detect_fvgs(candles, Timeframe.M1) == ()


def test_inverse_fvg_on_bullish_fvg_closed_through():
    candles = (
        _candle(100, 95, 97, i=0), _candle(103, 101, 102, i=1), _candle(110, 105, 107, i=2),
        _candle(99, 90, 92, open_=95, i=3),
    )
    fvgs = SMCEngine()._detect_fvgs(candles, Timeframe.M1)
    inverses = SMCEngine()._detect_inverse_fvgs(candles, fvgs, Timeframe.M1)

    assert len(inverses) == 1
    inv = inverses[0]
    assert inv.direction == TrendDirection.BEARISH
    assert inv.candle_index == 3
    assert inv.price_high == 105
    assert inv.price_low == 100
    assert inv.strength == 5
    assert inv.source_fvg is fvgs[0]


def test_no_inverse_fvg_if_never_closed_through():
    candles = (
        _candle(100, 95, 97, i=0), _candle(103, 101, 102, i=1), _candle(110, 105, 107, i=2),
        _candle(108, 101, 104, i=3),
    )
    fvgs = SMCEngine()._detect_fvgs(candles, Timeframe.M1)
    assert SMCEngine()._detect_inverse_fvgs(candles, fvgs, Timeframe.M1) == ()


# --------------------------------------------------------------------------- order blocks

def test_order_block_detected_for_bullish_bos():
    broken_swing = _swing(2, 105.0, SwingType.HIGH)
    event = _event(candle_index=5, direction=TrendDirection.BULLISH, break_price=110.0, broken_swing=broken_swing)
    candles = (
        _candle(100, 95, 97, i=0),
        _candle(102, 98, 100, i=1),
        _candle(105, 101, 103, i=2),
        _candle(104, 100, 101, open_=103, i=3),  # bearish body -> eligible
        _candle(108, 105, 107, open_=106, i=4),  # bullish body -> checked first, ineligible
        _candle(112, 105, 110, i=5),
    )
    obs = SMCEngine()._detect_order_blocks(candles, (event,), Timeframe.M1)

    assert len(obs) == 1
    ob = obs[0]
    assert ob.candle_index == 3
    assert ob.direction == TrendDirection.BULLISH
    assert ob.price_high == 104
    assert ob.price_low == 100
    assert ob.strength == abs(110.0 - 101)


def test_no_order_block_when_no_opposite_candle_in_window():
    broken_swing = _swing(2, 105.0, SwingType.HIGH)
    event = _event(candle_index=4, direction=TrendDirection.BULLISH, break_price=110.0, broken_swing=broken_swing)
    candles = (
        _candle(100, 95, 97, i=0), _candle(102, 98, 100, i=1), _candle(105, 101, 103, i=2),
        _candle(107, 103, 106, open_=104, i=3),  # only candle in window, bullish -> ineligible
        _candle(112, 105, 110, i=4),
    )
    assert SMCEngine()._detect_order_blocks(candles, (event,), Timeframe.M1) == ()


def test_order_block_detected_for_bearish_choch():
    broken_swing = _swing(2, 90.0, SwingType.LOW)
    event = _event(
        candle_index=5, direction=TrendDirection.BEARISH, break_price=85.0,
        broken_swing=broken_swing, type_=StructureEventType.CHOCH,
    )
    candles = (
        _candle(100, 95, 97, i=0), _candle(98, 90, 92, i=1), _candle(95, 88, 90, i=2),
        _candle(96, 92, 95, open_=93, i=3),  # bullish body -> eligible
        _candle(93, 88, 89, open_=92, i=4),  # bearish body -> checked first, ineligible
        _candle(88, 82, 85, i=5),
    )
    obs = SMCEngine()._detect_order_blocks(candles, (event,), Timeframe.M1)

    assert len(obs) == 1
    assert obs[0].candle_index == 3
    assert obs[0].direction == TrendDirection.BEARISH


# --------------------------------------------------------------------------- breaker / mitigation

def test_breaker_block_on_close_through_bullish_ob():
    ob = _order_block(index=2, high=104, low=100, direction=TrendDirection.BULLISH)
    candles = (
        _candle(100, 95, 97, i=0), _candle(102, 98, 100, i=1), _candle(104, 100, 102, i=2),
        _candle(103, 98, 99, i=3),
    )
    breakers, mitigations = SMCEngine()._detect_breakers_and_mitigations(candles, (ob,), Timeframe.M1)

    assert len(breakers) == 1
    assert mitigations == ()
    b = breakers[0]
    assert b.candle_index == 3
    assert b.direction == TrendDirection.BEARISH
    assert b.price_high == 104 and b.price_low == 100
    assert b.strength == ob.strength
    assert b.source_order_block is ob


def test_mitigation_block_on_touch_without_close_through():
    ob = _order_block(index=2, high=104, low=100, direction=TrendDirection.BULLISH)
    candles = (
        _candle(100, 95, 97, i=0), _candle(102, 98, 100, i=1), _candle(104, 100, 102, i=2),
        _candle(103, 99, 101, i=3),
    )
    breakers, mitigations = SMCEngine()._detect_breakers_and_mitigations(candles, (ob,), Timeframe.M1)

    assert breakers == ()
    assert len(mitigations) == 1
    m = mitigations[0]
    assert m.candle_index == 3
    assert m.direction == TrendDirection.BULLISH
    assert m.strength == ob.strength
    assert m.source_order_block is ob


def test_no_interaction_produces_neither_breaker_nor_mitigation():
    ob = _order_block(index=2, high=104, low=100, direction=TrendDirection.BULLISH)
    candles = (
        _candle(100, 95, 97, i=0), _candle(102, 98, 100, i=1), _candle(104, 100, 102, i=2),
        _candle(120, 110, 115, i=3),
    )
    breakers, mitigations = SMCEngine()._detect_breakers_and_mitigations(candles, (ob,), Timeframe.M1)
    assert breakers == () and mitigations == ()


def test_bearish_ob_breaker_flips_bullish():
    ob = _order_block(index=2, high=104, low=100, direction=TrendDirection.BEARISH)
    candles = (
        _candle(100, 95, 97, i=0), _candle(102, 98, 100, i=1), _candle(104, 100, 102, i=2),
        _candle(106, 101, 105, i=3),  # touches, closes above 104 -> invalidated
    )
    breakers, mitigations = SMCEngine()._detect_breakers_and_mitigations(candles, (ob,), Timeframe.M1)

    assert len(breakers) == 1
    assert breakers[0].direction == TrendDirection.BULLISH
    assert mitigations == ()


# --------------------------------------------------------------------------- supply / demand (integration)

def test_supply_and_demand_zones_exclude_broken_order_blocks_but_not_active_ones():
    swing_a = _swing(2, 105.0, SwingType.HIGH)
    swing_b = _swing(8, 90.0, SwingType.LOW)
    event1 = _event(candle_index=5, direction=TrendDirection.BULLISH, break_price=112.0, broken_swing=swing_a)
    event2 = _event(candle_index=12, direction=TrendDirection.BEARISH, break_price=85.0, broken_swing=swing_b)

    candles = (
        _candle(100, 95, 97, open_=97, i=0),
        _candle(102, 98, 100, open_=99, i=1),
        _candle(105, 100, 102, open_=102, i=2),   # swing_a's candle
        _candle(104, 99, 100, open_=103, i=3),    # bearish -> OB1 (bullish, demand candidate)
        _candle(108, 105, 107, open_=106, i=4),   # bullish, ineligible; doesn't touch OB1 zone
        _candle(113, 108, 112, open_=109, i=5),   # breakout for event1; doesn't touch OB1 zone
        _candle(100, 90, 92, open_=98, i=6),      # touches OB1 zone[99,104], closes 92<99 -> BREAKS OB1
        _candle(95, 88, 90, open_=91, i=7),
        _candle(93, 90, 91, open_=91, i=8),       # swing_b's candle
        _candle(96, 91, 95, open_=92, i=9),       # bullish -> OB2 (bearish, supply candidate)
        _candle(84, 80, 81, open_=83, i=10),      # bearish, ineligible; below OB2 zone[91,96]
        _candle(86, 82, 83, open_=85, i=11),      # bearish, ineligible; below OB2 zone
        _candle(90, 83, 85, open_=89, i=12),      # breakout for event2; below OB2 zone
        _candle(80, 75, 77, open_=78, i=13),      # never touches OB2 zone
    )
    series = _series(candles)
    structure = _structure(swings=(swing_a, swing_b), events=(event1, event2))

    result = SMCEngine().analyze(series, structure)

    assert len(result.order_blocks) == 2
    assert len(result.breaker_blocks) == 1
    assert result.breaker_blocks[0].source_order_block.candle_index == 3
    assert result.mitigation_blocks == ()
    assert result.demand_zones == ()  # OB1 (bullish) was broken -> excluded
    assert len(result.supply_zones) == 1  # OB2 (bearish) never interacted -> still active
    assert result.supply_zones[0].source_order_block.candle_index == 9


# --------------------------------------------------------------------------- premium / discount

def test_premium_discount_zones_split_the_dealing_range_at_equilibrium():
    swings = (_swing(2, 90.0, SwingType.LOW), _swing(5, 110.0, SwingType.HIGH))
    premium, discount = SMCEngine()._detect_premium_discount(swings, Timeframe.M1)

    assert premium.price_low == 100.0 and premium.price_high == 110.0
    assert discount.price_low == 90.0 and discount.price_high == 100.0
    assert premium.strength == 20.0 and discount.strength == 20.0
    assert premium.candle_index == 5
    assert discount.candle_index == 5


def test_premium_discount_none_without_both_swing_types():
    swings = (_swing(2, 90.0, SwingType.LOW),)
    premium, discount = SMCEngine()._detect_premium_discount(swings, Timeframe.M1)
    assert premium is None and discount is None


def test_premium_discount_none_on_zero_width_range():
    swings = (_swing(2, 100.0, SwingType.LOW), _swing(5, 100.0, SwingType.HIGH))
    premium, discount = SMCEngine()._detect_premium_discount(swings, Timeframe.M1)
    assert premium is None and discount is None


# --------------------------------------------------------------------------- OTE

def test_ote_bullish_on_low_to_high_leg():
    swings = (_swing(2, 100.0, SwingType.LOW), _swing(6, 200.0, SwingType.HIGH))
    ote = SMCEngine()._detect_ote(swings, Timeframe.M1)

    leg = 100.0
    assert ote.direction == TrendDirection.BULLISH
    assert ote.candle_index == 6
    assert ote.price_high == 200.0 - 0.618 * leg
    assert ote.price_low == 200.0 - 0.79 * leg
    assert ote.strength == leg


def test_ote_bearish_on_high_to_low_leg():
    swings = (_swing(2, 200.0, SwingType.HIGH), _swing(6, 100.0, SwingType.LOW))
    ote = SMCEngine()._detect_ote(swings, Timeframe.M1)

    leg = 100.0
    assert ote.direction == TrendDirection.BEARISH
    assert ote.candle_index == 6
    assert ote.price_low == 100.0 + 0.618 * leg
    assert ote.price_high == 100.0 + 0.79 * leg


def test_ote_none_on_tied_indices():
    swings = (_swing(4, 200.0, SwingType.HIGH), _swing(4, 100.0, SwingType.LOW))
    assert SMCEngine()._detect_ote(swings, Timeframe.M1) is None


def test_ote_none_without_both_swing_types():
    swings = (_swing(4, 200.0, SwingType.HIGH),)
    assert SMCEngine()._detect_ote(swings, Timeframe.M1) is None


# --------------------------------------------------------------------------- edge cases / invalid data

def test_analyze_raises_on_mismatched_symbol():
    series = _series([_candle(100, 95, 97, i=0)], symbol="XAUUSD")
    structure = _structure(symbol="EURUSD")
    try:
        SMCEngine().analyze(series, structure)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched symbol")


def test_analyze_raises_on_mismatched_timeframe():
    series = _series([_candle(100, 95, 97, i=0)], timeframe=Timeframe.M1)
    structure = _structure(timeframe=Timeframe.H1)
    try:
        SMCEngine().analyze(series, structure)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched timeframe")


def test_analyze_on_empty_series_returns_all_empty():
    series = _series([])
    structure = _structure()
    result = SMCEngine().analyze(series, structure)

    assert result.liquidity_pools == ()
    assert result.liquidity_sweeps == ()
    assert result.equal_highs == () and result.equal_lows == ()
    assert result.fair_value_gaps == () and result.inverse_fvgs == ()
    assert result.order_blocks == () and result.breaker_blocks == () and result.mitigation_blocks == ()
    assert result.supply_zones == () and result.demand_zones == ()
    assert result.premium_zone is None and result.discount_zone is None
    assert result.ote_zone is None


def test_invalid_equal_tolerance_raises():
    try:
        SMCEngine(equal_tolerance=-0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative equal_tolerance")


# --------------------------------------------------------------------------- full pipeline integration

def test_full_pipeline_structure_then_smc_on_a_realistic_series():
    prices = [
        (100, 98), (103, 100), (101, 99), (106, 102), (104, 101),
        (110, 105), (107, 103), (113, 108), (109, 106), (116, 110),
        (112, 108), (105, 100), (100, 94), (96, 90), (92, 86), (88, 82),
    ]
    candles = tuple(
        Candle(timestamp=_ts(i), open=(h + l) / 2, high=h, low=l, close=(h + l) / 2, volume=10.0)
        for i, (h, l) in enumerate(prices)
    )
    series = CandleSeries(symbol="XAUUSD", timeframe=Timeframe.H1, provider="test", candles=candles)
    structure = StructureEngine(swing_lookback=1).analyze(series)

    result = SMCEngine().analyze(series, structure)

    assert result.symbol == "XAUUSD"
    assert result.timeframe == Timeframe.H1
    for ob in result.order_blocks:
        assert ob.price_low <= ob.price_high
    for fvg in result.fair_value_gaps:
        assert fvg.price_low <= fvg.price_high

    # Determinism check.
    result2 = SMCEngine().analyze(series, structure)
    assert result == result2
