"""Closed candles, D1 repair, fresh structure and room before major levels."""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone

import pytest

import entry_quality as eq
from market_context import MarketContextBuilder
from market_data import (
    Candle, CandleSeries, MarketDataError, MarketDataService, Quote, Timeframe,
    clean_candle_series, closed_candles,
)
from models import TradeSetup
from scoring_engine import ScoringEngine
from smc_engine import SMCAnalysis
from structure_engine import (
    StructureAnalysis, StructureEvent, StructureEventType, SwingPoint, SwingType, TrendDirection,
)
from trade_decision import decide_deterministically

NOW = datetime(2026, 10, 7, 14, 37, tzinfo=timezone.utc)


def _bar(ts, o, h, l, c, v=None):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)


def _hourly_closes(n=300):
    return [100.0 + 0.04 * i + 3.0 * math.sin(i / 4.0) for i in range(n)]


def _series(tf, closes, last_open):
    step = tf.duration
    first = last_open - step * (len(closes) - 1)
    return CandleSeries(symbol="XAUUSD", timeframe=tf, provider="stub", candles=tuple(
        _bar(first + step * i, c, c + 0.6, c - 0.6, c) for i, c in enumerate(closes)))


class _Provider:
    """Every timeframe ends with the bar still forming at NOW."""
    name = "stub"

    def __init__(self, forming_close=None, price=None):
        self.forming_close, self.price = forming_close, price

    async def get_quote(self, symbol):
        closes = _hourly_closes()
        return Quote(symbol=symbol, price=self.price or closes[-1], timestamp=NOW, provider="stub")

    async def get_candles(self, symbol, timeframe, limit):
        closes = _hourly_closes()
        forming_open = {
            Timeframe.M15: NOW.replace(minute=30, second=0),
            Timeframe.H1: NOW.replace(minute=0, second=0),
            Timeframe.H4: NOW.replace(hour=12, minute=0, second=0),
            Timeframe.D1: NOW.replace(hour=0, minute=0, second=0),
        }[timeframe]
        closes = closes[-limit:]
        if self.forming_close is not None:
            closes = closes + [self.forming_close]     # same closed bars, plus one forming
        else:
            forming_open -= timeframe.duration
        return _series(timeframe, closes, forming_open)


def _build(provider, setup):
    builder = MarketContextBuilder(MarketDataService(provider, cache_ttl=0))
    return asyncio.run(builder.build("XAUUSD", setup, now=NOW))


def _long(entry=None, stop=95.0, tps=(130.0,)):
    return TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                      entries=[entry] if entry else [], stop_loss=stop, take_profits=list(tps))


# ------------------------------------------------------------ closed candles

def test_closed_candles_drops_only_the_bar_still_forming():
    series = _series(Timeframe.H1, [1.0, 2.0, 3.0], NOW.replace(minute=0, second=0))
    kept = closed_candles(series.candles, Timeframe.H1, NOW)
    assert [c.close for c in kept] == [1.0, 2.0]
    assert kept[-1].timestamp + Timeframe.H1.duration <= NOW


def test_a_forming_bar_never_reaches_structure_scoring_or_indicators():
    """A violent forming bar — the kind that used to create a BOS mid-bar —
    must leave every structural number exactly where the closed bars put it."""
    setup = _long(entry=105.0)
    closed = _build(_Provider(), setup)
    spiking = _build(_Provider(forming_close=_hourly_closes()[-1] + 40.0), setup)

    assert spiking.scoring.breakdown == closed.scoring.breakdown
    assert spiking.scoring.confidence == closed.scoring.confidence
    assert spiking.structure.events == closed.structure.events
    assert spiking.primary.atr == closed.primary.atr
    assert spiking.primary_candles == closed.primary_candles
    assert spiking.closed_through == closed.closed_through
    assert spiking.closed_through["H1"] == NOW.replace(minute=0, second=0) - timedelta(hours=1)


def test_the_live_price_still_reaches_entry_and_latest_move():
    """Live price is allowed for the current entry, location, R:R and the
    latest move — and only there."""
    calm = _build(_Provider(price=_hourly_closes()[-1]), _long())
    spiked = _build(_Provider(price=_hourly_closes()[-1] + 15.0), _long())

    assert spiked.scoring.breakdown == calm.scoring.breakdown              # structure untouched
    assert spiked.quality.entry_price == calm.quality.entry_price + 15.0   # market entry moved
    assert spiked.entry.extension.move_atr > calm.entry.extension.move_atr  # latest move seen


def test_a_replay_at_a_past_time_sees_only_what_had_closed_then():
    setup = _long(entry=105.0)
    ctx = _build(_Provider(), setup)
    assert ctx.generated_at == NOW
    assert all(ts + Timeframe[tf].duration <= NOW for tf, ts in ctx.closed_through.items())


# ------------------------------------------------------------ D1 repair

def _daily(rows):
    return CandleSeries(symbol="XAUUSD", timeframe=Timeframe.D1, provider="stub", candles=tuple(
        _bar(datetime.fromisoformat(d).replace(tzinfo=timezone.utc), o, h, l, c) for d, o, h, l, c in rows))


_WEEK = [("2026-09-28", 100, 190, 90, 150), ("2026-09-29", 150, 240, 140, 200),
         ("2026-09-30", 200, 290, 190, 250), ("2026-10-01", 250, 340, 240, 300),
         ("2026-10-02", 300, 390, 290, 350)]


def test_a_duplicated_daily_bar_is_merged_not_fatal():
    """The live failure: two bars labelled 2026-10-04 made the whole D1
    series 'out of order' and the daily trend vanished."""
    rows = _WEEK + [("2026-10-05", 350, 440, 340, 400), ("2026-10-05", 400, 450, 330, 420)]
    cleaned = clean_candle_series(_daily(rows))
    MarketDataService._validate_candle_series(cleaned, expected_symbol="XAUUSD",
                                              expected_timeframe=Timeframe.D1)
    last = cleaned.candles[-1]
    assert len(cleaned.candles) == 6
    assert (last.open, last.high, last.low, last.close) == (350, 450, 330, 420)


def test_flat_weekend_daily_stubs_are_dropped():
    rows = _WEEK + [("2026-10-03", 350, 353, 349, 351), ("2026-10-04", 351, 351.2, 350.9, 351),
                    ("2026-10-04", 351, 351.3, 350.9, 351.1), ("2026-10-05", 351, 440, 340, 400)]
    cleaned = clean_candle_series(_daily(rows))
    dates = [c.timestamp.date().isoformat() for c in cleaned.candles]
    assert "2026-10-03" not in dates and "2026-10-04" not in dates
    assert dates[-1] == "2026-10-05"


def test_real_weekend_sessions_are_kept():
    """Instruments that trade at weekends (crypto) keep their weekend bars."""
    rows = _WEEK + [("2026-10-03", 350, 430, 340, 400), ("2026-10-04", 400, 470, 380, 450)]
    assert len(clean_candle_series(_daily(rows)).candles) == 7


def test_intraday_gold_bars_inside_the_weekly_closure_are_dropped():
    """Saturday quotes for gold are stubs (no market); crypto keeps its weekend."""
    sat = [_bar(datetime(2026, 10, 3, h, tzinfo=timezone.utc), 1, 1.01, 0.99, 1) for h in (1, 2)]
    fri = _bar(datetime(2026, 10, 2, 15, tzinfo=timezone.utc), 1, 1.01, 0.99, 1)
    gold = CandleSeries(symbol="XAUUSD", timeframe=Timeframe.H1, provider="stub", candles=(fri, *sat))
    assert clean_candle_series(gold).candles == (fri,)
    crypto = CandleSeries(symbol="BTCUSDT", timeframe=Timeframe.H1, provider="stub", candles=tuple(sat))
    assert clean_candle_series(crypto) is crypto


def test_genuinely_disordered_bars_are_still_rejected():
    rows = [_WEEK[1], _WEEK[0]]
    with pytest.raises(MarketDataError):
        MarketDataService._validate_candle_series(clean_candle_series(_daily(rows)),
                                                  expected_symbol="XAUUSD",
                                                  expected_timeframe=Timeframe.D1)


def test_d1_is_analysed_when_the_provider_sends_a_duplicate():
    class Duplicating(_Provider):
        async def get_candles(self, symbol, timeframe, limit):
            series = await super().get_candles(symbol, timeframe, limit)
            if timeframe is Timeframe.D1:
                return CandleSeries(symbol=symbol, timeframe=timeframe, provider="stub",
                                    candles=series.candles[:-1] + series.candles[-2:-1] + series.candles[-1:])
            return series
    ctx = _build(Duplicating(), _long(entry=105.0))
    assert "D1" in ctx.timeframes
    assert not any("D1" in w for w in ctx.warnings)


def test_a_warning_appears_only_when_d1_is_genuinely_unavailable():
    class NoDaily(_Provider):
        async def get_candles(self, symbol, timeframe, limit):
            if timeframe is Timeframe.D1:
                raise MarketDataError("daily series down", retryable=False)
            return await super().get_candles(symbol, timeframe, limit)
    ctx = _build(NoDaily(), _long(entry=105.0))
    assert "D1" not in ctx.timeframes
    assert any(w.startswith("D1 candles unavailable") for w in ctx.warnings)


# ------------------------------------------------------------ fresh structure

def _event(index, direction, type_=StructureEventType.BOS):
    return StructureEvent(type=type_, direction=direction, timestamp=NOW, candle_index=index,
                          break_price=100.0,
                          broken_swing=SwingPoint(index=max(index - 1, 0), timestamp=NOW,
                                                  price=100.0, type=SwingType.HIGH, strength=1))


def _score(events, last_index, trend=TrendDirection.UNKNOWN, **kw):
    structure = StructureAnalysis(symbol="XAUUSD", timeframe=Timeframe.H1, trend=trend,
                                  swing_points=(), events=tuple(events),
                                  last_event=events[-1] if events else None)
    smc = SMCAnalysis(symbol="XAUUSD", timeframe=Timeframe.H1, liquidity_pools=(),
                      liquidity_sweeps=(), equal_highs=(), equal_lows=(), fair_value_gaps=(),
                      inverse_fvgs=(), order_blocks=(), breaker_blocks=(), mitigation_blocks=(),
                      supply_zones=(), demand_zones=(), premium_zone=None, discount_zone=None,
                      ote_zone=None)
    return ScoringEngine(**kw).score(structure, smc, last_index=last_index)


def _bos(result):
    return next(f for f in result.breakdown if f.name == "bos")


def test_a_fresh_break_counts_more_than_an_old_one():
    fresh = _bos(_score([_event(98, TrendDirection.BULLISH)], last_index=100))
    old = _bos(_score([_event(10, TrendDirection.BULLISH)], last_index=100))
    assert fresh.contribution > old.contribution > 0
    assert old.freshness < 0.02                      # 90 bars old: almost nothing


def test_freshness_halves_every_half_life():
    one = _bos(_score([_event(86, TrendDirection.BULLISH)], last_index=100))
    two = _bos(_score([_event(72, TrendDirection.BULLISH)], last_index=100))
    assert one.freshness == pytest.approx(0.5)
    assert two.freshness == pytest.approx(0.25)
    assert "14 bars ago" in one.reason


def test_freshness_is_smooth_and_never_increases_with_age():
    weights = [_bos(_score([_event(100 - age, TrendDirection.BULLISH)], last_index=100)).freshness
               for age in range(0, 101)]
    assert weights == sorted(weights, reverse=True)
    assert max(a - b for a, b in zip(weights, weights[1:])) < 0.05


def test_trend_is_a_state_and_does_not_age():
    result = _score([_event(10, TrendDirection.BULLISH)], last_index=100,
                    trend=TrendDirection.BULLISH)
    trend = next(f for f in result.breakdown if f.name == "trend")
    assert trend.freshness == 1.0


def test_without_a_bar_index_nothing_ages():
    """Backward compatible: callers that do not say where the series ends
    get the original full-weight votes."""
    result = _score([_event(10, TrendDirection.BULLISH)], last_index=None)
    assert _bos(result).freshness == 1.0


def test_a_non_positive_half_life_is_rejected():
    with pytest.raises(ValueError):
        ScoringEngine(freshness_half_life=0)


# ------------------------------------------------------------ room

@pytest.mark.parametrize("level,expected_factor,label", [
    (100.2, 0.6, "tight"),     # 0.2 ATR ahead: buying into the level
    (100.5, 0.75, "tight"),
    (101.0, 1.0, "ample"),     # one ATR of room first
    (103.0, 1.0, "ample"),
    (106.0, 1.0, "clear"),     # beyond the target
])
def test_room_grades_the_first_major_level_in_atr(level, expected_factor, label):
    room = eq.assess_room("long", 100.0, [105.0], [(level, "H4 swing high")], atr=1.0)
    assert room.factor == expected_factor
    assert room.label == label


def test_room_is_mirrored_for_shorts():
    long_ = eq.assess_room("long", 100.0, [105.0], [(100.3, "x")], atr=1.0)
    short = eq.assess_room("short", 100.0, [95.0], [(99.7, "x")], atr=1.0)
    assert long_.factor == short.factor


def test_levels_behind_the_entry_are_not_in_the_way():
    room = eq.assess_room("long", 100.0, [105.0], [(99.5, "equal lows"), (99.9, "previous day low")],
                          atr=1.0)
    assert room.label == "clear" and room.factor == 1.0


def test_the_nearest_target_is_the_one_checked():
    """Targets listed far-first (as in a live signal) are still read
    nearest-first: the first target is the one price must reach."""
    room = eq.assess_room("short", 100.0, [90.0, 98.0], [(97.0, "previous day low")], atr=1.0)
    assert room.label == "clear"


def test_room_never_depends_on_the_stop():
    """The stop is the trader's to move by hand. Room is in ATR for exactly
    that reason — no stop argument exists."""
    import inspect
    assert "stop" not in inspect.signature(eq.assess_room).parameters


@pytest.mark.parametrize("args", [
    (None, 100.0, [105.0], [], 1.0),
    ("long", None, [105.0], [], 1.0),
    ("long", 100.0, [], [], 1.0),
    ("long", 100.0, [105.0], [], None),
    ("long", 100.0, [105.0], [], float("nan")),
])
def test_unknown_room_fails_safe(args):
    room = eq.assess_room(*args)
    assert room.factor == 1.0 and room.label == "unknown"


def test_location_and_room_count_once():
    """Both look at the structure between entry and target, so the worse of
    the two applies — an entry under the range high is not charged twice."""
    q = eq.EntryQuality(
        location=eq.assess_location("long", 109.0, 100.0, 110.0),           # 0.6
        extension=eq.assess_extension("long", []),                           # unknown -> 1.0
        room=eq.assess_room("long", 109.0, [115.0], [(109.3, "x")], atr=1.0),  # 0.65
    )
    assert q.space == min(q.location.factor, q.room.factor) == 0.6
    assert q.factor == pytest.approx(0.6)


# ------------------------------------------------------------ major levels

def test_major_levels_come_from_closed_data_and_skip_swept_levels():
    ctx = _build(_Provider(), _long(entry=105.0))
    labels = {label for _, label in ctx.major_levels}
    assert {"previous day high", "previous day low"} <= labels


def test_an_h4_swing_price_has_since_exceeded_is_no_longer_a_level():
    from market_context import MarketContext
    closes = [10, 11, 12, 20, 12, 11, 10, 11, 12, 25, 12, 11, 10, 13, 21.5, 15, 14]
    h4 = CandleSeries(symbol="XAUUSD", timeframe=Timeframe.H4, provider="stub", candles=tuple(
        _bar(NOW - timedelta(hours=4 * (len(closes) - i)), c, c + 0.5, c - 0.5, c)
        for i, c in enumerate(closes)))
    builder = MarketContextBuilder(MarketDataService(_Provider(), cache_ttl=0))
    ctx = MarketContext(symbol="XAUUSD", generated_at=NOW, primary_timeframe=Timeframe.H1)
    builder._add_major_levels(ctx, {Timeframe.H4: h4})

    highs = sorted(p for p, label in ctx.major_levels if label == "H4 swing high")
    assert 20.5 not in highs          # traded through later by the 22.0 high
    assert highs == [22.0, 25.5]      # the breaking high is itself an unswept swing


def test_previous_day_levels_come_from_the_last_closed_daily_bar():
    from market_context import MarketContext
    d1 = _daily(_WEEK)
    builder = MarketContextBuilder(MarketDataService(_Provider(), cache_ttl=0))
    ctx = MarketContext(symbol="XAUUSD", generated_at=NOW, primary_timeframe=Timeframe.H1)
    builder._add_major_levels(ctx, {Timeframe.D1: d1})
    assert (390, "previous day high") in ctx.major_levels
    assert (290, "previous day low") in ctx.major_levels


def test_the_decision_uses_the_worse_of_location_and_room():
    ctx = _build(_Provider(), _long(entry=105.0, tps=(140.0,)))
    trace = decide_deterministically(ctx)
    if trace.adjusted_score is not None:
        expected = round(trace.adjusted_score * min(trace.location_factor, trace.room_factor)
                         * trace.extension_factor)
        assert trace.confluence_score == expected
        assert 0 <= trace.confluence_score <= 100
