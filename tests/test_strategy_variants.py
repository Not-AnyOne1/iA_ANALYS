"""Candidate production changes (strategy_variants.py) and the clean-candle
mode of the context builder. All prices are TEST DATA (tests/analyst_synth.py).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta

import pytest

from indicators import ADXResult
from market_context import MarketContextBuilder, clean_request_size, clean_series
from market_data import Candle, CandleSeries, MarketDataService, Quote, Timeframe
from models import TradeSetup
from news_filter import NewsFilter
from structure_engine import StructureEventType, TrendDirection
from trade_decision import Action, decide_deterministically

import strategy_variants as sv
from analyst_synth import aggregate, scenario

UTC = __import__("datetime").timezone.utc


def _daily(h4):
    days = {}
    for c in h4:
        key = c.timestamp.replace(hour=0)
        days.setdefault(key, []).append(c)
    return tuple(Candle(k, g[0].open, max(x.high for x in g), min(x.low for x in g), g[-1].close)
                 for k, g in sorted(days.items()))


class _Provider:
    """Serves a synthetic scenario's bars; ``mirror`` reflects every price."""

    name = "test"

    def __init__(self, name="b_long", mirror=None):
        bars, frames, self.at, self.price = scenario(name)
        h4_all = aggregate(bars, Timeframe.H4)
        self.frames = {**frames, Timeframe.D1: _daily(h4_all)}
        self.mirror = mirror
        if mirror is not None:
            k = mirror
            self.frames = {tf: tuple(Candle(c.timestamp, k - c.open, k - c.low, k - c.high, k - c.close)
                                     for c in cs) for tf, cs in self.frames.items()}
            self.price = k - self.price

    async def get_quote(self, symbol):
        return Quote(symbol=symbol, price=self.price, timestamp=self.at, provider=self.name)

    async def get_candles(self, symbol, timeframe, limit):
        return CandleSeries(symbol, timeframe, self.name, self.frames[timeframe][-limit:])


def context(direction="long", name="b_long", mirror=None, clean=False):
    provider = _Provider(name, mirror)
    price = provider.price
    s = 1.0 if direction == "long" else -1.0
    setup = TradeSetup(symbol="XAUUSD", direction=direction, order_type="market",
                       stop_loss=round(price - s * 6.0, 2), take_profits=[round(price + s * 12.0, 2)])
    builder = MarketContextBuilder(MarketDataService(provider), news_filter=NewsFilter(),
                                   clean_candles=clean)
    return asyncio.run(builder.build("XAUUSD", setup, now=provider.at))


# --------------------------------------------------------------- symmetry

def test_production_verdict_is_symmetric_under_a_price_mirror():
    """BUY on the data and SELL on its mirror image get the same score."""
    long_ctx = context("long")
    k = 2 * long_ctx.current_price
    short_ctx = context("short", mirror=k)
    a, b = decide_deterministically(long_ctx), decide_deterministically(short_ctx)
    assert (a.verdict, a.deterministic_score, a.adjusted_score) == \
        (b.verdict, b.deterministic_score, b.adjusted_score)


def test_baseline_and_clean_data_variants_are_the_production_decision():
    ctx = context("long")
    assert sv.decide(ctx, "baseline") == decide_deterministically(ctx)
    assert sv.decide(ctx, "clean_data") == decide_deterministically(ctx)
    with pytest.raises(ValueError):
        sv.decide(ctx, "something_else")


# -------------------------------------------------------------- htf_regime

def test_regime_needs_h4_and_h1_to_agree():
    ctx = context("long")
    assert sv.timeframe_bias(ctx, Timeframe.H1) == "bullish"
    assert sv.timeframe_bias(ctx, Timeframe.H4) == "bullish"
    assert sv.regime(ctx) == "long"
    h4 = ctx.timeframes["H4"]
    flat = replace(ctx, timeframes={**ctx.timeframes, "H4": replace(h4, ema=replace(h4.ema, ema_50=h4.last_close + 50))})
    assert sv.timeframe_bias(flat, Timeframe.H4) == "neutral" and sv.regime(flat) is None


def test_htf_regime_skips_against_the_regime_and_without_one():
    short_ctx = context("short")                  # bullish regime, short signal
    assert sv.decide(short_ctx, "htf_regime").deterministic_score == 0
    assert sv.decide(short_ctx, "htf_regime").verdict is Action.SKIP
    h4 = short_ctx.timeframes["H4"]
    none = replace(short_ctx, timeframes={**short_ctx.timeframes,
                                          "H4": replace(h4, ema=replace(h4.ema, ema_50=h4.last_close + 50))})
    assert sv.decide(none, "htf_regime").deterministic_score == 0
    long_ctx = context("long")
    assert sv.decide(long_ctx, "htf_regime").deterministic_score == long_ctx.scoring.confidence


# ------------------------------------------------------------ valid_events

def test_an_event_no_longer_holding_loses_its_vote_in_both_directions():
    ctx = context("long")
    bos = max((e for e in ctx.structure.events if e.type == StructureEventType.BOS),
              key=lambda e: e.candle_index)
    assert bos.direction is TrendDirection.BULLISH
    n = len(ctx.primary_candles)
    above = sv.ValidEventScoringEngine(bos.broken_swing.price + 1, n).score(ctx.structure, ctx.smc)
    below = sv.ValidEventScoringEngine(bos.broken_swing.price - 1, n).score(ctx.structure, ctx.smc)
    vote = lambda s: {f.name: f.contribution for f in s.breakdown}["bos"]  # noqa: E731
    assert vote(above) == 15.0 and vote(below) == 0.0
    mirror = context("short", mirror=2 * ctx.current_price)
    mbos = max((e for e in mirror.structure.events if e.type == StructureEventType.BOS),
               key=lambda e: e.candle_index)
    assert mbos.direction is TrendDirection.BEARISH
    held = sv.ValidEventScoringEngine(mbos.broken_swing.price - 1, n).score(mirror.structure, mirror.smc)
    gone = sv.ValidEventScoringEngine(mbos.broken_swing.price + 1, n).score(mirror.structure, mirror.smc)
    assert vote(held) == -15.0 and vote(gone) == 0.0


def test_valid_events_rescoring_reruns_the_review():
    ctx = context("long")
    re = sv.rescored(ctx)
    assert re.scoring is not ctx.scoring and re.validation is not ctx.validation
    assert sv.decide(ctx, "valid_events").verdict in (Action.ENTER, Action.WAIT, Action.SKIP)


# ------------------------------------------------------------ range_filter

def test_range_filter_skips_only_a_ranging_low_adx_market():
    ctx = context("long")
    assert decide_deterministically(ctx).verdict is Action.ENTER
    h1 = ctx.timeframes["H1"]
    ranging = replace(ctx, timeframes={**ctx.timeframes, "H1": replace(h1, trend="ranging",
                                                                     adx=ADXResult(15.0, 20.0, 20.0))})
    t = sv.decide(ranging, "range_filter")
    assert t.verdict is Action.SKIP and "ranging market" in t.reason
    strong = replace(ctx, timeframes={**ctx.timeframes, "H1": replace(h1, trend="ranging",
                                                                    adx=ADXResult(30.0, 20.0, 20.0))})
    assert sv.decide(strong, "range_filter").verdict is decide_deterministically(strong).verdict


# -------------------------------------------------------------- clean data

def test_clean_series_drops_forming_bars_stubs_and_weekend_days():
    friday = __import__("datetime").datetime(2026, 7, 3, 20, 0, tzinfo=UTC)
    bars = [Candle(friday + timedelta(minutes=15 * i), 1, 2, 0.5, 1.5) for i in range(12)]   # into the closure
    series = CandleSeries("XAUUSD", Timeframe.M15, "test", tuple(bars))
    now = friday + timedelta(hours=2, minutes=5)
    kept = clean_series(series, now, 300).candles
    assert all(c.timestamp + timedelta(minutes=15) <= now for c in kept)
    assert kept[-1].timestamp == friday + timedelta(minutes=45)        # 21:00 onward is the closure
    days = [Candle(friday.replace(hour=0) + timedelta(days=i), 1, 2, 0.5, 1.5) for i in range(4)]
    d1 = clean_series(CandleSeries("XAUUSD", Timeframe.D1, "test", tuple(days)),
                      friday + timedelta(days=5), 300).candles
    assert [c.timestamp.weekday() for c in d1] == [4, 0]               # Saturday and Sunday dropped
    assert clean_request_size(Timeframe.M15, 300) > 300


def test_clean_mode_builds_on_closed_bars_only():
    provider = _Provider()
    ctx = context("long", clean=True)
    for tf, series in ctx.candle_series.items():
        dur = {Timeframe.M15: 15, Timeframe.H1: 60, Timeframe.H4: 240, Timeframe.D1: 1440}[tf]
        assert all(c.timestamp + timedelta(minutes=dur) <= provider.at for c in series.candles), tf


def test_the_live_builder_is_unchanged_by_default():
    ctx = context("long")
    provider = _Provider()
    assert ctx.candle_series[Timeframe.M15].candles == provider.frames[Timeframe.M15][-300:]


# ------------------------------------------------------- the score display

def test_the_card_shows_a_labelled_score_never_a_percentage():
    from report import render_telegram_report, score_line
    from trade_decision import DecisionSource, TradeDecision
    ctx = context("long")
    trace = decide_deterministically(ctx)

    def decision(**kw):
        base = dict(symbol="XAUUSD", verdict=trace.verdict, confidence=trace.adjusted_score or 0,
                    reasoning="TEST", strengths=[], risks=[], execution_plan="TEST",
                    source=DecisionSource.CLAUDE, deterministic_score=trace.deterministic_score,
                    validation_bonus=trace.validation_bonus, validation_penalty=trace.validation_penalty,
                    adjusted_score=trace.adjusted_score, risk_approved=True)
        return TradeDecision(**{**base, **kw})

    card = render_telegram_report(ctx, decision())
    assert f"Setup score:</b> {trace.adjusted_score}/100 (heuristic, not a win probability)" in card
    assert "%" not in card.split("Setup score")[1].split("\n")[0]
    blocked = score_line(decision(adjusted_score=None, confidence=0, risk_approved=False,
                                  risk_rejections=["risk_reward: R:R 0.80 is below 1.10"]), ctx)
    assert "Setup score:</b> Not scored — blocked by the risk gate: risk_reward" in blocked
    assert "/100" not in blocked and "%" not in blocked
    unscored = score_line(decision(adjusted_score=None, confidence=0,
                                   decision_reason="no deterministic score could be computed"), ctx)
    assert unscored.endswith("Not scored — no deterministic score could be computed")
    against = score_line(decision(deterministic_score=0, adjusted_score=5, confidence=5), ctx)
    assert "5/100 — engine reads BUY" in against
