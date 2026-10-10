"""End-to-end checks of the confluence redesign through the REAL engines.

Market data is synthetic and the Claude CLI is never called; everything in
between — context builder, structure, SMC, scoring, levels, trade quality,
risk gate, entry quality, validation, decision — runs for real.
"""

from __future__ import annotations

import asyncio
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from config import Settings
from market_context import MarketContextBuilder
from market_data import Candle, CandleSeries, MarketDataError, MarketDataService, Quote, Timeframe
from models import TradeSetup
from trade_decision import Action, TradeDecisionEngine, decide_deterministically


@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")


def _closes_trend_then_run(n: int = 300, run: int = 20, step: float = 1.5):
    """A rising zigzag (real swings for structure/SMC to find), ending in an
    outsized straight run — the shape of the losing 99-100 trades."""
    closes = [100.0 + 0.05 * i + 3.0 * math.sin(i / 4.0) for i in range(n - run)]
    for _ in range(run):
        closes.append(closes[-1] + step)
    return closes


def _series(symbol, timeframe, closes):
    base = datetime(2026, 3, 10, tzinfo=timezone.utc) - timedelta(hours=len(closes))
    candles = tuple(
        Candle(timestamp=base + timedelta(hours=i), open=c, high=c + 0.6, low=c - 0.6,
               close=c, volume=10.0)
        for i, c in enumerate(closes))
    return CandleSeries(symbol=symbol, timeframe=timeframe, provider="stub", candles=candles)


class _Provider:
    name = "stub"

    def __init__(self, closes=None, fail: bool = False):
        self._closes = closes or _closes_trend_then_run()
        self._fail = fail

    async def get_quote(self, symbol):
        if self._fail:
            raise MarketDataError("provider down", retryable=False)
        price = self._closes[-1]
        return Quote(symbol=symbol, price=price, timestamp=datetime.now(timezone.utc),
                     provider=self.name, bid=price - 0.05, ask=price + 0.05)

    async def get_candles(self, symbol, timeframe, limit):
        if self._fail:
            raise MarketDataError("provider down", retryable=False)
        return _series(symbol, timeframe, self._closes)


def _build(setup, provider=None):
    builder = MarketContextBuilder(MarketDataService(provider or _Provider()))
    return asyncio.run(builder.build(setup.symbol, setup))


def _settings() -> Settings:
    return Settings(
        api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
        claude_cli_path="claude", model="", claude_max_turns=3,
        signal_parser_mode="off", worker_count=1, queue_maxsize=10,
        max_message_chars=8000, analyse_edits=False, max_retries=0,
        request_timeout=5.0, log_level="CRITICAL", log_file=None, jsonl_output=None,
        color=False, show_json=False, storage_db_path=Path("data/x.db"),
        twelve_data_api_key="k", market_data_cache_ttl=30.0,
        market_data_timeframe=Timeframe.H1)


# --------------------------------------------------- impossible trades

def test_the_invalid_short_that_was_approved_at_93_is_now_skipped():
    """Phase 10 #4 and #11: a SHORT with a long's levels never reaches ENTER,
    whatever the market says, and the reason is stated."""
    price = _closes_trend_then_run()[-1]
    setup = TradeSetup(symbol="XAUUSD", direction="short", order_type="limit",
                       entries=[price], stop_loss=price - 10, take_profits=[price + 20])
    ctx = _build(setup)
    trace = decide_deterministically(ctx)

    assert not ctx.risk.approved
    assert trace.verdict is Action.SKIP
    assert trace.blocked
    assert trace.confluence_score is None          # no valid score for it
    assert "invalid SHORT" in trace.reason


@pytest.mark.parametrize("direction,stop_delta,tp_delta", [
    ("long", +10, +30),      # stop above entry
    ("long", -10, -30),      # target below entry
    ("long", 0, +30),        # zero risk
    ("short", +10, 0),       # zero reward
    ("long", -0.5, +0.5),    # valid geometry but R:R 1.0, below the 1.10 minimum
])
def test_no_impossible_or_sub_minimum_trade_can_enter(direction, stop_delta, tp_delta):
    price = _closes_trend_then_run()[-1]
    setup = TradeSetup(symbol="XAUUSD", direction=direction, order_type="limit",
                       entries=[price], stop_loss=price + stop_delta,
                       take_profits=[price + tp_delta])
    trace = decide_deterministically(_build(setup))
    assert trace.verdict is Action.SKIP
    assert trace.blocked


# --------------------------------------------------- late entries

def test_a_long_after_an_outsized_run_is_measured_as_late():
    """Phase 10 #8 through the real engines: the entry assessment sees both
    the location (above the dealing range) and the run, and the confluence
    score is the adjusted score scaled by them — never above it."""
    price = _closes_trend_then_run()[-1]
    setup = TradeSetup(symbol="XAUUSD", direction="long", order_type="market",
                       entries=[], stop_loss=price - 8, take_profits=[price + 24])
    ctx = _build(setup)
    trace = decide_deterministically(ctx)

    assert ctx.entry is not None
    assert ctx.entry.extension.label == "strong"
    assert ctx.entry.extension.factor < 0.6
    assert ctx.entry.location.factor < 1.0
    assert trace.confluence_score == round(
        trace.adjusted_score * trace.location_factor * trace.extension_factor)
    assert trace.confluence_score < trace.adjusted_score or trace.adjusted_score == 0
    assert trace.verdict is not Action.ENTER


def test_the_same_run_does_not_penalise_a_short_and_never_proposes_one():
    """Extension is not a reversal signal: a short after the run gets no
    extension penalty — and its score is still measured from the short's
    side, where the bullish evidence gives it nothing."""
    price = _closes_trend_then_run()[-1]
    setup = TradeSetup(symbol="XAUUSD", direction="short", order_type="market",
                       entries=[], stop_loss=price + 8, take_profits=[price - 24])
    ctx = _build(setup)
    trace = decide_deterministically(ctx)

    assert ctx.entry.extension.factor == 1.0
    assert trace.verdict is not Action.ENTER


# --------------------------------------------------- safety

def test_missing_market_data_fails_safely():
    """Phase 10 #14: no candles, no quote — the decision is SKIP, nothing
    raises, and Claude is not asked to judge a blank."""
    setup = TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                       entries=[100.0], stop_loss=95.0, take_profits=[115.0])
    ctx = _build(setup, _Provider(fail=True))
    assert not ctx.is_usable

    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock()
    result = asyncio.run(engine.decide(ctx))

    assert result.verdict is Action.SKIP
    engine._run_cli.assert_not_awaited()
    assert decide_deterministically(ctx).verdict is Action.SKIP


@pytest.mark.parametrize("direction", ["long", "short"])
def test_the_real_pipeline_score_is_bounded_and_finite(direction):
    price = _closes_trend_then_run()[-1]
    sign = 1 if direction == "long" else -1
    for entry_offset in (-15, -5, 0, 5):
        entry = price + entry_offset
        setup = TradeSetup(symbol="XAUUSD", direction=direction, order_type="limit",
                           entries=[entry], stop_loss=entry - sign * 6,
                           take_profits=[entry + sign * 18])
        trace = decide_deterministically(_build(setup))
        for value in (trace.deterministic_score, trace.adjusted_score, trace.confluence_score):
            assert value is not None and 0 <= value <= 100
        for factor in (trace.location_factor, trace.extension_factor):
            assert math.isfinite(factor) and 0.5 <= factor <= 1.0


def test_moving_the_stop_never_changes_the_decision():
    """Through the real engines: same market, same entry, same target, the
    stop moved from razor-thin to very wide. Each stop gets its own SL risk
    finding (here every one is HIGH, for different reasons — inside the noise,
    inside the typical heat, unusually wide); the verdict and every number
    behind it are identical. Room is measured in ATR, not in units of the
    stop, precisely so that this holds."""
    from trade_decision import SLRisk, stop_loss_risk
    closes = [100.0 + 3.0 * math.sin(i / 4.0) for i in range(300)]
    entry = min(closes[-30:]) + 0.5
    results, findings = set(), set()
    for risk in (0.2, 1.0, 2.5, 8.0):
        setup = TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                           entries=[entry], stop_loss=entry - risk,
                           take_profits=[entry + 30.0])
        ctx = _build(setup, _Provider(closes))
        trace = decide_deterministically(ctx)
        results.add((trace.verdict, trace.confluence_score, trace.adjusted_score, trace.reason))
        sl = stop_loss_risk(ctx)
        assert sl.level is not SLRisk.UNKNOWN
        findings.add(sl.reasons)
    assert len(results) == 1, results          # the decision never moved
    assert len(findings) == 4                  # while each stop was assessed on its own


def test_only_enter_or_skip_comes_out_of_the_real_pipeline():
    price = _closes_trend_then_run()[-1]
    for direction, sign in (("long", 1), ("short", -1)):
        for offset in (-15, -5, 0, 5):
            entry = price + offset
            setup = TradeSetup(symbol="XAUUSD", direction=direction, order_type="limit",
                               entries=[entry], stop_loss=entry - sign * 6,
                               take_profits=[entry + sign * 18])
            trace = decide_deterministically(_build(setup))
            expected = (Action.ENTER if not trace.blocked and trace.confluence_score >= 50
                        else Action.SKIP)
            assert trace.verdict is expected


def test_an_existing_valid_signal_is_still_assessed_normally():
    """Phase 10 #15: a clean, valid setup passes the geometry gate and gets
    a full score — the redesign blocks the impossible, not the ordinary."""
    closes = [100.0 + 3.0 * math.sin(i / 4.0) for i in range(300)]     # ranging market
    low_entry = min(closes[-30:]) + 0.5
    setup = TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                       entries=[low_entry], stop_loss=low_entry - 4,
                       take_profits=[low_entry + 8])
    ctx = _build(setup, _Provider(closes))
    trace = decide_deterministically(ctx)

    geometry = next(c for c in ctx.risk.checks if c.name == "trade_geometry")
    assert geometry.verdict.value == "pass"
    assert not trace.blocked
    assert trace.confluence_score is not None
