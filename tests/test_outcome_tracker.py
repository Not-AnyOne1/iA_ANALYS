"""Automatic outcome logging: the evaluation dataset must be honest."""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import outcome_tracker as ot
from market_data import Candle, CandleSeries, MarketDataError, Timeframe
from models import TradeSetup
from storage import Storage

sys.path.insert(0, str(Path(__file__).parent))
from test_confluence_redesign import _build, _closes_trend_then_run, _settings  # noqa: E402

T0 = datetime(2026, 6, 1, 10, 7, tzinfo=timezone.utc)      # decision time, mid-bar


@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")


def _bar(start: datetime, o, h, l, c) -> Candle:
    return Candle(timestamp=start, open=o, high=h, low=l, close=c, volume=1.0)


def _quiet(start: datetime, n: int, price: float = 100.0):
    """``n`` M15 bars from ``start`` that touch nothing: 99.5 - 100.5."""
    return [_bar(start + i * ot.BAR, price, price + 0.5, price - 0.5, price) for i in range(n)]


def _row(**over):
    row = {"id": 1, "recorded_at": T0.isoformat(), "direction": "long", "order_type": "market",
           "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0, "price_at_decision": 100.0}
    row.update(over)
    return row


BAR0 = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)    # the bar in progress at T0
NEXT = BAR0 + ot.BAR                                       # first bar wholly after T0
LATER = NEXT + timedelta(days=10)                          # "now", long after


# ------------------------------------------------------------- the snapshot

def _decision(ctx):
    from trade_decision import TradeDecisionEngine, TradeDecisionError
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(side_effect=TradeDecisionError("offline", retryable=False))
    return asyncio.run(engine.decide(ctx))


def test_the_snapshot_records_every_required_field():
    price = _closes_trend_then_run()[-1]
    ctx = _build(TradeSetup(symbol="XAUUSD", direction="long", order_type="market",
                            entries=[], stop_loss=price - 8, take_profits=[price + 24]))
    row = ot.snapshot(ctx, _decision(ctx))

    required = ("recorded_at", "symbol", "analysis_timeframe", "direction", "entry",
                "stop_loss", "take_profit", "confluence_score", "components_json",
                "range_position", "extension_atr", "atr", "market_regime", "decision")
    for key in required:
        assert row[key] is not None, key
    assert row["status"] == "open"
    components = json.loads(row["components_json"])
    for part in ("direction", "review", "entry", "adjusted_score", "confluence_score", "risk"):
        assert part in components
    assert components["direction"]["groups"]["market_structure"] is not None
    assert row["recorded_at"] == ctx.generated_at.isoformat()


def test_an_impossible_trade_is_recorded_but_never_tracked():
    price = _closes_trend_then_run()[-1]
    ctx = _build(TradeSetup(symbol="XAUUSD", direction="short", order_type="limit",
                            entries=[price], stop_loss=price - 10, take_profits=[price + 20]))
    row = ot.snapshot(ctx, _decision(ctx))
    assert row["status"] == "invalid"
    assert "invalid SHORT" in row["outcome_note"]
    assert row["decision"] == "skip"


# ------------------------------------------------------------- resolution

def test_take_profit_first_is_a_win_with_excursions():
    candles = _quiet(BAR0, 3) + [_bar(NEXT + 3 * ot.BAR, 100, 111, 98, 110)]
    r = ot.resolve(_row(), candles, LATER)
    assert r.status == "tp" and r.outcome == "win"
    assert r.mfe == pytest.approx(11.0) and r.mfe_r == pytest.approx(2.2)
    assert r.mae == pytest.approx(2.0) and r.mae_r == pytest.approx(0.4)
    assert r.minutes_to_tp == pytest.approx(53.0)       # 10:07 -> the 11:00 bar
    assert r.minutes_to_sl is None


def test_stop_loss_first_is_a_loss():
    candles = _quiet(BAR0, 2) + [_bar(NEXT + 2 * ot.BAR, 100, 101, 94, 95)]
    r = ot.resolve(_row(), candles, LATER)
    assert r.status == "sl" and r.outcome == "loss"
    assert r.minutes_to_sl is not None and r.minutes_to_tp is None


def test_both_levels_in_one_bar_is_a_loss():
    """A8: the order inside the bar is unknown, so the conservative reading."""
    candles = _quiet(BAR0, 2) + [_bar(NEXT + ot.BAR, 100, 111, 94, 100)]
    r = ot.resolve(_row(), candles, LATER)
    assert r.outcome == "loss"
    assert "same bar" in r.note


def test_a_short_resolves_as_the_mirror_image():
    row = _row(direction="short", stop_loss=105.0, take_profit=90.0)
    candles = _quiet(BAR0, 2) + [_bar(NEXT + ot.BAR, 100, 101, 89, 90)]
    r = ot.resolve(row, candles, LATER)
    assert r.status == "tp" and r.outcome == "win"


def test_price_before_the_signal_never_counts():
    """No peeking backwards: a target touched before the decision — even in
    the bar that was open when the signal arrived — is not a win."""
    early = [_bar(BAR0 - ot.BAR, 100, 115, 99, 100),     # before the signal
             _bar(BAR0, 100, 115, 99, 100)]              # in progress at the signal
    r = ot.resolve(_row(), early + _quiet(NEXT, 10), NEXT + 11 * ot.BAR)
    assert r is None                                      # still open: nothing decided


def test_bars_that_have_not_closed_never_count():
    """No lookahead: the forming bar is not information yet."""
    candles = _quiet(BAR0, 2) + [_bar(NEXT + ot.BAR, 100, 111, 99, 110)]
    still_forming = NEXT + ot.BAR + timedelta(minutes=5)
    assert ot.resolve(_row(), candles, still_forming) is None
    assert ot.resolve(_row(), candles, still_forming + ot.BAR).outcome == "win"


def test_a_resting_order_must_fill_first():
    row = _row(order_type="limit", entry=98.0, stop_loss=94.0, take_profit=106.0)
    no_fill = _quiet(BAR0, 200)                           # never trades down to 98
    assert ot.resolve(row, no_fill[:20], NEXT + 20 * ot.BAR) is None          # window open
    assert ot.resolve(row, no_fill, NEXT + 200 * ot.BAR).status == "not_filled"


def test_a_target_on_the_fill_bar_does_not_count():
    """It may have been touched before the order filled."""
    row = _row(order_type="limit", entry=98.0, stop_loss=94.0, take_profit=106.0)
    fill_and_spike = _bar(NEXT, 100, 107, 97.5, 99)
    r = ot.resolve(row, _quiet(BAR0, 1) + [fill_and_spike] + _quiet(NEXT + ot.BAR, 5, 99),
                   NEXT + 6 * ot.BAR)
    assert r is None                                       # filled, still open
    r = ot.resolve(row, _quiet(BAR0, 1) + [fill_and_spike, _bar(NEXT + ot.BAR, 99, 106.5, 98.5, 106)],
                   LATER)
    assert r.status == "tp"
    assert r.filled_at == NEXT


def test_a_stop_on_the_fill_bar_is_a_loss():
    row = _row(order_type="limit", entry=98.0, stop_loss=94.0, take_profit=106.0)
    r = ot.resolve(row, _quiet(BAR0, 1) + [_bar(NEXT, 100, 100, 93, 94)], LATER)
    assert r.outcome == "loss"


def test_nothing_touched_within_the_horizon_expires():
    candles = _quiet(BAR0, 600)
    r = ot.resolve(_row(), candles, BAR0 + 600 * ot.BAR, horizon=timedelta(days=5))
    assert r.status == "expired" and r.outcome is None
    assert r.mfe is not None and r.mae is not None


def test_history_that_does_not_reach_the_signal_is_no_data():
    r = ot.resolve(_row(), _quiet(NEXT + timedelta(hours=3), 10), LATER)
    assert r.status == "no_data"


def test_every_resolved_number_is_finite():
    for candles in (_quiet(BAR0, 2) + [_bar(NEXT + ot.BAR, 100, 111, 98, 110)],
                    _quiet(BAR0, 2) + [_bar(NEXT + ot.BAR, 100, 101, 94, 95)]):
        r = ot.resolve(_row(), candles, LATER)
        for v in (r.mfe, r.mae, r.mfe_r, r.mae_r):
            assert v is not None and math.isfinite(v)


# ------------------------------------------------------------- storage

def _storage(tmp_path) -> Storage:
    s = Storage(tmp_path / "o.db")
    asyncio.run(s.initialize())
    return s


def _snapshot_row(**over):
    row = {"recorded_at": T0.isoformat(), "symbol": "XAUUSD", "direction": "long",
           "order_type": "market", "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0,
           "price_at_decision": 100.0, "decision": "enter", "confluence_score": 52,
           "status": "open"}
    row.update(over)
    return row


def test_a_resolved_outcome_can_never_be_overwritten(tmp_path):
    s = _storage(tmp_path)
    row_id = asyncio.run(s.record_signal_outcome(_snapshot_row()))
    assert [r["id"] for r in asyncio.run(s.open_signal_outcomes())] == [row_id]

    win = ot.Resolution("tp", "win", T0, NEXT, 11.0, 2.0, 2.2, 0.4, 53.0, None).as_row()
    assert asyncio.run(s.resolve_signal_outcome(row_id, win)) is True
    loss = ot.Resolution("sl", "loss", T0, NEXT, 0.0, 5.0, 0.0, 1.0, None, 10.0).as_row()
    assert asyncio.run(s.resolve_signal_outcome(row_id, loss)) is False      # no-op

    stored = asyncio.run(s.fetch_signal_outcomes())[0]
    assert stored["status"] == "tp" and stored["outcome"] == "win"
    assert asyncio.run(s.open_signal_outcomes()) == []
    with pytest.raises(ValueError):
        asyncio.run(s.resolve_signal_outcome(row_id, {"status": "open"}))


def test_recording_never_raises(tmp_path):
    s = _storage(tmp_path)
    s._db_path = tmp_path / "missing-dir" / "x" / "o.db"     # unwritable path
    assert asyncio.run(s.record_signal_outcome(_snapshot_row())) is None


# ------------------------------------------------------------- the tracker

class _Market:
    def __init__(self, candles=None, fail=False):
        self.calls = 0
        self._candles, self._fail = candles, fail

    async def get_candles(self, symbol, timeframe, limit):
        self.calls += 1
        if self._fail:
            raise MarketDataError("down", retryable=False)
        assert timeframe is Timeframe.M15
        return CandleSeries(symbol=symbol, timeframe=timeframe, provider="stub",
                            candles=tuple(self._candles))


def test_the_tracker_resolves_open_rows_one_fetch_per_symbol(tmp_path):
    s = _storage(tmp_path)
    for _ in range(3):
        asyncio.run(s.record_signal_outcome(_snapshot_row()))
    market = _Market(_quiet(BAR0, 2) + [_bar(NEXT + ot.BAR, 100, 111, 98, 110)])
    tracker = ot.OutcomeTracker(s, market, clock=lambda: LATER)

    assert asyncio.run(tracker.resolve_pending()) == 3
    assert market.calls == 1
    assert {r["outcome"] for r in asyncio.run(s.fetch_signal_outcomes())} == {"win"}


def test_a_data_outage_leaves_rows_open(tmp_path):
    s = _storage(tmp_path)
    asyncio.run(s.record_signal_outcome(_snapshot_row()))
    tracker = ot.OutcomeTracker(s, _Market(fail=True), clock=lambda: LATER)
    assert asyncio.run(tracker.resolve_pending()) == 0
    assert len(asyncio.run(s.open_signal_outcomes())) == 1


def test_the_pipeline_records_every_decision(tmp_path):
    """SKIP included — refused trades are part of the dataset."""
    from formatter import Formatter
    from market_context import MarketContextBuilder
    from market_data import MarketDataService
    from models import SignalAnalysis
    from pipeline import AnalysisPipeline
    from telegram_client import IncomingMessage
    from trade_decision import TradeDecisionEngine, TradeDecisionError
    from test_confluence_redesign import _Provider

    price = _closes_trend_then_run()[-1]
    analysis = SignalAnalysis(
        is_signal=True, category="signal",
        setup=TradeSetup(symbol="XAUUSD", direction="long", order_type="market",
                         entries=[], stop_loss=price - 8, take_profits=[price + 24]),
        summary="s", confidence=0.9, missing_fields=[], notes=None, source="regex")

    class _Analyzer:
        async def analyze(self, message):
            return analysis

    settings = _settings()
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(side_effect=TradeDecisionError("offline", retryable=False))
    tracker = AsyncMock()
    message = IncomingMessage(id=7, chat_id=1, chat_title="g", sender="s",
                              text="BUY XAUUSD", timestamp=T0)

    async def go():
        async with AnalysisPipeline(
                settings, asyncio.Queue(), _Analyzer(), Formatter(color=False, show_json=False),
                context_builder=MarketContextBuilder(MarketDataService(_Provider())),
                trade_decision_engine=engine, emit_reports=False,
                outcome_tracker=tracker) as pipeline:
            await pipeline._process(message)

    asyncio.run(go())
    tracker.record.assert_awaited_once()
    context, decision, recorded_message = tracker.record.await_args.args
    assert recorded_message is message
    assert decision.verdict.value in ("enter", "skip")
