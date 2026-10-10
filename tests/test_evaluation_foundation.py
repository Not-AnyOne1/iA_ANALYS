"""P1 evaluation foundation: versions, costs, storage, shadow mode, tracker, dashboard."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import engine_version as ev
import shadow
from costs import CostModel, Costs
from market_data import Candle, CandleSeries, Quote
from models import TradeSetup
from outcome_tracker import OutcomeTracker
from storage import Storage
from trade_decision import Action, DecisionTrace, decide_deterministically

sys.path.insert(0, str(Path(__file__).parent))
from test_confluence_redesign import _build, _closes_trend_then_run, _settings  # noqa: E402

T0 = datetime(2026, 10, 12, 10, 7, tzinfo=timezone.utc)
BAR0 = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)
BAR = timedelta(minutes=15)


@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")


def _storage(tmp_path) -> Storage:
    s = Storage(tmp_path / "e.db")
    asyncio.run(s.initialize())
    return s


def _outcome_row(**over):
    row = {"recorded_at": T0.isoformat(), "symbol": "XAUUSD", "direction": "long",
           "order_type": "market", "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0,
           "take_profits_json": "[110.0]", "price_at_decision": 100.0, "decision": "enter",
           "confluence_score": 60, "status": "open"}
    row.update(over)
    return row


# ------------------------------------------------------------------ version

def test_engine_version_is_code_plus_parameter_hash(monkeypatch):
    monkeypatch.setenv("ENGINE_COMMIT", "abc1234")
    ev.code_version.cache_clear()
    try:
        version = ev.engine_version()
        assert version.startswith("abc1234+p") and len(version.split("+p")[1]) == 10
    finally:
        ev.code_version.cache_clear()


def test_a_railway_image_is_stamped_with_the_commit_it_was_built_from(monkeypatch):
    """The image has no .git; without this every deployed decision would read 'nogit'."""
    monkeypatch.delenv("ENGINE_COMMIT", raising=False)
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "0123456789abcdef0123456789abcdef01234567")
    ev.code_version.cache_clear()
    try:
        assert ev.code_version() == "0123456"
    finally:
        ev.code_version.cache_clear()


def test_changing_a_decision_parameter_changes_the_stamp(monkeypatch):
    import trade_decision
    before = ev.parameter_hash()
    monkeypatch.setattr(trade_decision, "ENTER_MIN_SCORE", 55)
    assert ev.parameter_hash() != before


def test_the_stamp_is_stable_for_unchanged_rules():
    assert ev.parameter_hash() == ev.parameter_hash()


# ------------------------------------------------------------------ costs

def test_the_assumption_is_used_and_labelled_when_no_bid_ask_exists():
    model = CostModel()
    costs = asyncio.run(model.costs_for("XAUUSD", Quote("XAUUSD", 4100.0, T0, "twelvedata")))
    assert costs == Costs(0.30, 0.10, "assumed")


def test_real_bid_ask_wins_over_the_assumption():
    quote = Quote("XAUUSD", 4100.0, T0, "oanda", bid=4099.9, ask=4100.15)
    costs = asyncio.run(CostModel().costs_for("XAUUSD", quote))
    assert costs.spread == pytest.approx(0.25) and costs.source == "bid/ask:oanda"
    assert costs.slippage == pytest.approx(0.10)          # slippage is still assumed


def test_a_symbol_without_an_assumption_has_no_cost_figure():
    assert asyncio.run(CostModel().costs_for("BTCUSDT", None)) is None


def test_assumptions_can_be_overridden_and_must_be_sane(monkeypatch):
    monkeypatch.setenv("COST_SPREAD_XAUUSD", "0.45")
    assert CostModel.from_env().assumption("XAUUSD").spread == pytest.approx(0.45)
    monkeypatch.setenv("COST_SPREAD_XAUUSD", "-1")
    with pytest.raises(ValueError):
        CostModel.from_env()


def test_a_dedicated_bid_ask_source_is_used_when_configured():
    class Source:
        async def get_quote(self, symbol):
            return Quote(symbol, 4100.0, T0, "mt5", bid=4099.8, ask=4100.1)
    costs = asyncio.run(CostModel(quote_source=Source()).costs_for("XAUUSD", None))
    assert costs.source == "bid/ask:mt5" and costs.spread == pytest.approx(0.3)


# ------------------------------------------------------------------ storage

def test_an_existing_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    from storage import _SCHEMA
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO signal_outcomes (recorded_at, symbol, status) VALUES (?, ?, 'open')",
                 (T0.isoformat(), "XAUUSD"))
    conn.commit()
    conn.close()
    s = Storage(path)
    asyncio.run(s.initialize())
    rows = asyncio.run(s.fetch_performance_rows())
    assert len(rows) == 1 and rows[0]["engine_version"] is None and rows[0]["cost_source"] is None


def test_duplicates_point_at_the_first_signal_of_the_cluster(tmp_path):
    s = _storage(tmp_path)
    first = asyncio.run(s.record_signal_outcome(_outcome_row()))
    second_row = _outcome_row(recorded_at=(T0 + timedelta(minutes=5)).isoformat(), entry=100.05)
    second_row["duplicate_of"] = asyncio.run(s.find_duplicate(second_row))
    second = asyncio.run(s.record_signal_outcome(second_row))
    third_row = _outcome_row(recorded_at=(T0 + timedelta(minutes=9)).isoformat())
    assert second_row["duplicate_of"] == first
    assert asyncio.run(s.find_duplicate(third_row)) == first      # not the second
    assert second != first
    different = _outcome_row(recorded_at=(T0 + timedelta(minutes=9)).isoformat(), stop_loss=94.0)
    assert asyncio.run(s.find_duplicate(different)) is None


def test_storage_and_report_apply_the_same_duplicate_rule(tmp_path):
    import performance as pf
    s = _storage(tmp_path)
    variants = [_outcome_row(), _outcome_row(recorded_at=(T0 + timedelta(minutes=5)).isoformat()),
                _outcome_row(recorded_at=(T0 + timedelta(hours=7)).isoformat()),
                _outcome_row(recorded_at=(T0 + timedelta(hours=7, minutes=2)).isoformat(), entry=103.0)]
    tagged = set()
    for row in variants:
        dup = asyncio.run(s.find_duplicate(row))
        row_id = asyncio.run(s.record_signal_outcome(row))
        if dup is not None:
            tagged.add(row_id)
    rows = asyncio.run(s.fetch_performance_rows())
    for r in rows:
        r["duplicate_of"] = None                          # force the pure rule
    assert pf.duplicate_ids(rows) == tagged


def test_accounting_is_written_once(tmp_path):
    s = _storage(tmp_path)
    oid = asyncio.run(s.record_signal_outcome(_outcome_row()))
    result = {"basis": "net", "status": "tp", "r": 1.9}
    assert asyncio.run(s.record_accounting(oid, result, 7200)) is True
    assert asyncio.run(s.record_accounting(oid, {**result, "r": -1.0}, 7200)) is False
    row = asyncio.run(s.fetch_performance_rows())[0]
    assert row["accounting"]["net"]["r"] == pytest.approx(1.9)
    with pytest.raises(ValueError):
        asyncio.run(s.record_accounting(oid, {"basis": "gross", "status": "open"}, 7200))


# ------------------------------------------------------------------ shadow

def _trace(verdict=Action.SKIP, score=40):
    return DecisionTrace(deterministic_score=score, validation_bonus=0, validation_penalty=0,
                         adjusted_score=score, verdict=verdict, reason="x", blocked=False,
                         confluence_score=score)


class _Ctx:
    generated_at = T0


def test_the_champion_is_recorded_as_decided_never_recomputed():
    records = shadow.shadow_decisions(_Ctx(), _trace(Action.SKIP, 41), challengers=())
    assert records == [records[0]] and records[0]["variant"] == "champion"
    assert records[0]["verdict"] == "skip" and records[0]["confluence_score"] == 41
    assert records[0]["engine_version"] == ev.engine_version()


def test_a_failing_challenger_is_recorded_and_isolated():
    def boom(ctx):
        raise RuntimeError("bad challenger")
    good = shadow.Variant("eager", "enters more", lambda ctx: _trace(Action.ENTER, 70), {"t": 1})
    bad = shadow.Variant("broken", "fails", boom)
    records = shadow.shadow_decisions(_Ctx(), _trace(), challengers=(good, bad))
    by_name = {r["variant"]: r for r in records}
    assert by_name["champion"]["verdict"] == "skip"
    assert by_name["eager"]["verdict"] == "enter"
    assert by_name["eager"]["engine_version"] != by_name["champion"]["engine_version"]
    assert by_name["broken"]["verdict"] == "error" and "bad challenger" in by_name["broken"]["reason"]


def test_no_challenger_is_live_in_p1():
    assert tuple(shadow.CHALLENGERS) == ()


def test_a_challenger_can_never_change_what_is_sent(monkeypatch, tmp_path):
    """The live card is the champion's, whatever a challenger says or does."""
    from formatter import Formatter
    from market_context import MarketContextBuilder
    from market_data import MarketDataService
    from models import SignalAnalysis
    from pipeline import AnalysisPipeline
    from telegram_client import IncomingMessage
    from trade_decision import TradeDecisionEngine, TradeDecisionError
    from test_confluence_redesign import _Provider

    eager = shadow.Variant("eager", "always enters", lambda ctx: _trace(Action.ENTER, 99), {"e": 1})
    def boom(ctx):
        raise RuntimeError("bad challenger")
    monkeypatch.setattr(shadow, "CHALLENGERS", (eager, shadow.Variant("broken", "fails", boom)))

    price = _closes_trend_then_run()[-1]
    analysis = SignalAnalysis(is_signal=True, category="signal",
                              setup=TradeSetup(symbol="XAUUSD", direction="long", order_type="market",
                                               entries=[], stop_loss=price - 8,
                                               take_profits=[price + 24]),
                              summary="s", confidence=0.9, missing_fields=[], notes=None,
                              source="regex")

    class _Analyzer:
        async def analyze(self, message):
            return analysis

    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(side_effect=TradeDecisionError("offline", retryable=False))
    tracker = AsyncMock()
    sent = []
    bot = AsyncMock()
    bot.send_report = AsyncMock(side_effect=lambda msg, text: sent.append(text))
    message = IncomingMessage(id=7, chat_id=1, chat_title="g", sender="s", text="BUY", timestamp=T0)

    async def go():
        async with AnalysisPipeline(_settings(), asyncio.Queue(), _Analyzer(),
                                    Formatter(color=False, show_json=False), bot=bot,
                                    context_builder=MarketContextBuilder(MarketDataService(_Provider())),
                                    trade_decision_engine=engine, emit_reports=False,
                                    outcome_tracker=tracker) as pipeline:
            await pipeline._process(message)

    asyncio.run(go())
    context, decision, _ = tracker.record.await_args.args
    variants = {v["variant"]: v for v in tracker.record.await_args.kwargs["variants"]}
    assert decision.verdict.value == decide_deterministically(context).verdict.value
    assert variants["champion"]["verdict"] == decision.verdict.value
    assert variants["eager"]["verdict"] == "enter" and variants["broken"]["verdict"] == "error"
    assert len(sent) == 1 and f"{decision.verdict.value.upper()}" in sent[0]


# ------------------------------------------------------------------ tracker

def _closes_to_m15(at, n=40, price=100.0, rally_at=None):
    bars = []
    for i in range(n):
        ts = at - 4 * BAR + i * BAR
        hi = price + (12 if rally_at is not None and i == rally_at else 0.5)
        bars.append(Candle(timestamp=ts, open=price, high=hi, low=price - 0.5, close=price, volume=1.0))
    return bars


class _Market:
    def __init__(self, candles):
        self.candles = candles

    async def get_candles(self, symbol, timeframe, limit):
        return CandleSeries(symbol=symbol, timeframe=timeframe, provider="stub",
                            candles=tuple(self.candles))


def test_recording_stamps_version_costs_and_variants(tmp_path):
    s = _storage(tmp_path)
    price = _closes_trend_then_run()[-1]
    ctx = _build(TradeSetup(symbol="XAUUSD", direction="long", order_type="market", entries=[],
                            stop_loss=price - 8, take_profits=[price + 24]))
    from trade_decision import TradeDecisionEngine, TradeDecisionError
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(side_effect=TradeDecisionError("offline", retryable=False))
    decision = asyncio.run(engine.decide(ctx))
    tracker = OutcomeTracker(s, _Market([]), cost_model=CostModel())
    variants = shadow.shadow_decisions(ctx, decide_deterministically(ctx), challengers=())
    oid = asyncio.run(tracker.record(ctx, decision, variants=variants))
    row = asyncio.run(s.fetch_performance_rows())[0]
    assert row["id"] == oid
    assert row["engine_version"] == ev.engine_version()
    # The test quote carries a bid/ask (price +/- 0.05), so the MEASURED spread
    # is used rather than the assumption, and the source says so.
    assert row["cost_spread"] == pytest.approx(0.10)
    assert row["cost_slippage"] == pytest.approx(0.10)
    assert row["cost_source"] == "bid/ask:stub"
    assert row["variants"]["champion"]["verdict"] == decision.verdict.value


def test_resolved_history_is_backfilled_once_without_touching_outcomes(tmp_path):
    s = _storage(tmp_path)
    oid = asyncio.run(s.record_signal_outcome(_outcome_row()))     # pre-P1 row: no costs
    asyncio.run(s.resolve_signal_outcome(oid, {"status": "tp", "outcome": "win"}))
    candles = _closes_to_m15(BAR0, rally_at=6)
    later = BAR0 + 50 * BAR
    tracker = OutcomeTracker(s, _Market(candles), cost_model=CostModel(), clock=lambda: later)
    asyncio.run(tracker.resolve_pending())
    row = asyncio.run(s.fetch_performance_rows())[0]
    assert row["status"] == "tp" and row["outcome"] == "win"          # raw outcome untouched
    assert row["accounting"]["gross"]["status"] == "tp"
    assert row["accounting"]["net"]["cost_source"] == "assumed (backfill)"
    assert row["accounting"]["net"]["r"] < row["accounting"]["gross"]["r"]
    stamp = row["accounting"]["net"]["computed_at"]
    asyncio.run(tracker.resolve_pending())
    assert asyncio.run(s.fetch_performance_rows())[0]["accounting"]["net"]["computed_at"] == stamp


# ------------------------------------------------------------------ dashboard

def test_the_performance_endpoint_returns_a_labelled_report(tmp_path):
    import dashboard
    s = _storage(tmp_path)
    data = asyncio.run(dashboard.get_performance(s, basis="net", period="all"))
    assert data["basis"] == "net" and data["period"] == "all"
    assert {"enter", "skip", "all_signals", "missed_winners", "by_direction"} <= set(data)
    json.dumps(data)


def test_the_performance_route_rejects_bad_parameters(tmp_path):
    import dashboard
    s = _storage(tmp_path)
    for params in ({"basis": "blended"}, {"period": "future"}):
        with pytest.raises(ValueError):
            asyncio.run(dashboard._route_performance(s, None, params))
