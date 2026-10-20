"""Strategy shadow evaluation (strategy_shadow/). All prices are TEST DATA."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from costs import Costs
from market_data import Candle, CandleSeries, MarketDataError, MarketDataService, Timeframe
from models import SignalAnalysis, TradeSetup
from storage import Storage
from telegram_bot import BotSettings, TelegramBot
from telegram_client import IncomingMessage
from trade_decision import decide_deterministically

import strategy_variants
from strategy_shadow import report as shadow_report
from strategy_shadow.evaluate import evaluate
from strategy_shadow.outcomes import StrategyOutcomeResolver
from strategy_shadow.service import StrategyShadow, signal_trade
from strategy_shadow.snapshot import LabelCalibrator, assess, fingerprint
from strategy_shadow.store import SCHEMA_VERSION, StrategyShadowStore, load_readonly
from strategy_shadow.versions import STRATEGIES, rules_hash
from test_strategy_variants import _Provider, context

UTC = timezone.utc
COSTS = Costs(0.30, 0.10, "assumed")


def run(coro):
    return asyncio.run(coro)


def analysis_for(ctx, direction="long", entries=(), order="market"):
    price = ctx.current_price
    s = 1.0 if direction == "long" else -1.0
    setup = TradeSetup(symbol="XAUUSD", direction=direction, order_type=order, entries=list(entries),
                       stop_loss=round(price - s * 6.0, 2), take_profits=[round(price + s * 12.0, 2)])
    return SignalAnalysis(is_signal=True, category="signal", setup=setup, summary="TEST",
                          confidence=0.9, missing_fields=[], notes=None, source="regex")


def message(ctx, message_id=1, chat_id=-100):
    return IncomingMessage(id=message_id, chat_id=chat_id, chat_title="TEST", sender="TEST",
                           timestamp=ctx.generated_at - timedelta(seconds=30), text="TEST")


def live_decision(ctx):
    t = decide_deterministically(ctx)
    return SimpleNamespace(verdict=t.verdict, adjusted_score=t.adjusted_score, decision_reason=t.reason)


class _UtcProvider:
    """Serves the synthetic scenario's M15 bars as UTC-labelled bars."""

    name = "utc"

    def __init__(self, fail=False):
        self.provider, self.fail, self.calls = _Provider(), fail, 0

    async def get_quote(self, symbol):
        raise AssertionError("no quote is needed")

    async def get_candles(self, symbol, timeframe, limit):
        self.calls += 1
        if self.fail:
            raise MarketDataError("provider down", retryable=False)
        return CandleSeries(symbol, timeframe, "utc", self.provider.frames[timeframe][-limit:])


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "signals.db"
    run(Storage(path).initialize())                  # the live tables exist first
    s = StrategyShadowStore(path)
    run(s.initialize())
    return s


def shadow(store, fail=False, **kw):
    utc = _UtcProvider(fail=fail)
    return StrategyShadow(store, LabelCalibrator(MarketDataService(utc, max_retries=0)), **kw), utc


# --------------------------------------------------------- isolation & flags

def test_shadow_mode_is_off_by_default_and_flags_are_strict(monkeypatch):
    from config import ConfigError, Settings
    for name in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_TARGET_CHAT"):
        monkeypatch.setenv(name, "1" if name == "TELEGRAM_API_ID" else "test")
    for name in ("STRATEGY_SHADOW_MODE", "STRATEGY_SHADOW_REPORTING", "STRATEGY_SHADOW_CONFIRMATION_START"):
        monkeypatch.delenv(name, raising=False)
    s = Settings.load(env_file=None)
    assert (s.strategy_shadow_mode, s.strategy_shadow_reporting, s.strategy_shadow_confirmation_start) == \
        ("off", "off", None)
    monkeypatch.setenv("STRATEGY_SHADOW_MODE", "ON")
    monkeypatch.setenv("STRATEGY_SHADOW_CONFIRMATION_START", "2027-01-01")
    s = Settings.load(env_file=None)
    assert s.strategy_shadow_mode == "on" and s.strategy_shadow_confirmation_start == date(2027, 1, 1)
    monkeypatch.setenv("STRATEGY_SHADOW_MODE", "live")
    with pytest.raises(ConfigError, match="STRATEGY_SHADOW_MODE"):
        Settings.load(env_file=None)
    monkeypatch.setenv("STRATEGY_SHADOW_MODE", "on")
    monkeypatch.setenv("STRATEGY_SHADOW_CONFIRMATION_START", "next week")
    with pytest.raises(ConfigError, match="CONFIRMATION_START"):
        Settings.load(env_file=None)


def test_recording_does_not_touch_the_live_decision(store):
    ctx = context("long")
    before = decide_deterministically(ctx)
    decision = live_decision(ctx)
    sh, _ = shadow(store)
    run(sh.on_signal(message(ctx), analysis_for(ctx), ctx, decision))
    assert decide_deterministically(ctx) == before                 # the context is untouched
    assert decision.verdict is before.verdict


def test_each_strategy_is_isolated(monkeypatch):
    ctx = context("long")
    real = strategy_variants.decide

    def flaky(c, variant):
        if variant == "valid_events":
            raise RuntimeError("boom")
        return real(c, variant)
    monkeypatch.setattr(strategy_variants, "decide", flaky)
    decisions = evaluate(ctx)
    assert decisions["valid_events"].verdict is None and "boom" in decisions["valid_events"].error
    for s in ("baseline", "htf_regime", "range_filter"):
        assert decisions[s].verdict in ("ENTER", "WAIT", "SKIP") and decisions[s].error is None
    assert decisions["valid_events"].score_label == "Setup score: Not scored — evaluation error"


def test_the_same_snapshot_feeds_every_strategy(store, monkeypatch):
    ctx = context("long")
    seen = []
    real = strategy_variants.decide
    monkeypatch.setattr(strategy_variants, "decide", lambda c, v: (seen.append(id(c)), real(c, v))[1])
    sh, _ = shadow(store)
    run(sh.on_signal(message(ctx), analysis_for(ctx), ctx, live_decision(ctx)))
    assert set(seen) == {id(ctx)} and len(seen) == len(STRATEGIES)
    row = load_readonly(store.path)["signals"][0]
    assert row["snapshot_fingerprint"] == fingerprint(ctx)
    assert row["baseline_matches_live"] == 1 and row["rules_hash"] == rules_hash()


def test_buy_sell_symmetry_of_every_strategy():
    long_ctx = context("long")
    short_ctx = context("short", mirror=2 * long_ctx.current_price)
    a, b = evaluate(long_ctx), evaluate(short_ctx)
    for s in STRATEGIES:
        assert (a[s].verdict, a[s].setup_score) == (b[s].verdict, b[s].setup_score), s


# ------------------------------------------------------------- data quality

def test_fresh_snapshot_with_a_measured_offset():
    ctx = context("long")
    q = assess(ctx, timedelta(0), analysis_for(ctx).setup)
    assert q.status == "fresh", q.problems
    # Like the live provider, the scenario's last M15 bar is still forming.
    assert q.frames["M15"]["forming_bar"] is True and q.frames["M15"]["weekday_gaps"] == 0
    assert q.frames["M15"]["last_closed_age_min"] <= 15


def test_label_shift_is_measured_and_unknown_shift_is_unverified():
    ctx = context("long")
    shifted = [Candle(c.timestamp + timedelta(hours=11), c.open, c.high, c.low, c.close)
               for c in ctx.candle_series[Timeframe.M15].candles]
    cal = LabelCalibrator(MarketDataService(_UtcProvider(), max_retries=0))
    assert run(cal.offset(shifted, ctx.generated_at)) == timedelta(hours=11)
    assert run(cal.offset(shifted, ctx.generated_at)) == timedelta(hours=11) and cal.requests == 1   # cached
    assert assess(ctx, None).status == "unverified"


def test_an_ambiguous_recalibration_keeps_the_measured_offset():
    ctx = context("long")
    live = list(ctx.candle_series[Timeframe.M15].candles)
    utc = _UtcProvider()
    cal = LabelCalibrator(MarketDataService(utc, max_retries=0), refresh=timedelta(seconds=1))
    assert run(cal.offset(live, ctx.generated_at)) == timedelta(0)
    flat = [Candle(c.timestamp, 1.0, 1.0, 1.0, 1.0) for c in utc.provider.frames[Timeframe.M15]]
    utc.provider.frames = {**utc.provider.frames, Timeframe.M15: tuple(flat)}       # stub-like, ambiguous
    later = ctx.generated_at + timedelta(hours=1)
    assert run(cal.offset(live, later)) == timedelta(0) and cal.requests == 2
    fresh = LabelCalibrator(MarketDataService(utc, max_retries=0))
    assert run(fresh.offset(live, later)) is None                 # never measured: unknown


def test_stale_incomplete_closed_and_unit_problems():
    ctx = context("long")
    late = replace(ctx, generated_at=ctx.generated_at + timedelta(hours=3))
    assert assess(late, timedelta(0)).status == "stale"
    missing = replace(ctx, candle_series={k: v for k, v in ctx.candle_series.items() if k is not Timeframe.H4})
    assert assess(missing, timedelta(0)).status == "incomplete"
    weekend = replace(ctx, generated_at=datetime(2026, 3, 21, 12, tzinfo=UTC))
    assert assess(weekend, timedelta(0)).status == "market_closed"
    odd = TradeSetup(symbol="XAUUSD", direction="long", stop_loss=10.0, take_profits=[ctx.current_price + 5])
    assert any("units" in p for p in assess(ctx, timedelta(0), odd).problems)


def test_a_provider_failure_still_records_the_evaluation(store):
    ctx = context("long")
    sh, utc = shadow(store, fail=True)
    assert run(sh.on_signal(message(ctx), analysis_for(ctx), ctx, live_decision(ctx))) is not None
    row = load_readonly(store.path)["signals"][0]
    assert row["data_status"] == "unverified" and utc.calls == 1


# ------------------------------------------------------------ persistence

def test_schema_is_added_beside_the_live_tables_idempotently(store):
    run(store.initialize())
    con = sqlite3.connect(store.path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"analyses", "subscribers", "strategy_shadow_signals", "strategy_shadow_decisions",
            "strategy_shadow_outcomes"} <= tables
    assert con.execute("SELECT value FROM strategy_shadow_meta").fetchone()[0] == str(SCHEMA_VERSION)
    con.execute("UPDATE strategy_shadow_meta SET value = '99'")
    con.commit()
    with pytest.raises(RuntimeError, match="newer than this code"):
        run(store.initialize())


def test_duplicates_after_restart_and_forwarded_duplicates(store):
    ctx = context("long")
    sh, _ = shadow(store)
    first = run(sh.on_signal(message(ctx, 7), analysis_for(ctx), ctx, live_decision(ctx)))
    restarted, _ = shadow(StrategyShadowStore(store.path))          # a new process, same file
    assert run(restarted.on_signal(message(ctx, 7), analysis_for(ctx), ctx, live_decision(ctx))) is None
    forwarded = run(restarted.on_signal(message(ctx, 8), analysis_for(ctx), ctx, live_decision(ctx)))
    rows = load_readonly(store.path)["signals"]
    assert len(rows) == 2 and rows[1]["id"] == forwarded and rows[1]["duplicate_of"] == first
    assert len(load_readonly(store.path)["decisions"]) == 2 * len(STRATEGIES)


def test_records_are_immutable(store):
    ctx = context("long")
    sh, _ = shadow(store)
    sid = run(sh.on_signal(message(ctx), analysis_for(ctx), ctx, live_decision(ctx)))
    con = sqlite3.connect(store.path)
    for table in ("strategy_shadow_signals", "strategy_shadow_decisions"):
        with pytest.raises(sqlite3.IntegrityError, match="write-once"):
            con.execute(f"UPDATE {table} SET id = id")
        with pytest.raises(sqlite3.IntegrityError, match="write-once"):
            con.execute(f"DELETE FROM {table}")
    assert sid is not None


# --------------------------------------------------------------- outcomes

def _bars(start, path):
    out, ts, prev = [], start, path[0][2]
    for high, low, close in path:
        out.append(Candle(ts, prev, high, low, close))
        ts, prev = ts + timedelta(minutes=15), close
    return out


class _Bars:
    def __init__(self, bars, fail=False):
        self.bars, self.fail, self.calls = bars, fail, 0

    async def get_candles(self, symbol, timeframe, limit):
        self.calls += 1
        if self.fail:
            raise MarketDataError("down", retryable=False)
        return CandleSeries(symbol, timeframe, "test", tuple(self.bars))


def _seed(store, ctx, setup_kwargs=None, message_id=1):
    sh, _ = shadow(store)
    a = analysis_for(ctx, **(setup_kwargs or {}))
    return run(sh.on_signal(message(ctx, message_id), a, ctx, live_decision(ctx))), a.setup


def test_tp_sl_costs_and_corrections(store):
    ctx = context("long")
    sid, setup = _seed(store, ctx)
    p, at = ctx.current_price, ctx.generated_at
    bars = _bars(at - timedelta(minutes=15), [(p + 0.1, p - 0.1, p), (p + 13, p - 1, p + 12.5)])
    resolver = StrategyOutcomeResolver(store, _Bars(bars), costs=COSTS)
    assert run(resolver.resolve_once(at + timedelta(minutes=30))) == 2
    out = {o["basis"]: o for o in load_readonly(store.path)["outcomes"]}
    assert out["gross"]["status"] == "tp" and out["gross"]["r"] == pytest.approx(2.0)
    assert out["net"]["r"] < out["gross"]["r"] and out["net"]["cost_source"] == "assumed"
    assert run(resolver.resolve_once(at + timedelta(minutes=45))) == 0            # never overwritten
    from trade_accounting import TradeResult
    fix = {"net": TradeResult("net", "sl", r=-1.05, costs=COSTS, note="TEST correction")}
    run(store.record_outcome(sid, fix, at + timedelta(days=1), correction_reason="TEST: data fix"))
    con = sqlite3.connect(store.path)
    revs = con.execute("SELECT revision, status FROM strategy_shadow_outcomes WHERE basis='net' "
                       "ORDER BY revision").fetchall()
    assert revs == [(1, "tp"), (2, "sl")]                          # the original stays readable
    latest = {o["basis"]: o for o in load_readonly(store.path)["outcomes"]}
    assert latest["net"]["status"] == "sl" and latest["net"]["correction_reason"] == "TEST: data fix"


def test_unfilled_and_unresolved_trades_are_not_wins_or_losses(store):
    ctx = context("long")
    p, at = ctx.current_price, ctx.generated_at
    _seed(store, ctx, {"entries": (p - 4,), "order": "limit"}, message_id=1)      # never reached
    _seed(store, ctx, {}, message_id=2)                                            # still open
    flat = _bars(at - timedelta(minutes=15), [(p + 0.5, p - 0.5, p)] * 120)        # 30 h, inside the band
    resolver = StrategyOutcomeResolver(store, _Bars(flat), costs=COSTS)
    run(resolver.resolve_once(at + timedelta(hours=29)))
    data = load_readonly(store.path)
    status = {o["signal_id"]: o["status"] for o in data["outcomes"] if o["basis"] == "net"}
    assert list(status.values()) == ["not_filled"]
    rep = shadow_report.build(data, include_duplicates=True)
    assert rep["resolved"] == 0 and rep["outcome_status"] == {"not_filled": 1, "unresolved": 1}
    for e in rep["strategies"].values():
        assert e["enter_net"]["n"] == 0


def test_the_resolver_survives_an_outage_and_idles_without_requests(store):
    ctx = context("long")
    _seed(store, ctx)
    down = _Bars([], fail=True)
    assert run(StrategyOutcomeResolver(store, down, costs=COSTS).resolve_once(
        ctx.generated_at + timedelta(hours=1))) == 0
    assert len(run(store.pending())) == 1
    idle = _Bars([])
    empty = StrategyShadowStore(store.path.parent / "empty.db")
    run(empty.initialize())
    assert run(StrategyOutcomeResolver(empty, idle, costs=COSTS).resolve_once()) == 0 and idle.calls == 0


def test_untrackable_signals_are_recorded_with_the_reason(store):
    ctx = context("long")
    setup = TradeSetup(symbol="XAUUSD", direction="long", stop_loss=None, take_profits=[])
    trade, why = signal_trade(setup, ctx.current_price, ctx.generated_at)
    assert trade is None and why == "no stop or no take-profit"


# ----------------------------------------------------------------- report

def _dataset(rows):
    """Hand-built report input: (verdicts by strategy, net R or status, cohort, day)."""
    signals, decisions, outcomes = [], [], []
    for i, (verdicts, result, cohort, day, direction) in enumerate(rows, start=1):
        signals.append({"id": i, "decided_at": f"2027-01-{day:02d}T10:00:00+00:00", "cohort": cohort,
                        "duplicate_of": None, "trade_json": "{}", "data_status": "fresh",
                        "baseline_matches_live": 1, "direction": direction,
                        "regime_json": json.dumps({"h4": "bullish", "htf_regime": "long", "session": "london"})})
        for s, v in zip(STRATEGIES, verdicts):
            decisions.append({"signal_id": i, "strategy": s, "verdict": v, "error": None})
        if isinstance(result, float):
            outcomes.append({"signal_id": i, "basis": "net", "status": "tp" if result > 0 else "sl", "r": result})
            outcomes.append({"signal_id": i, "basis": "gross", "status": "tp" if result > 0 else "sl", "r": result + 0.1})
        elif result:
            outcomes.append({"signal_id": i, "basis": "net", "status": result, "r": None})
    return {"signals": signals, "decisions": decisions, "outcomes": outcomes}


def test_report_counts_paired_discordance_and_warnings():
    E, S = "ENTER", "SKIP"
    data = _dataset([
        ((E, S, E, E), -1.0, "exploration", 1, "long"),     # baseline FP; htf avoided it
        ((S, E, S, S), 2.0, "exploration", 2, "short"),     # baseline FN; htf caught it
        ((E, E, E, E), 2.0, "exploration", 3, "long"),
        ((S, S, S, S), -1.0, "exploration", 4, "short"),
        ((E, E, E, E), "not_filled", "exploration", 5, "long"),
        ((E, E, E, E), 5.0, "confirmation", 6, "long"),     # hidden by default
    ])
    rep = shadow_report.build(data)
    assert rep["signals"] == 5 and rep["cohorts_recorded"] == {"exploration": 5, "confirmation": 1}
    base, htf = rep["strategies"]["baseline"], rep["strategies"]["htf_regime"]
    assert base["false_positives"] == 1 and base["false_negatives"] == 1
    assert htf["false_positives"] == 0 and htf["false_negatives"] == 0
    assert base["enter_trades"] == {"tracked": 3, "resolved": 2, "not_filled": 1, "unresolved": 0}
    paired = rep["paired_vs_baseline"]["htf_regime"]
    assert paired["discordant"]["candidate_only"]["expectancy_r"] == 2.0
    assert paired["discordant"]["baseline_only"]["expectancy_r"] == -1.0
    assert paired["delta_r_per_enter"] == pytest.approx(2.0 - 0.5)
    assert base["warning"].startswith("INSUFFICIENT SAMPLE")
    assert rep["agreement"]["baseline~htf_regime"] == pytest.approx(3 / 5)
    full = shadow_report.build(data, include_confirmation=True)
    assert full["signals"] == 6
    text = shadow_report.render(rep, compact=True)
    assert "EXPERIMENTAL, NOT A TRADE INSTRUCTION" in text and "INSUFFICIENT SAMPLE" in text
    assert "historical replay" in text


# --------------------------------------------------------------- telegram

class _Update:
    def __init__(self, chat_id):
        self.effective_chat = MagicMock(id=chat_id, username=None)
        self.message = MagicMock()
        self.message.reply_text = AsyncMock()


def test_commands_answer_only_the_configured_chat(store, monkeypatch):
    monkeypatch.setattr("telegram_bot.Bot", lambda token: MagicMock())
    ctx = context("long")
    sh, _ = shadow(store)
    run(sh.on_signal(message(ctx), analysis_for(ctx), ctx, live_decision(ctx)))
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=12345))
    bot.attach_strategy_shadow(store)
    bot._subscribers = {777}
    owner, subscriber = _Update(12345), _Update(777)
    for update in (owner, subscriber):
        run(bot._cmd_strategy_shadow(update, None))
        run(bot._cmd_strategy_shadow_latest(update, None))
    assert subscriber.message.reply_text.await_count == 0
    summary, latest = [c.args[0] for c in owner.message.reply_text.await_args_list]
    assert "NOT A TRADE INSTRUCTION" in summary and "INSUFFICIENT SAMPLE" in summary
    assert "not a trade instruction" in latest and "baseline" in latest and "Outcome: pending" in latest


# ---------------------------------------------------------- railway paths

def test_storage_must_be_on_the_volume(monkeypatch, tmp_path):
    from main import storage_on_volume
    monkeypatch.delenv("RAILWAY_VOLUME_MOUNT_PATH", raising=False)
    assert storage_on_volume(tmp_path / "x.db") is None
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(tmp_path))
    assert storage_on_volume(tmp_path / "signals.db") is True
    assert storage_on_volume(tmp_path.parent / "elsewhere" / "signals.db") is False


# ------------------------------------------------- the real pipeline, on/off

@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")


def _pipeline_run(tmp_path, strategy_shadow_factory=None):
    from formatter import Formatter
    from market_context import MarketContextBuilder
    from pipeline import AnalysisPipeline
    from risk_engine import RiskSettings
    from trade_decision import TradeDecisionEngine
    from test_telegram_report_delivery import _Analyzer, _msg, _Provider as _StubProvider, _settings, \
        _verdict_envelope

    settings = _settings(tmp_path)
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=12345))
    bot._bot.send_message = AsyncMock()
    builder = MarketContextBuilder(MarketDataService(_StubProvider()),
                                   risk_settings=RiskSettings(min_confidence=0))
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, _verdict_envelope(), ""))

    async def go():
        storage = Storage(settings.storage_db_path)
        await storage.initialize()
        strategy = None
        if strategy_shadow_factory is not None:
            strategy = await strategy_shadow_factory(settings)
        pipeline = AnalysisPipeline(settings, asyncio.Queue(), _Analyzer(),
                                    Formatter(color=False, show_json=False), bot=bot, storage=storage,
                                    context_builder=builder, trade_decision_engine=engine,
                                    emit_reports=False, strategy_shadow=strategy)
        async with pipeline:
            await pipeline._process(_msg())
        await storage.close()
        return pipeline

    pipeline = asyncio.run(go())
    con = sqlite3.connect(settings.storage_db_path)
    live = con.execute("SELECT decision_verdict, decision_confidence FROM analyses").fetchall()
    sent = [c.kwargs["text"] for c in bot._bot.send_message.await_args_list]
    return live, sent, settings, pipeline


def _mask_times(text):
    import re
    return re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", "<t>", text)


def test_live_verdict_and_message_are_identical_with_shadow_on_and_off(tmp_path, monkeypatch):
    monkeypatch.setattr("telegram_bot.Bot", lambda token: MagicMock())

    async def real(settings):
        s = StrategyShadowStore(settings.storage_db_path)
        await s.initialize()
        return StrategyShadow(s, LabelCalibrator(None))

    live_off, sent_off, _, _ = _pipeline_run(tmp_path / "off")
    live_on, sent_on, settings, _ = _pipeline_run(tmp_path / "on", real)
    assert live_on == live_off and len(sent_on) == len(sent_off) == 1
    assert _mask_times(sent_on[0]) == _mask_times(sent_off[0])
    assert "Setup score:" in sent_on[0] and "%" not in sent_on[0].split("Setup score:")[1].split("\n")[0]
    data = load_readonly(settings.storage_db_path)
    assert len(data["signals"]) == 1 and len(data["decisions"]) == len(STRATEGIES)
    row = data["signals"][0]
    assert row["live_verdict"] == live_on[0][0].upper() and row["baseline_matches_live"] == 1
    assert row["data_status"] == "unverified"        # no UTC calibration source in this run


def test_a_failing_strategy_shadow_cannot_break_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr("telegram_bot.Bot", lambda token: MagicMock())

    class Exploding:
        async def on_signal(self, *args):
            raise RuntimeError("shadow exploded")

    async def factory(settings):
        return Exploding()

    live, sent, _, _ = _pipeline_run(tmp_path, factory)
    assert len(live) == 1 and len(sent) == 1


def test_shadow_tasks_are_bounded(tmp_path):
    import pipeline as pipeline_mod

    class Slow:
        async def on_signal(self, *args):
            await asyncio.sleep(10)

    async def go():
        p = pipeline_mod.AnalysisPipeline.__new__(pipeline_mod.AnalysisPipeline)
        p._independent_tasks, p.shadow_dropped = set(), 0
        for i in range(pipeline_mod._MAX_SHADOW_TASKS + 5):
            p._spawn_shadow("strategy-shadow", Slow(), SimpleNamespace(id=i), None, None, None)
        in_flight, dropped = len(p._independent_tasks), p.shadow_dropped
        for t in list(p._independent_tasks):
            t.cancel()
        await asyncio.gather(*p._independent_tasks, return_exceptions=True)
        return in_flight, dropped

    assert asyncio.run(go()) == (16, 5)


def test_the_startup_persistence_line_reports_paths_only(monkeypatch, tmp_path):
    from config import Settings
    from main import persistence_report
    from market_data import Timeframe as TF
    (tmp_path / "signal_monitor.session").write_bytes(b"TEST SESSION BYTES - never logged")
    monkeypatch.setenv("RAILWAY_VOLUME_MOUNT_PATH", str(tmp_path))
    settings = Settings(api_id=1, api_hash="x", session_name=str(tmp_path / "signal_monitor"), phone="",
                        target_chat="t", claude_cli_path="claude", model="", claude_max_turns=3,
                        signal_parser_mode="off", worker_count=1, queue_maxsize=10, max_message_chars=8000,
                        analyse_edits=False, max_retries=0, request_timeout=5.0, log_level="INFO",
                        log_file=None, jsonl_output=None, color=False, show_json=False,
                        storage_db_path=tmp_path / "signals.db", twelve_data_api_key="",
                        market_data_cache_ttl=30.0, market_data_timeframe=TF.H1)
    r = persistence_report(settings)
    assert r["session_exists"] is True and r["session_on_volume"] is True
    assert r["storage_on_volume"] is True and r["storage_db_exists"] is False
    assert "TEST SESSION BYTES" not in repr(r)
    off = replace(settings, storage_db_path=tmp_path.parent / "elsewhere.db")
    assert persistence_report(off)["storage_on_volume"] is False
