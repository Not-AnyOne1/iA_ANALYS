"""Local integration check for strategy shadow mode, on the recorded signals.

Every recorded XAUUSD signal goes through the real ``AnalysisPipeline``:
real context builder, engines, decision and report, real storage and the
real strategy shadow. Replaced: Telegram delivery (captured, not sent),
Claude (a fixed narrative; the verdict never comes from it), and Twelve Data
(``research/production_replay.py`` serves the research bars as the live
provider would have at each signal's time). Nothing touches the network or
the bot's database.

Checks, printed as PASS/FAIL:

1. live verdicts and stored rows identical with shadow ON and OFF;
2. one shadow record per signal, four decisions each, baseline = live;
3. outcomes resolved and equal to an independent P1 computation;
4. a restart on the same database adds nothing and loses nothing;
5. Telegram messages identical ON vs OFF, and against production
   ``197b2a0``'s own renderer only the score line differs;
6. provider failures (calibration and outcome source) do not stop the
   pipeline and are recorded as such;
7. time and memory per shadow evaluation, and requests made.

    python research/strategy_shadow_integration.py --db <copy of signals.db> \\
        --prod-tree <git archive of 197b2a0> --out integration.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import market_context as market_context_mod  # noqa: E402
from config import Settings  # noqa: E402
from costs import Costs  # noqa: E402
from formatter import Formatter  # noqa: E402
from market_context import MarketContextBuilder  # noqa: E402
from market_data import CandleSeries, MarketDataError, MarketDataService, Timeframe  # noqa: E402
from models import SignalAnalysis  # noqa: E402
from news_filter import NewsFilter  # noqa: E402
from pipeline import AnalysisPipeline  # noqa: E402
from storage import Storage  # noqa: E402
from telegram_bot import BotSettings, TelegramBot  # noqa: E402
from telegram_client import IncomingMessage  # noqa: E402
from trade_accounting import account_both  # noqa: E402
import report as branch_report  # noqa: E402
import telegram_bot  # noqa: E402
from trade_decision import DecisionSource, TradeDecision, TradeDecisionEngine,     decide_deterministically  # noqa: E402

from analyst.data import drop_closure_stubs  # noqa: E402
from analyst.tracker import accounting_row  # noqa: E402
from production_replay import Bars, ReplayProvider, service  # noqa: E402
from strategy_shadow.outcomes import StrategyOutcomeResolver  # noqa: E402
from strategy_shadow.report import build, render  # noqa: E402
from strategy_shadow.service import StrategyShadow  # noqa: E402
from strategy_shadow.snapshot import LabelCalibrator  # noqa: E402
from strategy_shadow.store import StrategyShadowStore, load_readonly  # noqa: E402

UTC = timezone.utc
NET = Costs(0.30, 0.10, "assumed")
ENVELOPE = json.dumps({"structured_output": {
    "verdict": "wait", "confidence": 50, "reasoning": "TEST narrative.", "strengths": ["TEST"],
    "risks": ["TEST"], "strongest_reason_against": "TEST", "alternative_scenario": "TEST",
    "worst_case": "TEST", "best_case": "TEST", "execution_plan": "TEST"}, "is_error": False})


class FrozenClock(datetime):
    """``market_context.datetime`` during a replayed message: now() is the
    signal's decision time, as it was live."""
    current: datetime = datetime.now(UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz is None else cls.current.astimezone(tz)


class UtcBars:
    """The UTC-labelled provider of shadow mode: research M15 bars that had
    opened by ``at`` (closed ones only matter; the forming one is harmless)."""

    name = "utc-replay"

    def __init__(self, bars: Bars) -> None:
        self.bars, self.at, self.calls, self.fail = bars, None, 0, False

    async def get_quote(self, symbol):
        raise AssertionError("no quote is requested by shadow mode")

    async def get_candles(self, symbol, timeframe, limit):
        self.calls += 1
        if self.fail:
            raise MarketDataError("injected provider failure", retryable=False)
        return CandleSeries(symbol, timeframe, self.name, self.bars.live_window(timeframe, self.at, limit))


class CountingProvider(ReplayProvider):
    def __init__(self, bars):
        super().__init__(bars)
        self.calls = 0

    async def get_quote(self, symbol):
        self.calls += 1
        return await super().get_quote(symbol)

    async def get_candles(self, symbol, timeframe, limit):
        self.calls += 1
        return await super().get_candles(symbol, timeframe, limit)


def settings_for(db: Path) -> Settings:
    return Settings(api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
                    claude_cli_path="claude", model="", claude_max_turns=3, signal_parser_mode="off",
                    worker_count=1, queue_maxsize=10, max_message_chars=8000, analyse_edits=False,
                    max_retries=0, request_timeout=5.0, log_level="CRITICAL", log_file=None,
                    jsonl_output=None, color=False, show_json=False, storage_db_path=db,
                    twelve_data_api_key="k", market_data_cache_ttl=0.0,
                    market_data_timeframe=Timeframe.H1)


def load_signals(db: str):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    live_price = {r["message_id"]: r["price_at_decision"]
                  for r in con.execute("SELECT message_id, price_at_decision FROM signal_outcomes")}
    rows = con.execute("SELECT * FROM analyses WHERE is_signal = 1 AND symbol = 'XAUUSD' "
                       "AND recorded_at >= '2026-08-20' ORDER BY id").fetchall()
    return [(r, SignalAnalysis.model_validate(json.loads(r["analysis_json"])), live_price.get(r["message_id"]))
            for r in rows]


async def run_pipeline(signals, bars, db: Path, *, shadow_on: bool, fail_calibration_every: int = 0):
    settings = settings_for(db)
    telegram_bot.Bot = lambda token: MagicMock()           # nothing can reach Telegram
    live = CountingProvider(bars)
    builder = MarketContextBuilder(service(live), news_filter=NewsFilter())
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, ENVELOPE, ""))
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=12345))
    bot._bot = MagicMock()
    bot._bot.send_message = AsyncMock()
    storage = Storage(db)
    await storage.initialize()
    utc = UtcBars(bars)
    strategy, timings, contexts = None, [], []
    if shadow_on:
        store = StrategyShadowStore(db)
        await store.initialize()
        strategy = StrategyShadow(store, LabelCalibrator(MarketDataService(utc, cache_ttl=0, max_retries=0)))
        original = strategy.on_signal

        async def timed(*args):
            t0 = time.perf_counter()
            result = await original(*args)
            timings.append(time.perf_counter() - t0)
            return result
        strategy.on_signal = timed

    analyses = {}

    class Analyzer:
        async def analyze(self, message):
            return analyses[message.id]

    pipeline = AnalysisPipeline(settings, asyncio.Queue(), Analyzer(), Formatter(color=False, show_json=False),
                                bot=bot, storage=storage, context_builder=builder,
                                trade_decision_engine=engine, emit_reports=False, strategy_shadow=strategy)
    original_build = builder.build

    async def build_and_keep(*args, **kwargs):
        ctx = await original_build(*args, **kwargs)
        contexts.append(ctx)
        return ctx
    builder.build = build_and_keep
    market_context_mod.datetime = FrozenClock
    try:
        async with pipeline:
            for i, (row, analysis, price) in enumerate(signals):
                at = datetime.fromisoformat(row["recorded_at"])
                FrozenClock.current = at
                live.at, live.price = at, price or bars.last_close(at)
                utc.at = at
                utc.fail = bool(fail_calibration_every) and i % fail_calibration_every == 0
                analyses[row["message_id"]] = analysis
                message = IncomingMessage(id=row["message_id"], chat_id=row["chat_id"],
                                          chat_title="TEST", sender="TEST",
                                          timestamp=datetime.fromisoformat(row["message_timestamp"]),
                                          text="TEST")
                await pipeline._process(message)
                if pipeline._independent_tasks:                # let this signal's shadow finish
                    await asyncio.wait(set(pipeline._independent_tasks))
    finally:
        market_context_mod.datetime = datetime
    await storage.close()
    sent = [c.kwargs["text"] for c in bot._bot.send_message.await_args_list]
    con = sqlite3.connect(db)
    stored = con.execute("SELECT message_id, decision_verdict, decision_confidence FROM analyses ORDER BY id").fetchall()
    con.close()
    return {"sent": sent, "stored": stored, "timings": timings, "live_requests": live.calls,
            "utc_requests": utc.calls, "contexts": contexts, "engine": engine, "dropped": pipeline.shadow_dropped}


def mask(text: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", "<t>", text)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", required=True)
    p.add_argument("--prod-tree", required=True, help="git archive of 197b2a0 (report.py is read from it)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    results, checks = {}, []

    def check(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    bars = Bars()
    signals = load_signals(args.db)
    work = Path(tempfile.mkdtemp(prefix="shadow-integration-"))
    try:
        off = asyncio.run(run_pipeline(signals, bars, work / "off.db", shadow_on=False))
        tracemalloc.start()
        on = asyncio.run(run_pipeline(signals, bars, work / "on.db", shadow_on=True))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 1 and 5: the live path
        check("1 live verdicts identical ON vs OFF", on["stored"] == off["stored"],
              f"{len(on['stored'])} signals")
        check("5a one Telegram message per signal, identical ON vs OFF",
              len(on["sent"]) == len(off["sent"]) == len(signals)
              and [mask(t) for t in on["sent"]] == [mask(t) for t in off["sent"]],
              f"{len(on['sent'])} messages")
        prod_report = load_module(Path(args.prod_tree) / "report.py", "prod_report_197b2a0")
        diffs = set()
        # Render both cards for every replayed context with one decision each.
        for ctx in on["contexts"]:
            t = decide_deterministically(ctx)
            d = TradeDecision(symbol=ctx.symbol, verdict=t.verdict, confidence=t.adjusted_score or 0,
                              reasoning="TEST", strengths=["TEST"], risks=["TEST"], execution_plan="TEST",
                              source=DecisionSource.CLAUDE, deterministic_score=t.deterministic_score,
                              validation_bonus=t.validation_bonus, validation_penalty=t.validation_penalty,
                              adjusted_score=t.adjusted_score, decision_reason=t.reason,
                              risk_approved=ctx.risk.approved if ctx.risk else None,
                              risk_rejections=[f"{c.name}: {c.detail}" for c in ctx.risk.rejections] if ctx.risk else [])
            a = branch_report.render_telegram_report(ctx, d).split("\n")
            b = prod_report.render_telegram_report(ctx, d).split("\n")
            if len(a) != len(b):
                diffs.add("line count differs")
                continue
            for x, y in zip(a, b):
                if x != y:
                    diffs.add("score line" if "Setup score" in x and "Confidence" in y else f"other: {y[:40]}")
        check("5b card vs production 197b2a0: only the score line differs", diffs <= {"score line"},
              f"differences: {sorted(diffs) or 'none'}")

        # 2: shadow records
        data = load_readonly(work / "on.db")
        per = {}
        for d in data["decisions"]:
            per.setdefault(d["signal_id"], set()).add(d["strategy"])
        check("2a one shadow record per signal", len(data["signals"]) == len(signals),
              f"{len(data['signals'])} records")
        check("2b four strategy decisions per record", all(len(v) == 4 for v in per.values()))
        mismatches = [s for s in data["signals"] if s["baseline_matches_live"] != 1]
        check("2c replayed baseline equals the live verdict", not mismatches, f"{len(mismatches)} mismatches")
        statuses = {}
        for s in data["signals"]:
            statuses[s["data_status"]] = statuses.get(s["data_status"], 0) + 1
        check("2d snapshot quality recorded (UTC calibration offset measured)",
              all(json.loads(s["data_quality_json"])["label_offset_minutes"] == 0 for s in data["signals"]),
              f"statuses {statuses}")

        # 3: outcomes
        end = datetime(2026, 10, 18, 2, 45, tzinfo=UTC)
        utc = UtcBars(bars)
        utc.at = end
        store = StrategyShadowStore(work / "on.db")
        resolver = StrategyOutcomeResolver(store, MarketDataService(utc, cache_ttl=0, max_retries=0), costs=NET)
        written = asyncio.run(resolver.resolve_once(end))
        data = load_readonly(work / "on.db")
        m15 = drop_closure_stubs(bars.bars[Timeframe.M15], Timeframe.M15, "XAUUSD")
        expected, got, mismatch = 0, {}, []
        for o in data["outcomes"]:
            if o["basis"] == "net":
                got[o["signal_id"]] = (o["status"], o["r"])
        for s in data["signals"]:
            if not s["trade_json"]:
                continue
            trade = json.loads(s["trade_json"])
            ref = account_both(accounting_row(trade), m15, end, costs=NET, horizon=timedelta(days=5),
                               fill_window=timedelta(hours=24))["net"]
            if ref.final:
                expected += 1
                if got.get(s["id"]) != (ref.status, ref.r):
                    mismatch.append(s["id"])
        check("3 outcomes resolved and equal to an independent P1 computation",
              written > 0 and not mismatch and len(got) == expected,
              f"{len(got)} resolved (net), {len(mismatch)} mismatches, {resolver.requests} M15 request(s)")

        # 4: restart on the same database
        before = load_readonly(work / "on.db")
        again = asyncio.run(run_pipeline(signals, bars, work / "on.db", shadow_on=True))
        after = load_readonly(work / "on.db")
        check("4 restart: no new records, nothing lost",
              len(after["signals"]) == len(before["signals"]) and len(after["outcomes"]) == len(before["outcomes"])
              and [s["snapshot_fingerprint"] for s in after["signals"]] ==
              [s["snapshot_fingerprint"] for s in before["signals"]],
              f"{len(after['signals'])} records, {len(after['outcomes'])} outcome rows; "
              f"shadow evaluations re-run: {len(again['timings'])} (all skipped as already recorded)")

        # 6: provider failures
        failing = asyncio.run(run_pipeline(signals, bars, work / "fail.db", shadow_on=True,
                                           fail_calibration_every=1))
        fdata = load_readonly(work / "fail.db")
        check("6a calibration source down: every signal processed and recorded 'unverified'",
              len(failing["sent"]) == len(signals) and len(fdata["signals"]) == len(signals)
              and all(s["data_status"] in ("unverified", "market_closed") for s in fdata["signals"]),
              f"{len(fdata['signals'])} records")
        down = UtcBars(bars)
        down.at, down.fail = end, True
        lost = asyncio.run(StrategyOutcomeResolver(StrategyShadowStore(work / "fail.db"),
                                                   MarketDataService(down, cache_ttl=0, max_retries=0),
                                                   costs=NET).resolve_once(end))
        check("6b outcome source down: nothing written, trades stay pending",
              lost == 0 and len(asyncio.run(StrategyShadowStore(work / "fail.db").pending())) > 0)

        # 7: resources
        t = sorted(on["timings"])
        results["resources"] = {
            "shadow_eval_seconds": {"mean": round(sum(t) / len(t), 4), "p95": round(t[int(0.95 * (len(t) - 1))], 4),
                                    "max": round(t[-1], 4)},
            "peak_traced_memory_mb_whole_run": round(peak / 1e6, 1),
            "live_provider_requests": {"off": off["live_requests"], "on": on["live_requests"]},
            "utc_requests_for_calibration": on["utc_requests"], "signals": len(signals),
            "shadow_tasks_dropped": on["dropped"],
        }
        first = datetime.fromisoformat(signals[0][0]["recorded_at"])
        last = datetime.fromisoformat(signals[-1][0]["recorded_at"])
        bound = int((last - first) / timedelta(hours=6)) + 1      # one calibration per 6 h at most
        check("7 live requests unchanged; calibration at most once per 6 h of signal time",
              on["live_requests"] == off["live_requests"] and on["utc_requests"] <= bound,
              json.dumps(results["resources"]) + f" (bound {bound})")
        results["report"] = build(load_readonly(work / "on.db"), include_duplicates=False)
        print(render(results["report"]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    results["checks"] = checks
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    failed = [c for c in checks if not c["pass"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
