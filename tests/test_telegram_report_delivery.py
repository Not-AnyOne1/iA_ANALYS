"""Integration tests for Telegram delivery of the final report.

Verifies the four guarantees requested:

  1. exactly one Telegram message per signal,
  2. it is the final report (not the parser's extraction),
  3. no parser output reaches Telegram at all,
  4. a delivery failure never stops the pipeline.

The Telegram boundary is mocked at ``telegram.Bot.send_message``; everything
above it — pipeline, context builder, engines, report rendering — is real.
"""

from __future__ import annotations

import asyncio
import html
import json
import shutil
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest, TimedOut

import report as report_mod
from config import Settings
from formatter import Formatter
from market_context import MarketContextBuilder
from market_data import Candle, CandleSeries, MarketDataService, Quote, Timeframe
from models import SignalAnalysis, TradeSetup
from pipeline import AnalysisPipeline
from risk_engine import RiskSettings
from storage import Storage
from telegram_bot import BotSettings, TelegramBot
from telegram_client import IncomingMessage
from trade_decision import TradeDecisionEngine


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())


@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")


@pytest.fixture(autouse=True)
def _fake_bot_class(monkeypatch):
    def _factory(token: str) -> MagicMock:
        fake = MagicMock()
        fake.send_message = AsyncMock()
        return fake
    monkeypatch.setattr("telegram_bot.Bot", _factory)


# --------------------------------------------------------------------------- fixtures

def _series(symbol, timeframe, n=300):
    base = datetime(2026, 3, 10, tzinfo=timezone.utc) - timedelta(hours=n)
    candles = []
    for i in range(n):
        price = 100.0 + i * 0.5
        candles.append(Candle(timestamp=base + timedelta(hours=i), open=price,
                              high=price + 1.0, low=price - 1.0, close=price, volume=10.0))
    return CandleSeries(symbol=symbol, timeframe=timeframe, provider="stub",
                        candles=tuple(candles))


class _Provider:
    name = "stub"

    async def get_quote(self, symbol):
        return Quote(symbol=symbol, price=250.0, timestamp=datetime.now(timezone.utc),
                     provider=self.name, bid=249.9, ask=250.1)

    async def get_candles(self, symbol, timeframe, limit):
        return _series(symbol, timeframe)


_PARSER_SUMMARY = "PARSER-EXTRACTION-SUMMARY-SHOULD-NEVER-BE-SENT"


def _analysis() -> SignalAnalysis:
    return SignalAnalysis(
        is_signal=True, category="signal",
        setup=TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                         entries=[250.0], stop_loss=240.0, take_profits=[280.0]),
        summary=_PARSER_SUMMARY, confidence=0.97,
        missing_fields=[], notes=None, source="regex")


class _Analyzer:
    async def analyze(self, message):
        return _analysis()


def _settings(tmp_path) -> Settings:
    return Settings(
        api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
        claude_cli_path="claude", model="", claude_max_turns=3,
        signal_parser_mode="off", worker_count=1, queue_maxsize=10,
        max_message_chars=8000, analyse_edits=False, max_retries=0,
        request_timeout=5.0, log_level="CRITICAL", log_file=None, jsonl_output=None,
        color=False, show_json=False, storage_db_path=tmp_path / "s.db",
        twelve_data_api_key="k", market_data_cache_ttl=30.0,
        market_data_timeframe=Timeframe.H1)


def _verdict_envelope() -> str:
    return json.dumps({"structured_output": {
        "verdict": "wait", "confidence": 55,
        "reasoning": "Structure aligns but the stop sits near liquidity.",
        "strengths": ["H1 trend bullish"], "risks": ["M15 has not confirmed"],
        "strongest_reason_against": "The stop sits beyond a swing level.",
        "alternative_scenario": "Price sweeps the low first.",
        "worst_case": "Stop hit for 1R.", "best_case": "Target reached for 3R.",
        "execution_plan": "Wait for a pullback.",
    }, "is_error": False})


def _msg(message_id: int = 4242) -> IncomingMessage:
    return IncomingMessage(id=message_id, chat_id=-100999, chat_title="VIP Signals",
                           sender="Analyst", timestamp=datetime.now(timezone.utc),
                           text="BUY GOLD NOW\nTP:280\nSL:240")


def _run(tmp_path, *, send_effect=None, message=None, emit_reports=False,
         bot_chat_id=12345, subscribers=()):
    """Run one message through the real pipeline; return (bot, stats, sends)."""
    settings = _settings(tmp_path)
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=bot_chat_id))
    bot._subscribers = set(subscribers)
    if send_effect is not None:
        bot._bot.send_message = AsyncMock(side_effect=send_effect)

    builder = MarketContextBuilder(
        MarketDataService(_Provider()),
        risk_settings=RiskSettings(min_confidence=0))
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, _verdict_envelope(), ""))

    async def go():
        storage = Storage(settings.storage_db_path)
        await storage.initialize()
        pipeline = AnalysisPipeline(
            settings, asyncio.Queue(), _Analyzer(),
            Formatter(color=False, show_json=False), bot=bot, storage=storage,
            context_builder=builder, trade_decision_engine=engine,
            emit_reports=emit_reports)
        async with pipeline:
            await pipeline._process(message or _msg())
        await storage.close()
        return pipeline.stats

    stats = asyncio.run(go())
    return bot, stats, bot._bot.send_message.await_args_list


def _sent_text(call) -> str:
    return html.unescape(call.kwargs["text"])


# --------------------------------------------------------------- exactly one message

def test_exactly_one_telegram_message_is_sent(tmp_path):
    bot, _, sends = _run(tmp_path)
    assert len(sends) == 1, f"expected 1 message, got {len(sends)}"


def test_reprocessing_the_same_signal_sends_nothing_further(tmp_path):
    """The duplicate guard covers the report path too."""
    settings = _settings(tmp_path)
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=12345))
    builder = MarketContextBuilder(MarketDataService(_Provider()),
                                   risk_settings=RiskSettings(min_confidence=0))
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, _verdict_envelope(), ""))

    async def go():
        pipeline = AnalysisPipeline(
            settings, asyncio.Queue(), _Analyzer(),
            Formatter(color=False, show_json=False), bot=bot,
            context_builder=builder, trade_decision_engine=engine,
            emit_reports=False)
        async with pipeline:
            await pipeline._process(_msg(7))
            await pipeline._process(_msg(7))     # same signal again

    asyncio.run(go())

    assert bot._bot.send_message.await_count == 1
    assert bot._stats.duplicates_skipped == 1


# ----------------------------------------------------------- it is the final report

def test_the_message_is_a_short_card_not_the_long_report(tmp_path):
    """The terminal report is thousands of characters of tables and is
    unreadable on a phone. It stays in the terminal; Telegram gets a card."""
    _, _, sends = _run(tmp_path)
    text = _sent_text(sends[0])

    assert "📊" in text and "XAUUSD" in text
    assert "Decision:" in text
    assert "Confidence:" in text
    assert "Summary:" in text

    for long_report_marker in ("MARKET REPORT", "INDICATORS", "SMART MONEY",
                               "LEVELS", "STRUCTURE", "VALIDATION",
                               "Bands", "deterministic score"):
        assert long_report_marker not in text, f"{long_report_marker!r} leaked into the card"


def test_the_card_is_at_most_twenty_lines(tmp_path):
    _, _, sends = _run(tmp_path)
    lines = _sent_text(sends[0]).splitlines()
    assert len(lines) <= 20, f"{len(lines)} lines:\n" + "\n".join(lines)


def test_the_card_carries_the_levels_a_trader_needs(tmp_path):
    _, _, sends = _run(tmp_path)
    text = _sent_text(sends[0])
    assert "Entry:" in text
    assert "SL:" in text
    assert "TP1:" in text


def test_the_card_has_no_json_and_no_raw_lists(tmp_path):
    text = _sent_text(_run(tmp_path)[2][0])
    for banned in ("{", "}", "[", "]", "severity", "contribution", "weight="):
        assert banned not in text, f"{banned!r} leaked into the card"


def test_the_message_uses_telegram_html(tmp_path):
    _, _, sends = _run(tmp_path)
    call = sends[0]
    assert call.kwargs["parse_mode"] == "HTML"
    assert "<b>" in call.kwargs["text"]
    assert "<pre>" not in call.kwargs["text"]   # no monospace dump any more


def test_the_message_fits_telegram_limit(tmp_path):
    _, _, sends = _run(tmp_path)
    assert len(sends[0].kwargs["text"]) <= 4096


# ------------------------------------------------------------- no parser output

def test_the_parser_summary_is_never_sent(tmp_path):
    """The extraction summary stays in the terminal and the logs."""
    _, _, sends = _run(tmp_path)
    for call in sends:
        assert _PARSER_SUMMARY not in _sent_text(call)


def test_parser_confidence_is_never_sent(tmp_path):
    """The parser reported 0.97; only the final (much lower) confidence
    derived from scoring/validation/Claude may appear."""
    _, _, sends = _run(tmp_path)
    text = _sent_text(sends[0])
    assert "97%" not in text
    assert "0.97" not in text


def test_nothing_is_sent_before_the_report(tmp_path):
    """A single send, and it already contains the decision — so no
    intermediate message preceded it."""
    _, _, sends = _run(tmp_path)
    assert len(sends) == 1
    assert "Decision:" in _sent_text(sends[0])


def test_no_message_at_all_when_market_validation_is_disabled(tmp_path):
    """Without a provider there is no final report, and the parser output
    must not be sent as a substitute."""
    settings = _settings(tmp_path)
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=12345))

    async def go():
        pipeline = AnalysisPipeline(
            settings, asyncio.Queue(), _Analyzer(),
            Formatter(color=False, show_json=False), bot=bot,
            context_builder=None, emit_reports=False)   # no validation path
        async with pipeline:
            await pipeline._process(_msg())

    asyncio.run(go())

    bot._bot.send_message.assert_not_awaited()


# ------------------------------------------------------------------- replying

def test_the_report_replies_when_the_destination_is_the_source_chat(tmp_path):
    """A reply only means anything in the chat the signal came from."""
    _, _, sends = _run(tmp_path, message=_msg(4242), bot_chat_id=-100999)
    assert sends[0].kwargs.get("reply_to_message_id") == 4242


def test_no_reply_is_attempted_for_a_subscriber(tmp_path):
    """A subscriber's private chat does not contain the group's message, so
    attempting a reply would be a guaranteed rejection."""
    _, _, sends = _run(tmp_path, message=_msg(4242), subscribers=(777,))
    assert sends[0].kwargs["chat_id"] == 777
    assert "reply_to_message_id" not in sends[0].kwargs


def test_falls_back_to_a_normal_message_when_the_reply_target_is_missing(tmp_path):
    """The bot's chat is usually not the monitored group, so the group's
    message id does not exist there and Telegram rejects the reply."""
    calls = {"n": 0}

    def effect(*args, **kwargs):
        calls["n"] += 1
        if "reply_to_message_id" in kwargs:
            raise BadRequest("Replied message not found")
        return None

    bot, _, sends = _run(tmp_path, send_effect=effect, bot_chat_id=-100999)

    assert calls["n"] == 2                                   # reply attempt, then plain
    assert "reply_to_message_id" not in sends[-1].kwargs      # the retry had no reply
    assert bot._stats.notifications_sent == 1                # still counted as delivered


def test_a_non_reply_bad_request_is_not_retried_as_plain(tmp_path):
    """Only a missing reply target triggers the fallback; a genuinely
    malformed request must not be sent twice."""
    bot, _, sends = _run(tmp_path, send_effect=BadRequest("chat not found"))
    assert len(sends) == 1
    assert bot._stats.notifications_failed == 1


# --------------------------------------------------- delivery never breaks anything

def test_a_send_failure_does_not_stop_the_pipeline(tmp_path):
    bot, stats, _ = _run(tmp_path, send_effect=TimedOut())

    assert stats.received == 1                     # analysis completed
    assert bot._stats.notifications_failed == 1    # delivery recorded as failed


def test_a_send_failure_still_persists_the_analysis(tmp_path):
    settings = _settings(tmp_path)
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=12345))
    bot._bot.send_message = AsyncMock(side_effect=RuntimeError("telegram exploded"))
    builder = MarketContextBuilder(MarketDataService(_Provider()),
                                   risk_settings=RiskSettings(min_confidence=0))
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, _verdict_envelope(), ""))

    async def go():
        storage = Storage(settings.storage_db_path)
        await storage.initialize()
        pipeline = AnalysisPipeline(
            settings, asyncio.Queue(), _Analyzer(),
            Formatter(color=False, show_json=False), bot=bot, storage=storage,
            context_builder=builder, trade_decision_engine=engine,
            emit_reports=False)
        async with pipeline:
            await pipeline._process(_msg())        # must not raise
        rows = await storage.fetch_analyses()
        await storage.close()
        return rows

    rows = asyncio.run(go())
    assert len(rows) == 1                          # storage unaffected
    assert rows[0].decision_verdict == "skip"      # capped by the score band


def test_an_unexpected_send_exception_is_swallowed(tmp_path):
    bot, stats, _ = _run(tmp_path, send_effect=ValueError("something weird"))
    assert stats.received == 1


def test_terminal_output_is_unaffected_by_telegram(tmp_path, capsys):
    """The terminal report still renders even when delivery fails."""
    _run(tmp_path, send_effect=TimedOut(), emit_reports=True)
    printed = capsys.readouterr().out
    assert "MARKET REPORT" in printed
    assert "DECISION" in printed


# --------------------------------------------------------- renderer-level guarantees

def _long_decision(verdict, **over):
    from trade_decision import TradeDecision
    from decision_engine import DecisionSource
    base = dict(
        symbol="XAUUSD", verdict=verdict, confidence=0,
        reasoning="R" * 400, strengths=["S" * 100] * 5, risks=["W" * 100] * 5,
        strongest_reason_against="A" * 300, alternative_scenario="B" * 300,
        worst_case="C" * 300, best_case="D" * 300, execution_plan="E" * 200,
        source=DecisionSource.CLAUDE, fatal_problems=["F" * 200],
        decision_reason="Z" * 300)
    base.update(over)
    return TradeDecision(**base)


def _built_context():
    builder = MarketContextBuilder(MarketDataService(_Provider()),
                                   risk_settings=RiskSettings(min_confidence=0))
    return asyncio.run(builder.build("XAUUSD", _analysis().setup))


def test_the_card_stays_short_even_with_pathological_model_output():
    """Every narrative field maxed out must still produce a readable card."""
    from trade_decision import Action

    out = report_mod.render_telegram_report(_built_context(), _long_decision(Action.SKIP))
    lines = out.splitlines()

    assert len(out) <= 4096
    assert len(lines) <= 20, f"{len(lines)} lines"
    assert max(len(l) for l in lines) <= 160, "a bullet ran away"


def test_a_refusal_leads_with_the_reason_it_was_refused():
    """On SKIP the trader needs to know what stopped it, first."""
    from trade_decision import Action

    out = report_mod.render_telegram_report(
        _built_context(),
        _long_decision(Action.SKIP, decision_reason="risk engine rejected: spread too wide",
                       strengths=["Structure is clean."]))
    summary = out.split("Summary:")[1]
    assert summary.index("risk engine rejected") < summary.index("Structure is clean")


def test_an_entry_leads_with_the_confluence():
    from trade_decision import Action

    out = report_mod.render_telegram_report(
        _built_context(),
        _long_decision(Action.ENTER, confidence=72,
                       strengths=["Bullish BOS aligns with the trade.",
                                  "Risk/Reward is acceptable."],
                       strongest_reason_against="Higher timeframes are mixed."))
    summary = out.split("Summary:")[1]
    assert summary.index("Bullish BOS") < summary.index("Higher timeframes are mixed")
    assert "72%" in out


def test_the_card_renders_without_a_decision():
    """A context with no decision must still produce a valid message."""
    from market_context import MarketContext

    ctx = MarketContext(symbol="XAUUSD", generated_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
                        primary_timeframe=Timeframe.H1)
    out = report_mod.render_telegram_report(ctx)
    assert "XAUUSD" in out
    assert len(out.splitlines()) <= 20


def test_prices_are_written_the_way_a_trader_writes_them():
    assert report_mod._price(4100.0) == "4100"
    assert report_mod._price(4085.5) == "4085.5"
    assert report_mod._price(1.08432) == "1.08432"
    assert report_mod._price(0.00012345) == "0.00012345"
    assert report_mod._price(None) == "—"


# ------------------------------------------------------------- subscribers

def test_every_subscriber_receives_the_report(tmp_path):
    _, _, sends = _run(tmp_path, subscribers=(111, 222, 333))
    assert sorted(c.kwargs["chat_id"] for c in sends) == [111, 222, 333]


def test_subscribers_replace_the_configured_chat_as_the_destination(tmp_path):
    """TELEGRAM_BOT_CHAT_ID is a fallback, not the primary target."""
    _, _, sends = _run(tmp_path, bot_chat_id=12345, subscribers=(777,))
    assert [c.kwargs["chat_id"] for c in sends] == [777]


def test_the_configured_chat_is_used_when_nobody_has_subscribed(tmp_path):
    """Backward compatibility: an existing deployment keeps working."""
    _, _, sends = _run(tmp_path, bot_chat_id=12345, subscribers=())
    assert [c.kwargs["chat_id"] for c in sends] == [12345]


def test_nothing_is_sent_with_no_subscribers_and_no_fallback(tmp_path):
    _, _, sends = _run(tmp_path, bot_chat_id=None, subscribers=())
    assert sends == []


def test_one_failing_subscriber_does_not_cost_the_others(tmp_path):
    """A user who blocked the bot must not silence everyone else."""
    def effect(*args, **kwargs):
        if kwargs.get("chat_id") == 222:
            raise BadRequest("bot was blocked by the user")
        return None

    bot, _, sends = _run(tmp_path, subscribers=(111, 222, 333), send_effect=effect)

    assert sorted(c.kwargs["chat_id"] for c in sends) == [111, 222, 333]
    assert bot._stats.notifications_failed == 1
    assert bot._stats.notifications_sent == 1      # the signal was still delivered


def test_the_report_is_still_deduplicated_across_all_subscribers(tmp_path):
    """Three destinations is one report, not three chances to re-send."""
    settings = _settings(tmp_path)
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=None))
    bot._subscribers = {111, 222}
    builder = MarketContextBuilder(MarketDataService(_Provider()),
                                   risk_settings=RiskSettings(min_confidence=0))
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, _verdict_envelope(), ""))

    async def go():
        pipeline = AnalysisPipeline(
            settings, asyncio.Queue(), _Analyzer(),
            Formatter(color=False, show_json=False), bot=bot,
            context_builder=builder, trade_decision_engine=engine,
            emit_reports=False)
        async with pipeline:
            await pipeline._process(_msg(7))
            await pipeline._process(_msg(7))

    asyncio.run(go())

    assert bot._bot.send_message.await_count == 2      # two subscribers, once each
    assert bot._stats.duplicates_skipped == 1
