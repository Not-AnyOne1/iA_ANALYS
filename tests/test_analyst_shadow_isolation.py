"""Shadow mode is isolated: the live verdict, report and delivery are unchanged.

Uses the real pipeline, context builder and engines with a stub provider,
as ``test_telegram_report_delivery`` does. All prices are TEST DATA.
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import ConfigError, Settings
from formatter import Formatter
from market_context import MarketContextBuilder
from market_data import MarketDataService, Timeframe
from pipeline import AnalysisPipeline
from risk_engine import RiskSettings
from storage import Storage
from telegram_bot import BotSettings, TelegramBot
from trade_decision import TradeDecisionEngine, build_facts

from analyst.service import ShadowAnalyst
from analyst.store import AnalystStore
from test_telegram_report_delivery import _Analyzer, _msg, _Provider, _settings, _verdict_envelope


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


class _Spy:
    """Stands in for ShadowAnalyst and records how it was called."""

    def __init__(self, bot, fail=False):
        self.bot, self.fail, self.calls = bot, fail, []

    async def on_signal(self, message, analysis, context, decision):
        self.calls.append({"sent_before": self.bot._bot.send_message.await_count,
                           "decision": decision, "context": context})
        if self.fail:
            raise RuntimeError("shadow analyst exploded")


def _pipeline(tmp_path, analyst_factory=None):
    settings = _settings(tmp_path)
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=12345))
    market_data = MarketDataService(_Provider())
    builder = MarketContextBuilder(market_data, risk_settings=RiskSettings(min_confidence=0))
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, _verdict_envelope(), ""))
    analyst = analyst_factory(bot, market_data, settings) if analyst_factory else None

    async def go():
        storage = Storage(settings.storage_db_path)
        await storage.initialize()
        pipeline = AnalysisPipeline(settings, asyncio.Queue(), _Analyzer(),
                                    Formatter(color=False, show_json=False), bot=bot,
                                    storage=storage, context_builder=builder,
                                    trade_decision_engine=engine, emit_reports=False,
                                    independent_analyst=analyst)
        async with pipeline:
            await pipeline._process(_msg())
        await storage.close()

    asyncio.run(go())
    con = sqlite3.connect(settings.storage_db_path)
    live = con.execute("SELECT decision_verdict, decision_confidence FROM analyses").fetchall()
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return bot, analyst, live, tables, settings


def test_without_the_analyst_nothing_extra_exists(tmp_path):
    bot, _, live, tables, _ = _pipeline(tmp_path)
    assert bot._bot.send_message.await_count == 1 and len(live) == 1
    assert not {"independent_analyses", "independent_outcomes"} & tables


def test_the_analyst_runs_after_the_live_report_and_sends_nothing(tmp_path):
    bot, spy, live, _, _ = _pipeline(tmp_path, lambda bot, md, s: _Spy(bot))
    assert len(spy.calls) == 1 and spy.calls[0]["sent_before"] == 1
    assert bot._bot.send_message.await_count == 1           # still exactly one message
    sent = bot._bot.send_message.await_args.kwargs["text"]
    assert "INDEPENDENT" not in sent and "SHADOW" not in sent


def test_live_verdict_and_storage_are_identical_with_and_without_shadow(tmp_path):
    _, _, live_off, _, _ = _pipeline(tmp_path / "off")
    _, spy, live_on, _, _ = _pipeline(tmp_path / "on", lambda bot, md, s: _Spy(bot))
    assert live_on == live_off
    assert spy.calls[0]["decision"].verdict.value == live_on[0][0]


def test_a_failing_analyst_cannot_break_the_pipeline(tmp_path):
    bot, spy, live, _, _ = _pipeline(tmp_path, lambda bot, md, s: _Spy(bot, fail=True))
    assert len(spy.calls) == 1 and bot._bot.send_message.await_count == 1 and len(live) == 1


def test_the_real_analyst_records_one_row_with_the_live_verdict(tmp_path):
    def real(bot, market_data, settings):
        store = AnalystStore(settings.storage_db_path)
        asyncio.run(store.initialize())
        return ShadowAnalyst(market_data, store)

    bot, _, live, tables, settings = _pipeline(tmp_path, real)
    assert {"independent_analyses", "independent_outcomes"} <= tables
    con = sqlite3.connect(settings.storage_db_path)
    rows = con.execute("SELECT live_verdict, verdict, data_status, card_text FROM independent_analyses").fetchall()
    assert len(rows) == 1
    live_verdict, verdict, data_status, card = rows[0]
    assert live_verdict == live[0][0]
    # The stub provider serves hourly bars dated months ago for every
    # timeframe: the analyst must refuse to judge on it, not guess.
    assert verdict == "NO TRADE" and data_status != "fresh"
    assert card.rstrip().endswith("Mode: SHADOW — NOT A LIVE TRADE INSTRUCTION")
    assert bot._bot.send_message.await_count == 1


def test_the_analyst_ignores_other_symbols(tmp_path):
    store = AnalystStore(tmp_path / "x.db")
    asyncio.run(store.initialize())
    analyst = ShadowAnalyst(MarketDataService(_Provider()), store)
    analysis = _Analyzer()
    a = asyncio.run(analysis.analyze(None))
    other = a.model_copy(update={"setup": a.setup.model_copy(update={"symbol": "EURUSD"})})
    assert asyncio.run(analyst.on_signal(_msg(), other, None, None)) is None


def test_the_retained_series_are_not_shown_to_claude(tmp_path):
    builder = MarketContextBuilder(MarketDataService(_Provider()))
    ctx = asyncio.run(builder.build("XAUUSD"))
    assert set(ctx.candle_series) == {Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1}
    assert "candle_series" not in repr(build_facts(ctx))


# ----------------------------------------------------------------- config

def _env(monkeypatch, **values):
    for name in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_TARGET_CHAT"):
        monkeypatch.setenv(name, {"TELEGRAM_API_ID": "1"}.get(name, "test"))
    monkeypatch.delenv("INDEPENDENT_ANALYSIS_MODE", raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return Settings.load(env_file=None)


def test_independent_analysis_is_off_by_default(monkeypatch):
    assert _env(monkeypatch).independent_analysis_mode == "off"


def test_shadow_is_accepted_and_anything_else_refused(monkeypatch):
    assert _env(monkeypatch, INDEPENDENT_ANALYSIS_MODE="Shadow").independent_analysis_mode == "shadow"
    with pytest.raises(ConfigError, match="INDEPENDENT_ANALYSIS_MODE"):
        _env(monkeypatch, INDEPENDENT_ANALYSIS_MODE="live")


# ---------------------------------------------------------------- /shadow

class _Update:
    def __init__(self, chat_id):
        self.effective_chat = MagicMock(id=chat_id, username=None)
        self.message = MagicMock()
        self.message.reply_text = AsyncMock()


def test_shadow_command_answers_only_the_configured_chat(tmp_path):
    store = AnalystStore(tmp_path / "x.db")
    asyncio.run(store.initialize())
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=12345))
    bot.attach_shadow_store(store)
    bot._subscribers = {777}
    owner, subscriber = _Update(12345), _Update(777)
    asyncio.run(bot._cmd_shadow(owner, None))
    asyncio.run(bot._cmd_shadow(subscriber, None))
    assert owner.message.reply_text.await_args.args[0] == "No shadow analysis recorded yet."
    assert subscriber.message.reply_text.await_count == 0
