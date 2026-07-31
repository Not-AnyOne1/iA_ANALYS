"""Tests for RFC-001's mode-based routing in AnalysisPipeline._get_analysis.

Uses a stub analyzer (no real claude_client/CLI involved) to verify: shadow
mode always calls Claude and only compares; active mode skips Claude on a
confident regex match and falls back on an ambiguous one; off mode never
consults the regex parser at all.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from claude_client import AnalysisError
from config import Settings
from formatter import Formatter
from market_data import Timeframe
from models import SignalAnalysis, TradeSetup
from pipeline import AnalysisPipeline
from telegram_client import IncomingMessage


class StubAnalyzer:
    """Records every call; returns a canned result or raises on demand."""

    def __init__(
        self,
        result: Optional[SignalAnalysis] = None,
        error: Optional[AnalysisError] = None,
    ) -> None:
        self.result = result
        self.error = error
        self.call_count = 0

    async def analyze(self, message: IncomingMessage) -> SignalAnalysis:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _settings(mode: str) -> Settings:
    """Construct Settings directly — no env vars, no I/O, parallel-safe."""
    return Settings(
        api_id=123456,
        api_hash="deadbeef",
        session_name="test",
        phone="",
        target_chat="test_chat",
        claude_cli_path="claude",
        model="",
        claude_max_turns=3,
        signal_parser_mode=mode,
        worker_count=2,
        queue_maxsize=200,
        max_message_chars=8000,
        analyse_edits=False,
        max_retries=3,
        request_timeout=180.0,
        log_level="INFO",
        log_file=None,
        jsonl_output=None,
        color=False,
        show_json=False,
        storage_db_path=Path("data/signals.db"),
        twelve_data_api_key="",
        market_data_cache_ttl=30.0,
        market_data_timeframe=Timeframe.H1,
    )


def _msg(text: str) -> IncomingMessage:
    return IncomingMessage(
        id=1, chat_id=-1001234567890, chat_title="Test", sender="Analyst",
        timestamp=datetime.now(timezone.utc), text=text,
    )


def _claude_result(is_signal: bool = False, category: str = "commentary") -> SignalAnalysis:
    return SignalAnalysis(
        is_signal=is_signal, category=category, setup=TradeSetup(),
        summary="stub claude result", confidence=0.7,
        missing_fields=[], notes=None, source="claude",
    )


def _make_pipeline(mode: str, analyzer: StubAnalyzer) -> AnalysisPipeline:
    return AnalysisPipeline(_settings(mode), asyncio.Queue(), analyzer, Formatter(color=False, show_json=False))


_CLEAN_SIGNAL_TEXT = "BTCUSDT LONG Entry: 61200 SL: 60350 TP1: 62400"


def test_off_mode_always_calls_claude_even_for_a_clean_signal():
    analyzer = StubAnalyzer(result=_claude_result())
    pipeline = _make_pipeline("off", analyzer)

    result = asyncio.run(pipeline._get_analysis(_msg(_CLEAN_SIGNAL_TEXT)))

    assert analyzer.call_count == 1
    assert result is analyzer.result


def test_shadow_mode_always_calls_claude_and_records_agreement():
    # Claude's canned answer matches what regex would say for this message.
    claude_answer = SignalAnalysis(
        is_signal=True, category="signal",
        setup=TradeSetup(
            symbol="BTCUSDT", direction="long", order_type="unknown",
            entries=[61200.0], stop_loss=60350.0, take_profits=[62400.0],
        ),
        summary="claude's own summary", confidence=0.95,
        missing_fields=[], notes=None, source="claude",
    )
    analyzer = StubAnalyzer(result=claude_answer)
    pipeline = _make_pipeline("shadow", analyzer)

    result = asyncio.run(pipeline._get_analysis(_msg(_CLEAN_SIGNAL_TEXT)))

    assert analyzer.call_count == 1  # shadow mode never skips Claude
    assert result is claude_answer   # shadow mode never overrides Claude's answer
    assert pipeline.stats.shadow_agree == 1
    assert pipeline.stats.shadow_disagree == 0


def test_shadow_mode_records_disagreement():
    claude_answer = _claude_result(is_signal=False, category="commentary")
    analyzer = StubAnalyzer(result=claude_answer)
    pipeline = _make_pipeline("shadow", analyzer)

    asyncio.run(pipeline._get_analysis(_msg(_CLEAN_SIGNAL_TEXT)))

    assert pipeline.stats.shadow_agree == 0
    assert pipeline.stats.shadow_disagree == 1


def test_shadow_mode_logs_deferred_regex_verdict_without_counting_it():
    # An ambiguous message: regex defers, so there's no regex verdict to
    # agree/disagree with — shadow stats should stay untouched.
    analyzer = StubAnalyzer(result=_claude_result(is_signal=False, category="update"))
    pipeline = _make_pipeline("shadow", analyzer)

    asyncio.run(pipeline._get_analysis(_msg("Move SL to entry on BTCUSDT")))

    assert analyzer.call_count == 1
    assert pipeline.stats.shadow_agree == 0
    assert pipeline.stats.shadow_disagree == 0


def test_active_mode_skips_claude_on_confident_regex_match():
    analyzer = StubAnalyzer(result=_claude_result())
    pipeline = _make_pipeline("active", analyzer)

    result = asyncio.run(pipeline._get_analysis(_msg(_CLEAN_SIGNAL_TEXT)))

    assert analyzer.call_count == 0  # Claude never consulted
    assert result is not None
    assert result.source == "regex"
    assert result.setup.symbol == "BTCUSDT"


def test_active_mode_falls_back_to_claude_on_ambiguous_message():
    analyzer = StubAnalyzer(result=_claude_result(is_signal=False, category="update"))
    pipeline = _make_pipeline("active", analyzer)

    result = asyncio.run(pipeline._get_analysis(_msg("Move SL to entry on BTCUSDT")))

    assert analyzer.call_count == 1
    assert result is analyzer.result
    assert result.source == "claude"


def test_analysis_failure_is_isolated_and_counted():
    analyzer = StubAnalyzer(error=AnalysisError("boom", retryable=False))
    pipeline = _make_pipeline("off", analyzer)

    result = asyncio.run(pipeline._get_analysis(_msg("anything")))

    assert result is None
    assert pipeline.stats.failures == 1


# --------------------------------------------------------------------- RFC-002 wiring

def test_a_wired_bot_does_not_interfere_with_analysis():
    """A real (disabled) TelegramBot wired into the pipeline must not affect
    the analysis, and must send nothing on the _process path."""
    from telegram_bot import BotSettings, TelegramBot

    analyzer = StubAnalyzer(result=_claude_result(is_signal=True, category="signal"))
    bot = TelegramBot(BotSettings(bot_token=None, chat_id=None))  # disabled
    pipeline = AnalysisPipeline(
        _settings("off"), asyncio.Queue(), analyzer,
        Formatter(color=False, show_json=False), bot=bot,
    )

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))

    assert pipeline.stats.received == 1
    assert bot._stats.notifications_sent == 0


def test_process_never_sends_the_parser_extraction_to_telegram(monkeypatch):
    """_process must NOT call notify(): at that point only the parser's
    extraction exists, which is an intermediate guess made before any market
    data was consulted. The single Telegram message is the finished report,
    sent from _validate_with_market_data() — see
    tests/test_telegram_report_delivery.py.

    (This replaced an RFC-002 test that asserted the opposite; notify()
    itself is unchanged and still covered by tests/test_telegram_bot.py.)
    """
    from unittest.mock import AsyncMock, MagicMock
    from telegram_bot import BotSettings, TelegramBot

    # Avoid real telegram.Bot() construction (~1s of unrelated client/SSL setup).
    monkeypatch.setattr("telegram_bot.Bot", lambda token: MagicMock())

    analyzer = StubAnalyzer(result=_claude_result(is_signal=True, category="signal"))
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=1))
    bot.notify = AsyncMock()
    bot.send_report = AsyncMock()
    pipeline = AnalysisPipeline(
        _settings("off"), asyncio.Queue(), analyzer,
        Formatter(color=False, show_json=False), bot=bot,
    )

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))

    bot.notify.assert_not_awaited()
    # No context builder configured -> no report exists -> nothing sent at all.
    bot.send_report.assert_not_awaited()
    assert pipeline.stats.received == 1     # the analysis itself still completed


def test_bot_send_report_exception_never_stops_the_pipeline(monkeypatch):
    """Even if telegram_bot.py's own failure isolation somehow regressed,
    pipeline.py's wrapper around send_report() must still swallow it."""
    from unittest.mock import AsyncMock, MagicMock
    from telegram_bot import BotSettings, TelegramBot

    monkeypatch.setattr("telegram_bot.Bot", lambda token: MagicMock())

    analyzer = StubAnalyzer(result=_claude_result(is_signal=True, category="signal"))
    bot = TelegramBot(BotSettings(bot_token="t", chat_id=1))
    bot.send_report = AsyncMock(side_effect=RuntimeError("bot module regressed"))
    pipeline = AnalysisPipeline(
        _settings("off"), asyncio.Queue(), analyzer,
        Formatter(color=False, show_json=False), bot=bot,
    )

    # Must not raise.
    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))

    assert pipeline.stats.received == 1


# --------------------------------------------------------------------- RFC-003 wiring

def test_process_records_the_analysis_to_storage():
    """_process must call storage.record(message, analysis) exactly once per
    successful analysis, alongside (not instead of) the existing JSONL write."""
    from unittest.mock import AsyncMock

    analyzer = StubAnalyzer(result=_claude_result(is_signal=True, category="signal"))
    storage = AsyncMock()
    pipeline = AnalysisPipeline(
        _settings("off"), asyncio.Queue(), analyzer,
        Formatter(color=False, show_json=False), storage=storage,
    )
    message = _msg(_CLEAN_SIGNAL_TEXT)

    asyncio.run(pipeline._process(message))

    # The stub analysis has no setup.symbol, so enrichment short-circuits
    # and decision/structure/smc are all None — still explicitly passed,
    # per storage.record()'s RFC-009 signature.
    storage.record.assert_awaited_once_with(
        message, analyzer.result, decision=None, structure=None, smc=None
    )


def test_storage_record_exception_never_stops_the_pipeline():
    """Even if storage.py's own failure isolation somehow regressed,
    pipeline.py's wrapper around record() must still swallow it."""
    from unittest.mock import AsyncMock

    analyzer = StubAnalyzer(result=_claude_result(is_signal=True, category="signal"))
    storage = AsyncMock()
    storage.record = AsyncMock(side_effect=RuntimeError("storage module regressed"))
    pipeline = AnalysisPipeline(
        _settings("off"), asyncio.Queue(), analyzer,
        Formatter(color=False, show_json=False), storage=storage,
    )

    # Must not raise.
    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))

    assert pipeline.stats.received == 1


# ----------------------------------------------------- integration phase (RFC-004..008)

def _signal_with_symbol(symbol: str = "BTCUSDT") -> SignalAnalysis:
    return SignalAnalysis(
        is_signal=True, category="signal",
        setup=TradeSetup(symbol=symbol, direction="long", order_type="limit",
                          entries=[100.0], stop_loss=95.0, take_profits=[110.0]),
        summary="stub signal", confidence=0.8,
        missing_fields=[], notes=None, source="claude",
    )


def _make_enrichment_pipeline(analysis: SignalAnalysis, **engines):
    from unittest.mock import AsyncMock
    analyzer = StubAnalyzer(result=analysis)
    storage = AsyncMock()
    pipeline = AnalysisPipeline(
        _settings("off"), asyncio.Queue(), analyzer,
        Formatter(color=False, show_json=False), storage=storage, **engines,
    )
    return pipeline, storage


def test_enrichment_populates_storage_with_structure_smc_and_decision():
    from unittest.mock import AsyncMock, MagicMock
    from market_data import CandleSeries, Timeframe
    from structure_engine import StructureAnalysis, TrendDirection
    from smc_engine import SMCAnalysis
    from decision_engine import DecisionResult, DecisionSource, Verdict

    series = CandleSeries(symbol="BTCUSDT", timeframe=Timeframe.H1, provider="stub", candles=())
    structure = StructureAnalysis(symbol="BTCUSDT", timeframe=Timeframe.H1, trend=TrendDirection.BULLISH,
                                   swing_points=(), events=(), last_event=None)
    smc = SMCAnalysis(symbol="BTCUSDT", timeframe=Timeframe.H1,
                       liquidity_pools=(), liquidity_sweeps=(), equal_highs=(), equal_lows=(),
                       fair_value_gaps=(), inverse_fvgs=(), order_blocks=(), breaker_blocks=(),
                       mitigation_blocks=(), supply_zones=(), demand_zones=(),
                       premium_zone=None, discount_zone=None, ote_zone=None)
    decision = DecisionResult(symbol="BTCUSDT", timeframe=Timeframe.H1, verdict=Verdict.BUY,
                               confidence=70, reasoning="r", strengths=[], risks=[],
                               execution_plan="e", source=DecisionSource.FALLBACK)

    market_data = AsyncMock()
    market_data.get_candles = AsyncMock(return_value=series)
    structure_engine = MagicMock()
    structure_engine.analyze = MagicMock(return_value=structure)
    smc_engine = MagicMock()
    smc_engine.analyze = MagicMock(return_value=smc)
    scoring_engine = MagicMock()
    scoring_engine.score = MagicMock(return_value="stub-scoring-result")
    decision_engine = AsyncMock()
    decision_engine.decide = AsyncMock(return_value=decision)

    pipeline, storage = _make_enrichment_pipeline(
        _signal_with_symbol("BTCUSDT"),
        market_data=market_data, structure_engine=structure_engine,
        smc_engine=smc_engine, scoring_engine=scoring_engine, decision_engine=decision_engine,
    )

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))

    market_data.get_candles.assert_awaited_once_with("BTCUSDT", pipeline._settings.market_data_timeframe)
    structure_engine.analyze.assert_called_once_with(series)
    smc_engine.analyze.assert_called_once_with(series, structure)
    scoring_engine.score.assert_called_once_with(structure, smc)
    decision_engine.decide.assert_awaited_once()
    storage.record.assert_awaited_once()
    _, kwargs = storage.record.await_args
    assert kwargs["structure"] is structure
    assert kwargs["smc"] is smc
    assert kwargs["decision"] is decision


def test_enrichment_skipped_when_signal_has_no_symbol():
    from unittest.mock import AsyncMock
    market_data = AsyncMock()
    pipeline, storage = _make_enrichment_pipeline(
        _claude_result(is_signal=True, category="signal"),  # setup.symbol is None
        market_data=market_data,
    )

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))

    market_data.get_candles.assert_not_called()
    _, kwargs = storage.record.await_args
    assert kwargs == {"decision": None, "structure": None, "smc": None}


def test_enrichment_skipped_for_non_signal():
    from unittest.mock import AsyncMock
    market_data = AsyncMock()
    pipeline, storage = _make_enrichment_pipeline(
        _claude_result(is_signal=False, category="commentary"),
        market_data=market_data,
    )

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))

    market_data.get_candles.assert_not_called()


def test_enrichment_skipped_when_market_data_not_configured():
    # market_data=None (the default) -> _enrich must not touch it at all.
    pipeline, storage = _make_enrichment_pipeline(_signal_with_symbol())

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))  # must not raise

    _, kwargs = storage.record.await_args
    assert kwargs == {"decision": None, "structure": None, "smc": None}


def test_enrichment_degrades_gracefully_on_market_data_error():
    from unittest.mock import AsyncMock
    from market_data import MarketDataError

    market_data = AsyncMock()
    market_data.get_candles = AsyncMock(side_effect=MarketDataError("unavailable", retryable=False))
    pipeline, storage = _make_enrichment_pipeline(_signal_with_symbol(), market_data=market_data)

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))  # must not raise

    _, kwargs = storage.record.await_args
    assert kwargs == {"decision": None, "structure": None, "smc": None}
    # The base pipeline still proceeded: analysis was still counted and stored.
    assert pipeline.stats.received == 1
    storage.record.assert_awaited_once()


def test_enrichment_unexpected_exception_never_stops_the_pipeline():
    from unittest.mock import AsyncMock, MagicMock
    from market_data import CandleSeries, Timeframe

    series = CandleSeries(symbol="BTCUSDT", timeframe=Timeframe.H1, provider="stub", candles=())
    market_data = AsyncMock()
    market_data.get_candles = AsyncMock(return_value=series)
    structure_engine = MagicMock()
    structure_engine.analyze = MagicMock(side_effect=RuntimeError("structure engine regressed"))

    pipeline, storage = _make_enrichment_pipeline(
        _signal_with_symbol(), market_data=market_data, structure_engine=structure_engine,
        smc_engine=MagicMock(), scoring_engine=MagicMock(),
    )

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))  # must not raise

    _, kwargs = storage.record.await_args
    assert kwargs == {"decision": None, "structure": None, "smc": None}
    assert pipeline.stats.received == 1


def test_enrichment_without_decision_engine_still_stores_structure_and_smc():
    from unittest.mock import AsyncMock, MagicMock
    from market_data import CandleSeries, Timeframe
    from structure_engine import StructureAnalysis, TrendDirection
    from smc_engine import SMCAnalysis

    series = CandleSeries(symbol="BTCUSDT", timeframe=Timeframe.H1, provider="stub", candles=())
    structure = StructureAnalysis(symbol="BTCUSDT", timeframe=Timeframe.H1, trend=TrendDirection.RANGING,
                                   swing_points=(), events=(), last_event=None)
    smc = SMCAnalysis(symbol="BTCUSDT", timeframe=Timeframe.H1,
                       liquidity_pools=(), liquidity_sweeps=(), equal_highs=(), equal_lows=(),
                       fair_value_gaps=(), inverse_fvgs=(), order_blocks=(), breaker_blocks=(),
                       mitigation_blocks=(), supply_zones=(), demand_zones=(),
                       premium_zone=None, discount_zone=None, ote_zone=None)

    market_data = AsyncMock()
    market_data.get_candles = AsyncMock(return_value=series)
    structure_engine = MagicMock()
    structure_engine.analyze = MagicMock(return_value=structure)
    smc_engine = MagicMock()
    smc_engine.analyze = MagicMock(return_value=smc)
    scoring_engine = MagicMock()
    scoring_engine.score = MagicMock(return_value="stub-scoring-result")

    pipeline, storage = _make_enrichment_pipeline(
        _signal_with_symbol(), market_data=market_data, structure_engine=structure_engine,
        smc_engine=smc_engine, scoring_engine=scoring_engine,  # decision_engine intentionally omitted
    )

    asyncio.run(pipeline._process(_msg(_CLEAN_SIGNAL_TEXT)))

    _, kwargs = storage.record.await_args
    assert kwargs["structure"] is structure
    assert kwargs["smc"] is smc
    assert kwargs["decision"] is None


# ----------------------------------------------------- hostile input (regression)

def test_lone_surrogates_in_message_do_not_break_the_jsonl_write(tmp_path):
    """Telegram text can carry lone surrogates (malformed UTF-16 from some
    clients). A plain utf-8 file handle raises UnicodeEncodeError mid-write,
    which cost the whole message because the JSONL write happens before
    storage and notification in _process."""
    from unittest.mock import AsyncMock
    import dataclasses

    analyzer = StubAnalyzer(result=_claude_result(is_signal=True, category="signal"))
    storage = AsyncMock()
    base = _settings("off")
    settings = dataclasses.replace(base, jsonl_output=tmp_path / "signals.jsonl")

    pipeline = AnalysisPipeline(
        settings, asyncio.Queue(), analyzer,
        Formatter(color=False, show_json=False), storage=storage,
    )
    message = _msg("BUY \ud800\udc00 GOLD NOW")

    async def run():
        async with pipeline:
            await pipeline._process(message)   # must not raise

    asyncio.run(run())

    assert pipeline.stats.received == 1
    storage.record.assert_awaited_once()               # reached storage
    assert (tmp_path / "signals.jsonl").stat().st_size > 0   # and got written
