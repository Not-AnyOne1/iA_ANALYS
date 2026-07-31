"""Integration tests for the Python Analysis Engine.

Exercises the real MarketContextBuilder against a stub provider through the
real MarketDataService — so the whole chain (fetch -> structure -> SMC ->
indicators -> levels -> session -> quality -> risk) runs for real, with only
the network boundary replaced.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from market_context import MarketContextBuilder, TimeframeAnalysis
from market_data import Candle, CandleSeries, MarketDataError, MarketDataService, Quote, Timeframe
from models import TradeSetup
from news_filter import EconomicEvent, NewsFilter, StaticCalendar
from trade_decision import build_facts


def _series(symbol: str, timeframe: Timeframe, n: int = 300, rising: bool = True) -> CandleSeries:
    base = datetime(2026, 3, 10, tzinfo=timezone.utc) - timedelta(hours=n)
    candles = []
    for i in range(n):
        price = 100.0 + (i * 0.5 if rising else -i * 0.5)
        candles.append(Candle(
            timestamp=base + timedelta(hours=i),
            open=price, high=price + 1.0, low=price - 1.0, close=price, volume=10.0,
        ))
    return CandleSeries(symbol=symbol, timeframe=timeframe, provider="stub",
                        candles=tuple(candles))


class StubProvider:
    """Answers every timeframe; can be told to fail specific ones."""

    name = "stub"

    def __init__(self, *, fail_timeframes=(), fail_quote=False,
                 bid=None, ask=None, rising=True):
        self.fail_timeframes = set(fail_timeframes)
        self.fail_quote = fail_quote
        self.bid, self.ask = bid, ask
        self.rising = rising
        self.candle_calls = []

    async def get_quote(self, symbol: str) -> Quote:
        if self.fail_quote:
            raise MarketDataError("stub quote failure", retryable=False)
        return Quote(symbol=symbol, price=250.0, timestamp=datetime.now(timezone.utc),
                     provider=self.name, bid=self.bid, ask=self.ask)

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        self.candle_calls.append(timeframe)
        if timeframe in self.fail_timeframes:
            raise MarketDataError(f"stub failure for {timeframe.value}", retryable=True)
        return _series(symbol, timeframe, rising=self.rising)


def _builder(provider, **kwargs) -> MarketContextBuilder:
    return MarketContextBuilder(MarketDataService(provider), **kwargs)


def _setup(**overrides) -> TradeSetup:
    base = dict(symbol="XAUUSD", direction="long", order_type="limit",
                entries=[250.0], stop_loss=240.0, take_profits=[280.0])
    base.update(overrides)
    return TradeSetup(**base)


# --------------------------------------------------------------------------- happy path

def test_builder_populates_every_section():
    ctx = asyncio.run(_builder(StubProvider(bid=249.9, ask=250.1)).build("XAUUSD", _setup()))

    assert ctx.is_usable
    assert ctx.current_price == 250.0
    assert ctx.spread == pytest.approx(0.2)
    assert ctx.structure is not None
    assert ctx.smc is not None
    assert ctx.scoring is not None
    assert ctx.levels is not None
    assert ctx.session is not None
    assert ctx.news is not None
    assert ctx.quality is not None
    assert ctx.risk is not None


def test_every_requested_timeframe_is_analysed():
    builder = _builder(StubProvider())
    ctx = asyncio.run(builder.build("XAUUSD"))

    for tf in (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1):
        assert tf.value in ctx.timeframes, f"{tf.value} missing"
        analysis = ctx.timeframes[tf.value]
        assert isinstance(analysis, TimeframeAnalysis)
        assert analysis.candles_analysed == 300


def test_indicators_are_computed_per_timeframe():
    ctx = asyncio.run(_builder(StubProvider()).build("XAUUSD"))
    primary = ctx.primary

    assert primary.atr is not None
    assert primary.rsi is not None
    assert primary.ema.ema_20 is not None
    assert primary.ema.ema_200 is not None
    assert primary.adx.adx is not None
    assert primary.volatility in ("low", "normal", "high")


def test_rising_series_produces_bullish_ema_alignment():
    ctx = asyncio.run(_builder(StubProvider(rising=True)).build("XAUUSD"))
    assert ctx.primary.ema_alignment == "bullish"


def test_falling_series_produces_bearish_ema_alignment():
    ctx = asyncio.run(_builder(StubProvider(rising=False)).build("XAUUSD"))
    assert ctx.primary.ema_alignment == "bearish"


def test_primary_timeframe_is_configurable():
    builder = _builder(StubProvider(), primary_timeframe=Timeframe.M15)
    ctx = asyncio.run(builder.build("XAUUSD"))
    assert ctx.primary_timeframe is Timeframe.M15
    assert ctx.primary is not None


def test_primary_timeframe_is_always_fetched_even_if_not_listed():
    builder = _builder(StubProvider(), timeframes=(Timeframe.D1,),
                       primary_timeframe=Timeframe.H4)
    ctx = asyncio.run(builder.build("XAUUSD"))
    assert "H4" in ctx.timeframes


# --------------------------------------------------------------------------- degradation

def test_missing_quote_degrades_without_raising():
    ctx = asyncio.run(_builder(StubProvider(fail_quote=True)).build("XAUUSD", _setup()))

    assert ctx.current_price is None
    assert any("quote unavailable" in w for w in ctx.warnings)
    assert ctx.is_usable          # candles still analysed
    assert ctx.structure is not None


def test_one_failing_timeframe_does_not_sink_the_others():
    ctx = asyncio.run(_builder(StubProvider(fail_timeframes={Timeframe.H4})).build("XAUUSD"))

    assert "H4" not in ctx.timeframes
    assert "H1" in ctx.timeframes
    assert any("H4 candles unavailable" in w for w in ctx.warnings)


def test_failing_primary_timeframe_marks_context_unusable():
    builder = _builder(StubProvider(fail_timeframes={Timeframe.H1}))
    ctx = asyncio.run(builder.build("XAUUSD"))

    assert not ctx.is_usable
    assert ctx.structure is None
    assert any("primary timeframe" in w for w in ctx.warnings)


def test_provider_without_bid_ask_records_a_warning_and_no_spread():
    ctx = asyncio.run(_builder(StubProvider()).build("XAUUSD"))
    assert ctx.spread is None
    assert any("no bid/ask" in w for w in ctx.warnings)


def test_total_data_failure_still_returns_a_context():
    provider = StubProvider(fail_quote=True,
                            fail_timeframes=set(Timeframe))
    ctx = asyncio.run(_builder(provider).build("XAUUSD", _setup()))

    assert not ctx.is_usable
    assert ctx.timeframes == {}
    assert len(ctx.warnings) >= 5     # every failure recorded
    assert ctx.session is not None    # session needs no market data


# --------------------------------------------------------------------------- risk wiring

def test_news_block_flows_into_the_risk_gate():
    now = datetime.now(timezone.utc)
    calendar = StaticCalendar([EconomicEvent.create("FOMC Statement", now, "USD")])
    builder = _builder(StubProvider(bid=249.9, ask=250.1),
                       news_filter=NewsFilter(calendar))
    ctx = asyncio.run(builder.build("XAUUSD", _setup()))

    assert ctx.news.blocked is True
    assert not ctx.risk.approved
    assert any(c.name == "news" for c in ctx.risk.rejections)


def test_no_setup_means_no_quality_or_risk():
    ctx = asyncio.run(_builder(StubProvider()).build("XAUUSD"))
    assert ctx.quality is None
    assert ctx.risk is None


def test_context_building_never_raises_on_a_broken_provider():
    class Exploding:
        name = "boom"

        async def get_quote(self, symbol):
            raise RuntimeError("provider exploded")

        async def get_candles(self, symbol, timeframe, limit):
            raise RuntimeError("provider exploded")

    ctx = asyncio.run(_builder(Exploding()).build("XAUUSD", _setup()))
    assert not ctx.is_usable
    assert ctx.warnings


# --------------------------------------------------------------------------- facts for Claude

def test_facts_contain_no_raw_candles():
    """The core guarantee: Claude gets finished numbers, never a series it
    could compute an indicator from."""
    ctx = asyncio.run(_builder(StubProvider(bid=249.9, ask=250.1)).build("XAUUSD", _setup()))
    facts = build_facts(ctx)

    import json
    blob = json.dumps(facts)
    assert "candles" not in blob or "candles_analysed" in blob
    # No OHLC arrays anywhere.
    assert '"open"' not in blob
    assert '"high"' not in blob
    assert '"low"' not in blob


def test_facts_include_every_computed_section():
    ctx = asyncio.run(_builder(StubProvider(bid=249.9, ask=250.1)).build("XAUUSD", _setup()))
    facts = build_facts(ctx)

    for key in ("symbol", "current_price", "spread", "session", "timeframes",
                "structure", "smc", "levels", "scoring", "trade_quality",
                "news", "risk"):
        assert key in facts, f"{key} missing from the facts given to Claude"


def test_facts_are_json_serialisable():
    import json
    ctx = asyncio.run(_builder(StubProvider()).build("XAUUSD", _setup()))
    json.dumps(build_facts(ctx))   # must not raise


def test_facts_surface_data_gaps():
    ctx = asyncio.run(_builder(StubProvider(fail_quote=True)).build("XAUUSD", _setup()))
    facts = build_facts(ctx)
    assert facts["current_price"] is None
    assert facts["warnings"]


def test_context_is_deterministic_for_fixed_data():
    """Same candles in, same analysis out — only the fetch varies."""
    builder = _builder(StubProvider(bid=249.9, ask=250.1))
    a = asyncio.run(builder.build("XAUUSD", _setup()))
    b = asyncio.run(builder.build("XAUUSD", _setup()))

    assert a.primary.atr == b.primary.atr
    assert a.primary.rsi == b.primary.rsi
    assert a.primary.adx.adx == b.primary.adx.adx
    assert a.structure.trend == b.structure.trend
    assert a.scoring.confidence == b.scoring.confidence


# ------------------------------------------------- end-to-end pipeline integration

def test_full_pipeline_telegram_to_report(tmp_path, monkeypatch):
    """Telegram signal -> extraction -> live data -> Python analysis ->
    Claude reasoning -> decision -> report, with only the two external
    boundaries (market provider, Claude CLI) stubbed."""
    import shutil
    from unittest.mock import AsyncMock
    from datetime import datetime as dt

    from config import Settings
    from formatter import Formatter
    from models import SignalAnalysis
    from pipeline import AnalysisPipeline
    from storage import Storage
    from telegram_client import IncomingMessage
    from trade_decision import TradeDecisionEngine

    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")

    settings = Settings(
        api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
        claude_cli_path="claude", model="", claude_max_turns=3,
        signal_parser_mode="off", worker_count=1, queue_maxsize=10,
        max_message_chars=8000, analyse_edits=False, max_retries=0,
        request_timeout=5.0, log_level="CRITICAL", log_file=None, jsonl_output=None,
        color=False, show_json=False, storage_db_path=tmp_path / "s.db",
        twelve_data_api_key="k", market_data_cache_ttl=30.0,
        market_data_timeframe=Timeframe.H1,
    )

    analysis = SignalAnalysis(
        is_signal=True, category="signal",
        setup=_setup(), summary="Buy gold.", confidence=0.8,
        missing_fields=[], notes=None, source="claude")

    class StubAnalyzer:
        async def analyze(self, message):
            return analysis

    provider = StubProvider(bid=249.9, ask=250.1)
    # The synthetic series is a straight line, so it has no swing structure
    # and the scoring engine correctly returns 0 confidence. Relax just that
    # threshold so the gate approves and the Claude path actually runs —
    # the gate's own thresholds are covered in test_risk_and_quality.py, and
    # the skip-on-rejection path has its own test below.
    from risk_engine import RiskSettings
    builder = MarketContextBuilder(
        MarketDataService(provider), risk_settings=RiskSettings(min_confidence=0))

    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, __import__("json").dumps({
        "structured_output": {
            "verdict": "wait", "confidence": 55,
            "reasoning": "Trend aligns but the stop sits near liquidity.",
            "strengths": ["H1 trend bullish"], "risks": ["spread is wide"],
            "strongest_reason_against": "The stop sits beyond a swing level.",
            "alternative_scenario": "Price sweeps the low first.",
            "worst_case": "Stop hit for 1R.",
            "best_case": "Target reached for 2.6R.",
            "execution_plan": "Wait for a pullback to the stated entry.",
        }, "is_error": False}), ""))

    async def run():
        storage = Storage(settings.storage_db_path)
        await storage.initialize()
        pipeline = AnalysisPipeline(
            settings, asyncio.Queue(), StubAnalyzer(),
            Formatter(color=False, show_json=False), storage=storage,
            context_builder=builder, trade_decision_engine=engine,
            emit_reports=False,
        )
        msg = IncomingMessage(id=1, chat_id=-1, chat_title="c", sender="s",
                              timestamp=dt.now(timezone.utc),
                              text="BUY GOLD NOW\nTP:280\nSL:240")
        async with pipeline:
            await pipeline._process(msg)
        rows = await storage.fetch_analyses()
        await storage.close()
        return pipeline, rows

    pipeline, rows = asyncio.run(run())

    # The whole chain ran and persisted an enriched row.
    assert pipeline.stats.received == 1
    assert len(rows) == 1
    row = rows[0]
    assert row.symbol == "XAUUSD"
    # ENTER/WAIT/SKIP, not BUY/SELL. Claude asked for "wait", but this
    # straight-line synthetic series has no swing structure so it validates
    # at 12/100 — below the 40 threshold — and the band caps it at SKIP.
    assert row.decision_verdict == "skip"
    assert row.decision_source == "claude"     # the model really was consulted
    # The persisted confidence is the ONE confidence: the deterministic score
    # plus the review's bonus, minus its penalty. This structureless
    # straight-line series scores near zero and the review can only nudge it,
    # so it lands far below the 40 band and the verdict is SKIP.
    assert row.decision_confidence < 40
    assert row.structure_trend is not None
    assert row.timeframe == "H1"

    # Claude was asked, and asked only about facts.
    engine._run_cli.assert_awaited_once()
    payload = engine._run_cli.await_args.args[0]
    assert "<facts>" in payload
    assert '"open"' not in payload      # no candle series reached the model


def test_report_renders_for_a_real_context():
    import report as report_mod
    ctx = asyncio.run(_builder(StubProvider(bid=249.9, ask=250.1)).build("XAUUSD", _setup()))
    text = report_mod.render_text(ctx)

    assert "XAUUSD" in text
    assert "INDICATORS" in text
    assert "H1" in text
    for heading in ("PRICE", "STRUCTURE", "SMART MONEY", "LEVELS",
                    "CONTEXT", "SETUP QUALITY", "SCORING & RISK"):
        assert heading in text


def test_the_deterministic_bands_cap_the_stored_verdict(tmp_path, monkeypatch):
    """The deterministic thresholds are final. Claude is consulted (so the
    findings get explained) but its verdict is capped by the validation
    score band — it cannot argue its way past the rules.

    This fixture validates at 12/100, below the 40 threshold, so the stored
    verdict is SKIP no matter what the model asks for."""
    import shutil
    from unittest.mock import AsyncMock
    from datetime import datetime as dt

    from config import Settings
    from formatter import Formatter
    from models import SignalAnalysis
    from pipeline import AnalysisPipeline
    from storage import Storage
    from telegram_client import IncomingMessage
    from trade_decision import TradeDecisionEngine

    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")
    settings = Settings(
        api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
        claude_cli_path="claude", model="", claude_max_turns=3,
        signal_parser_mode="off", worker_count=1, queue_maxsize=10,
        max_message_chars=8000, analyse_edits=False, max_retries=0,
        request_timeout=5.0, log_level="CRITICAL", log_file=None, jsonl_output=None,
        color=False, show_json=False, storage_db_path=tmp_path / "s.db",
        twelve_data_api_key="k", market_data_cache_ttl=30.0,
        market_data_timeframe=Timeframe.H1,
    )
    analysis = SignalAnalysis(
        is_signal=True, category="signal", setup=_setup(), summary="Buy gold.",
        confidence=0.8, missing_fields=[], notes=None, source="claude")

    class StubAnalyzer:
        async def analyze(self, message):
            return analysis

    # Default risk settings -> min_confidence 40 -> the flat series scores 0
    # -> rejected.
    builder = MarketContextBuilder(MarketDataService(StubProvider(bid=249.9, ask=250.1)))
    engine = TradeDecisionEngine(settings)
    engine._run_cli = AsyncMock(return_value=(0, __import__("json").dumps({
        "structured_output": {
            "verdict": "enter", "confidence": 95, "reasoning": "Looks good.",
            "strengths": ["trend"], "risks": ["none stated"],
            "strongest_reason_against": "M15 has not confirmed.",
            "alternative_scenario": "Reversal.", "worst_case": "1R loss.",
            "best_case": "3R gain.", "execution_plan": "Enter now.",
        }, "is_error": False}), ""))

    async def run():
        storage = Storage(settings.storage_db_path)
        await storage.initialize()
        pipeline = AnalysisPipeline(
            settings, asyncio.Queue(), StubAnalyzer(),
            Formatter(color=False, show_json=False), storage=storage,
            context_builder=builder, trade_decision_engine=engine, emit_reports=False,
        )
        msg = IncomingMessage(id=1, chat_id=-1, chat_title="c", sender="s",
                              timestamp=dt.now(timezone.utc), text="BUY GOLD NOW")
        async with pipeline:
            await pipeline._process(msg)
        rows = await storage.fetch_analyses()
        await storage.close()
        return rows

    rows = asyncio.run(run())

    engine._run_cli.assert_awaited_once()          # consulted, for the explanation
    assert rows[0].decision_verdict == "skip"      # but capped by the band
    assert rows[0].decision_source == "claude"
