"""Unit tests for decision_engine.py (RFC-008).

No real Claude CLI subprocess is ever spawned: DecisionEngine._run_cli is
replaced with an AsyncMock per test (the same boundary-mocking approach
test_telegram_bot.py uses for telegram.Bot), returning canned CLI envelope
strings. shutil.which is patched so DecisionEngine's constructor doesn't
require a real `claude` executable on PATH — matching how test_telegram_bot
avoids the real ~1s telegram.Bot() constructor cost, this avoids any
dependency on the local machine's actual CLI installation.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from config import Settings
from decision_engine import (
    ClaudeDecision,
    DecisionEngine,
    DecisionError,
    DecisionSource,
    Verdict,
    _build_stdin_payload,
    _to_json,
)
from market_data import Timeframe
from models import SignalAnalysis, TradeSetup
from scoring_engine import FactorScore, ScoreDirection, ScoringResult
from smc_engine import SMCAnalysis
from structure_engine import StructureAnalysis, SwingPoint, SwingType, TrendDirection


# --------------------------------------------------------------------------- helpers

def _ts(i: int = 0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)


def _settings() -> Settings:
    return Settings(
        api_id=123456, api_hash="deadbeef", session_name="test", phone="",
        target_chat="test_chat", claude_cli_path="claude", model="",
        claude_max_turns=3, signal_parser_mode="off", worker_count=2,
        queue_maxsize=200, max_message_chars=8000, analyse_edits=False,
        max_retries=2, request_timeout=5.0, log_level="INFO", log_file=None,
        jsonl_output=None, color=False, show_json=False,
        storage_db_path=__import__("pathlib").Path("data/signals.db"),
        twelve_data_api_key="", market_data_cache_ttl=30.0,
        market_data_timeframe=Timeframe.H1,
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())


@pytest.fixture(autouse=True)
def _fake_cli_path(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "C:/fake/claude.cmd")


def _engine() -> DecisionEngine:
    return DecisionEngine(_settings())


def _signal(
    is_signal: bool = True, category: str = "signal",
    entries=(61200.0,), stop_loss=60350.0, take_profits=(62400.0,),
) -> SignalAnalysis:
    return SignalAnalysis(
        is_signal=is_signal, category=category,
        setup=TradeSetup(
            symbol="BTCUSDT", direction="long", order_type="limit",
            entries=list(entries), stop_loss=stop_loss, take_profits=list(take_profits),
        ),
        summary="Long setup on BTCUSDT.", confidence=0.9,
        missing_fields=[], notes=None, source="regex",
    )


def _structure(
    symbol: str = "BTCUSDT", timeframe: Timeframe = Timeframe.M1,
    trend: TrendDirection = TrendDirection.BULLISH, swings=(), events=(),
) -> StructureAnalysis:
    events = tuple(events)
    return StructureAnalysis(
        symbol=symbol, timeframe=timeframe, trend=trend,
        swing_points=tuple(swings), events=events,
        last_event=events[-1] if events else None,
    )


def _smc(symbol: str = "BTCUSDT", timeframe: Timeframe = Timeframe.M1, **overrides) -> SMCAnalysis:
    defaults = dict(
        liquidity_pools=(), liquidity_sweeps=(), equal_highs=(), equal_lows=(),
        fair_value_gaps=(), inverse_fvgs=(), order_blocks=(), breaker_blocks=(),
        mitigation_blocks=(), supply_zones=(), demand_zones=(),
        premium_zone=None, discount_zone=None, ote_zone=None,
    )
    defaults.update(overrides)
    return SMCAnalysis(symbol=symbol, timeframe=timeframe, **defaults)


def _scoring(
    symbol: str = "BTCUSDT", timeframe: Timeframe = Timeframe.M1,
    direction: ScoreDirection = ScoreDirection.BUY, confidence: int = 80,
) -> ScoringResult:
    breakdown = (
        FactorScore("trend", 20.0, 20.0, TrendDirection.BULLISH, "Structure trend is bullish"),
    )
    return ScoringResult(
        symbol=symbol, timeframe=timeframe, direction=direction, confidence=confidence,
        net_score=20.0, total_possible=106.0, breakdown=breakdown,
        reasons=tuple(f.reason for f in breakdown),
    )


def _valid_claude_decision(**overrides) -> dict:
    base = dict(
        verdict="buy", confidence=75, reasoning="Strong bullish confluence across all engines.",
        strengths=["Trend is bullish", "BOS confirmed continuation"],
        risks=["No take profit stated"],
        execution_plan="Enter long using the stated entry, stop loss, and take profit.",
    )
    base.update(overrides)
    return base


def _cli_envelope(structured_output, *, is_error: bool = False, result: str = "ok") -> str:
    return json.dumps({"structured_output": structured_output, "is_error": is_error, "result": result})


# --------------------------------------------------------------------------- prompt generation

def test_stdin_payload_contains_exactly_the_four_engine_sections():
    payload = _build_stdin_payload(_signal(), _structure(), _smc(), _scoring())

    assert payload.startswith("<analysis>\n")
    assert payload.rstrip().endswith("</analysis>")
    body = payload[len("<analysis>\n"):payload.rindex("</analysis>")].strip()
    data = json.loads(body)

    assert set(data.keys()) == {"signal", "structure", "smc", "scoring"}
    assert data["signal"]["is_signal"] is True
    assert data["structure"]["trend"] == "bullish"
    assert data["scoring"]["direction"] == "buy"


def test_stdin_payload_never_contains_raw_candle_data():
    payload = _build_stdin_payload(_signal(), _structure(), _smc(), _scoring())
    assert "candle" not in payload.lower()
    assert "open" not in payload.lower()  # candle OHLC fields never appear anywhere


def test_to_json_serializes_enums_to_their_value():
    structure = _structure(trend=TrendDirection.BEARISH)
    assert _to_json(structure)["trend"] == "bearish"


def test_to_json_serializes_datetimes_to_iso_strings():
    swing = SwingPoint(index=1, timestamp=_ts(5), price=100.0, type=SwingType.HIGH, strength=2)
    data = _to_json(swing)
    assert data["timestamp"] == _ts(5).isoformat()
    assert isinstance(data["timestamp"], str)


def test_to_json_serializes_nested_dataclasses_and_pydantic_models():
    data = _to_json(_signal())
    assert data["setup"]["symbol"] == "BTCUSDT"
    assert data["setup"]["entries"] == [61200.0]


# --------------------------------------------------------------------------- successful Claude path

def test_decide_returns_claude_sourced_result_on_success():
    engine = _engine()
    engine._run_cli = AsyncMock(return_value=(0, _cli_envelope(_valid_claude_decision()), ""))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))

    assert result.source == DecisionSource.CLAUDE
    assert result.verdict == Verdict.BUY
    assert result.confidence == 75
    assert "Strong bullish" in result.reasoning
    assert result.strengths == ["Trend is bullish", "BOS confirmed continuation"]
    assert result.risks == ["No take profit stated"]
    engine._run_cli.assert_awaited_once()


def test_decide_does_not_call_claude_for_a_non_signal():
    engine = _engine()
    engine._run_cli = AsyncMock()

    result = asyncio.run(engine.decide(_signal(is_signal=False), _structure(), _smc(), _scoring()))

    assert result.source == DecisionSource.FALLBACK
    assert result.verdict == Verdict.REJECT
    engine._run_cli.assert_not_awaited()


def test_decide_clamps_claude_confidence_to_scoring_confidence():
    engine = _engine()
    engine._run_cli = AsyncMock(return_value=(0, _cli_envelope(_valid_claude_decision(confidence=99)), ""))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring(confidence=40)))

    assert result.confidence == 40  # never inflated beyond the deterministic score
    assert result.source == DecisionSource.CLAUDE


def test_decide_never_lowers_claude_confidence_below_its_own_value():
    engine = _engine()
    engine._run_cli = AsyncMock(return_value=(0, _cli_envelope(_valid_claude_decision(confidence=30)), ""))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring(confidence=80)))

    assert result.confidence == 30  # Claude's own (lower) number passes through unchanged


# --------------------------------------------------------------------------- invalid output -> fallback

def test_decide_falls_back_on_non_json_output():
    engine = _engine()
    engine._run_cli = AsyncMock(return_value=(0, "not json at all", ""))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))
    assert result.source == DecisionSource.FALLBACK


def test_decide_falls_back_when_structured_output_missing():
    engine = _engine()
    envelope = json.dumps({"result": "some prose", "is_error": False})
    engine._run_cli = AsyncMock(return_value=(0, envelope, ""))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))
    assert result.source == DecisionSource.FALLBACK


def test_decide_falls_back_on_schema_invalid_structured_output():
    engine = _engine()
    bad = _valid_claude_decision(confidence=150)  # outside the 0-100 schema range
    engine._run_cli = AsyncMock(return_value=(0, _cli_envelope(bad), ""))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))
    assert result.source == DecisionSource.FALLBACK


def test_decide_falls_back_on_missing_required_field():
    engine = _engine()
    bad = _valid_claude_decision()
    del bad["execution_plan"]
    engine._run_cli = AsyncMock(return_value=(0, _cli_envelope(bad), ""))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))
    assert result.source == DecisionSource.FALLBACK


def test_decide_falls_back_when_cli_reports_an_error():
    engine = _engine()
    envelope = json.dumps({"is_error": True, "result": "authentication_failed"})
    engine._run_cli = AsyncMock(return_value=(1, envelope, ""))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))
    assert result.source == DecisionSource.FALLBACK


def test_decide_falls_back_on_empty_output():
    engine = _engine()
    engine._run_cli = AsyncMock(return_value=(1, "", "some stderr text"))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))
    assert result.source == DecisionSource.FALLBACK


# --------------------------------------------------------------------------- deterministic fallback content

def test_fallback_verdict_matches_scoring_direction_buy():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("boom", retryable=False))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring(direction=ScoreDirection.BUY)))
    assert result.verdict == Verdict.BUY
    assert result.source == DecisionSource.FALLBACK


def test_fallback_verdict_matches_scoring_direction_sell():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("boom", retryable=False))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring(direction=ScoreDirection.SELL)))
    assert result.verdict == Verdict.SELL


def test_fallback_verdict_is_wait_when_scoring_direction_is_none():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("boom", retryable=False))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring(direction=ScoreDirection.NONE)))
    assert result.verdict == Verdict.WAIT


def test_fallback_confidence_matches_scoring_confidence_exactly():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("boom", retryable=False))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring(confidence=37)))
    assert result.confidence == 37


def test_fallback_flags_missing_risk_management_fields():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("boom", retryable=False))
    signal = _signal(entries=(), stop_loss=None, take_profits=())

    result = asyncio.run(engine.decide(signal, _structure(), _smc(), _scoring()))

    risks_text = " ".join(result.risks).lower()
    assert "entry" in risks_text
    assert "stop loss" in risks_text
    assert "take profit" in risks_text


def test_fallback_strengths_come_from_scoring_reasons():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("boom", retryable=False))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))
    assert result.strengths == list(_scoring().reasons)


def test_fallback_never_invents_an_execution_plan():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("boom", retryable=False))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))
    assert "no execution plan" in result.execution_plan.lower()


def test_reject_fallback_for_non_signal_states_no_setup_to_evaluate():
    engine = _engine()
    result = asyncio.run(engine.decide(_signal(is_signal=False), _structure(), _smc(), _scoring()))

    assert result.verdict == Verdict.REJECT
    assert result.confidence == 0
    assert "not identified as an actionable" in result.reasoning.lower()


def test_fallback_is_deterministic_across_repeated_calls():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("boom", retryable=False))
    signal, structure, smc, scoring = _signal(), _structure(), _smc(), _scoring()

    r1 = asyncio.run(engine.decide(signal, structure, smc, scoring))
    r2 = asyncio.run(engine.decide(signal, structure, smc, scoring))
    assert r1 == r2


# --------------------------------------------------------------------------- retry behaviour

def test_decide_retries_a_retryable_failure_then_succeeds():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=[
        DecisionError("transient", retryable=True),
        (0, _cli_envelope(_valid_claude_decision()), ""),
    ])

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))

    assert result.source == DecisionSource.CLAUDE
    assert engine._run_cli.await_count == 2


def test_decide_does_not_retry_a_non_retryable_failure():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=DecisionError("permanent", retryable=False))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))

    assert result.source == DecisionSource.FALLBACK
    assert engine._run_cli.await_count == 1


def test_decide_falls_back_after_exhausting_retries():
    engine = _engine()  # max_retries=2 -> 3 attempts total
    engine._run_cli = AsyncMock(side_effect=DecisionError("always fails", retryable=True))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))

    assert result.source == DecisionSource.FALLBACK
    assert engine._run_cli.await_count == 3


def test_decide_falls_back_on_unexpected_non_decision_error_exception():
    engine = _engine()
    engine._run_cli = AsyncMock(side_effect=RuntimeError("totally unexpected"))

    result = asyncio.run(engine.decide(_signal(), _structure(), _smc(), _scoring()))  # must not raise

    assert result.source == DecisionSource.FALLBACK
    assert engine._run_cli.await_count == 1  # unknown exceptions are treated as non-retryable


# --------------------------------------------------------------------------- input validation

def test_decide_raises_on_mismatched_symbol():
    engine = _engine()
    structure = _structure(symbol="BTCUSDT")
    smc = _smc(symbol="ETHUSDT")
    with pytest.raises(ValueError):
        asyncio.run(engine.decide(_signal(), structure, smc, _scoring(symbol="BTCUSDT")))


def test_decide_raises_on_mismatched_timeframe():
    engine = _engine()
    structure = _structure(timeframe=Timeframe.M1)
    scoring = _scoring(timeframe=Timeframe.H1)
    with pytest.raises(ValueError):
        asyncio.run(engine.decide(_signal(), structure, _smc(), scoring))


# --------------------------------------------------------------------------- schema shape

def test_claude_decision_schema_has_no_symbol_or_timeframe_or_source_fields():
    from decision_engine import _build_schema
    schema = _build_schema()
    properties = schema.get("properties", {})
    assert "symbol" not in properties
    assert "timeframe" not in properties
    assert "source" not in properties
    assert "verdict" in properties
    assert "confidence" in properties


def test_decision_result_symbol_and_timeframe_come_from_inputs_not_claude():
    engine = _engine()
    engine._run_cli = AsyncMock(return_value=(0, _cli_envelope(_valid_claude_decision()), ""))

    result = asyncio.run(engine.decide(
        _signal(), _structure(symbol="ETHUSDT", timeframe=Timeframe.H4),
        _smc(symbol="ETHUSDT", timeframe=Timeframe.H4),
        _scoring(symbol="ETHUSDT", timeframe=Timeframe.H4),
    ))

    assert result.symbol == "ETHUSDT"
    assert result.timeframe == Timeframe.H4
