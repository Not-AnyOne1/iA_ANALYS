"""Unit tests for trade_decision.py and report.py.

The Claude CLI boundary is mocked throughout (`_run_cli`), so no subprocess
runs. shutil.which is patched so constructing the engine needs no real CLI.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import report as report_mod
from config import Settings
from decision_engine import DecisionSource
from market_context import MarketContext, TimeframeAnalysis
from market_data import Timeframe
from indicators import ADXResult, EMASet
from risk_engine import RiskAssessment, RiskCheck, RiskVerdict
from scoring_engine import FactorScore, ScoreDirection, ScoringResult
from structure_engine import TrendDirection
from trade_decision import Action, TradeDecisionEngine, TradeDecisionError, build_facts


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())


@pytest.fixture(autouse=True)
def _fake_cli(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/fake/claude")


def _settings() -> Settings:
    return Settings(
        api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
        claude_cli_path="claude", model="", claude_max_turns=3,
        signal_parser_mode="off", worker_count=1, queue_maxsize=10,
        max_message_chars=8000, analyse_edits=False, max_retries=1,
        request_timeout=5.0, log_level="INFO", log_file=None, jsonl_output=None,
        color=False, show_json=False, storage_db_path=Path("data/signals.db"),
        twelve_data_api_key="", market_data_cache_ttl=30.0,
        market_data_timeframe=Timeframe.H1,
    )


def _tf(name: str = "H1") -> TimeframeAnalysis:
    return TimeframeAnalysis(
        timeframe=Timeframe.H1, candles_analysed=300, trend="bullish",
        ema=EMASet(100.0, 99.0, 98.0), rsi=55.0, atr=2.0, atr_percent=2.0,
        volatility="normal", adx=ADXResult(30.0, 25.0, 15.0), last_close=100.0,
    )


def _scoring(confidence: int = 70) -> ScoringResult:
    breakdown = (FactorScore("trend", 20.0, 20.0, TrendDirection.BULLISH, "trend bullish"),)
    return ScoringResult(symbol="XAUUSD", timeframe=Timeframe.H1,
                         direction=ScoreDirection.BUY, confidence=confidence,
                         net_score=20.0, total_possible=106.0,
                         breakdown=breakdown, reasons=("trend bullish",))


def _risk(approved: bool = True) -> RiskAssessment:
    checks = (
        RiskCheck("risk_reward", RiskVerdict.PASS if approved else RiskVerdict.REJECT,
                  "R:R 3.00" if approved else "R:R 0.40 below the 1.00 minimum"),
    )
    return RiskAssessment(approved=approved, checks=checks)


def _context(*, usable: bool = True, approved: bool = True,
             confidence: int = 70, bonus: int = 5, penalty: int = 0,
             validation: bool = True, fatal: bool = False) -> MarketContext:
    """A context that decides ENTER by default.

    ``confidence`` is the deterministic score — the ONE confidence. ``bonus``
    and ``penalty`` are the review's adjustments to it, so the adjusted score
    is ``confidence + bonus - penalty``.
    """
    ctx = MarketContext(symbol="XAUUSD", generated_at=datetime(2026, 5, 1, 14, tzinfo=timezone.utc),
                        primary_timeframe=Timeframe.H1)
    if usable:
        ctx.timeframes["H1"] = _tf()
    ctx.current_price = 100.0
    ctx.spread = 0.2
    ctx.scoring = _scoring(confidence)
    ctx.risk = _risk(approved)
    if validation:
        ctx.validation = _validation(bonus=bonus, penalty=penalty, fatal=fatal)
    return ctx


def _envelope(**overrides) -> str:
    verdict = dict(reasoning="Facts align.",
                   strengths=["trend bullish"], risks=["spread unknown"],
                   strongest_reason_against="The M15 has not confirmed yet.",
                   alternative_scenario="Price rejects from resistance.",
                   worst_case="Stop is hit for 1R.",
                   best_case="Target reached for 3R.",
                   execution_plan="Enter at the stated entry.")
    verdict.update(overrides)
    return json.dumps({"structured_output": verdict, "is_error": False})


# --------------------------------------------------------------------------- happy path

def test_returns_the_deterministic_verdict_with_claudes_narrative():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    result = asyncio.run(engine.decide(_context(confidence=70, bonus=5)))

    assert result.source is DecisionSource.CLAUDE
    assert result.verdict is Action.ENTER
    assert result.confidence == 75            # 70 + 5 - 0, computed not quoted
    assert result.reasoning == "Facts align."  # the narrative came from Claude
    assert result.symbol == "XAUUSD"


def test_confidence_is_the_deterministic_score_plus_review():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    result = asyncio.run(engine.decide(_context(confidence=55, bonus=12, penalty=4)))

    assert result.deterministic_score == 55
    assert result.validation_bonus == 12
    assert result.validation_penalty == 4
    assert result.adjusted_score == 63
    assert result.confidence == 63            # the ONE confidence


def test_the_single_clamp_is_applied_once_at_each_end():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    high = asyncio.run(engine.decide(_context(confidence=95, bonus=25, penalty=0)))
    low = asyncio.run(engine.decide(_context(confidence=5, bonus=0, penalty=25)))

    assert high.confidence == 100             # 120 clamped
    assert low.confidence == 0                # -20 clamped


def test_claude_has_no_channel_to_change_the_decision():
    """Structural, not prompt-enforced: the schema has no verdict field and
    no confidence field, so there is nothing to override with."""
    from trade_decision import ClaudeTradeVerdict
    fields = set(ClaudeTradeVerdict.model_fields)
    assert "verdict" not in fields
    assert "confidence" not in fields
    schema = json.loads(engine_schema())
    assert "verdict" not in schema["properties"]
    assert "confidence" not in schema["properties"]


def engine_schema() -> str:
    from trade_decision import _build_schema
    return json.dumps(_build_schema())


# --------------------------------------------------------------- deterministic gates

def test_risk_rejection_caps_the_verdict_at_skip():
    """The risk engine is the single owner of blocking, so its REJECT means
    the trade cannot be placed — SKIP, whatever the score.

    Claude is still consulted, because its job is to explain the finding.
    Only an unusable context skips the model entirely.
    """
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    result = asyncio.run(engine.decide(_context(approved=False, confidence=90)))

    assert result.verdict is Action.SKIP
    assert result.risk_approved is False
    assert result.risk_rejections
    engine._run_cli.assert_awaited_once()


def test_an_objective_blocker_reaches_the_decision_by_exactly_one_route():
    """Regression: news blackout used to be a validation FATAL (-> SKIP) and
    a risk REJECT (-> WAIT) at once, so the outcome depended on which branch
    ran first. Only the risk engine blocks now."""
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    result = asyncio.run(engine.decide(_context(approved=False, confidence=90)))

    assert result.verdict is Action.SKIP
    assert not result.fatal_problems          # validation no longer vetoes
    assert "objective blocker" in result.decision_reason


def test_a_blocked_trade_reports_no_adjusted_score():
    """Nothing about the arithmetic can rescue a trade that cannot be placed,
    so the score is not computed and not shown as if it mattered."""
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    result = asyncio.run(engine.decide(_context(approved=False, confidence=95)))

    assert result.verdict is Action.SKIP
    assert result.adjusted_score is None


def test_unusable_context_skips_without_consulting_claude():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock()

    result = asyncio.run(engine.decide(_context(usable=False)))

    assert result.verdict is Action.SKIP
    engine._run_cli.assert_not_awaited()


# --------------------------------------------------------------------- fallbacks

def test_cli_failure_keeps_the_same_verdict_and_loses_only_the_narrative():
    """The decision is deterministic, so losing Claude costs the explanation,
    not the answer. Deciding something different when the model is
    unreachable would be a hidden override."""
    ctx = _context(confidence=70, bonus=5)
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))
    with_claude = asyncio.run(engine.decide(ctx))

    broken = TradeDecisionEngine(_settings())
    broken._run_cli = AsyncMock(side_effect=TradeDecisionError("boom", retryable=False))
    without = asyncio.run(broken.decide(ctx))

    assert without.source is DecisionSource.FALLBACK
    assert without.verdict is with_claude.verdict is Action.ENTER
    assert without.confidence == with_claude.confidence == 75
    assert "Claude was unavailable" in without.reasoning


def test_malformed_output_falls_back():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, "not json", ""))
    assert asyncio.run(engine.decide(_context())).source is DecisionSource.FALLBACK


def test_schema_violation_falls_back():
    engine = TradeDecisionEngine(_settings())
    # strongest_reason_against is required: the mandatory self-critique.
    broken = json.loads(_envelope())
    del broken["structured_output"]["strongest_reason_against"]
    engine._run_cli = AsyncMock(return_value=(0, json.dumps(broken), ""))
    assert asyncio.run(engine.decide(_context())).source is DecisionSource.FALLBACK


def test_error_envelope_surfaces_the_real_reason():
    engine = TradeDecisionEngine(_settings())
    envelope = json.dumps({"is_error": True, "subtype": "error_max_turns",
                           "errors": ["Reached maximum number of turns (3)"],
                           "permission_denials": [{"tool_name": "StructuredOutput"}]})
    engine._run_cli = AsyncMock(return_value=(1, envelope, ""))

    result = asyncio.run(engine.decide(_context()))

    assert result.source is DecisionSource.FALLBACK
    assert "unknown error" not in result.reasoning


def test_retryable_failure_is_retried_then_succeeds():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(side_effect=[
        TradeDecisionError("transient", retryable=True),
        (0, _envelope(), ""),
    ])
    result = asyncio.run(engine.decide(_context()))
    assert result.source is DecisionSource.CLAUDE
    assert engine._run_cli.await_count == 2


def test_decide_never_raises_on_an_unexpected_exception():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(side_effect=RuntimeError("totally unexpected"))
    result = asyncio.run(engine.decide(_context()))   # must not raise
    assert result.source is DecisionSource.FALLBACK


# --------------------------------------------------------------- CLI arguments

def test_structured_output_tool_is_allowed(monkeypatch):
    """Regression guard: --disallowedTools '*' also denies StructuredOutput
    and breaks every call (see claude_client.py)."""
    engine = TradeDecisionEngine(_settings())
    seen = {}

    async def fake_exec(path, *args, **kwargs):
        seen["args"] = list(args)
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(_envelope().encode(), b""))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    asyncio.run(engine._run_cli("payload"))

    assert "--allowedTools" in seen["args"]
    assert seen["args"][seen["args"].index("--allowedTools") + 1] == "StructuredOutput"
    assert "--disallowedTools" not in seen["args"]


# --------------------------------------------------------------------------- report

def test_text_report_contains_every_section():
    ctx = _context()
    text = report_mod.render_text(ctx)
    for heading in ("MARKET REPORT", "PRICE", "STRUCTURE", "SMART MONEY",
                    "INDICATORS", "LEVELS", "CONTEXT", "SETUP QUALITY",
                    "SCORING & RISK"):
        assert heading in text


def test_text_report_includes_the_decision_when_given():
    engine_result = _decision()
    text = report_mod.render_text(_context(), engine_result)
    assert "DECISION  ENTER" in text
    assert engine_result.reasoning in text


def test_report_shows_dashes_for_missing_values():
    ctx = MarketContext(symbol="X", generated_at=datetime.now(timezone.utc),
                        primary_timeframe=Timeframe.H1)
    text = report_mod.render_text(ctx)
    assert "—" in text          # gaps are visible, not silently omitted


def test_report_lists_data_gaps():
    ctx = _context()
    ctx.warnings.append("H4 candles unavailable: provider timeout")
    assert "DATA GAPS" in report_mod.render_text(ctx)
    assert "H4 candles unavailable" in report_mod.render_text(ctx)


def _decision():
    from trade_decision import TradeDecision
    return TradeDecision(symbol="XAUUSD", verdict=Action.ENTER, confidence=65,
                         reasoning="Facts align with the higher timeframe trend.",
                         strengths=["ADX 30"], risks=["spread unknown"],
                         execution_plan="Enter at 100.", source=DecisionSource.CLAUDE)


def test_telegram_report_escapes_html():
    ctx = _context()
    ctx.symbol = "<script>alert(1)</script>"
    html_out = report_mod.render_telegram(ctx)
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_telegram_report_is_length_capped():
    ctx = _context()
    ctx.warnings.extend(["x" * 500] * 50)
    decision = _decision()
    decision = decision.model_copy(update={"reasoning": "y" * 8000})
    out = report_mod.render_telegram(ctx, decision)
    assert len(out) <= 3800 + len("\n[... truncated]")


def test_telegram_report_shows_risk_rejections():
    ctx = _context(approved=False)
    out = report_mod.render_telegram(ctx)
    assert "REJECTED" in out
    assert "risk_reward" in out


def test_reports_are_deterministic():
    ctx = _context()
    assert report_mod.render_text(ctx) == report_mod.render_text(ctx)
    assert report_mod.render_telegram(ctx) == report_mod.render_telegram(ctx)


# --------------------------------------------------------------------------- facts

def test_facts_never_include_a_candle_series():
    facts = build_facts(_context())
    blob = json.dumps(facts)
    assert '"candles"' not in blob
    assert '"open"' not in blob


# ------------------------------------------------- validation layer (final phase)

def _validation(*, bonus: int = 5, penalty: int = 0, fatal: bool = False):
    from validation_engine import Severity, ValidationCheck, ValidationResult
    checks = (
        ValidationCheck("news", Severity.FATAL if fatal else Severity.STRENGTH,
                        -8.0 if fatal else 8.0, 8.0,
                        "high-impact event imminent" if fatal else "window clear"),
    )
    return ValidationResult(bonus=bonus, penalty=penalty, checks=checks)


def test_self_critique_and_scenarios_are_carried_through():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    result = asyncio.run(engine.decide(_context()))

    assert result.strongest_reason_against == "The M15 has not confirmed yet."
    assert result.alternative_scenario == "Price rejects from resistance."
    assert result.worst_case == "Stop is hit for 1R."
    assert result.best_case == "Target reached for 3R."


def test_the_schema_requires_the_self_critique():
    """Claude cannot approve without arguing the opposing case."""
    from trade_decision import _build_schema
    schema = _build_schema()
    for field in ("strongest_reason_against", "alternative_scenario",
                  "worst_case", "best_case"):
        assert field in schema["properties"]
        assert field in schema["required"]


# ---------------------------------------------------------------- traceability

def test_fallback_records_the_arithmetic_and_fatal_problems():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(side_effect=TradeDecisionError("boom", retryable=False))
    ctx = _context(fatal=True)

    result = asyncio.run(engine.decide(ctx))

    assert result.source is DecisionSource.FALLBACK
    assert result.verdict is Action.SKIP          # the retained fatal path
    assert result.fatal_problems
    assert result.deterministic_score == 70


def test_facts_give_claude_the_review_and_the_decision_to_explain():
    ctx = _context(confidence=70, bonus=9, penalty=3)
    facts = build_facts(ctx)

    assert facts["validation"]["bonus"] == 9
    assert facts["validation"]["penalty"] == 3
    assert "score" not in facts["validation"]        # no second confidence
    for key in ("has_fatal", "fatal_problems", "weaknesses", "strengths",
                "unverified", "summary", "max_adjustment"):
        assert key in facts["validation"]

    decision = facts["decision"]
    assert decision["verdict"] == "enter"
    assert decision["deterministic_score"] == 70
    assert decision["adjusted_score"] == 76
    assert decision["arithmetic"] == "70 + 9 - 3 = 76"
    assert decision["bands"] == {"skip": "0-39", "wait": "40-59", "enter": "60-100"}


def test_report_shows_the_validation_section():
    ctx = _context(fatal=True)
    text = report_mod.render_text(ctx)

    assert "VALIDATION" in text
    assert "FATAL PROBLEM" in text
    assert "high-impact event imminent" in text


def test_report_shows_the_whole_calculation():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))
    ctx = _context(confidence=62, bonus=8, penalty=5)
    decision = asyncio.run(engine.decide(ctx))

    text = report_mod.render_text(ctx, decision)

    assert "Strongest reason NOT to take this trade" in text
    assert "Alternative scenario" in text
    assert "Worst case" in text
    assert "Best case" in text

    # Every term of the one confidence, and the band that used it.
    assert "deterministic score" in text
    assert "validation bonus" in text
    assert "validation penalty" in text
    assert "adjusted score" in text
    assert "Bands" in text
    assert "Risk gate" in text
    for number in ("62", "8", "5", "65"):
        assert number in text


def test_the_report_shows_no_second_confidence():
    """Regression: the report used to print three numbers and a min()."""
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))
    ctx = _context()
    text = report_mod.render_text(ctx, asyncio.run(engine.decide(ctx)))

    assert "min(" not in text
    assert "Validation score" not in text
    assert "Confidence inputs" not in text


# --------------------------------------------------------- decision thresholds

@pytest.mark.parametrize("adjusted,expected", [
    (100, Action.ENTER),
    (61, Action.ENTER),
    (60, Action.ENTER),    # boundary: >= 60 enters
    (59, Action.WAIT),     # boundary: 40-59 waits
    (50, Action.WAIT),
    (40, Action.WAIT),     # boundary: >= 40 waits
    (39, Action.SKIP),     # boundary: < 40 skips
    (10, Action.SKIP),
    (0, Action.SKIP),
])
def test_the_bands_decide(adjusted, expected):
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    result = asyncio.run(engine.decide(_context(confidence=adjusted, bonus=0)))

    assert result.verdict is expected, f"{adjusted} should be {expected.value}"
    assert result.adjusted_score == adjusted


def test_the_review_can_move_a_verdict_across_a_band():
    """A middling conviction with an excellent review reaches ENTER; a strong
    one with a damning review does not."""
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    lifted = asyncio.run(engine.decide(_context(confidence=50, bonus=15, penalty=0)))
    sunk = asyncio.run(engine.decide(_context(confidence=65, bonus=0, penalty=20)))

    assert lifted.verdict is Action.ENTER and lifted.confidence == 65
    assert sunk.verdict is Action.WAIT and sunk.confidence == 45


def test_the_review_can_never_manufacture_a_verdict_on_its_own():
    """Bounded by MAX_ADJUSTMENT, so market conviction stays the base."""
    from validation_engine import MAX_ADJUSTMENT
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    best = asyncio.run(engine.decide(_context(confidence=0, bonus=MAX_ADJUSTMENT)))
    worst = asyncio.run(engine.decide(_context(confidence=100, penalty=MAX_ADJUSTMENT, bonus=0)))

    assert best.verdict is Action.SKIP     # 0 + 25 = 25, still below 40
    assert worst.verdict is Action.ENTER   # 100 - 25 = 75, still at or above 60


@pytest.mark.parametrize("score", [60, 75, 100])
def test_a_risk_rejection_skips_at_every_score(score):
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    approved = asyncio.run(engine.decide(_context(confidence=score, approved=True)))
    rejected = asyncio.run(engine.decide(_context(confidence=score, approved=False)))

    assert approved.verdict is Action.ENTER
    assert rejected.verdict is Action.SKIP


@pytest.mark.parametrize("score", [0, 39, 40, 59, 60, 100])
def test_a_fatal_problem_forces_skip_at_every_score(score):
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    assert asyncio.run(engine.decide(_context(confidence=score, fatal=True))).verdict is Action.SKIP


def test_anything_unmeasured_skips():
    """Unverified is never treated as acceptable."""
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    no_validation = _context(validation=False)
    no_scoring = _context(); no_scoring.scoring = None
    no_risk = _context(); no_risk.risk = None

    for label, ctx in (("validation", no_validation), ("scoring", no_scoring),
                       ("risk", no_risk)):
        result = asyncio.run(engine.decide(ctx))
        assert result.verdict is Action.SKIP, f"missing {label} should skip"


def test_claude_is_consulted_even_when_the_verdict_is_skip():
    """A refusal still gets an explanation in the report."""
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    asyncio.run(engine.decide(_context(confidence=10)))

    engine._run_cli.assert_awaited_once()


def test_thresholds_are_named_constants():
    from trade_decision import ENTER_MIN_SCORE, WAIT_MIN_SCORE
    assert ENTER_MIN_SCORE == 60
    assert WAIT_MIN_SCORE == 40


def test_there_is_exactly_one_confidence_on_the_decision():
    """Regression for the architecture: no field named for a second one."""
    from trade_decision import TradeDecision
    fields = set(TradeDecision.model_fields)
    for gone in ("validation_score", "claude_confidence", "scoring_confidence",
                 "confidence_bound_by", "ceiling", "ceiling_reason"):
        assert gone not in fields, f"{gone} is a second confidence or a ceiling"
    assert "confidence" in fields and "adjusted_score" in fields
