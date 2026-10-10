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


def _entry(location: float = 1.0, extension: float = 1.0):
    """An entry assessment with the given factors. 1.0 / 1.0 is a clean
    entry: favourable location, normal-sized move."""
    from entry_quality import EntryQuality, ExtensionAssessment, LocationAssessment
    loc_label = "favourable" if location == 1.0 else "unfavourable"
    ext_label = "normal" if extension == 1.0 else "strong"
    return EntryQuality(
        location=LocationAssessment(location, loc_label, "lower" if location == 1.0 else "upper",
                                    0.2 if location == 1.0 else 1.0, None, 95.0, 105.0,
                                    f"buying in the {loc_label} part of the range"),
        extension=ExtensionAssessment(extension, ext_label, 0.5, 0.3 if extension == 1.0 else 0.99,
                                      14, 186, f"move is {ext_label}"),
    )


def _context(*, usable: bool = True, approved: bool = True,
             confidence: int = 70, bonus: int = 5, penalty: int = 0,
             validation: bool = True, fatal: bool = False,
             location: float = 1.0, extension: float = 1.0) -> MarketContext:
    """A context that decides ENTER by default.

    ``confidence`` is the deterministic score — the direction base. ``bonus``
    and ``penalty`` are the review's adjustments to it, so the adjusted score
    is ``confidence + bonus - penalty``; ``location`` and ``extension`` are
    the entry factors that scale it into the confluence score. The defaults
    describe a clean entry, so confluence equals the adjusted score.
    """
    ctx = MarketContext(symbol="XAUUSD", generated_at=datetime(2026, 5, 1, 14, tzinfo=timezone.utc),
                        primary_timeframe=Timeframe.H1)
    if usable:
        ctx.timeframes["H1"] = _tf()
    ctx.current_price = 100.0
    ctx.spread = 0.2
    ctx.scoring = _scoring(confidence)
    ctx.risk = _risk(approved)
    ctx.entry = _entry(location, extension)
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
    """Regression: news blackout used to be a validation FATAL and a risk
    REJECT at once, each with a different consequence, so the outcome
    depended on which branch ran first. Only the risk engine blocks now."""
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
    assert decision["bands"] == {"skip": "0-49", "enter": "50-100"}


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
    (60, Action.ENTER),
    (51, Action.ENTER),
    (50, Action.ENTER),    # boundary: >= 50 enters
    (49, Action.SKIP),     # boundary: < 50 skips
    (40, Action.SKIP),     # what used to be WAIT is now SKIP
    (10, Action.SKIP),
    (0, Action.SKIP),
])
def test_the_threshold_decides(adjusted, expected):
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    result = asyncio.run(engine.decide(_context(confidence=adjusted, bonus=0)))

    assert result.verdict is expected, f"{adjusted} should be {expected.value}"
    assert result.adjusted_score == adjusted


def test_there_are_only_two_verdicts():
    """WAIT is gone completely — from the enum, the bands and every score."""
    from trade_decision import decide_deterministically
    assert {a.value for a in Action} == {"enter", "skip"}
    for score in range(0, 101):
        verdict = decide_deterministically(_context(confidence=score, bonus=0)).verdict
        assert verdict is (Action.ENTER if score >= 50 else Action.SKIP), score


def test_the_review_can_move_a_verdict_across_the_threshold():
    """A middling conviction with an excellent review reaches ENTER; a strong
    one with a damning review does not."""
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    lifted = asyncio.run(engine.decide(_context(confidence=40, bonus=15, penalty=0)))
    sunk = asyncio.run(engine.decide(_context(confidence=65, bonus=0, penalty=20)))

    assert lifted.verdict is Action.ENTER and lifted.confidence == 55
    assert sunk.verdict is Action.SKIP and sunk.confidence == 45


def test_the_review_can_never_manufacture_a_verdict_on_its_own():
    """Bounded by MAX_ADJUSTMENT, so market conviction stays the base."""
    from validation_engine import MAX_ADJUSTMENT
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    best = asyncio.run(engine.decide(_context(confidence=0, bonus=MAX_ADJUSTMENT)))
    worst = asyncio.run(engine.decide(_context(confidence=100, penalty=MAX_ADJUSTMENT, bonus=0)))

    assert best.verdict is Action.SKIP     # 0 + 25 = 25, still below 50
    assert worst.verdict is Action.ENTER   # 100 - 25 = 75, still at or above 50


@pytest.mark.parametrize("score", [50, 75, 100])
def test_a_risk_rejection_skips_at_every_score(score):
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))

    approved = asyncio.run(engine.decide(_context(confidence=score, approved=True)))
    rejected = asyncio.run(engine.decide(_context(confidence=score, approved=False)))

    assert approved.verdict is Action.ENTER
    assert rejected.verdict is Action.SKIP


@pytest.mark.parametrize("score", [0, 49, 50, 100])
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


def test_the_threshold_is_a_named_constant():
    import trade_decision
    assert trade_decision.ENTER_MIN_SCORE == 50
    assert not hasattr(trade_decision, "WAIT_MIN_SCORE")


def test_there_is_exactly_one_confidence_on_the_decision():
    """Regression for the architecture: no field named for a second one."""
    from trade_decision import TradeDecision
    fields = set(TradeDecision.model_fields)
    for gone in ("validation_score", "claude_confidence", "scoring_confidence",
                 "confidence_bound_by", "ceiling", "ceiling_reason"):
        assert gone not in fields, f"{gone} is a second confidence or a ceiling"
    assert "confidence" in fields and "adjusted_score" in fields


# ------------------------------------------- confidence follows the TRADE

def _scoring_for(net_score: float, confidence: int = 85) -> ScoringResult:
    direction = (ScoreDirection.BUY if net_score > 0 else
                 ScoreDirection.SELL if net_score < 0 else ScoreDirection.NONE)
    trend = TrendDirection.BULLISH if net_score > 0 else TrendDirection.BEARISH
    return ScoringResult(
        symbol="XAUUSD", timeframe=Timeframe.H1, direction=direction,
        confidence=confidence, net_score=net_score, total_possible=106.0,
        breakdown=(FactorScore("trend", 20.0, 20.0, trend, "t"),), reasons=("t",))


def _context_trading(direction, *, net_score, confidence=85, bonus=5, penalty=0):
    from models import TradeSetup
    ctx = _context(confidence=confidence, bonus=bonus, penalty=penalty)
    ctx.scoring = _scoring_for(net_score, confidence)
    ctx.setup = TradeSetup(symbol="XAUUSD", direction=direction, order_type="limit",
                           entries=[100.0], stop_loss=95.0, take_profits=[115.0])
    return ctx


def test_a_signal_against_the_analysis_is_not_confident():
    """The bug: a strongly BULLISH read (engine 85% confident, BUY) reported
    85 as the base for a SELL signal, giving SELL conf 70 -> ENTER. The system
    confidently recommended the trade against its own analysis."""
    from trade_decision import decide_deterministically
    ctx = _context_trading("short", net_score=+80.0, confidence=85)

    trace = decide_deterministically(ctx)

    assert trace.deterministic_score == 0
    assert trace.verdict is not Action.ENTER


def test_a_signal_with_the_analysis_keeps_its_confidence():
    from trade_decision import decide_deterministically
    trace = decide_deterministically(_context_trading("long", net_score=+80.0, confidence=85))

    assert trace.deterministic_score == 85
    assert trace.verdict is Action.ENTER


@pytest.mark.parametrize("trade,net,expected", [
    ("long",  +80.0, 85),    # engine bullish, trade long  -> with the analysis
    ("short", -80.0, 85),    # engine bearish, trade short -> with the analysis
    ("long",  -80.0, 0),     # engine bearish, trade long  -> against it
    ("short", +80.0, 0),     # engine bullish, trade short -> against it
])
def test_confidence_is_measured_from_the_trades_side(trade, net, expected):
    from trade_decision import directional_confidence
    assert directional_confidence(_scoring_for(net, 85), trade) == expected


def test_buy_and_sell_are_treated_symmetrically():
    from trade_decision import directional_confidence
    for conf in (10, 40, 60, 85, 100):
        assert (directional_confidence(_scoring_for(+50.0, conf), "long")
                == directional_confidence(_scoring_for(-50.0, conf), "short"))
        assert (directional_confidence(_scoring_for(+50.0, conf), "short")
                == directional_confidence(_scoring_for(-50.0, conf), "long") == 0)


def test_the_favoured_side_comes_from_net_score_not_the_reported_direction():
    """``direction`` is forced to NONE below the reporting threshold while
    the evidence still leans one way. The lean must still count."""
    from trade_decision import directional_confidence
    weak_bull = ScoringResult(
        symbol="XAUUSD", timeframe=Timeframe.H1, direction=ScoreDirection.NONE,
        confidence=8, net_score=+3.0, total_possible=106.0, breakdown=(), reasons=())
    assert directional_confidence(weak_bull, "long") == 8
    assert directional_confidence(weak_bull, "short") == 0


def test_a_signal_with_no_direction_uses_the_engines_own_read():
    from trade_decision import directional_confidence
    assert directional_confidence(_scoring_for(+80.0, 85), None) == 85
    assert directional_confidence(_scoring_for(-80.0, 85), None) == 85


def test_a_dead_tie_has_no_confidence_either_way():
    from trade_decision import directional_confidence
    tie = ScoringResult(symbol="XAUUSD", timeframe=Timeframe.H1,
                        direction=ScoreDirection.NONE, confidence=0, net_score=0.0,
                        total_possible=106.0, breakdown=(), reasons=())
    assert directional_confidence(tie, "long") == directional_confidence(tie, "short") == 0


# ------------------------------------------- direction vs entry quality

def test_high_direction_with_a_poor_entry_does_not_enter():
    """Acceptance #7: perfect direction at the top of the range after an
    outsized run is a late entry. 100 x 0.5 x 0.5 = 25 — SKIP."""
    from trade_decision import decide_deterministically
    trace = decide_deterministically(_context(confidence=100, bonus=0, location=0.5, extension=0.5))

    assert trace.adjusted_score == 100
    assert trace.confluence_score == 25
    assert trace.verdict is Action.SKIP
    assert "entry is poor" in trace.reason


@pytest.mark.parametrize("location,extension,expected", [
    (1.0, 1.0, Action.ENTER),     # clean entry: direction carries it
    (0.75, 1.0, Action.ENTER),    # 90 x 0.75 = 68
    (0.75, 0.75, Action.ENTER),   # 90 x 0.5625 = 51
    (0.5, 1.0, Action.SKIP),      # 90 x 0.5 = 45
    (0.5, 0.6, Action.SKIP),      # 90 x 0.3 = 27
])
def test_entry_quality_moves_a_strong_direction_across_the_threshold(location, extension, expected):
    from trade_decision import decide_deterministically
    trace = decide_deterministically(_context(confidence=90, bonus=0,
                                              location=location, extension=extension))
    assert trace.adjusted_score == 90
    assert trace.verdict is expected, trace.reason


def test_a_good_entry_never_raises_the_score():
    from trade_decision import decide_deterministically
    for score in (0, 49, 50, 100):
        trace = decide_deterministically(_context(confidence=score, bonus=0))
        assert trace.confluence_score == trace.adjusted_score


# ------------------------------------------------ SL risk: a warning, never a gate

def _with_stop(ctx, *, stop_atr: float, mae: str = "unknown"):
    """Attach a stop ``stop_atr`` ATR from entry (R:R held at 3) and a
    max-adverse-excursion finding of the given grade."""
    import validation_engine as ve
    from models import TradeSetup
    from trade_quality import assess
    risk = stop_atr * 2.0
    ctx.setup = TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                           entries=[100.0], stop_loss=100.0 - risk,
                           take_profits=[100.0 + 3 * risk])
    ctx.quality = assess(ctx.setup, current_price=100.0, atr_value=2.0)
    finding = {
        "high": ve._check("max_adverse_excursion", ve.Severity.WEAKNESS, "inside the heat", ratio=1.0),
        "medium": ve._check("max_adverse_excursion", ve.Severity.WEAKNESS, "barely clears", ratio=0.5),
        "low": ve._check("max_adverse_excursion", ve.Severity.STRENGTH, "comfortably clears"),
        "unknown": ve._check("max_adverse_excursion", ve.Severity.UNKNOWN, "no history"),
    }[mae]
    ctx.validation = ve.ValidationResult(bonus=ctx.validation.bonus, penalty=ctx.validation.penalty,
                                         checks=ctx.validation.checks + (finding,))
    return ctx


@pytest.mark.parametrize("stop_atr,mae,expected", [
    (2.0, "low", "LOW"),          # good stop that clears the typical heat
    (2.0, "medium", "MEDIUM"),    # barely clears the heat
    (2.0, "high", "HIGH"),        # inside the heat
    (0.7, "low", "MEDIUM"),       # acceptable but tight stop
    (0.3, "low", "HIGH"),         # inside the noise band
    (6.0, "low", "HIGH"),         # unusually wide
    (2.0, "unknown", "LOW"),      # one source is enough
])
def test_sl_risk_grades_the_stop(stop_atr, mae, expected):
    from trade_decision import stop_loss_risk
    sl = stop_loss_risk(_with_stop(_context(), stop_atr=stop_atr, mae=mae))
    assert sl.level.value == expected
    assert sl.reasons


def test_sl_risk_is_unknown_without_a_stop_assessment():
    from trade_decision import stop_loss_risk
    assert stop_loss_risk(_context()).level.value == "UNKNOWN"


@pytest.mark.parametrize("confidence", [30, 49, 50, 80])
def test_sl_risk_never_changes_the_decision(confidence):
    """Same market, same R:R, stops from razor-thin to very wide and from
    LOW to HIGH risk: the verdict and every number behind it are identical."""
    from trade_decision import decide_deterministically, stop_loss_risk
    traces, levels = [], set()
    for stop_atr, mae in ((2.0, "low"), (2.0, "high"), (0.3, "medium"), (6.0, "high")):
        ctx = _with_stop(_context(confidence=confidence, bonus=0), stop_atr=stop_atr, mae=mae)
        traces.append(decide_deterministically(ctx))
        levels.add(stop_loss_risk(ctx).level)
    assert len(levels) > 1                                  # the warning did vary...
    assert len({(t.verdict, t.confluence_score, t.reason) for t in traces}) == 1   # ...the decision did not


def test_the_decision_carries_the_sl_warning():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))
    ctx = _with_stop(_context(confidence=80), stop_atr=0.3, mae="high")
    result = asyncio.run(engine.decide(ctx))

    assert result.verdict is Action.ENTER                   # HIGH SL risk, still ENTER
    assert result.sl_risk == "HIGH"
    assert result.sl_risk_reasons
    facts = build_facts(ctx)
    assert facts["sl_risk"]["level"] == "HIGH"
    assert "did not affect the verdict" in facts["sl_risk"]["note"]


def test_the_card_shows_the_sl_risk_next_to_the_stop():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))
    ctx = _with_stop(_context(confidence=80), stop_atr=0.3, mae="high")
    card = report_mod.render_telegram_report(ctx, asyncio.run(engine.decide(ctx)))

    sl_line = next(line for line in card.splitlines() if "SL:" in line)
    assert "SL risk: HIGH" in sl_line
    assert "SL risk HIGH" in card.split("Summary:")[1]       # with the reason to adjust it
    assert len(card.splitlines()) <= 20
    assert "WAIT" not in card


def test_the_text_report_shows_the_sl_warning_and_two_bands():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))
    ctx = _with_stop(_context(confidence=80), stop_atr=2.0, mae="medium")
    text = report_mod.render_text(ctx, asyncio.run(engine.decide(ctx)))

    assert "SL risk            MEDIUM" in text
    assert "not part of the decision" in text
    assert "SKIP 0-49   ENTER 50-100" in text
    assert "WAIT" not in text


def test_a_missing_entry_assessment_skips():
    """Unmeasured is not acceptable — same rule as scoring, review and risk."""
    from trade_decision import decide_deterministically
    ctx = _context(confidence=95)
    ctx.entry = None
    trace = decide_deterministically(ctx)
    assert trace.verdict is Action.SKIP
    assert "entry-quality" in trace.reason


def test_the_score_is_bounded_and_finite_everywhere():
    import math
    from trade_decision import decide_deterministically
    for score in (0, 20, 50, 80, 100):
        for bonus, penalty in ((0, 0), (25, 0), (0, 80), (25, 80)):
            for loc in (0.5, 0.75, 1.0):
                for ext in (0.5, 0.8, 1.0):
                    trace = decide_deterministically(_context(
                        confidence=score, bonus=bonus, penalty=penalty,
                        location=loc, extension=ext))
                    assert 0 <= trace.confluence_score <= 100
                    assert math.isfinite(trace.confluence_score)


def test_the_decision_carries_the_whole_calculation():
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))
    result = asyncio.run(engine.decide(_context(confidence=80, bonus=4, penalty=2,
                                                location=0.75, extension=0.8)))

    assert result.adjusted_score == 82
    assert result.location_factor == 0.75 and result.extension_factor == 0.8
    assert result.confluence_score == round(82 * 0.75 * 0.8) == result.confidence
    assert result.entry_findings                     # location and extension, in words

    text = report_mod.render_text(_context(confidence=80, bonus=4, penalty=2,
                                           location=0.75, extension=0.8), result)
    assert "x location factor" in text and "x extension factor" in text
    assert "= confluence score" in text
    assert "NOT a win probability" in text


def test_facts_tell_claude_the_score_is_not_a_probability():
    facts = build_facts(_context(location=0.5, extension=1.0))
    assert "not a probability" in facts["decision"]["score_meaning"]
    assert facts["decision"]["confluence_arithmetic"].endswith(
        f"= {facts['decision']['confluence_score']}")
    assert facts["entry_quality"]["location"]["factor"] == 0.5
    from trade_decision import SYSTEM_PROMPT
    assert "NOT a probability" in SYSTEM_PROMPT


def test_the_report_and_telegram_card_show_the_directional_number():
    """The number a trader reads must be confidence in the side shown."""
    import asyncio as _asyncio
    engine = TradeDecisionEngine(_settings())
    engine._run_cli = AsyncMock(return_value=(0, _envelope(), ""))
    ctx = _context_trading("short", net_score=+80.0, confidence=85)

    decision = _asyncio.run(engine.decide(ctx))
    card = report_mod.render_telegram_report(ctx, decision)

    assert decision.deterministic_score == 0
    assert "SELL" in card
    assert "85%" not in card
