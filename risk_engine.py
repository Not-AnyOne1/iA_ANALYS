"""Deterministic pre-trade risk gate.

Pure rules over already-computed facts. No I/O, no model involvement —
this runs *before* Claude and can reject a setup outright, so Claude is
never asked to reason about a trade that fails a hard risk rule.

Every rule is explicit, individually reported, and configurable. Each
returns a :class:`RiskCheck` so the report can show exactly which rule
fired and why, rather than a single opaque "rejected".

An `unknown` input never rejects. If ATR is unavailable, the volatility
rule abstains rather than assuming the worst — the risk engine's job is to
reject *known* bad setups, not to reject everything it can't measure. Those
abstentions are surfaced so the reader knows the check didn't run.

Most rules report rather than gate. ``trend_alignment``, ``volatility``,
``confidence``, ``stop_quality``, ``target_quality`` and ``spread`` return
:attr:`RiskVerdict.WARN` on a bad finding: they are still evaluated, still
reported with their full diagnostics, and still weighed by
``validation_engine`` (which scores the same facts at the full weight of its
own checks), but they no longer block the analysis or force SKIP on their
own. Each is a judgement or a matter of degree — a counter-trend entry is a
legitimate style, low volatility can change within the life of the trade, a
stop or target placement is an opinion about levels, a wide spread is a cost
— rather than a fact about whether the trade can be placed at all.

Two rules still gate, and they are the objective blockers: ``risk_reward``
(a payoff too thin to survive costs) and ``news`` (a scheduled release that
makes price unmodellable). Only ``REJECT`` sets ``approved = False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from news_filter import NewsStatus
from structure_engine import TrendDirection
from trade_quality import Quality, TradeQuality


class RiskVerdict(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    WARN = "warn"         # the rule fired, but the finding does not block
    ABSTAIN = "abstain"   # the input needed for this rule wasn't available


@dataclass(frozen=True)
class RiskCheck:
    """One rule's outcome."""

    name: str
    verdict: RiskVerdict
    detail: str

    @property
    def rejected(self) -> bool:
        return self.verdict is RiskVerdict.REJECT

    @property
    def warned(self) -> bool:
        return self.verdict is RiskVerdict.WARN


@dataclass(frozen=True)
class RiskSettings:
    """Thresholds, all overridable per deployment."""

    # Objective blocker: a payoff this thin cannot survive costs, so the
    # trade is not placeable rather than merely unattractive. Anything at or
    # above it is a preference — the validation engine still scores R:R
    # below 1:2 as a weakness (unchanged).
    min_risk_reward: float = 1.10         # reject strictly below 1.10
    max_spread_atr: float = 0.25          # spread wider than 1/4 ATR is too much
    min_confidence: int = 40              # scoring-engine confidence, 0-100
    reject_counter_trend: bool = True
    reject_low_volatility: bool = True
    reject_on_news: bool = True


@dataclass(frozen=True)
class RiskAssessment:
    """The full gate result."""

    approved: bool
    checks: Tuple[RiskCheck, ...]

    @property
    def rejections(self) -> Tuple[RiskCheck, ...]:
        return tuple(c for c in self.checks if c.rejected)

    @property
    def warnings(self) -> Tuple[RiskCheck, ...]:
        return tuple(c for c in self.checks if c.warned)

    @property
    def abstentions(self) -> Tuple[RiskCheck, ...]:
        return tuple(c for c in self.checks if c.verdict is RiskVerdict.ABSTAIN)

    @property
    def summary(self) -> str:
        if self.approved:
            notes = []
            if self.warnings:
                notes.append(f"{len(self.warnings)} warning(s)")
            if self.abstentions:
                notes.append(f"{len(self.abstentions)} check(s) had no data")
            return "approved" + (f" ({', '.join(notes)})" if notes else "")
        return "; ".join(f"{c.name}: {c.detail}" for c in self.rejections)


def _check_risk_reward(quality: TradeQuality, settings: RiskSettings) -> RiskCheck:
    rr = quality.risk_reward
    if rr is None:
        return RiskCheck("risk_reward", RiskVerdict.ABSTAIN,
                         "entry, stop or target missing — R:R not computable")
    if rr < settings.min_risk_reward:
        return RiskCheck("risk_reward", RiskVerdict.REJECT,
                         f"R:R {rr:.2f} below the {settings.min_risk_reward:.2f} minimum")
    return RiskCheck("risk_reward", RiskVerdict.PASS, f"R:R {rr:.2f}")


def _check_stop(quality: TradeQuality) -> RiskCheck:
    stop = quality.stop
    if stop.quality is Quality.UNKNOWN:
        return RiskCheck("stop_quality", RiskVerdict.ABSTAIN, "; ".join(stop.reasons))
    if stop.quality is Quality.POOR:
        # WARN, not REJECT: a poorly placed stop is a judgement about where
        # price is likely to go, not a fact about whether the trade can be
        # placed. Reported and weighed; the validation engine scores the same
        # facts. Calculations, messages and diagnostics are unchanged.
        return RiskCheck("stop_quality", RiskVerdict.WARN, "; ".join(stop.reasons))
    return RiskCheck("stop_quality", RiskVerdict.PASS, "; ".join(stop.reasons))


def _check_target(quality: TradeQuality) -> RiskCheck:
    target = quality.target
    if target.quality is Quality.UNKNOWN:
        return RiskCheck("target_quality", RiskVerdict.ABSTAIN, "; ".join(target.reasons))
    if target.quality is Quality.POOR:
        # WARN, not REJECT — same reasoning as the stop check above.
        return RiskCheck("target_quality", RiskVerdict.WARN, "; ".join(target.reasons))
    return RiskCheck("target_quality", RiskVerdict.PASS, "; ".join(target.reasons))


def _check_spread(
    spread: Optional[float], atr_value: Optional[float], settings: RiskSettings
) -> RiskCheck:
    if spread is None or atr_value is None or atr_value == 0:
        return RiskCheck("spread", RiskVerdict.ABSTAIN, "spread or ATR unavailable")
    ratio = spread / atr_value
    if ratio > settings.max_spread_atr:
        # WARN, not REJECT: a wide spread is a cost, and cost is a matter of
        # degree. The ratio, the threshold and the message are unchanged —
        # only the verdict, so it no longer sets approved = False.
        return RiskCheck("spread", RiskVerdict.WARN,
                         f"spread is {ratio:.2f} ATR (max {settings.max_spread_atr:.2f})")
    return RiskCheck("spread", RiskVerdict.PASS, f"spread {ratio:.2f} ATR")


def _check_trend(
    direction: Optional[str], trend: Optional[TrendDirection], settings: RiskSettings
) -> RiskCheck:
    if not settings.reject_counter_trend:
        return RiskCheck("trend_alignment", RiskVerdict.PASS, "counter-trend check disabled")
    if direction is None or trend is None:
        return RiskCheck("trend_alignment", RiskVerdict.ABSTAIN, "direction or trend unknown")
    if trend in (TrendDirection.RANGING, TrendDirection.UNKNOWN):
        # Not counter-trend if there is no trend to be counter to.
        return RiskCheck("trend_alignment", RiskVerdict.ABSTAIN,
                         f"structure trend is {trend.value}")
    aligned = (
        (direction == "long" and trend is TrendDirection.BULLISH)
        or (direction == "short" and trend is TrendDirection.BEARISH)
    )
    if not aligned:
        # WARN, not REJECT: counter-trend entries are a legitimate style
        # (reversals, mean reversion), so this is reported and weighed but
        # does not gate the analysis. The Validation Engine scores the same
        # fact at the full weight of its ``higher_timeframe_trend`` check.
        return RiskCheck("trend_alignment", RiskVerdict.WARN,
                         f"{direction} against a {trend.value} structure trend")
    return RiskCheck("trend_alignment", RiskVerdict.PASS,
                     f"{direction} aligns with {trend.value} trend")


def _check_volatility(volatility: Optional[str], settings: RiskSettings) -> RiskCheck:
    if not settings.reject_low_volatility:
        return RiskCheck("volatility", RiskVerdict.PASS, "volatility check disabled")
    if volatility in (None, "unknown"):
        return RiskCheck("volatility", RiskVerdict.ABSTAIN, "volatility unavailable")
    if volatility == "low":
        # WARN, not REJECT: low volatility is a market condition that can
        # change within the life of the trade, so it is reported and weighed
        # but does not gate the analysis. The Validation Engine scores the
        # same fact at the full weight of its ``volatility`` check.
        return RiskCheck("volatility", RiskVerdict.WARN,
                         "volatility is low — moves may not reach the target")
    return RiskCheck("volatility", RiskVerdict.PASS, f"volatility {volatility}")


def _check_confidence(confidence: Optional[int], settings: RiskSettings) -> RiskCheck:
    if confidence is None:
        return RiskCheck("confidence", RiskVerdict.ABSTAIN, "no confidence score")
    if confidence < settings.min_confidence:
        # WARN, not REJECT. A low deterministic score is a market *opinion* —
        # "the evidence does not strongly favour this" — not an objective
        # blocker, and this gate duplicated the decision bands in
        # ``trade_decision.py``, which already send a low score to WAIT or
        # SKIP. Rejecting here as well made the same fact veto twice.
        return RiskCheck("confidence", RiskVerdict.WARN,
                         f"confidence {confidence} below the {settings.min_confidence} minimum")
    return RiskCheck("confidence", RiskVerdict.PASS, f"confidence {confidence}")


def _check_news(news: Optional[NewsStatus], settings: RiskSettings) -> RiskCheck:
    if not settings.reject_on_news:
        return RiskCheck("news", RiskVerdict.PASS, "news check disabled")
    if news is None or not news.available:
        return RiskCheck("news", RiskVerdict.ABSTAIN,
                         news.reason if news else "no news filter configured")
    if news.blocked:
        return RiskCheck("news", RiskVerdict.REJECT, news.reason)
    return RiskCheck("news", RiskVerdict.PASS, news.reason)


def evaluate(
    quality: TradeQuality,
    *,
    direction: Optional[str] = None,
    trend: Optional[TrendDirection] = None,
    atr_value: Optional[float] = None,
    spread: Optional[float] = None,
    volatility: Optional[str] = None,
    confidence: Optional[int] = None,
    news: Optional[NewsStatus] = None,
    settings: Optional[RiskSettings] = None,
) -> RiskAssessment:
    """Run every rule. Approved only when nothing rejected."""
    cfg = settings or RiskSettings()
    checks: List[RiskCheck] = [
        _check_risk_reward(quality, cfg),
        _check_stop(quality),
        _check_target(quality),
        _check_spread(spread, atr_value, cfg),
        _check_trend(direction, trend, cfg),
        _check_volatility(volatility, cfg),
        _check_confidence(confidence, cfg),
        _check_news(news, cfg),
    ]
    return RiskAssessment(
        approved=not any(c.rejected for c in checks),
        checks=tuple(checks),
    )
