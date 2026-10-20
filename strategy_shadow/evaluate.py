"""The four strategies' decisions on one snapshot. Pure; never raises.

Each strategy runs in isolation: an exception in one is recorded as that
strategy's error and does not touch the others. The baseline is
production's own ``decide_deterministically`` on the very context the live
decision used, so its verdict must equal the live one; ``baseline_matches``
reports whether it does.

Scores are labelled what they are — a 0-100 heuristic setup score — and a
setup that was blocked before scoring is "Not scored", never 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import strategy_variants

from strategy_shadow.versions import STRATEGIES, STRATEGY_VERSIONS

SCORE_LABEL = "heuristic, not a win probability"


@dataclass(frozen=True)
class StrategyDecision:
    strategy: str
    version: str
    verdict: Optional[str]          # ENTER / WAIT / SKIP; None when the strategy failed
    reason: str = ""
    base_score: Optional[int] = None
    bonus: Optional[int] = None
    penalty: Optional[int] = None
    setup_score: Optional[int] = None   # None = not scored
    blocked: bool = False
    error: Optional[str] = None

    @property
    def score_label(self) -> str:
        if self.error is not None:
            return "Setup score: Not scored — evaluation error"
        if self.setup_score is None:
            return f"Setup score: Not scored — {self.reason}"[:160]
        return f"Setup score: {self.setup_score}/100 ({SCORE_LABEL})"


def _one(context, strategy: str) -> StrategyDecision:
    version = STRATEGY_VERSIONS[strategy]
    try:
        trace = strategy_variants.decide(context, strategy)
    except Exception as exc:  # noqa: BLE001 - one strategy's failure is recorded, not raised
        return StrategyDecision(strategy, version, None, error=f"{type(exc).__name__}: {exc}"[:300])
    return StrategyDecision(
        strategy, version, trace.verdict.value.upper(), trace.reason,
        trace.deterministic_score, trace.validation_bonus, trace.validation_penalty,
        trace.adjusted_score, trace.blocked)


def evaluate(context) -> Dict[str, StrategyDecision]:
    return {strategy: _one(context, strategy) for strategy in STRATEGIES}


def baseline_matches(decisions: Dict[str, StrategyDecision], live_verdict: Optional[str]) -> Optional[bool]:
    """Did the replayed baseline reach the live verdict? None if either is missing."""
    baseline = decisions.get("baseline")
    if baseline is None or baseline.verdict is None or live_verdict is None:
        return None
    return baseline.verdict == live_verdict.upper()
