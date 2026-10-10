"""Champion/challenger shadow mode.

The **champion** is the live strategy: ``trade_decision.decide_deterministically``.
Its verdict is the only one ever sent. **Challengers** are candidate
strategies that run on exactly the same signal and the same market context,
AFTER the champion has decided; their verdicts are only recorded.

Because every variant judges the same signals, a challenger can later be
compared with the champion trade for trade — paired, out-of-sample, with
the same fills and costs — instead of on a different sample.

Rules that keep it honest:

* a challenger receives the finished context and may only READ it; it can
  never change the champion's decision, the card, or the context;
* a challenger that raises is recorded as having failed, and nothing else is
  affected;
* every recorded verdict carries the variant's engine version, so results
  are only ever compared across known rule sets.

P1 ships the mechanism with no challengers registered. Candidates (P3) are
added to :data:`CHALLENGERS` with a written hypothesis; promoting one to
champion is a deliberate decision made on out-of-sample evidence, never
automatic.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from engine_version import engine_version
from trade_decision import DecisionTrace, decide_deterministically

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    name: str
    hypothesis: str                                  # what it is expected to improve, and why
    decide: Callable[[Any], DecisionTrace]           # reads a MarketContext, returns a trace
    parameters: Dict[str, Any] = field(default_factory=dict)

    def version(self) -> str:
        """The champion's engine version plus this variant's own parameters."""
        if not self.parameters:
            return engine_version()
        blob = json.dumps({"name": self.name, **self.parameters}, sort_keys=True, default=str)
        return f"{engine_version()}+v{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


CHAMPION = Variant(
    name="champion",
    hypothesis="the live strategy",
    decide=decide_deterministically,
)

# Challengers registered for shadow evaluation. Empty in P1.
CHALLENGERS: Sequence[Variant] = ()


def _record(variant: Variant, trace: DecisionTrace, recorded_at: str) -> Dict[str, Any]:
    return {
        "variant": variant.name,
        "engine_version": variant.version(),
        "recorded_at": recorded_at,
        "verdict": trace.verdict.value,
        "confluence_score": trace.confluence_score,
        "adjusted_score": trace.adjusted_score,
        "reason": trace.reason,
        "details_json": json.dumps({
            "deterministic_score": trace.deterministic_score,
            "validation_bonus": trace.validation_bonus,
            "validation_penalty": trace.validation_penalty,
            "location_factor": trace.location_factor,
            "room_factor": trace.room_factor,
            "extension_factor": trace.extension_factor,
            "blocked": trace.blocked,
        }),
    }


def shadow_decisions(context: Any, champion: DecisionTrace,
                     challengers: Sequence[Variant] = None) -> List[Dict[str, Any]]:
    """Records for the champion's decision and every challenger's.

    ``champion`` is the trace the live decision was made from; it is
    recorded as-is, never recomputed. Challengers are evaluated here, each
    in isolation. Never raises.
    """
    challengers = CHALLENGERS if challengers is None else challengers
    recorded_at = context.generated_at.isoformat()
    out = [_record(CHAMPION, champion, recorded_at)]
    for variant in challengers:
        try:
            out.append(_record(variant, variant.decide(context), recorded_at))
        except Exception as exc:  # noqa: BLE001 - a challenger can never break the pipeline
            log.exception("challenger_failed variant=%s", variant.name)
            out.append({"variant": variant.name, "engine_version": variant.version(),
                        "recorded_at": recorded_at, "verdict": "error",
                        "reason": f"{type(exc).__name__}: {exc}"})
    return out
