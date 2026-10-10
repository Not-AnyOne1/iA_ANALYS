"""Which engine made a decision.

Every decision is stamped with ``<code>+p<parameters>``:

* **code** — ``ENGINE_COMMIT`` from the environment when set; else the
  commit Railway built the image from (``RAILWAY_GIT_COMMIT_SHA`` — the image
  has no ``.git``, see .dockerignore); else the git commit of the checkout
  (``-dirty`` when the working tree had uncommitted changes); else ``nogit``;
* **parameters** — a short hash of every value that can change a decision:
  scoring weights and curves, freshness, evidence groups, review weights and
  exclusions, entry-quality constants, risk settings, the ENTER threshold and
  the data-cleaning rule.

Two decisions with the same stamp were made by the same rules. Outcomes are
only ever compared within a stamp, or across stamps on purpose — which is
what makes out-of-sample evaluation honest: a result can always be traced to
the exact engine that produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent


def decision_parameters() -> Dict[str, Any]:
    """Every parameter that can change ENTER/SKIP, read from the live modules.

    Read at call time, not import time, so a changed constant is reflected
    (and tested). Imports are local to keep this module free of import
    cycles — the engines import nothing from here.
    """
    import entry_quality
    import evidence_groups
    import market_context
    import market_data
    import risk_engine
    import scoring_engine
    import trade_decision
    import validation_engine

    engine = scoring_engine.ScoringEngine()
    return {
        "scoring": {
            "weights": engine._weights,
            "freshness_half_life": engine._half_life,
            "min_confidence_for_direction": engine._min_confidence_for_direction,
            "margin_curve": [scoring_engine._MARGIN_W, scoring_engine._MARGIN_A1,
                             scoring_engine._MARGIN_B1, scoring_engine._MARGIN_A2,
                             scoring_engine._MARGIN_B2],
            "participation_curve": [scoring_engine._BASE_B, scoring_engine._BASE_C],
            "groups": [dataclasses.asdict(g) for g in evidence_groups.SCORING_GROUPS],
        },
        "review": {
            "weights": validation_engine._WEIGHTS,
            "max_adjustment": validation_engine.MAX_ADJUSTMENT,
            "max_penalty": validation_engine.MAX_PENALTY,
            "excluded": sorted(validation_engine._CONFIDENCE_EXCLUDED),
            "groups": [dataclasses.asdict(g) for g in evidence_groups.VALIDATION_GROUPS],
        },
        "entry": {
            "extension_window": entry_quality.EXTENSION_WINDOW,
            "extension_lookback": entry_quality.EXTENSION_LOOKBACK,
            "extension_min_samples": entry_quality.EXTENSION_MIN_SAMPLES,
            "moderate_from": entry_quality._MODERATE_FROM,
            "strong_from": entry_quality._STRONG_FROM,
        },
        "risk": dataclasses.asdict(risk_engine.RiskSettings()),
        "decision": {"enter_min_score": trade_decision.ENTER_MIN_SCORE},
        "data": {
            "timeframes": [tf.value for tf in market_context.DEFAULT_TIMEFRAMES],
            "primary": market_context.DEFAULT_PRIMARY.value,
            "candle_limit": market_context.DEFAULT_CANDLE_LIMIT,
            "weekend_stub_fraction": market_data._WEEKEND_STUB_FRACTION,
        },
    }


def parameter_hash() -> str:
    blob = json.dumps(decision_parameters(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


@lru_cache(maxsize=1)
def code_version() -> str:
    """The commit this process runs. Cached: it cannot change under a process."""
    pinned = os.getenv("ENGINE_COMMIT", "").strip()
    if pinned:
        return pinned
    built_from = os.getenv("RAILWAY_GIT_COMMIT_SHA", "").strip()
    if built_from:
        return built_from[:7]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=_ROOT,
                               capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    return f"{commit}-dirty" if dirty else (commit or "nogit")


def engine_version() -> str:
    """``<code>+p<parameters>`` — see the module docstring."""
    return f"{code_version()}+p{parameter_hash()}"
