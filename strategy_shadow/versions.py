"""Which code made each decision.

Every recorded decision carries three identifiers:

* ``strategy_version`` — the definition's name and revision; bumped only if
  a strategy's rules are deliberately changed (they must not be during an
  evaluation);
* ``rules_hash`` — a hash of the source files the decisions depend on, so a
  silent change of any rule is visible in the data;
* ``code_version`` — the deployed commit (``RAILWAY_GIT_COMMIT_SHA`` on
  Railway, else ``git rev-parse`` locally, else "unknown").
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STRATEGIES = ("baseline", "htf_regime", "valid_events", "range_filter")
STRATEGY_VERSIONS = {
    "baseline": "baseline/1",          # trade_decision.decide_deterministically as on main 197b2a0
    "htf_regime": "htf_regime/1",      # strategy_variants.py as tested in PS_results.md
    "valid_events": "valid_events/1",
    "range_filter": "range_filter/1",
}
ACCOUNTING_VERSION = "p1-trade_accounting/1"

_RULE_FILES = ("trade_decision.py", "scoring_engine.py", "validation_engine.py", "risk_engine.py",
               "structure_engine.py", "smc_engine.py", "market_context.py", "indicators.py",
               "trade_quality.py", "levels.py", "sessions.py", "strategy_variants.py")


@lru_cache(maxsize=1)
def rules_hash() -> str:
    digest = hashlib.sha256()
    for name in _RULE_FILES:
        path = ROOT / name
        data = path.read_bytes().replace(b"\r\n", b"\n") if path.exists() else b"<missing>"
        digest.update(name.encode() + b"\0" + data + b"\0")
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def code_version() -> str:
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "").strip()
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"
