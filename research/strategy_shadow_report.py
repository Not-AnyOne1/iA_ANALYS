"""Strategy shadow comparison from the bot's database, read-only.

    python research/strategy_shadow_report.py --db <copy of signals.db>
    python research/strategy_shadow_report.py --db <copy> --json
    python research/strategy_shadow_report.py --db <copy> --include-confirmation

The confirmation cohort (signals on/after STRATEGY_SHADOW_CONFIRMATION_START)
is hidden unless --include-confirmation is given: open it once, after the
hypotheses are fixed. See strategy_shadow/report.py for every definition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy_shadow.report import build, render  # noqa: E402
from strategy_shadow.store import load_readonly  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", required=True)
    p.add_argument("--json", action="store_true")
    p.add_argument("--include-confirmation", action="store_true")
    p.add_argument("--include-duplicates", action="store_true")
    args = p.parse_args(argv)
    report = build(load_readonly(Path(args.db)), include_confirmation=args.include_confirmation,
                   include_duplicates=args.include_duplicates)
    print(json.dumps(report, indent=1) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
