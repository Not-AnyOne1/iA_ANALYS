"""Verify the metals weekly-session boundaries against provider timestamps.

For every week in the raw Twelve Data M15 history, and separately for US
summer (EDT) and winter (EST) weeks, the share of weeks with at least one bar
in each UTC hour around the Friday close and the Sunday reopen. If the
boundaries in ``market_hours`` are right, the hour just inside trading has
bars in almost every week and the hour just inside the closure only in the
weeks that also carry weekend stub quotes.

Usage: python research/session_boundaries.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market_hours import bar_inside_weekly_closure, new_york_offset  # noqa: E402

Bar = namedtuple("Bar", "ts")


def raw_bars(path: Path) -> list:
    """The provider's bar timestamps, sorted and de-duplicated, nothing removed."""
    stamps = {datetime.fromisoformat(r[0]) for r in json.loads(path.read_text())["bars"]}
    return [Bar(ts) for ts in sorted(stamps)]


def main() -> int:
    raw = raw_bars(ROOT / "data" / "research" / "xauusd_15min.json")
    hours = {"Fri": range(18, 24), "Sun": range(19, 24), "Sat": (12,)}
    seen = defaultdict(set)                  # (dst, day, hour) -> ISO weeks with a bar there
    weeks = defaultdict(set)
    for b in raw:
        dst = "summer (EDT)" if new_york_offset(b.ts) == timedelta(hours=-4) else "winter (EST)"
        day = b.ts.strftime("%a")
        week = tuple(b.ts.isocalendar())[:2]
        weeks[dst].add(week)
        if day in hours and b.ts.hour in hours[day]:
            seen[(dst, day, b.ts.hour)].add(week)
    for dst in ("summer (EDT)", "winter (EST)"):
        n = len(weeks[dst])
        print(f"{dst}: {n} weeks — share of weeks with >= 1 M15 bar in each UTC hour "
              f"(* = the rule treats the whole hour as closed)")
        for day, hrs in hours.items():
            cells = []
            for h in hrs:
                ts_probe = next((b.ts for b in raw if b.ts.strftime("%a") == day and b.ts.hour == h
                                 and ("EDT" in dst) == (new_york_offset(b.ts) == timedelta(hours=-4))), None)
                closed = ts_probe is not None and bar_inside_weekly_closure(ts_probe.replace(minute=0),
                                                                            timedelta(hours=1))
                cells.append(f"{h:02d}h {len(seen[(dst, day, h)]) / n:5.0%}{'*' if closed else ' '}")
            print(f"  {day}: " + "  ".join(cells))
    dropped = sum(1 for b in raw if bar_inside_weekly_closure(b.ts, timedelta(minutes=15)))
    print(f"M15 bars the rule drops: {dropped} of {len(raw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
