"""Weekend filter on the provider's real timestamps: DST-transition weekends
and every Friday close / Sunday reopen.

For each US DST transition weekend in the history (second Sunday of March,
first Sunday of November) it prints the raw Twelve Data M15 bars from an hour
before to an hour after the expected Friday close and Sunday reopen, with the
filter's keep/drop decision. Across ALL weekends it checks the two failure
modes that matter: a bar at or after the reopen being dropped (losing real
data), and a bar inside the closure being kept, per timeframe.

Usage: python research/session_transitions.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import market_hours as mh  # noqa: E402

UTC = timezone.utc
DATA = ROOT / "data" / "research"
DUR = {"15min": timedelta(minutes=15), "1h": timedelta(hours=1), "4h": timedelta(hours=4)}


def raw(interval):
    return [(datetime.fromisoformat(r[0]), r) for r in json.loads((DATA / f"xauusd_{interval}.json").read_text())["bars"]]


def boundaries(friday: date):
    """Expected (close, reopen) in UTC for the weekend starting on ``friday``."""
    close = next(t for t in (datetime(friday.year, friday.month, friday.day, h, m, tzinfo=UTC)
                             for h in range(19, 24) for m in (0, 15, 30, 45))
                 if mh.in_weekly_closure(t))
    sunday = friday + timedelta(days=2)
    reopen = next(t for t in (datetime(sunday.year, sunday.month, sunday.day, h, m, tzinfo=UTC)
                              for h in range(19, 24) for m in (0, 15, 30, 45))
                  if not mh.in_weekly_closure(t))
    return close, reopen


def transitions(first_year=2021, last=date(2026, 10, 16)):
    out = []
    for y in range(first_year, last.year + 1):
        for sunday in (mh._nth_sunday(y, 3, 2), mh._nth_sunday(y, 11, 1)):
            if sunday <= last:
                out.append(("DST starts" if sunday.month == 3 else "DST ends", sunday - timedelta(days=2)))
    return out


def main() -> int:
    m15 = raw("15min")
    by_ts = {ts: r for ts, r in m15}
    print("== DST-transition weekends: raw provider M15 bars around each boundary (K = kept, D = dropped)")
    for label, friday in transitions():
        close, reopen = boundaries(friday)
        def window(center):
            t, cells = center - timedelta(hours=1), []
            while t < center + timedelta(hours=1):
                if t in by_ts:
                    cells.append(f"{t:%H:%M}{'D' if mh.bar_inside_weekly_closure(t, DUR['15min']) else 'K'}")
                t += DUR["15min"]
            return " ".join(cells) or "(no bars)"
        print(f" {label} {friday + timedelta(days=2)}: Fri close {close:%H:%M} UTC | {window(close)}")
        print(f" {'':<{len(label) + 11}}  Sun reopen {reopen:%H:%M} UTC | {window(reopen)}")
    print("\n== independent check on stub-free weekends (no provider bar at all on Saturday):")
    print("   the provider's own first Sunday bar and last Friday bar ARE the session edges")
    sat = {ts.date() for ts, _ in m15 if ts.weekday() == 5}
    fridays = sorted({ts.date() for ts, _ in m15 if ts.weekday() == 4})
    stats = {}
    for friday in fridays:
        sunday = friday + timedelta(days=2)
        if friday + timedelta(days=1) in sat:
            continue                                      # stub weekend: edges not observable
        close, reopen = boundaries(friday)
        state = "summer" if mh.new_york_offset(close) == timedelta(hours=-4) else "winter"
        s = stats.setdefault(state, {"weekends": 0, "reopen_match": 0, "reopen_other": [],
                                     "close_match": 0, "close_early": 0, "close_late": []})
        s["weekends"] += 1
        sun_bars = [ts for ts, _ in m15 if ts.date() == sunday]
        fri_bars = [ts for ts, _ in m15 if ts.date() == friday]
        if sun_bars:
            if sun_bars[0] == reopen:
                s["reopen_match"] += 1
            else:
                s["reopen_other"].append(f"{sunday}:{sun_bars[0]:%H:%M}")
        if fri_bars:
            end = fri_bars[-1] + DUR["15min"]
            if end == close:
                s["close_match"] += 1
            elif end < close:
                s["close_early"] += 1
            else:
                s["close_late"].append(f"{friday}:{end:%H:%M}")
    for state, s in stats.items():
        print(f" {state}: {s['weekends']} stub-free weekends | first Sunday bar AT the expected reopen: "
              f"{s['reopen_match']}, elsewhere: {s['reopen_other'] or 'none'} | Friday data ends AT the "
              f"expected close: {s['close_match']}, earlier (early close / short feed): {s['close_early']}, "
              f"LATER (would be real data dropped): {s['close_late'] or 'none'}")
    for interval, dur in DUR.items():
        bars = raw(interval)
        print(f" {interval:>5}: provider bars {len(bars)}, dropped as wholly inside the closure "
              f"{sum(1 for ts, _ in bars if mh.bar_inside_weekly_closure(ts, dur))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
