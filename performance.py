"""Performance of the bot's decisions, in R — gross and net, never mixed.

Reads the tracked signals (``signal_outcomes``), their realised trade
accounting (``trade_accounting``) and every variant's verdict
(``shadow_decisions``), and reports, for a chosen variant:

* ENTER trades — what following the bot would have made;
* SKIP trades — what it declined: missed winners and avoided losers;
* every signal — what taking everything the channel sent would have made;

each with: count, win rate, average win and loss, expectancy (R per trade)
with a bootstrap interval, profit factor, total R, maximum drawdown and the
longest losing streak (both in R, in exit order), and median holding time —
and the same for ENTER broken down by direction, confluence band, session
and engine version.

Honesty rules built in:

* **Gross and net are separate reports.** A trade with no cost figure is
  left out of the net report and counted as excluded; gross is never shown
  as net.
* **Development data is labelled.** Everything recorded before
  :data:`DEV_CUTOFF` was seen while the current rules were designed. It is
  debugging data, never evidence, and is reported only when asked for.
* **Duplicates count once** by default (repeats of the same setup within six
  hours), and small samples are flagged.

Usage::

    python performance.py                         # net, champion, out-of-sample
    python performance.py --basis gross --period all --format json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from costs import DEFAULT_ASSUMPTIONS
from sessions import session_info

# The last signal the closed-candle redesign was built and checked on.
# Signals up to here were seen while the current rules were designed:
# development data only. (The redesigned engine itself was first committed
# after this, so decisions in the live database around here came from older
# code — the engine-version breakdown keeps them apart.)
DEV_CUTOFF = datetime(2026, 10, 8, 18, 36, tzinfo=timezone.utc)

DUPLICATE_WINDOW = timedelta(hours=6)
SMALL_SAMPLE = 30           # below this many resolved trades, say so
_BOOTSTRAP_SAMPLES = 2000
_BOOTSTRAP_SEED = 20261010

CONFLUENCE_BANDS = ((0, 49, "0-49"), (50, 59, "50-59"), (60, 69, "60-69"), (70, 100, "70-100"))


# ------------------------------------------------------------------- trades

@dataclass(frozen=True)
class Trade:
    id: int
    recorded_at: datetime
    direction: Optional[str]
    verdict: str                     # this variant's verdict: enter / skip / error
    confluence: Optional[int]
    engine_version: Optional[str]
    duplicate: bool
    status: str                      # accounting status for the basis, or "pending"
    r: Optional[float]
    exit_at: Optional[datetime]
    holding_minutes: Optional[float]
    session: str
    cost_source: Optional[str]
    has_costs: bool                  # a cost figure exists, so a net result can exist

    @property
    def resolved(self) -> bool:
        return self.r is not None


def duplicate_ids(rows: Sequence[Dict[str, Any]]) -> set:
    """Signals that repeat an earlier one — the same rule ``Storage.find_duplicate``
    applies at recording time: same symbol, direction, stop and first target,
    entry within 0.1, within six hours. Used for rows recorded before
    tagging existed; a stored ``duplicate_of`` always wins."""
    seen: List[Tuple[Dict[str, Any], datetime]] = []
    dups = set()
    for row in sorted(rows, key=lambda r: r["id"]):
        if row.get("duplicate_of") is not None:
            dups.add(row["id"])
            seen.append((row, datetime.fromisoformat(row["recorded_at"])))
            continue
        at = datetime.fromisoformat(row["recorded_at"])
        for earlier, earlier_at in seen:
            if (row.get("entry") is not None and earlier.get("entry") is not None
                    and earlier["symbol"] == row["symbol"]
                    and earlier["direction"] == row["direction"]
                    and earlier.get("stop_loss") == row.get("stop_loss")
                    and earlier.get("take_profit") == row.get("take_profit")
                    and abs(earlier["entry"] - row["entry"]) <= 0.1
                    and timedelta(0) <= at - earlier_at <= DUPLICATE_WINDOW):
                dups.add(row["id"])
                break
        seen.append((row, at))
    return dups


def trades_from_rows(rows: Sequence[Dict[str, Any]], *, basis: str,
                     variant: str = "champion",
                     assumed_symbols: frozenset = frozenset(DEFAULT_ASSUMPTIONS)) -> List[Trade]:
    """One Trade per signal the variant judged, with the accounting for ``basis``.

    ``assumed_symbols`` are the symbols with a cost assumption: a signal
    recorded before costs were has a net result only for those.
    """
    if basis not in ("gross", "net"):
        raise ValueError("basis must be 'gross' or 'net'")
    dups = duplicate_ids(rows)
    out = []
    for row in rows:
        if variant == "champion":
            verdict, confluence = row.get("decision"), row.get("confluence_score")
        else:
            shadow = (row.get("variants") or {}).get(variant)
            if shadow is None:
                continue
            verdict, confluence = shadow["verdict"], shadow.get("confluence_score")
        if verdict is None:
            continue
        acct = (row.get("accounting") or {}).get(basis)
        at = datetime.fromisoformat(row["recorded_at"])
        exit_at = acct.get("exit_at") if acct else None
        out.append(Trade(
            id=row["id"], recorded_at=at, direction=row.get("direction"),
            verdict=verdict, confluence=confluence, engine_version=row.get("engine_version"),
            duplicate=row["id"] in dups,
            status=acct["status"] if acct else "pending",
            r=acct.get("r") if acct else None,
            exit_at=datetime.fromisoformat(exit_at) if exit_at else None,
            holding_minutes=acct.get("holding_minutes") if acct else None,
            session=session_info(at).label,
            cost_source=acct.get("cost_source") if acct else None,
            has_costs=row.get("cost_spread") is not None or row.get("symbol") in assumed_symbols,
        ))
    return out


# ------------------------------------------------------------------ metrics

@dataclass(frozen=True)
class Summary:
    basis: str
    signals: int
    resolved: int
    wins: int
    losses: int
    timeouts: int
    not_filled: int
    pending: int
    win_rate: Optional[float]
    avg_win: Optional[float]
    avg_loss: Optional[float]
    expectancy: Optional[float]
    expectancy_interval: Optional[Tuple[float, float]]
    profit_factor: Optional[float]           # None when there is no losing trade
    total_r: float
    max_drawdown: float
    longest_losing_streak: int
    median_holding_minutes: Optional[float]
    small_sample: bool


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _interval(rs: List[float]) -> Optional[Tuple[float, float]]:
    """90% bootstrap interval of the mean. Seeded: the same data always
    gives the same interval."""
    if len(rs) < 5:
        return None
    rng = random.Random(_BOOTSTRAP_SEED)
    means = sorted(sum(rng.choice(rs) for _ in rs) / len(rs) for _ in range(_BOOTSTRAP_SAMPLES))
    return (round(means[int(0.05 * len(means))], 3), round(means[int(0.95 * len(means)) - 1], 3))


def summarize(trades: Sequence[Trade], basis: str) -> Summary:
    resolved = sorted((t for t in trades if t.resolved),
                      key=lambda t: (t.exit_at or t.recorded_at, t.id))
    rs = [t.r for t in resolved]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]

    equity = peak = drawdown = 0.0
    streak = longest = 0
    for r in rs:
        equity += r
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
        streak = streak + 1 if r < 0 else 0
        longest = max(longest, streak)

    return Summary(
        basis=basis,
        signals=len(trades),
        resolved=len(rs),
        wins=len(wins),
        losses=len(losses),
        timeouts=sum(1 for t in resolved if t.status == "timeout"),
        not_filled=sum(1 for t in trades if t.status == "not_filled"),
        pending=sum(1 for t in trades if t.status == "pending"),
        win_rate=round(len(wins) / len(rs), 4) if rs else None,
        avg_win=round(sum(wins) / len(wins), 4) if wins else None,
        avg_loss=round(sum(losses) / len(losses), 4) if losses else None,
        expectancy=round(sum(rs) / len(rs), 4) if rs else None,
        expectancy_interval=_interval(rs),
        profit_factor=round(sum(wins) / -sum(losses), 3) if losses else None,
        total_r=round(sum(rs), 4),
        max_drawdown=round(drawdown, 4),
        longest_losing_streak=longest,
        median_holding_minutes=_median([t.holding_minutes for t in resolved
                                        if t.holding_minutes is not None]),
        small_sample=len(rs) < SMALL_SAMPLE,
    )


def _band(confluence: Optional[int]) -> str:
    if confluence is None:
        return "no score"
    for low, high, label in CONFLUENCE_BANDS:
        if low <= confluence <= high:
            return label
    return "no score"


# ------------------------------------------------------------------- report

@dataclass
class Report:
    basis: str
    variant: str
    period: str
    deduplicated: bool
    cost_sources: Dict[str, int]
    excluded_no_costs: int
    enter: Summary
    skip: Summary
    all_signals: Summary
    missed_winners: Dict[str, float]
    avoided_losers: Dict[str, float]
    by_direction: Dict[str, Summary] = field(default_factory=dict)
    by_confluence: Dict[str, Summary] = field(default_factory=dict)
    by_session: Dict[str, Summary] = field(default_factory=dict)
    by_version: Dict[str, Summary] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _group(trades: Sequence[Trade], key, basis: str) -> Dict[str, Summary]:
    groups: Dict[str, List[Trade]] = {}
    for t in trades:
        groups.setdefault(key(t), []).append(t)
    return {k: summarize(v, basis) for k, v in sorted(groups.items())}


def build_report(rows: Sequence[Dict[str, Any]], *, basis: str = "net",
                 variant: str = "champion", period: str = "oos",
                 deduplicate: bool = True) -> Report:
    if period not in ("oos", "dev", "all"):
        raise ValueError("period must be 'oos', 'dev' or 'all'")
    trades = trades_from_rows(rows, basis=basis, variant=variant)
    if period == "oos":
        trades = [t for t in trades if t.recorded_at >= DEV_CUTOFF]
    elif period == "dev":
        trades = [t for t in trades if t.recorded_at < DEV_CUTOFF]
    if deduplicate:
        trades = [t for t in trades if not t.duplicate]

    excluded = 0
    if basis == "net":
        # A trade with no cost figure has no net result: out, and counted.
        excluded = sum(1 for t in trades if not t.has_costs)
        trades = [t for t in trades if t.has_costs]
    enter = [t for t in trades if t.verdict == "enter"]
    skip = [t for t in trades if t.verdict == "skip"]

    cost_sources: Dict[str, int] = {}
    for t in trades:
        if t.cost_source:
            cost_sources[t.cost_source] = cost_sources.get(t.cost_source, 0) + 1

    report = Report(
        basis=basis, variant=variant, period=period, deduplicated=deduplicate,
        cost_sources=cost_sources, excluded_no_costs=excluded,
        enter=summarize(enter, basis), skip=summarize(skip, basis),
        all_signals=summarize(trades, basis),
        missed_winners={"count": sum(1 for t in skip if t.r is not None and t.r > 0),
                        "total_r": round(sum(t.r for t in skip if t.r is not None and t.r > 0), 4)},
        avoided_losers={"count": sum(1 for t in skip if t.r is not None and t.r < 0),
                        "total_r": round(sum(t.r for t in skip if t.r is not None and t.r < 0), 4)},
        by_direction=_group(enter, lambda t: t.direction or "unknown", basis),
        by_confluence=_group(trades, lambda t: _band(t.confluence), basis),
        by_session=_group(enter, lambda t: t.session, basis),
        by_version=_group(enter, lambda t: t.engine_version or "unversioned (before P1)", basis),
    )
    if period != "oos":
        report.notes.append("Includes development data recorded before "
                            f"{DEV_CUTOFF:%Y-%m-%d %H:%M} UTC — debugging only, not evidence.")
    if report.enter.small_sample:
        report.notes.append(f"Only {report.enter.resolved} resolved ENTER trades — far too few "
                            "to judge performance.")
    if basis == "net" and any("assumed" in s for s in cost_sources):
        report.notes.append("Net figures use ASSUMED spread/slippage, not measured bid/ask.")
    return report


def report_json(report: Report) -> Dict[str, Any]:
    return asdict(report)


# ------------------------------------------------------------------ render

def _fmt(value: Optional[float], digits: int = 2, signed: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _row(label: str, s: Summary) -> str:
    interval = (f"[{s.expectancy_interval[0]:+.2f}, {s.expectancy_interval[1]:+.2f}]"
                if s.expectancy_interval else "—")
    return (f"| {label} | {s.resolved} | {_fmt(s.win_rate * 100 if s.win_rate is not None else None, 0, False)}% "
            f"| {_fmt(s.avg_win)} | {_fmt(s.avg_loss)} | {_fmt(s.expectancy)} | {interval} "
            f"| {_fmt(s.profit_factor, 2, False)} | {_fmt(s.total_r)} | {_fmt(s.max_drawdown, 2, False)} "
            f"| {s.longest_losing_streak} | {_fmt(s.median_holding_minutes, 0, False)} "
            f"| {s.timeouts}/{s.not_filled}/{s.pending} |{' small' if s.small_sample else ''}")


_HEADER = ("| | resolved | win rate | avg win R | avg loss R | expectancy R | 90% interval "
           "| profit factor | total R | max DD R | losing streak | median hold (min) "
           "| timeout/unfilled/pending | |\n|" + "---|" * 14)


def render_markdown(report: Report) -> str:
    title = {"oos": "out-of-sample", "dev": "DEVELOPMENT DATA (debugging only)",
             "all": "all data (includes development data)"}[report.period]
    lines = [
        f"# Performance — {report.variant}, {report.basis.upper()}, {title}",
        "",
        f"Basis: **{report.basis}**" + (
            " (after costs: " + ", ".join(f"{k}: {v}" for k, v in report.cost_sources.items()) + ")"
            if report.basis == "net" else " (before any cost — NOT a net result)"),
        f"Duplicates {'counted once' if report.deduplicated else 'included'}."
        + (f" {report.excluded_no_costs} signal(s) excluded: no cost figure."
           if report.excluded_no_costs else ""),
        "",
    ]
    lines += [f"> {n}" for n in report.notes] + ([""] if report.notes else [])
    lines += [_HEADER, _row("ENTER", report.enter), _row("SKIP", report.skip),
              _row("every signal", report.all_signals), ""]
    lines.append(f"Missed winners (SKIP that made money): {report.missed_winners['count']} "
                 f"({_fmt(report.missed_winners['total_r'])}R). Avoided losers: "
                 f"{report.avoided_losers['count']} ({_fmt(report.avoided_losers['total_r'])}R).")
    for title, table in (("ENTER by direction", report.by_direction),
                         ("Every signal by confluence band", report.by_confluence),
                         ("ENTER by session", report.by_session),
                         ("ENTER by engine version", report.by_version)):
        lines += ["", f"## {title}", "", _HEADER] + [_row(k, v) for k, v in table.items()]
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="performance", description=__doc__.split("\n\n")[0])
    p.add_argument("--db", default=None, help="SQLite path (default: STORAGE_DB_PATH from .env)")
    p.add_argument("--basis", choices=("net", "gross"), default="net")
    p.add_argument("--variant", default="champion")
    p.add_argument("--period", choices=("oos", "dev", "all"), default="oos")
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--format", choices=("md", "json"), default="md")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    from storage import Storage

    args = _parse_args(argv)
    db = args.db
    if db is None:
        from config import Settings
        db = Settings.load(".env").storage_db_path
    storage = Storage(Path(db))
    asyncio.run(storage.initialize())
    rows = asyncio.run(storage.fetch_performance_rows())
    report = build_report(rows, basis=args.basis, variant=args.variant, period=args.period,
                          deduplicate=not args.no_dedup)
    out = (json.dumps(report_json(report), indent=1, default=str) if args.format == "json"
           else render_markdown(report))
    # As in main.py: a Windows console defaults to a codepage that cannot
    # print the report's dashes and would crash the write.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
