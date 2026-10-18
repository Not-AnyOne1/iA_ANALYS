"""Pre-registered accounting experiments: challengers that change how a trade
is MANAGED, never whether it is taken.

E1 — breakeven stop after +0.5R — was registered on 2026-10-10 in
``data/experiment_E1_breakeven_2026-10-10.md`` before any of the data that
will judge it existed. The constants in :data:`E1` ARE that registration:
they are not tuned on results.

What an experiment can and cannot touch:

* It reads the recorded signals and the same M15 candles as the live
  strategy's accounting and writes only to ``experiment_accounting``.
  The decision, the Telegram card, ``signal_outcomes`` and
  ``trade_accounting`` never read anything back from it.
* No real stop is ever moved. The challenger arm is arithmetic on candles.

Arms: ``baseline`` is the live strategy's accounting (the P1 rules);
``challenger`` is the same trade with the stop moved to the fill once a closed
bar has carried the bid/ask ``breakeven_at_r`` x planned risk beyond it (see
``trade_accounting.account``). Bases: ``gross`` (no cost), ``net`` (recorded
or assumed costs) and ``net_2x`` (those costs doubled — the sensitivity check).

The report pairs the arms signal by signal (duplicates once), applies the
registered success criteria, and gives a verdict only at the registered
evaluation point: the earliest run of signals, in recording order, holding
enough resolved signals and ENTERs — so the result cannot be chosen by
picking when to look. Development data gets no verdict at all.

Usage::

    python experiments.py                  # E1, the bot's database
    python experiments.py --db path.db --format json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from costs import NO_COSTS, Costs
from market_data import Candle
from performance import _interval, duplicate_ids
from trade_accounting import TradeResult, account


@dataclass(frozen=True)
class AccountingExperiment:
    name: str
    title: str
    registration: str               # the pre-registration document
    breakeven_at_r: float           # the challenger's trigger
    min_signals: int                # evaluation point: resolved unique signals ...
    min_enter: int                  # ... and resolved unique ENTER signals
    max_duration: timedelta         # hard stop, counted from the start
    cost_multiplier: float          # the doubled-cost sensitivity
    direction_min_signals: int      # a direction is judged only with this many ...
    direction_floor_r: float        # ... and must not fall below this difference

    def parameters(self) -> Dict[str, Any]:
        out = asdict(self)
        out["max_duration"] = str(self.max_duration)
        return out


E1 = AccountingExperiment(
    name="E1",
    title="breakeven stop after +0.5R",
    registration="data/experiment_E1_breakeven_2026-10-10.md",
    breakeven_at_r=0.5,
    min_signals=150,
    min_enter=30,
    max_duration=timedelta(weeks=8),
    cost_multiplier=2.0,
    direction_min_signals=20,
    direction_floor_r=-0.10,
)

REGISTERED: Tuple[AccountingExperiment, ...] = (E1,)

ARMS = ("baseline", "challenger")
BASES = ("gross", "net", "net_2x")
POPULATIONS = ("every signal", "ENTER", "SKIP", "long", "short")


def account_arms(experiment: AccountingExperiment, row: Dict[str, Any],
                 candles: Sequence[Candle], now: datetime, *, costs: Optional[Costs],
                 **kw) -> Dict[str, TradeResult]:
    """Both arms on every basis, from the same candles and the same costs.

    Keyed ``"arm/basis"``. Without a cost figure only the gross basis exists,
    exactly as for the live strategy's own accounting.
    """
    bases: Dict[str, Costs] = {"gross": NO_COSTS}
    if costs is not None:
        m = experiment.cost_multiplier
        bases["net"] = costs
        bases["net_2x"] = Costs(costs.spread * m, costs.slippage * m, f"{costs.source} x{m:g}")
    out: Dict[str, TradeResult] = {}
    for arm, breakeven in (("baseline", None), ("challenger", experiment.breakeven_at_r)):
        for basis, c in bases.items():
            out[f"{arm}/{basis}"] = account(row, candles, now, costs=c, breakeven_at=breakeven, **kw)
    return out


EXPECTED_ROWS = len(ARMS) * len(BASES)


# ------------------------------------------------------------------ pairing

@dataclass(frozen=True)
class Pair:
    """One signal, both arms, one basis."""
    id: int
    recorded_at: datetime
    direction: Optional[str]
    verdict: Optional[str]
    baseline_status: str
    baseline_r: Optional[float]
    baseline_exit: Optional[datetime]
    challenger_status: str
    challenger_r: Optional[float]
    challenger_exit: Optional[datetime]

    @property
    def resolved(self) -> bool:
        return self.baseline_r is not None and self.challenger_r is not None


def _when(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


def pairs_from_rows(rows: Sequence[Dict[str, Any]], basis: str) -> List[Pair]:
    """One Pair per signal that is a trade at all (``invalid`` signals are not)."""
    out = []
    for row in rows:
        arms = row.get("arms") or {}
        base, chall = arms.get(f"baseline/{basis}"), arms.get(f"challenger/{basis}")
        if row.get("status") == "invalid" or (base or {}).get("status") == "invalid":
            continue
        out.append(Pair(
            id=row["id"], recorded_at=datetime.fromisoformat(row["recorded_at"]),
            direction=row.get("direction"), verdict=row.get("decision"),
            baseline_status=base["status"] if base else "pending",
            baseline_r=base.get("r") if base else None,
            baseline_exit=_when(base.get("exit_at")) if base else None,
            challenger_status=chall["status"] if chall else "pending",
            challenger_r=chall.get("r") if chall else None,
            challenger_exit=_when(chall.get("exit_at")) if chall else None,
        ))
    return out


def _drawdown(results: List[Tuple[datetime, int, float]]) -> float:
    """Largest peak-to-trough fall of cumulative R, in exit order."""
    equity = peak = worst = 0.0
    for _, _, r in sorted(results):
        equity += r
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return round(worst, 4)


@dataclass(frozen=True)
class Comparison:
    signals: int
    resolved: int
    not_filled: int
    pending: int
    baseline_r: Optional[float]         # mean R per resolved signal
    challenger_r: Optional[float]
    difference: Optional[float]         # mean paired difference, challenger - baseline
    interval: Optional[Tuple[float, float]]   # seeded 90% bootstrap of the difference
    baseline_drawdown: float
    challenger_drawdown: float
    winners_cut: int                    # baseline won, challenger left at breakeven
    losers_saved: int                   # baseline lost, challenger left at breakeven


def compare(pairs: Sequence[Pair]) -> Comparison:
    done = [p for p in pairs if p.resolved]
    diffs = [p.challenger_r - p.baseline_r for p in done]
    mean = (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)
    return Comparison(
        signals=len(pairs), resolved=len(done),
        not_filled=sum(1 for p in pairs if p.baseline_status == "not_filled"),
        pending=sum(1 for p in pairs if not p.resolved and p.baseline_status != "not_filled"),
        baseline_r=mean([p.baseline_r for p in done]),
        challenger_r=mean([p.challenger_r for p in done]),
        difference=mean(diffs), interval=_interval(diffs),
        baseline_drawdown=_drawdown([(p.baseline_exit or p.recorded_at, p.id, p.baseline_r)
                                     for p in done]),
        challenger_drawdown=_drawdown([(p.challenger_exit or p.recorded_at, p.id, p.challenger_r)
                                       for p in done]),
        winners_cut=sum(1 for p in done if p.baseline_r > 0 and p.challenger_status == "breakeven"),
        losers_saved=sum(1 for p in done if p.baseline_r < 0 and p.challenger_status == "breakeven"),
    )


def _populations(pairs: Sequence[Pair]) -> Dict[str, Comparison]:
    pick = {"every signal": lambda p: True, "ENTER": lambda p: p.verdict == "enter",
            "SKIP": lambda p: p.verdict == "skip", "long": lambda p: p.direction == "long",
            "short": lambda p: p.direction == "short"}
    return {name: compare([p for p in pairs if pick[name](p)]) for name in POPULATIONS}


# ------------------------------------------------------------------- report

@dataclass
class ExperimentReport:
    experiment: str
    title: str
    registration: str
    period: str                         # "experiment" or "development"
    started_at: Optional[str]
    as_of: str
    signals_seen: int
    duplicates_removed: int
    evaluation_set: int                 # signals the verdict is computed on
    cost_sources: Dict[str, int]
    net: Dict[str, Comparison]
    gross: Dict[str, Comparison]
    net_2x: Dict[str, Comparison]
    evaluation_point: Dict[str, Any]
    criteria: List[Dict[str, Any]]
    verdict: str
    notes: List[str] = field(default_factory=list)
    by_engine_version: Dict[str, Comparison] = field(default_factory=dict)   # net, descriptive


def _criteria(exp: AccountingExperiment, net: Dict[str, Comparison],
              net_2x: Dict[str, Comparison]) -> List[Dict[str, Any]]:
    """The six registered success criteria, verbatim from the registration."""
    every, enter = net["every signal"], net["ENTER"]
    judged = {d: net[d] for d in ("long", "short") if net[d].resolved >= exp.direction_min_signals}
    return [
        {"n": 1, "rule": "paired net difference per signal > 0 and its 90% interval's lower bound > 0",
         "value": {"difference": every.difference, "interval": every.interval},
         "passed": (every.difference is not None and every.difference > 0
                    and every.interval is not None and every.interval[0] > 0)},
        {"n": 2, "rule": "ENTER-only paired net difference >= 0",
         "value": enter.difference,
         "passed": enter.difference is not None and enter.difference >= 0},
        {"n": 3, "rule": "challenger max drawdown <= baseline, overall and for ENTER (net, R)",
         "value": {"overall": [every.baseline_drawdown, every.challenger_drawdown],
                   "ENTER": [enter.baseline_drawdown, enter.challenger_drawdown]},
         "passed": (every.resolved > 0 and enter.resolved > 0
                    and every.challenger_drawdown <= every.baseline_drawdown
                    and enter.challenger_drawdown <= enter.baseline_drawdown)},
        {"n": 4, "rule": "losers turned into breakeven > winners turned into breakeven",
         "value": {"losers_saved": every.losers_saved, "winners_cut": every.winners_cut},
         "passed": every.losers_saved > every.winners_cut},
        {"n": 5, "rule": f"difference still > 0 with costs x{exp.cost_multiplier:g}",
         "value": net_2x["every signal"].difference,
         "passed": (net_2x["every signal"].difference is not None
                    and net_2x["every signal"].difference > 0)},
        {"n": 6, "rule": f"no direction with >= {exp.direction_min_signals} resolved signals "
                         f"below {exp.direction_floor_r:+.2f}R",
         "value": {d: c.difference for d, c in judged.items()} or "no direction has enough signals",
         "passed": all(c.difference is not None and c.difference >= exp.direction_floor_r
                       for c in judged.values())},
    ]


def _evaluation_prefix(exp: AccountingExperiment, pairs: Sequence[Pair]) -> Optional[int]:
    """Length of the earliest run of signals (recording order) that reaches the
    registered sample, or None. Fixes the evaluation set, so later signals —
    good or bad — cannot change a verdict once the point is reached."""
    resolved = enters = 0
    for i, p in enumerate(pairs, start=1):
        if p.resolved:
            resolved += 1
            enters += p.verdict == "enter"
        if resolved >= exp.min_signals and enters >= exp.min_enter:
            return i
    return None


def build_experiment_report(rows: Sequence[Dict[str, Any]], experiment: AccountingExperiment = E1,
                            *, started_at: Optional[str], period: str = "experiment",
                            now: Optional[datetime] = None) -> ExperimentReport:
    """Side-by-side report. ``rows`` carry ``arms`` ("arm/basis" -> stored result).

    ``period="experiment"``: only signals recorded from ``started_at`` and
    before the hard stop count, and a verdict is given only at the
    evaluation point. ``period="development"``: every row is shown, labelled,
    and no verdict is given — those signals generated the hypothesis.
    """
    if period not in ("experiment", "development"):
        raise ValueError("period must be 'experiment' or 'development'")
    now = now or datetime.now(timezone.utc)
    rows = sorted(rows, key=lambda r: (r["recorded_at"], r["id"]))
    start = datetime.fromisoformat(started_at) if started_at else None
    hard_stop = start + experiment.max_duration if start else None
    if period == "experiment":
        rows = [r for r in rows if start is not None
                and start <= datetime.fromisoformat(r["recorded_at"]) < hard_stop]
    dups = duplicate_ids(rows)
    unique = [r for r in rows if r["id"] not in dups]

    by_basis = {b: pairs_from_rows(unique, b) for b in BASES}
    prefix = _evaluation_prefix(experiment, by_basis["net"])
    in_set = by_basis["net"][:prefix] if prefix else by_basis["net"]
    waiting = sum(1 for p in in_set if not p.resolved and p.baseline_status != "not_filled")
    if period == "experiment" and prefix:
        ids = {p.id for p in in_set}
        by_basis = {b: [p for p in ps if p.id in ids] for b, ps in by_basis.items()}
    tables = {b: _populations(ps) for b, ps in by_basis.items()}

    cost_sources: Dict[str, int] = {}
    for r in unique:
        src = ((r.get("arms") or {}).get("baseline/net") or {}).get("cost_source")
        if src:
            cost_sources[src] = cost_sources.get(src, 0) + 1

    resolved_now = sum(1 for p in by_basis["net"] if p.resolved)
    enter_now = sum(1 for p in by_basis["net"] if p.resolved and p.verdict == "enter")
    point = {"resolved_signals": resolved_now, "needed_signals": experiment.min_signals,
             "resolved_enter": enter_now, "needed_enter": experiment.min_enter,
             "reached": prefix is not None, "waiting_to_resolve": waiting if prefix else None,
             "hard_stop_at": hard_stop.isoformat() if hard_stop else None,
             "hard_stop_passed": bool(hard_stop and now >= hard_stop)}
    criteria = _criteria(experiment, tables["net"], tables["net_2x"])

    if period == "development":
        verdict = ("NO VERDICT — development data: these signals generated the hypothesis, "
                   "so they cannot test it")
    elif start is None:
        verdict = "NOT STARTED — no signal has been recorded by code carrying this experiment"
    elif prefix is None:
        verdict = ("INSUFFICIENT DATA — the hard stop passed before the evaluation point"
                   if point["hard_stop_passed"] else
                   "COLLECTING — evaluation point not reached; no decision before it")
    elif waiting:
        verdict = f"COLLECTING — {waiting} signal(s) in the evaluation set still open"
    elif tables["net"]["every signal"].difference is None or tables["net"]["every signal"].difference <= 0:
        verdict = "FAILED — the challenger did not beat the baseline; drop the idea"
    elif all(c["passed"] for c in criteria):
        verdict = "SUCCESS — every registered criterion met; adoption still needs the owner's approval"
    else:
        verdict = "INCONCLUSIVE — not every registered criterion met; not adopted"

    # The same evaluation set, split by the engine that recorded each signal.
    # Descriptive only: the registered verdict above pools them, as registered.
    version_of = {r["id"]: (r.get("engine_version") or "unversioned") for r in unique}
    by_version = {v: compare([p for p in by_basis["net"] if version_of[p.id] == v])
                  for v in sorted({version_of[p.id] for p in by_basis["net"]})}

    notes = []
    if period == "development":
        notes.append("IN-SAMPLE / DEVELOPMENT DATA — a debugging exercise, not validation.")
    if any("assumed" in s for s in cost_sources):
        notes.append("Net figures use ASSUMED spread/slippage, not measured bid/ask.")
    if tables["net"]["ENTER"].resolved < experiment.min_enter:
        notes.append(f"Only {tables['net']['ENTER'].resolved} resolved ENTER signals.")
    if len(by_version) > 1:
        notes.append(f"The evaluation set spans {len(by_version)} engine versions — see the "
                     "by-engine-version table; the registered verdict pools them "
                     "(research/reports/E1_engine_versions.md).")
    return ExperimentReport(
        experiment=experiment.name, title=experiment.title, registration=experiment.registration,
        period=period, started_at=started_at, as_of=now.isoformat(),
        signals_seen=len(rows), duplicates_removed=len(dups),
        evaluation_set=len(by_basis["net"]), cost_sources=cost_sources,
        net=tables["net"], gross=tables["gross"], net_2x=tables["net_2x"],
        evaluation_point=point, criteria=criteria, verdict=verdict, notes=notes,
        by_engine_version=by_version)


# ------------------------------------------------------------------- render

def _f(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:+.{digits}f}"


_TABLE = ("| | signals | resolved | baseline R/signal | challenger R/signal | difference "
          "| 90% interval | baseline max DD | challenger max DD | winners cut | losers saved |\n|"
          + "---|" * 11)


def _table(comparisons: Dict[str, Comparison]) -> List[str]:
    lines = [_TABLE]
    for name, c in comparisons.items():
        interval = f"[{c.interval[0]:+.2f}, {c.interval[1]:+.2f}]" if c.interval else "—"
        lines.append(f"| {name} | {c.signals} | {c.resolved} | {_f(c.baseline_r)} | "
                     f"{_f(c.challenger_r)} | {_f(c.difference)} | {interval} | "
                     f"{c.baseline_drawdown:.2f} | {c.challenger_drawdown:.2f} | "
                     f"{c.winners_cut} | {c.losers_saved} |")
    return lines


def render_markdown(report: ExperimentReport) -> str:
    title = ("DEVELOPMENT DATA — in-sample debugging, NOT validation"
             if report.period == "development" else "registered evaluation")
    p = report.evaluation_point
    lines = [f"# Experiment {report.experiment} — {report.title} ({title})", "",
             f"Registration: `{report.registration}`. Start: {report.started_at or 'not started'}. "
             f"As of {report.as_of[:16]} UTC.",
             f"Signals: {report.signals_seen} seen, {report.duplicates_removed} duplicates counted once, "
             f"{report.evaluation_set} in the evaluation set. Costs: "
             + (", ".join(f"{k}: {v}" for k, v in report.cost_sources.items()) or "none") + ".",
             f"Evaluation point: {p['resolved_signals']}/{p['needed_signals']} resolved signals, "
             f"{p['resolved_enter']}/{p['needed_enter']} resolved ENTER"
             + (f"; hard stop {p['hard_stop_at'][:10]}" if p["hard_stop_at"] else "") + ".", "",
             f"**Verdict: {report.verdict}**", ""]
    lines += [f"> {n}" for n in report.notes] + ([""] if report.notes else [])
    multiplier = next((e.cost_multiplier for e in REGISTERED if e.name == report.experiment), 2.0)
    for label, table in (("Net (after costs)", report.net),
                         (f"Net with costs x{multiplier:g} (sensitivity)", report.net_2x),
                         ("Gross (before any cost — NOT a net result)", report.gross),
                         ("Net by engine version (descriptive; the registered verdict pools them)",
                          report.by_engine_version)):
        lines += [f"## {label}", ""] + _table(table) + [""]
    lines += ["## Registered success criteria (net)", "", "| # | rule | value | met |", "|---|---|---|---|"]
    for c in report.criteria:
        lines.append(f"| {c['n']} | {c['rule']} | {json.dumps(c['value'], default=str)} | "
                     f"{'yes' if c['passed'] else 'no'} |")
    if report.period == "development":
        lines += ["", "Criteria shown for debugging only: development data cannot meet or fail them."]
    return "\n".join(lines)


def report_json(report: ExperimentReport) -> Dict[str, Any]:
    return asdict(report)


# --------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> int:
    from storage import Storage

    p = argparse.ArgumentParser(prog="experiments", description=__doc__.split("\n\n")[0])
    p.add_argument("--db", default=None, help="SQLite path (default: STORAGE_DB_PATH from .env)")
    p.add_argument("--experiment", default=E1.name, choices=[e.name for e in REGISTERED])
    p.add_argument("--format", choices=("md", "json"), default="md")
    args = p.parse_args(argv)
    db = args.db
    if db is None:
        from config import Settings
        db = Settings.load(".env").storage_db_path
    experiment = next(e for e in REGISTERED if e.name == args.experiment)
    storage = Storage(Path(db))
    asyncio.run(storage.initialize())
    started_at, rows = asyncio.run(storage.fetch_experiment_rows(experiment.name))
    report = build_experiment_report(rows, experiment, started_at=started_at)
    out = (json.dumps(report_json(report), indent=1, default=str) if args.format == "json"
           else render_markdown(report))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
