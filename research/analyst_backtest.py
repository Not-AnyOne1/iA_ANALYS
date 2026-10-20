"""Backtest of the independent analyst: recorded signals and systematic runs.

Runs exactly the shadow code (``analyst.engine`` / ``analyst.strategies``)
over research bars from ``data/research/`` (gitignored), served as the live
bot would see them: at decision time ``t`` each timeframe gets the last 300
bars that had opened before ``t``, the still-forming one included, and the
analyst itself drops unclosed bars and weekend stubs. Nothing later than
``t`` is visible to a decision.

Outcomes use ``trade_accounting`` unchanged: market fill at the decision
price, the stop first when a bar touches both levels, a bar opening through
the stop exits at its open, five-day timeout. R is reported gross, net of
the labelled cost assumption (spread 0.30, slippage 0.10) and net of twice
that.

    python research/analyst_backtest.py signals --db <copy of the bot's db> --out signals.json
    python research/analyst_backtest.py systematic --out systematic.json [--period development]

``systematic`` evaluates strategies B and C at every M15 close and keeps
every READY setup with its outcome; ``report`` (below) then applies one
position at a time per strategy, chronologically, and splits by period.
The periods are fixed here, before any result was seen; no threshold is
fitted on any of them.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from costs import NO_COSTS, Costs  # noqa: E402
from market_data import Candle, Timeframe  # noqa: E402
from models import SignalAnalysis  # noqa: E402
from trade_accounting import account  # noqa: E402
from trade_quality import effective_entry  # noqa: E402

from analyst.data import DURATION, TIMEFRAMES, build_view, drop_closure_stubs, m5_request  # noqa: E402
from analyst.engine import ENGINE_VERSION, analyse  # noqa: E402
from analyst.rules import Rules  # noqa: E402
from analyst.strategies import READY, STRATEGIES  # noqa: E402
from analyst.structure import LONG, SHORT, read_frame, read_market  # noqa: E402
from analyst.verdict import RecentSignal  # noqa: E402

UTC = timezone.utc
DATA = ROOT / "data" / "research"
FILES = {Timeframe.M5: "5min", Timeframe.M15: "15min", Timeframe.H1: "1h", Timeframe.H4: "4h"}
WINDOW = 300
COSTS = {"gross": NO_COSTS, "net": Costs(0.30, 0.10, "assumed"),
         "net_x2": Costs(0.60, 0.20, "assumed x2")}
HORIZON = timedelta(days=5)
PERIODS = (("development", "2021-01-01", "2024-01-01"),
           ("validation", "2024-01-01", "2025-01-01"),
           ("test", "2025-01-01", "2026-01-01"),
           ("holdout", "2026-01-01", "2026-10-18"))


# ------------------------------------------------------------------ history

class History:
    def __init__(self, data: Path = DATA) -> None:
        self.bars: Dict[Timeframe, Tuple[Candle, ...]] = {}
        self.opens: Dict[Timeframe, List[float]] = {}
        for tf, name in FILES.items():
            raw = json.loads((data / f"xauusd_{name}.json").read_text())["bars"]
            bars = tuple(Candle(datetime.fromisoformat(t), o, h, lo, c) for t, o, h, lo, c in raw)
            self.bars[tf] = bars
            self.opens[tf] = [b.timestamp.timestamp() for b in bars]
        self.m15_clean = drop_closure_stubs(self.bars[Timeframe.M15], Timeframe.M15, "XAUUSD")
        self.m15_clean_opens = [b.timestamp.timestamp() for b in self.m15_clean]

    def window(self, tf: Timeframe, at: datetime, n: Optional[int] = None) -> Tuple[Candle, ...]:
        """What the provider returns at ``at``: the last ``n`` bars that had
        opened — 300 for the live context's timeframes, the padded M5 request
        for M5, as the live shadow analyst asks."""
        n = n or (m5_request() if tf is Timeframe.M5 else WINDOW)
        end = bisect.bisect_left(self.opens[tf], at.timestamp())
        return self.bars[tf][max(0, end - n):end]

    def last_close(self, tf: Timeframe, at: datetime) -> Optional[float]:
        end = bisect.bisect_right(self.opens[tf], (at - DURATION[tf]).timestamp())
        return self.bars[tf][end - 1].close if end else None

    def outcome_bars(self, at: datetime) -> Tuple[Candle, ...]:
        """Clean M15 bars from one bar before ``at`` to the horizon plus a day."""
        lo = bisect.bisect_left(self.m15_clean_opens, (at - timedelta(minutes=15)).timestamp())
        hi = bisect.bisect_left(self.m15_clean_opens, (at + HORIZON + timedelta(days=1)).timestamp())
        return self.m15_clean[lo:hi]

    @property
    def end(self) -> datetime:
        return self.bars[Timeframe.M15][-1].timestamp + DURATION[Timeframe.M15]


def outcomes(history: History, trade: Dict, at: datetime) -> Dict[str, Dict]:
    bars = history.outcome_bars(at)
    now = min(at + HORIZON + timedelta(days=1), history.end)
    out = {}
    for basis, costs in COSTS.items():
        r = account(trade, bars, now, costs=costs, horizon=HORIZON)
        out[basis] = {"status": r.status, "r": r.r,
                      "exit_at": r.exit_at.isoformat() if r.exit_at else None}
    return out


class CachedReader:
    """read_frame, memoised on the exact window (H4/H1 windows repeat for
    many consecutive M15 steps). Same result as read_frame by construction."""

    def __init__(self, rules: Rules) -> None:
        self.rules, self.cache = rules, {}

    def __call__(self, candles, tf):
        key = (tf, candles[0].timestamp, candles[-1].timestamp, len(candles))
        hit = self.cache.get(tf)
        if hit is not None and hit[0] == key:
            return hit[1]
        frame = read_frame(candles, tf, self.rules)
        self.cache[tf] = (key, frame)
        return frame


def regime(read, direction: str, at: datetime) -> Dict[str, str]:
    want = "bullish" if direction == LONG else "bearish"

    def rel(bias):
        return "with" if bias == want else ("neutral" if bias == "neutral" else "against")
    hour = at.hour
    session = ("asia" if hour < 7 else "london" if hour < 12 else "overlap" if hour < 16
               else "new_york" if hour < 21 else "late")
    atr_pct = read.atr15 / read.price * 100
    return {"h4": rel(read.frame(Timeframe.H4).bias), "h1": rel(read.frame(Timeframe.H1).bias),
            "session": session, "atr_pct": round(atr_pct, 4)}


# --------------------------------------------------------------- systematic

def systematic_chunk(args) -> Dict:
    start_iso, end_iso = args
    history = History()
    rules = Rules()
    reader = CachedReader(rules)
    start, end = datetime.fromisoformat(start_iso), datetime.fromisoformat(end_iso)
    m15 = history.bars[Timeframe.M15]
    lo = bisect.bisect_left(history.opens[Timeframe.M15], (start - DURATION[Timeframe.M15]).timestamp())
    candidates, funnel = [], {name: {"READY": 0, "WAITING": 0, "INVALID": 0} for name in STRATEGIES}
    steps = fresh = 0
    for bar in m15[lo:]:
        at = bar.timestamp + DURATION[Timeframe.M15]
        if at < start:
            continue
        if at >= end:
            break
        steps += 1
        frames = {tf: history.window(tf, at) for tf in TIMEFRAMES}
        price = history.last_close(Timeframe.M5, at)
        view = build_view("XAUUSD", at, price, frames, rules)
        if not view.fresh:
            continue
        read = read_market(view, rules, reader)
        if read is None:
            continue
        fresh += 1
        for name, fn in STRATEGIES.items():
            for direction in (LONG, SHORT):
                p = fn(read, direction, COSTS["net"])
                if p is None:
                    continue
                funnel[name][p.status] += 1
                if p.status != READY:
                    continue
                trade = {"direction": direction, "order_type": "market", "price_at_decision": price,
                         "entry": p.entry, "stop_loss": p.stop_loss,
                         "take_profits_json": json.dumps([p.take_profit]),
                         "recorded_at": at.isoformat()}
                candidates.append({"strategy": name, "direction": direction, "at": at.isoformat(),
                                   "entry": p.entry, "stop_loss": p.stop_loss,
                                   "take_profit": p.take_profit, "rr_net": p.rr_net,
                                   "target": p.target, "regime": regime(read, direction, at),
                                   "outcome": outcomes(history, trade, at)})
    return {"start": start_iso, "end": end_iso, "steps": steps, "fresh": fresh,
            "funnel": funnel, "candidates": candidates}


def months(start: str, end: str) -> List[Tuple[str, str]]:
    a = datetime.fromisoformat(start).replace(tzinfo=UTC)
    stop = datetime.fromisoformat(end).replace(tzinfo=UTC)
    out = []
    while a < stop:
        b = (a.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((a.isoformat(), min(b, stop).isoformat()))
        a = b
    return out


def run_systematic(periods: Sequence[str], workers: int) -> Dict:
    chosen = [p for p in PERIODS if p[0] in periods]
    chunks = [c for _, s, e in chosen for c in months(s, e)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(systematic_chunk, chunks))
    print(f"systematic: {len(chunks)} monthly chunks in {time.time() - t0:.0f} s", flush=True)
    return {"engine_version": ENGINE_VERSION, "periods": [p[0] for p in chosen], "chunks": parts}


# ------------------------------------------------------------------ signals

def run_signals(db: str, audit: Optional[str]) -> Dict:
    history = History()
    rules = Rules()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    live_price = {r["message_id"]: r["price_at_decision"]
                  for r in con.execute("SELECT message_id, price_at_decision FROM signal_outcomes")}
    rows = con.execute("SELECT * FROM analyses WHERE is_signal = 1 AND symbol = 'XAUUSD' "
                       "AND recorded_at >= '2026-08-20' ORDER BY id").fetchall()
    baseline = {}
    if audit:
        baseline = {a["analysis_id"]: a for a in json.loads(Path(audit).read_text())}
    out, seen = [], []
    for row in rows:
        at = datetime.fromisoformat(row["recorded_at"])
        message_at = datetime.fromisoformat(row["message_timestamp"])
        setup = SignalAnalysis.model_validate(json.loads(row["analysis_json"])).setup
        price = live_price.get(row["message_id"]) or history.last_close(Timeframe.M5, at)
        frames = {tf: history.window(tf, at) for tf in TIMEFRAMES}
        recent = [s for s in seen if s.at >= message_at - rules.duplicate_window - rules.signal_max_age]
        a = analyse("XAUUSD", setup, message_at, at, price, frames, rules, COSTS["net"], recent=recent)
        seen.append(RecentSignal(message_at, setup.direction, setup.stop_loss, tuple(setup.take_profits)))
        signal = None
        if setup.direction and setup.stop_loss is not None and setup.take_profits:
            signal = outcomes(history, {
                "direction": setup.direction, "order_type": setup.order_type,
                "price_at_decision": price, "entry": effective_entry(setup, price),
                "stop_loss": setup.stop_loss, "take_profits_json": json.dumps(setup.take_profits),
                "recorded_at": at.isoformat()}, at)
        proposals = []
        for name, sides in a.proposals.items():
            for d, p in sides.items():
                if p is None:
                    continue
                item = {"strategy": name, "direction": d, "status": p.status, "rr_net": p.rr_net}
                if p.status == READY:
                    item["outcome"] = outcomes(history, {
                        "direction": d, "order_type": "market", "price_at_decision": price,
                        "entry": p.entry, "stop_loss": p.stop_loss,
                        "take_profits_json": json.dumps([p.take_profit]),
                        "recorded_at": at.isoformat()}, at)
                proposals.append(item)
        base = baseline.get(row["id"], {})
        out.append({"analysis_id": row["id"], "at": at.isoformat(), "direction": setup.direction,
                    "order_type": setup.order_type, "price": price,
                    "analyst": a.verdict.verdict, "codes": [r.code for r in a.verdict.reasons],
                    "checks_met": a.verdict.checks_met, "data_status": a.data_status,
                    "biases": a.biases,
                    "baseline": (base.get("verdict") or "").upper() or None,
                    "baseline_displayed": base.get("displayed_confidence"),
                    "live": (row["decision_verdict"] or "").upper() or None,
                    "signal": signal, "proposals": proposals})
    return {"engine_version": ENGINE_VERSION, "signals": out}


# ------------------------------------------------------------------- metrics

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def bootstrap_mean(values: Sequence[float], n: int = 4000, seed: int = 7) -> Tuple[Optional[float], Optional[float]]:
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)


def metrics(rs: Sequence[float]) -> Dict:
    n = len(rs)
    if n == 0:
        return {"n": 0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    equity = peak = dd = 0.0
    for r in rs:
        equity += r
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    lo, hi = bootstrap_mean(rs)
    wlo, whi = wilson(len(wins), n)
    return {"n": n, "win_rate": round(len(wins) / n, 4), "win_rate_ci95": [wlo, whi],
            "avg_win_r": round(sum(wins) / len(wins), 4) if wins else None,
            "avg_loss_r": round(sum(losses) / len(losses), 4) if losses else None,
            "expectancy_r": round(sum(rs) / n, 4), "expectancy_ci95": [lo, hi],
            "profit_factor": round(sum(wins) / -sum(losses), 3) if losses and sum(losses) < 0 else None,
            "total_r": round(sum(rs), 3), "max_drawdown_r": round(dd, 3)}


def gate(candidates: List[Dict]) -> List[Dict]:
    """One position at a time per strategy, in time order: a setup is taken
    only once the previous trade of the same strategy has exited."""
    taken, busy_until = [], {}
    for c in sorted(candidates, key=lambda c: (c["at"], c["strategy"], c["direction"])):
        status = c["outcome"]["net"]["status"]
        if status not in ("tp", "sl", "timeout"):
            continue
        at = datetime.fromisoformat(c["at"])
        if c["strategy"] in busy_until and at < busy_until[c["strategy"]]:
            continue
        exits = [datetime.fromisoformat(o["exit_at"]) for o in c["outcome"].values() if o["exit_at"]]
        busy_until[c["strategy"]] = max(exits) if exits else at + HORIZON
        taken.append(c)
    return taken


def period_of(at: str) -> str:
    for name, start, end in PERIODS:
        if start <= at[:10] < end:
            return name
    return "outside"


def report_systematic(result: Dict) -> Dict:
    candidates = [c for part in result["chunks"] for c in part["candidates"]]
    taken = gate(candidates)
    out = {"candidates": len(candidates), "taken": len(taken), "by_strategy": {}}
    for name in STRATEGIES:
        mine = [c for c in taken if c["strategy"] == name]
        entry = {"funnel": {k: sum(p["funnel"][name][k] for p in result["chunks"])
                            for k in ("READY", "WAITING", "INVALID")}, "periods": {}}
        groups = {}
        for c in mine:
            groups.setdefault(period_of(c["at"]), []).append(c)
        oos = [c for c in mine if period_of(c["at"]) in ("validation", "test", "holdout")]
        for label, items in [*sorted(groups.items()), ("out_of_sample", oos), ("all", mine)]:
            entry["periods"][label] = {basis: metrics([c["outcome"][basis]["r"] for c in items])
                                       for basis in COSTS}
        years = {}
        for c in mine:
            years.setdefault(c["at"][:4], []).append(c["outcome"]["net"]["r"])
        entry["by_year_net"] = {y: metrics(v) for y, v in sorted(years.items())}
        for key in ("h4", "h1", "session"):
            buckets = {}
            for c in oos:
                buckets.setdefault(c["regime"][key], []).append(c["outcome"]["net"]["r"])
            entry[f"oos_by_{key}_net"] = {k: metrics(v) for k, v in sorted(buckets.items())}
        entry["by_direction_net"] = {d: metrics([c["outcome"]["net"]["r"] for c in mine if c["direction"] == d])
                                     for d in (LONG, SHORT)}
        out["by_strategy"][name] = entry
    return out


def _verdict_table(signals: List[Dict], source: str) -> Dict:
    resolved = [s for s in signals if s["signal"] and s["signal"]["net"]["status"] in ("tp", "sl", "timeout")]
    counts, by_verdict = {}, {}
    for s in signals:
        counts[str(s[source])] = counts.get(str(s[source]), 0) + 1
    for s in resolved:
        by_verdict.setdefault(str(s[source]), []).append(s["signal"]["net"]["r"])
    enter, skip = by_verdict.get("ENTER", []), by_verdict.get("SKIP", [])
    return {"counts": counts,
            "by_verdict_net": {k: metrics(v) for k, v in sorted(by_verdict.items())},
            "false_positive_rate": round(sum(1 for r in enter if r <= 0) / len(enter), 4) if enter else None,
            "false_negative_rate": round(sum(1 for r in skip if r > 0) / len(skip), 4) if skip else None}


def report_signals(result: Dict) -> Dict:
    signals = result["signals"]
    # One per distinct signal: the same direction, stop and first target
    # forwarded again within four hours is the same trade idea.
    unique = [s for s in signals if "SIG_DUPLICATE" not in s["codes"]]
    resolved = [s for s in signals if s["signal"] and s["signal"]["net"]["status"] in ("tp", "sl", "timeout")]
    out = {"signals": len(signals), "unique": len(unique), "resolved": len(resolved),
           "all_signals_net": metrics([s["signal"]["net"]["r"] for s in resolved]),
           "sources": {src: _verdict_table(signals, src) for src in ("live", "baseline", "analyst")},
           "unique_sources": {src: _verdict_table(unique, src) for src in ("live", "baseline", "analyst")}}
    codes = {}
    for s in signals:
        for c in s["codes"]:
            codes[c] = codes.get(c, 0) + 1
    out["analyst_reason_codes"] = dict(sorted(codes.items(), key=lambda kv: -kv[1]))
    ready = [p for s in signals for p in s["proposals"] if p["status"] == READY
             and p["outcome"]["net"]["status"] in ("tp", "sl", "timeout")]
    out["ready_proposals_at_signal_times_net"] = metrics([p["outcome"]["net"]["r"] for p in ready])
    out["proposal_status_counts"] = {}
    for s in signals:
        for p in s["proposals"]:
            key = f"{p['strategy']} {p['status']}"
            out["proposal_status_counts"][key] = out["proposal_status_counts"].get(key, 0) + 1
    checks = {}
    for s in resolved:
        checks.setdefault(s["checks_met"], []).append(s["signal"]["net"]["r"])
    out["signal_r_by_checks_met"] = {str(k): metrics(v) for k, v in sorted(checks.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signals")
    s.add_argument("--db", required=True)
    s.add_argument("--audit", help="production replay (research/audit_production.py output)")
    s.add_argument("--out", required=True)
    y = sub.add_parser("systematic")
    y.add_argument("--period", action="append", choices=[x[0] for x in PERIODS])
    y.add_argument("--workers", type=int, default=12)
    y.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("kind", choices=["signals", "systematic"])
    r.add_argument("path")
    args = p.parse_args(argv)
    if args.cmd == "signals":
        result = run_signals(args.db, args.audit)
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(json.dumps(report_signals(result), indent=1))
    elif args.cmd == "systematic":
        result = run_systematic(args.period or [x[0] for x in PERIODS], args.workers)
        Path(args.out).write_text(json.dumps(result))
        print(json.dumps(report_systematic(result), indent=1))
    else:
        result = json.loads(Path(args.path).read_text())
        fn = report_signals if args.kind == "signals" else report_systematic
        print(json.dumps(fn(result), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
