"""Production verdict versus each candidate change, on standardized signals.

There are only 109 recorded Telegram signals, about two market episodes, and
the candidate changes were designed after studying them; they cannot also
be the test. So every two hours from 2021 to 2026 (market open, more than 30
minutes before the weekly close) this harness issues one BUY and one SELL
signal shaped like the recorded ones — a market entry at the price, stop
1.0 H1 ATR away (the recorded median, 0.98), target at 2.0R (the recorded
median) — and asks the production code, and each candidate
(``strategy_variants.py``), for its verdict. Every signal is then scored
with ``trade_accounting`` whatever the verdict, so ENTER and SKIP can both
be judged, gross and net of assumed costs (x1 and x2).

The production verdict is replayed as the live bot computed it
(``research/production_replay.py``): live-like windows with the forming bar
and the weekend stubs; ``clean_data`` gets the cleaned windows instead.

    python research/strategy_backtest.py run --out runs.json
    python research/strategy_backtest.py report runs.json
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import indicators  # noqa: E402
from costs import NO_COSTS, Costs  # noqa: E402
from market_context import MarketContextBuilder  # noqa: E402
from market_data import Timeframe  # noqa: E402
from market_hours import bar_inside_weekly_closure  # noqa: E402
from models import TradeSetup  # noqa: E402
from news_filter import NewsFilter  # noqa: E402
from trade_accounting import account  # noqa: E402

import strategy_variants  # noqa: E402
from analyst.data import market_state  # noqa: E402
from production_replay import Bars, ReplayProvider, service  # noqa: E402

UTC = timezone.utc
STEP = timedelta(hours=2)
STOP_ATR, TARGET_R = 1.0, 2.0
HORIZON = timedelta(days=5)
COSTS = {"gross": NO_COSTS, "net": Costs(0.30, 0.10, "assumed"), "net_x2": Costs(0.60, 0.20, "assumed x2")}
VARIANTS = strategy_variants.VARIANTS
PERIODS = (("development", "2021-01-01", "2024-01-01"),
           ("validation", "2024-01-01", "2025-01-01"),
           ("test", "2025-01-01", "2026-01-01"),
           ("holdout", "2026-01-01", "2026-10-17"))
# The window the candidate changes were designed from (the recorded signals,
# 2026-09-28..10-15) is excluded from the holdout.
EXCLUDED = ("2026-09-25", "2026-10-17")


def session_of(at: datetime) -> str:
    h = at.hour
    return ("asia" if h < 7 else "london" if h < 12 else "overlap" if h < 16
            else "new_york" if h < 21 else "late")


async def _chunk(start: datetime, end: datetime) -> Dict:
    bars = Bars()
    live, clean = ReplayProvider(bars), ReplayProvider(bars)
    live_builder = MarketContextBuilder(service(live), news_filter=NewsFilter())
    clean_builder = MarketContextBuilder(service(clean), news_filter=NewsFilter(), clean_candles=True)
    m15 = bars.bars[Timeframe.M15]
    m15_clean = [c for c in m15 if not bar_inside_weekly_closure(c.timestamp, timedelta(minutes=15))]
    opens = [c.timestamp.timestamp() for c in m15_clean]
    records = []
    at = start
    while at < end:
        if EXCLUDED[0] <= at.isoformat()[:10] < EXCLUDED[1] or \
                market_state(at, "XAUUSD", timedelta(minutes=30)) != "open":
            at += STEP
            continue
        price = bars.last_close(at)
        atr = indicators.atr(bars.live_window(Timeframe.H1, at, 300))
        if price is None or not atr:
            at += STEP
            continue
        lo = bisect.bisect_left(opens, (at - timedelta(minutes=15)).timestamp())
        hi = bisect.bisect_left(opens, (at + HORIZON + timedelta(days=1)).timestamp())
        outcome_bars = m15_clean[lo:hi]
        for d in ("long", "short"):
            s = 1.0 if d == "long" else -1.0
            stop, target = price - s * STOP_ATR * atr, price + s * STOP_ATR * atr * TARGET_R
            setup = TradeSetup(symbol="XAUUSD", direction=d, order_type="market",
                               stop_loss=round(stop, 2), take_profits=[round(target, 2)])
            live.at = clean.at = at
            live.price = clean.price = price
            ctx = await live_builder.build("XAUUSD", setup, now=at)
            verdicts = {v: strategy_variants.decide(ctx, v).verdict.value[0].upper()
                        for v in VARIANTS if v != "clean_data"}
            clean_ctx = await clean_builder.build("XAUUSD", setup, now=at)
            verdicts["clean_data"] = strategy_variants.decide(clean_ctx, "clean_data").verdict.value[0].upper()
            row = {"direction": d, "order_type": "market", "price_at_decision": price, "entry": price,
                   "stop_loss": setup.stop_loss, "take_profits_json": json.dumps(setup.take_profits),
                   "recorded_at": at.isoformat()}
            now = at + HORIZON + timedelta(days=1)
            out, exit_at = {}, None
            for basis, costs in COSTS.items():
                r = account(row, outcome_bars, now, costs=costs, horizon=HORIZON)
                out[basis] = r.r if r.status in ("tp", "sl", "timeout") else None
                if basis == "net" and r.exit_at is not None:
                    exit_at = r.exit_at.isoformat()
            h1 = ctx.timeframes.get("H1")
            vote = ctx.scoring.net_score if ctx.scoring else 0
            records.append({
                "at": at.isoformat(), "d": d, "r": out, "exit": exit_at, "v": verdicts,
                "reg": {"d1": ctx.timeframes["D1"].trend if "D1" in ctx.timeframes else None,
                        "h4": ctx.timeframes["H4"].trend if "H4" in ctx.timeframes else None,
                        "h1": h1.trend if h1 else None,
                        "adx": round(h1.adx.adx, 1) if h1 and h1.adx.adx is not None else None,
                        "session": session_of(at),
                        "vote": "long" if vote > 0 else ("short" if vote < 0 else "none"),
                        "regime": strategy_variants.regime(ctx)},
            })
        at += STEP
    return {"start": start.isoformat(), "end": end.isoformat(), "records": records}


def chunk(args) -> Dict:
    return asyncio.run(_chunk(datetime.fromisoformat(args[0]), datetime.fromisoformat(args[1])))


def months(start: str, end: str) -> List[Tuple[str, str]]:
    a = datetime.fromisoformat(start).replace(tzinfo=UTC)
    stop = datetime.fromisoformat(end).replace(tzinfo=UTC)
    out = []
    while a < stop:
        b = (a.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((a.isoformat(), min(b, stop).isoformat()))
        a = b
    return out


# ----------------------------------------------------------------- metrics

def week_of(at: str) -> Tuple[int, int]:
    y, w, _ = datetime.fromisoformat(at).isocalendar()
    return (y, w)


def summary(rs: Sequence[float]) -> Dict:
    n = len(rs)
    if not n:
        return {"n": 0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    eq = peak = dd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {"n": n, "win_rate": round(len(wins) / n, 4),
            "avg_win_r": round(sum(wins) / len(wins), 3) if wins else None,
            "avg_loss_r": round(sum(losses) / len(losses), 3) if losses else None,
            "expectancy_r": round(sum(rs) / n, 4),
            "profit_factor": round(sum(wins) / -sum(losses), 3) if losses and sum(losses) < 0 else None,
            "total_r": round(sum(rs), 2), "max_drawdown_r": round(dd, 2)}


class WeekBootstrap:
    """95% intervals that resample whole ISO weeks, because signals two hours
    apart with five-day horizons are not independent."""

    def __init__(self, records: Sequence[Dict], n: int = 2000, seed: int = 11) -> None:
        weeks = sorted({week_of(r["at"]) for r in records})
        rng = random.Random(seed)
        self.index = {w: i for i, w in enumerate(weeks)}
        self.draws = [[rng.randrange(len(weeks)) for _ in weeks] for _ in range(n)] if weeks else []
        self.weeks = len(weeks)

    def mean(self, records: Sequence[Dict], value) -> Tuple[Optional[float], Optional[float]]:
        sums, counts = [0.0] * self.weeks, [0] * self.weeks
        for r in records:
            v = value(r)
            if v is not None:
                i = self.index[week_of(r["at"])]
                sums[i] += v
                counts[i] += 1
        means = []
        for draw in self.draws:
            s = sum(sums[i] for i in draw)
            c = sum(counts[i] for i in draw)
            if c:
                means.append(s / c)
        if len(means) < 100:
            return None, None
        means.sort()
        return round(means[int(0.025 * len(means))], 4), round(means[int(0.975 * len(means)) - 1], 4)

    def diff(self, records: Sequence[Dict], a, b) -> Tuple[Optional[float], Optional[float]]:
        """Interval of mean(a) - mean(b), both on the same resampled weeks."""
        sa, ca, sb, cb = ([0.0] * self.weeks, [0] * self.weeks, [0.0] * self.weeks, [0] * self.weeks)
        for r in records:
            i = self.index[week_of(r["at"])]
            va, vb = a(r), b(r)
            if va is not None:
                sa[i] += va
                ca[i] += 1
            if vb is not None:
                sb[i] += vb
                cb[i] += 1
        out = []
        for draw in self.draws:
            na, nb = sum(ca[i] for i in draw), sum(cb[i] for i in draw)
            if na and nb:
                out.append(sum(sa[i] for i in draw) / na - sum(sb[i] for i in draw) / nb)
        if len(out) < 100:
            return None, None
        out.sort()
        return round(out[int(0.025 * len(out))], 4), round(out[int(0.975 * len(out)) - 1], 4)


def period_of(at: str) -> str:
    for name, start, end in PERIODS:
        if start <= at[:10] < end:
            return name
    return "outside"


def entered(variant: str, basis: str = "net"):
    return lambda r: r["r"][basis] if r["v"][variant] == "E" else None


def gated(records: Sequence[Dict], variant: str, basis: str = "net") -> List[float]:
    """One position at a time: an ENTER is taken only once the previously
    taken trade has exited (net basis exit time), in time order."""
    taken, free_at = [], None
    for r in sorted(records, key=lambda r: (r["at"], r["d"])):
        if r["v"][variant] != "E" or r["r"][basis] is None or r["exit"] is None:
            continue
        if free_at is not None and r["at"] < free_at:
            continue
        taken.append(r["r"][basis])
        free_at = r["exit"]
    return taken


def report(runs: Dict) -> Dict:
    records = [r for c in runs["chunks"] for r in c["records"]]
    out = {"signals": len(records), "periods": {}}
    groups = {}
    for r in records:
        groups.setdefault(period_of(r["at"]), []).append(r)
    groups["out_of_sample"] = [r for r in records if period_of(r["at"]) in ("validation", "test", "holdout")]
    for label, items in groups.items():
        boot = WeekBootstrap(items)
        block = {"signals": len(items), "all_signals_net": summary([r["r"]["net"] for r in items if r["r"]["net"] is not None]),
                 "variants": {}}
        for v in VARIANTS:
            counts = {k: sum(1 for r in items if r["v"][v] == k) for k in ("E", "W", "S")}
            ent = {basis: [r["r"][basis] for r in items if r["v"][v] == "E" and r["r"][basis] is not None]
                   for basis in COSTS}
            skip = [r["r"]["net"] for r in items if r["v"][v] == "S" and r["r"]["net"] is not None]
            wait = [r["r"]["net"] for r in items if r["v"][v] == "W" and r["r"]["net"] is not None]
            entry = {
                "counts": counts,
                "enter": {basis: summary(rs) for basis, rs in ent.items()},
                "enter_net_ci95": list(boot.mean(items, entered(v))),
                "skip_net": summary(skip), "wait_net": summary(wait),
                "false_positive_rate": round(sum(1 for x in ent["net"] if x <= 0) / len(ent["net"]), 4) if ent["net"] else None,
                "false_negative_rate": round(sum(1 for x in skip if x > 0) / len(skip), 4) if skip else None,
                "separation_net": round(sum(ent["net"]) / len(ent["net"]) - sum(skip) / len(skip), 4) if ent["net"] and skip else None,
                "enter_long_share": round(sum(1 for r in items if r["v"][v] == "E" and r["d"] == "long") / max(1, counts["E"]), 4),
                "gated_net": summary(gated(items, v)),
            }
            if v != "baseline":
                entry["delta_vs_baseline_net"] = (round(sum(ent["net"]) / len(ent["net"]) - (lambda b: sum(b) / len(b) if b else 0)(
                    [r["r"]["net"] for r in items if r["v"]["baseline"] == "E" and r["r"]["net"] is not None]), 4) if ent["net"] else None)
                entry["delta_ci95"] = list(boot.diff(items, entered(v), entered("baseline")))
            block["variants"][v] = entry
        out["periods"][label] = block
    oos = groups["out_of_sample"]
    out["oos_by_year"] = {}
    for year in ("2024", "2025", "2026"):
        items = [r for r in oos if r["at"][:4] == year]
        base = [r["r"]["net"] for r in items if r["v"]["baseline"] == "E" and r["r"]["net"] is not None]
        out["oos_by_year"][year] = {v: summary([r["r"]["net"] for r in items if r["v"][v] == "E" and r["r"]["net"] is not None])["expectancy_r"] if any(r["v"][v] == "E" for r in items) else None
                                    for v in VARIANTS}
        out["oos_by_year"][year]["baseline_n"] = len(base)
    out["oos_regimes"] = {}
    for key in ("d1", "h4", "session"):
        for v in VARIANTS:
            buckets = {}
            for r in oos:
                if r["v"][v] != "E" or r["r"]["net"] is None:
                    continue
                label = r["reg"][key]
                if key in ("d1", "h4") and label in ("bullish", "bearish"):
                    label = "with" if (label == "bullish") == (r["d"] == "long") else "against"
                buckets.setdefault(label, []).append(r["r"]["net"])
            out["oos_regimes"].setdefault(key, {})[v] = {k: [len(x), round(sum(x) / len(x), 3)] for k, x in sorted(buckets.items())}
    adx = {}
    for v in VARIANTS:
        b = {}
        for r in oos:
            if r["v"][v] != "E" or r["r"]["net"] is None or r["reg"]["adx"] is None:
                continue
            k = "<20" if r["reg"]["adx"] < 20 else ("20-25" if r["reg"]["adx"] < 25 else ">=25")
            b.setdefault(k, []).append(r["r"]["net"])
        adx[v] = {k: [len(x), round(sum(x) / len(x), 3)] for k, x in sorted(b.items())}
    out["oos_regimes"]["h1_adx"] = adx
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", default=PERIODS[0][1])
    r.add_argument("--end", default=PERIODS[-1][2])
    r.add_argument("--workers", type=int, default=14)
    r.add_argument("--out", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("path")
    args = p.parse_args(argv)
    if args.cmd == "run":
        chunks = months(args.start, args.end)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            parts = list(pool.map(chunk, chunks))
        Path(args.out).write_text(json.dumps({"variants": VARIANTS, "stop_atr": STOP_ATR,
                                              "target_r": TARGET_R, "chunks": parts}))
        print(f"{sum(len(c['records']) for c in parts)} signals in {time.time() - t0:.0f} s -> {args.out}",
              file=sys.stderr)
    else:
        print(json.dumps(report(json.loads(Path(args.path).read_text())), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
