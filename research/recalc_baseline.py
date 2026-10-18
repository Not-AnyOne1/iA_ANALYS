"""Recalculate the current strategy's baseline with corrected data/accounting.

Run the SAME script under two code trees — the pre-fix commit and the fix —
on the same provider bars and a read-only copy of the bot's database, then
compare. Nothing is written to any database.

    python research/recalc_baseline.py run --root <code tree> --db <db copy> --out old.json
    python research/recalc_baseline.py run --root .            --db <db copy> --out new.json
    python research/recalc_baseline.py compare old.json new.json --db <db copy>

``run`` computes, with the tree's own code:
  A. every recorded live signal: raw outcome, gross/net trade accounting, and
     E1's two arms for signals since E1's start (the tracker's exact inputs:
     M15 bars through the tree's ``clean_candle_series``, the recorded costs
     or the labelled backfill assumption);
  B. the replay of every Telegram signal through the CURRENT decision path
     (closed bars, the tree's data cleaning) — verdict, confluence, gross/net
     R and E1's arms under the assumed costs.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DUR = {"M15": timedelta(minutes=15), "H1": timedelta(hours=1), "H4": timedelta(hours=4),
       "D1": timedelta(days=1)}
FILES = {"M15": "15min", "H1": "1h", "H4": "4h", "D1": "1day"}


def _run(args) -> int:
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    logging.disable(logging.WARNING)
    import outcome_tracker as ot
    from costs import Costs
    from experiments import E1, account_arms
    from market_context import MarketContextBuilder
    from market_data import Candle, CandleSeries, MarketDataService, Quote, Timeframe, clean_candle_series
    from models import SignalAnalysis
    from trade_accounting import account_both
    from trade_decision import decide_deterministically
    from trade_quality import effective_entry

    raw = {tf: [Candle(timestamp=datetime.fromisoformat(t), open=o, high=h, low=lo, close=c, volume=None)
                for t, o, h, lo, c in json.loads((Path(args.data) / f"xauusd_{f}.json").read_text())["bars"]]
           for tf, f in FILES.items()}
    for tf in raw:
        raw[tf].sort(key=lambda c: c.timestamp)
    m15 = clean_candle_series(CandleSeries("XAUUSD", Timeframe.M15, "twelve_data", tuple(raw["M15"]))).candles
    now = raw["M15"][-1].timestamp + DUR["M15"]

    def arms_out(arms):
        return {k: {"status": v.status, "r": v.r} for k, v in arms.items()}

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    e1_start = None
    try:
        hit = con.execute("SELECT started_at FROM experiments WHERE name = 'E1'").fetchone()
        e1_start = hit[0] if hit else None
    except sqlite3.OperationalError:
        pass
    live = []
    for row in (dict(r) for r in con.execute("SELECT * FROM signal_outcomes ORDER BY id")):
        if row["status"] == "invalid":
            continue
        costs = (Costs(row["cost_spread"], row.get("cost_slippage") or 0.0, row.get("cost_source") or "assumed")
                 if row.get("cost_spread") is not None else Costs(0.30, 0.10, "assumed (backfill)"))
        acct = account_both(row, m15, now, costs=costs)
        outcome = ot.resolve(row, m15, now)
        item = {"id": row["id"], "recorded_at": row["recorded_at"], "decision": row["decision"],
                "direction": row["direction"], "raw_status": outcome.status if outcome else "open",
                "gross": {"status": acct["gross"].status, "r": acct["gross"].r},
                "net": {"status": acct["net"].status, "r": acct["net"].r}}
        if e1_start and row["recorded_at"] >= e1_start:
            item["e1"] = arms_out(account_arms(E1, row, m15, now, costs=costs))
        live.append(item)

    class ClosedBars:
        name = "replay"

        def __init__(self):
            self.opens = {tf: [c.timestamp for c in bars] for tf, bars in raw.items()}
            self.at = self.price = None

        async def get_quote(self, symbol):
            return Quote(symbol=symbol, price=self.price, timestamp=self.at, provider=self.name)

        async def get_candles(self, symbol, timeframe, limit):
            tf = timeframe.value
            end = bisect.bisect_right(self.opens[tf], self.at - DUR[tf])
            return CandleSeries(symbol=symbol, timeframe=timeframe, provider=self.name,
                                candles=tuple(raw[tf][max(0, end - limit):end]))

    live_price = {r["message_id"]: r["price_at_decision"]
                  for r in con.execute("SELECT message_id, price_at_decision FROM signal_outcomes")}
    signals = con.execute("SELECT * FROM analyses WHERE is_signal = 1 AND symbol = 'XAUUSD' "
                          "AND recorded_at >= '2026-08-20' ORDER BY id").fetchall()
    provider = ClosedBars()
    builder = MarketContextBuilder(MarketDataService(provider, cache_ttl=0))
    assumed = Costs(0.30, 0.10, "assumed")
    replay = []

    async def go():
        for s in signals:
            at = datetime.fromisoformat(s["recorded_at"])
            setup = SignalAnalysis.model_validate(json.loads(s["analysis_json"])).setup
            price = live_price.get(s["message_id"])
            if price is None:
                closed = [c for c in m15 if c.timestamp + DUR["M15"] <= at]
                price = closed[-1].close
            provider.at, provider.price = at, price
            ctx = await builder.build("XAUUSD", setup, now=at)
            t = decide_deterministically(ctx)
            direction = getattr(setup.direction, "value", setup.direction)
            tps = list(setup.take_profits or [])
            row = {"recorded_at": at.isoformat(), "direction": direction,
                   "order_type": getattr(setup.order_type, "value", setup.order_type),
                   "entry": effective_entry(setup, price), "stop_loss": setup.stop_loss,
                   "take_profit": tps[0] if tps else None, "take_profits_json": json.dumps(tps),
                   "price_at_decision": price}
            acct = account_both(row, m15, now, costs=assumed)
            replay.append({"analysis_id": s["id"], "recorded_at": s["recorded_at"], "direction": direction,
                           "entry": row["entry"], "stop_loss": row["stop_loss"],
                           "take_profit": row["take_profit"], "verdict": t.verdict.value,
                           "confluence": t.confluence_score,
                           "gross": {"status": acct["gross"].status, "r": acct["gross"].r},
                           "net": {"status": acct["net"].status, "r": acct["net"].r},
                           "e1": arms_out(account_arms(E1, row, m15, now, costs=assumed))})
    asyncio.run(go())
    Path(args.out).write_text(json.dumps({"root": str(root), "data_end": now.isoformat(),
                                          "m15_bars_after_cleaning": len(m15),
                                          "live": live, "replay": replay}, indent=1))
    print(f"{root.name}: live {len(live)} replay {len(replay)} m15 {len(m15)} -> {args.out}")
    return 0


def _stats(rs):
    rs = [r for r in rs if r is not None]
    if not rs:
        return {"n": 0, "mean": None, "win": None, "pf": None, "total": 0.0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    return {"n": len(rs), "mean": sum(rs) / len(rs), "win": len(wins) / len(rs),
            "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None, "total": sum(rs)}


def _fmt(s):
    if not s["n"]:
        return "n=0"
    pf = f"{s['pf']:.2f}" if s["pf"] is not None else "—"
    return f"n={s['n']:>3} win {s['win']:.0%} mean {s['mean']:+.3f}R total {s['total']:+.2f}R PF {pf}"


def _compare(args) -> int:
    sys.path.insert(0, str(HERE))
    import performance as pf
    old, new = (json.loads(Path(p).read_text()) for p in (args.old, args.new))
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    stored = {r["outcome_id"]: {} for r in con.execute("SELECT outcome_id FROM trade_accounting")}
    for r in con.execute("SELECT outcome_id, basis, status, r FROM trade_accounting"):
        stored[r["outcome_id"]][r["basis"]] = {"status": r["status"], "r": r["r"]}
    meta = {r["id"]: dict(r) for r in con.execute("SELECT * FROM signal_outcomes")}
    dups = pf.duplicate_ids(list(meta.values()))

    print("A. RECORDED LIVE SIGNALS (decisions as the bot made them; duplicates counted once)")
    o, n = {x["id"]: x for x in old["live"]}, {x["id"]: x for x in new["live"]}
    changed = [i for i in o if (o[i]["net"], o[i]["gross"], o[i]["raw_status"]) !=
               (n[i]["net"], n[i]["gross"], n[i]["raw_status"])]
    stored_vs_old = [i for i in o if i in stored and stored[i].get("net", {}).get("r") != o[i]["net"]["r"]]
    print(f"   signals {len(o)}; result changed by the fixes: {len(changed)} {changed or ''}")
    print(f"   stored row differs from the pre-fix recompute (more data since the bot stopped, or "
          f"provider revisions — not the fixes): {len(stored_vs_old)}")
    for i in changed:
        print(f"     id {i} {o[i]['recorded_at'][:16]} {o[i]['direction']}: raw {o[i]['raw_status']}->"
              f"{n[i]['raw_status']}, net {o[i]['net']}->{n[i]['net']}")
    for label, src in (("stored (as recorded)", None), ("pre-fix recompute", o), ("corrected", n)):
        for verdict in ("enter", "skip", None):
            ids = [i for i in o if i not in dups and (verdict is None or meta[i]["decision"] == verdict)]
            rs = [(stored.get(i, {}).get("net", {}).get("r") if src is None else src[i]["net"]["r"]) for i in ids]
            print(f"   {label:<22} {verdict or 'every signal':<12} net: {_fmt(_stats(rs))}")
    print("\nB. REPLAY OF THE TELEGRAM SIGNALS THROUGH THE CURRENT DECISION PATH (assumed costs)")
    ro, rn = old["replay"], new["replay"]
    rows = [{"id": i, "symbol": "XAUUSD", "direction": x["direction"], "entry": x["entry"],
             "stop_loss": x["stop_loss"], "take_profit": x["take_profit"], "recorded_at": x["recorded_at"]}
            for i, x in enumerate(ro)]
    rdups = pf.duplicate_ids(rows)
    flips = [(x["recorded_at"][:16], x["direction"], x["verdict"], y["verdict"], x["confluence"], y["confluence"])
             for x, y in zip(ro, rn) if x["verdict"] != y["verdict"]]
    conf_moves = sum(1 for x, y in zip(ro, rn) if x["confluence"] != y["confluence"])
    r_moves = sum(1 for x, y in zip(ro, rn) if x["net"]["r"] != y["net"]["r"])
    print(f"   signals {len(ro)} ({len(rdups)} duplicates counted once below); verdict changed: {len(flips)}; "
          f"confluence changed: {conf_moves}; net R changed: {r_moves}")
    for f in flips:
        print(f"     {f[0]} {f[1]}: {f[2]} -> {f[3]} (confluence {f[4]} -> {f[5]})")
    for label, src in (("pre-fix", ro), ("corrected", rn)):
        for verdict in ("enter", "skip", None):
            rs = [x["net"]["r"] for i, x in enumerate(src)
                  if i not in rdups and (verdict is None or x["verdict"] == verdict)]
            print(f"   {label:<10} {verdict or 'every signal':<12} net: {_fmt(_stats(rs))}")
    print("\nC. E1 (baseline vs breakeven arm, net, per signal; duplicates once)")
    for label, src in (("pre-fix", ro), ("corrected", rn)):
        pairs = [(x["e1"]["baseline/net"]["r"], x["e1"]["challenger/net"]["r"]) for i, x in enumerate(src)
                 if i not in rdups and x["e1"]["baseline/net"]["r"] is not None
                 and x["e1"]["challenger/net"]["r"] is not None]
        d = [b - a for a, b in pairs]
        print(f"   replay {label:<10} n={len(d)} baseline {sum(a for a, _ in pairs) / len(pairs):+.3f}R "
              f"challenger {sum(b for _, b in pairs) / len(pairs):+.3f}R difference {sum(d) / len(d):+.3f}R")
    live_e1 = [(o[i]["e1"], n[i]["e1"]) for i in o if "e1" in o[i]]
    same = sum(1 for a, b in live_e1 if a == b)
    print(f"   live E1 signals: {len(live_e1)}, results identical before/after: {same}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--db", required=True)
    r.add_argument("--data", default=str(HERE / "data" / "research"))
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    c.add_argument("--db", required=True)
    args = p.parse_args(argv)
    return _run(args) if args.cmd == "run" else _compare(args)


if __name__ == "__main__":
    sys.exit(main())
