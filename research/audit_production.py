"""Replay a code tree's decision path over the recorded Telegram signals and
record every number behind each verdict (audit of the deployed strategy).

Run it under the tree to audit — e.g. the production commit extracted with
``git archive`` — on the research bars and a read-only copy of the database:

    python research/audit_production.py --root <tree> --db <db copy> --out audit.json

The tree's own MarketDataService / MarketContextBuilder / decision code is
used unchanged; bars are served closed-only, as of each signal's time, in the
raw form the provider serves them. Nothing is written to any database.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import inspect
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

DUR = {"M15": timedelta(minutes=15), "H1": timedelta(hours=1), "H4": timedelta(hours=4),
       "D1": timedelta(days=1)}
FILES = {"M15": "15min", "H1": "1h", "H4": "4h", "D1": "1day"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data" / "research"))
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    sys.path.insert(0, str(Path(args.root).resolve()))
    logging.disable(logging.WARNING)
    from market_context import MarketContextBuilder
    from market_data import Candle, CandleSeries, MarketDataService, Quote
    from models import SignalAnalysis
    import trade_decision as td

    raw = {tf: sorted((Candle(timestamp=datetime.fromisoformat(t), open=o, high=h, low=lo, close=c, volume=None)
                       for t, o, h, lo, c in json.loads((Path(args.data) / f"xauusd_{f}.json").read_text())["bars"]),
                      key=lambda c: c.timestamp) for tf, f in FILES.items()}
    opens = {tf: [c.timestamp for c in v] for tf, v in raw.items()}

    class ClosedBars:
        name = "replay"

        def __init__(self):
            self.at = self.price = None

        async def get_quote(self, symbol):
            return Quote(symbol=symbol, price=self.price, timestamp=self.at, provider=self.name)

        async def get_candles(self, symbol, timeframe, limit):
            tf = timeframe.value
            end = bisect.bisect_right(opens[tf], self.at - DUR[tf])
            return CandleSeries(symbol=symbol, timeframe=timeframe, provider=self.name,
                                candles=tuple(raw[tf][max(0, end - limit):end]))

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    live_price = {r["message_id"]: r["price_at_decision"]
                  for r in con.execute("SELECT message_id, price_at_decision FROM signal_outcomes")}
    signals = con.execute("SELECT * FROM analyses WHERE is_signal = 1 AND symbol = 'XAUUSD' "
                          "AND recorded_at >= '2026-08-20' ORDER BY id").fetchall()
    provider = ClosedBars()
    builder = MarketContextBuilder(MarketDataService(provider, cache_ttl=0))
    takes_now = "now" in inspect.signature(MarketContextBuilder.build).parameters
    out = []

    async def go():
        for s in signals:
            at = datetime.fromisoformat(s["recorded_at"])
            setup = SignalAnalysis.model_validate(json.loads(s["analysis_json"])).setup
            price = live_price.get(s["message_id"])
            if price is None:
                closed = [c for c in raw["M15"] if c.timestamp + DUR["M15"] <= at]
                price = closed[-1].close
            provider.at, provider.price = at, price
            ctx = await (builder.build("XAUUSD", setup, now=at) if takes_now else builder.build("XAUUSD", setup))
            t = td.decide_deterministically(ctx)
            sc, risk = ctx.scoring, ctx.risk
            out.append({
                "analysis_id": s["id"], "recorded_at": s["recorded_at"],
                "direction": getattr(setup.direction, "value", setup.direction),
                "verdict": t.verdict.value, "reason": t.reason, "blocked": t.blocked,
                "engine_confidence": sc.confidence if sc else None,
                "engine_net_score": sc.net_score if sc else None,
                "engine_favours": (None if not sc or sc.net_score == 0 else ("long" if sc.net_score > 0 else "short")),
                "directional_base": t.deterministic_score, "bonus": t.validation_bonus,
                "penalty": t.validation_penalty, "adjusted": t.adjusted_score,
                "displayed_confidence": t.adjusted_score if t.adjusted_score is not None else 0,
                "risk_approved": risk.approved if risk else None,
                "risk_rejections": [c.name for c in risk.rejections] if risk else [],
                "d1_present": "D1" in ctx.timeframes, "warnings": list(ctx.warnings)[:6],
                "live_verdict": s["decision_verdict"], "live_confidence": s["decision_confidence"],
            })
    asyncio.run(go())
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"{Path(args.root).name}: audited {len(out)} signals -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
