"""Why did production favour BUY in a selloff? Factor-by-factor evidence.

Replays production's decision (``research/production_replay.py``) for every
recorded XAUUSD signal and records, per signal, every vote behind the
direction (the H1 scoring factors) and every validation check behind the
adjustment, plus the signal's outcome. Read-only on the database copy.

    python research/diagnose_bias.py --db <copy of the bot's db> --out diagnosis.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from costs import Costs  # noqa: E402
from market_context import MarketContextBuilder  # noqa: E402
from models import SignalAnalysis  # noqa: E402
from news_filter import NewsFilter  # noqa: E402
from trade_accounting import account  # noqa: E402
from trade_decision import decide_deterministically  # noqa: E402
from trade_quality import effective_entry  # noqa: E402

import strategy_variants  # noqa: E402
from production_replay import Bars, ReplayProvider, service  # noqa: E402

NET = Costs(0.30, 0.10, "assumed")


def context_facts(ctx, trace) -> dict:
    sc, val = ctx.scoring, ctx.validation
    return {
        "verdict": trace.verdict.value, "reason": trace.reason,
        "adjusted": trace.adjusted_score, "base": trace.deterministic_score,
        "bonus": trace.validation_bonus, "penalty": trace.validation_penalty,
        "net_score": sc.net_score if sc else None, "confidence": sc.confidence if sc else None,
        "factors": {f.name: f.contribution for f in sc.breakdown} if sc else {},
        "checks": {c.name: [c.severity.value, round(c.contribution, 2)] for c in val.checks} if val else {},
        "trends": {tf: a.trend for tf, a in ctx.timeframes.items()},
        "ema_alignment": {tf: a.ema_alignment for tf, a in ctx.timeframes.items()},
        "risk_rejections": [c.name for c in ctx.risk.rejections] if ctx.risk else [],
        "session": ctx.session.label if ctx.session else None,
    }


async def run(db: str):
    bars = Bars()
    provider = ReplayProvider(bars)
    builder = MarketContextBuilder(service(provider), news_filter=NewsFilter())
    clean_builder = MarketContextBuilder(service(provider), news_filter=NewsFilter(), clean_candles=True)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    live_price = {r["message_id"]: r["price_at_decision"]
                  for r in con.execute("SELECT message_id, price_at_decision FROM signal_outcomes")}
    rows = con.execute("SELECT * FROM analyses WHERE is_signal = 1 AND symbol = 'XAUUSD' "
                       "AND recorded_at >= '2026-08-20' ORDER BY id").fetchall()
    out = []
    m15 = bars.bars[__import__("market_data").Timeframe.M15]
    for row in rows:
        at = datetime.fromisoformat(row["recorded_at"])
        setup = SignalAnalysis.model_validate(json.loads(row["analysis_json"])).setup
        price = live_price.get(row["message_id"]) or bars.last_close(at)
        provider.at, provider.price = at, price
        ctx = await builder.build("XAUUSD", setup, now=at)
        trace = decide_deterministically(ctx)
        variants = {v: strategy_variants.decide(ctx, v).verdict.value
                    for v in strategy_variants.VARIANTS if v != "clean_data"}
        variants["clean_data"] = decide_deterministically(
            await clean_builder.build("XAUUSD", setup, now=at)).verdict.value
        outcome = None
        if setup.direction and setup.stop_loss is not None and setup.take_profits:
            r = account({"direction": setup.direction, "order_type": setup.order_type,
                         "price_at_decision": price, "entry": effective_entry(setup, price),
                         "stop_loss": setup.stop_loss,
                         "take_profits_json": json.dumps(setup.take_profits),
                         "recorded_at": at.isoformat()},
                        [c for c in m15 if at - timedelta(minutes=15) <= c.timestamp <= at + timedelta(days=6)],
                        at + timedelta(days=6), costs=NET)
            outcome = {"status": r.status, "r": r.r}
        out.append({"analysis_id": row["id"], "at": at.isoformat(), "direction": setup.direction,
                    "price": price, "outcome": outcome, "variants": variants,
                    **context_facts(ctx, trace)})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = asyncio.run(run(args.db))
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"diagnosed {len(out)} signals -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
