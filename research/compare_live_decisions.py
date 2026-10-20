"""The live ENTER/WAIT/SKIP decision of a code tree, on fixed inputs.

Run once with the production tree (``git archive 197b2a0``) and once with a
branch tree, then compare the two outputs: every verdict and every number
behind it must be identical. Only that tree's own modules are imported (this
script imports nothing else from the repository), the market data are the
research bars served as the live provider would have at each moment
(forming bar rebuilt from M5, weekend stubs included), and the decision
clock is frozen at each signal's time.

    python research/compare_live_decisions.py --root <tree> --db <copy of signals.db> \\
        --standard 1500 --out decisions.json
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import sqlite3
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

UTC = timezone.utc
DATA = Path(__file__).resolve().parent.parent / "data" / "research"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--standard", type=int, default=1500, help="standardized time points (x2 directions)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    sys.path.insert(0, str(Path(args.root).resolve()))

    import indicators
    import market_context
    from market_context import MarketContextBuilder
    from market_data import Candle, CandleSeries, MarketDataService, Quote, Timeframe
    from models import SignalAnalysis, TradeSetup
    from news_filter import NewsFilter
    from trade_decision import decide_deterministically

    files = {Timeframe.M5: "5min", Timeframe.M15: "15min", Timeframe.H1: "1h",
             Timeframe.H4: "4h", Timeframe.D1: "1day"}
    dur = {Timeframe.M5: timedelta(minutes=5), Timeframe.M15: timedelta(minutes=15),
           Timeframe.H1: timedelta(hours=1), Timeframe.H4: timedelta(hours=4), Timeframe.D1: timedelta(days=1)}

    def ny_offset(ts):
        year = ts.year
        def nth_sunday(month, n):
            first = date(year, month, 1)
            return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))
        start = datetime.combine(nth_sunday(3, 2), time(), UTC) + timedelta(hours=7)
        end = datetime.combine(nth_sunday(11, 1), time(), UTC) + timedelta(hours=6)
        return timedelta(hours=-4) if start <= ts < end else timedelta(hours=-5)

    def d1_start(day):
        local = datetime.combine(day - timedelta(days=1), time(17, 0))
        return (local - ny_offset(local.replace(tzinfo=UTC) + timedelta(hours=5))).replace(tzinfo=UTC)

    bars, starts, ends = {}, {}, {}
    for tf, name in files.items():
        raw = json.loads((DATA / f"xauusd_{name}.json").read_text())["bars"]
        bars[tf] = tuple(Candle(datetime.fromisoformat(t), o, h, lo, c) for t, o, h, lo, c in raw)
        st = [d1_start(b.timestamp.date()) if tf is Timeframe.D1 else b.timestamp for b in bars[tf]]
        starts[tf] = [s.timestamp() for s in st]
        ends[tf] = [(s + dur[tf]).timestamp() for s in st]

    def last_close(at):
        k = bisect.bisect_right(ends[Timeframe.M5], at.timestamp())
        return bars[Timeframe.M5][k - 1].close

    def window(tf, at, limit):
        t = at.timestamp()
        k = bisect.bisect_left(starts[tf], t)
        if k and ends[tf][k - 1] > t:
            lo = bisect.bisect_left(starts[Timeframe.M5], starts[tf][k - 1])
            hi = bisect.bisect_right(ends[Timeframe.M5], t)
            part = bars[Timeframe.M5][lo:hi]
            forming = (Candle(bars[tf][k - 1].timestamp, part[0].open, max(c.high for c in part),
                              min(c.low for c in part), part[-1].close),) if part else ()
            return bars[tf][max(0, k - limit):k - 1] + forming
        return bars[tf][max(0, k - limit):k]

    class Provider:
        name = "replay"
        at = price = None

        async def get_quote(self, symbol):
            return Quote(symbol=symbol, price=self.price, timestamp=self.at, provider=self.name)

        async def get_candles(self, symbol, timeframe, limit):
            return CandleSeries(symbol=symbol, timeframe=timeframe, provider=self.name,
                                candles=tuple(window(timeframe, self.at, limit)))

    class Frozen(datetime):
        current = datetime.now(UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current if tz is None else cls.current.astimezone(tz)

    market_context.datetime = Frozen
    provider = Provider()
    builder = MarketContextBuilder(MarketDataService(provider, cache_ttl=0, max_retries=0),
                                   news_filter=NewsFilter())

    inputs = []
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    live_price = {r["message_id"]: r["price_at_decision"]
                  for r in con.execute("SELECT message_id, price_at_decision FROM signal_outcomes")}
    for r in con.execute("SELECT * FROM analyses WHERE is_signal = 1 AND symbol = 'XAUUSD' "
                         "AND recorded_at >= '2026-08-20' ORDER BY id"):
        at = datetime.fromisoformat(r["recorded_at"])
        setup = SignalAnalysis.model_validate(json.loads(r["analysis_json"])).setup
        inputs.append((f"recorded:{r['id']}", at, live_price.get(r["message_id"]) or last_close(at), setup))
    first, last = datetime(2021, 1, 4, tzinfo=UTC), datetime(2026, 10, 16, tzinfo=UTC)
    step = (last - first) / args.standard
    for i in range(args.standard):
        at = (first + step * i).replace(minute=0, second=0, microsecond=0)
        if at.weekday() >= 5:
            continue
        price = last_close(at)
        atr = indicators.atr(window(Timeframe.H1, at, 300))
        if not atr:
            continue
        for d in ("long", "short"):
            s = 1.0 if d == "long" else -1.0
            inputs.append((f"standard:{at.isoformat()}:{d}", at, price, TradeSetup(
                symbol="XAUUSD", direction=d, order_type="market",
                stop_loss=round(price - s * atr, 2), take_profits=[round(price + 2 * s * atr, 2)])))

    async def go():
        out = {}
        for key, at, price, setup in inputs:
            Frozen.current = at
            provider.at, provider.price = at, price
            t = decide_deterministically(await builder.build("XAUUSD", setup))
            out[key] = [t.verdict.value, t.deterministic_score, t.validation_bonus,
                        t.validation_penalty, t.adjusted_score, t.blocked, t.reason]
        return out

    out = asyncio.run(go())
    Path(args.out).write_text(json.dumps(out))
    print(f"{Path(args.root).name}: {len(out)} decisions -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
