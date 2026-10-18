"""Gap-through-stop accounting on the provider's REAL reopen gaps.

For every weekend reopen in the M15 history (weekend stubs removed by the
corrected filter) whose open moved at least $5 from Friday's close, a trade
is opened at Friday's last bar with its stop halfway into the gap on the
losing side — a long before a down-gap, a short before an up-gap — and
accounted with ``trade_accounting.account``. Each must exit AT the Sunday
open's bid (long) / ask (short) plus slippage, never at the stop. E1's
breakeven arm is checked the same way with a stop already moved to the fill.

Usage: python research/gap_verification.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from costs import Costs  # noqa: E402
from market_data import Candle, CandleSeries, Timeframe, clean_candle_series  # noqa: E402
from trade_accounting import account  # noqa: E402

M15 = timedelta(minutes=15)
COSTS = Costs(0.30, 0.10, "assumed")


def main() -> int:
    raw = json.loads((ROOT / "data" / "research" / "xauusd_15min.json").read_text())["bars"]
    candles = clean_candle_series(CandleSeries("XAUUSD", Timeframe.M15, "twelve_data", tuple(
        Candle(timestamp=datetime.fromisoformat(t), open=o, high=h, low=lo, close=c, volume=None)
        for t, o, h, lo, c in raw))).candles
    checked = {"long": [0, 0], "short": [0, 0], "e1_long": [0, 0], "e1_short": [0, 0]}
    skipped: dict = {}
    worst = []
    for a, b in zip(candles, candles[1:]):
        if b.timestamp - a.timestamp < timedelta(hours=40):
            continue                                   # weekends only
        gap = b.open - a.close
        if abs(gap) < 5:
            continue
        direction = "long" if gap < 0 else "short"
        s = 1 if direction == "long" else -1
        entry = a.open
        stop = a.close + gap / 2                      # halfway into the gap, on the losing side
        target = entry + s * 10 * abs(entry - stop)
        window = [c for c in candles if a.timestamp - M15 <= c.timestamp <= b.timestamp + 4 * M15]
        row = {"recorded_at": (a.timestamp - timedelta(minutes=1)).isoformat(), "direction": direction,
               "order_type": "market", "entry": entry, "stop_loss": stop, "take_profit": target,
               "take_profits_json": json.dumps([target]), "price_at_decision": entry}
        now = b.timestamp + 5 * M15
        expected = b.open - s * (COSTS.spread / 2 + COSTS.slippage)
        r = account(row, window, now, costs=COSTS)
        if r.status != "sl" or r.exit_at != b.timestamp:
            # Not a gap case after all (e.g. Friday's last bar already traded through
            # the constructed stop, or no bar covers the decision): counted, not judged.
            skipped[r.status if r.status != "sl" else "stopped_before_the_gap"] = \
                skipped.get(r.status if r.status != "sl" else "stopped_before_the_gap", 0) + 1
            continue
        ok = abs(r.exit_price - expected) < 1e-6
        checked[direction][0] += 1
        checked[direction][1] += ok
        old_style = stop - s * COSTS.slippage          # what the old rule booked
        worst.append((abs(old_style - r.exit_price) / abs(entry - stop), a.timestamp.date(), direction))
        # E1: a stop already at the fill (armed) — the breakeven exit must also be the open.
        fill = entry + s * (COSTS.spread / 2 + COSTS.slippage)
        if s * (b.open - s * COSTS.spread / 2 - fill) < 0:          # the reopen is through the fill
            armed_row = dict(row)
            e1 = account(armed_row, window, now, costs=COSTS, breakeven_at=0.0)
            key = f"e1_{direction}"
            checked[key][0] += 1
            checked[key][1] += (e1.status in ("breakeven", "sl")
                                and abs(e1.exit_price - expected) < 1e-6)
    for k, (n, ok) in checked.items():
        print(f"{k:>9}: {n} real reopen gaps >= $5, exit at the open bid/ask + slippage in {ok}")
    print("constructed trades that were not gap cases (not judged):", skipped or "none")
    worst.sort(reverse=True)
    print("largest corrections vs the old stop-price booking, in R of the trade's risk:",
          [(f"{w:.2f}R", str(d), dr) for w, d, dr in worst[:5]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
