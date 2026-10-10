"""Realistic per-trade accounting: fills, exits, costs and realised R.

``outcome_tracker.resolve`` answers "which level did price touch first?" —
the raw outcome, kept unchanged for continuity. This module answers "what
would a trader have actually made, in R?", which is what expectancy, profit
factor and drawdown need:

* **Live fills.** A market order fills at the live price at decision time,
  not at whatever entry the signal wrote. A resting order fills when price
  trades through its entry within the fill window.
* **Nearest target first.** TP1 is the nearest target in the trade's
  direction, whatever order the signal listed them in.
* **Costs.** Candles are mid prices; bid and ask sit half a spread either
  side. A long buys at the ask and sells at the bid — so its take-profit only
  fills once the BID reaches it, and its stop triggers once the BID falls to
  it. Shorts mirror that. Fills that execute at market (market entries,
  stop-entry triggers, stop-loss exits, timeout exits) also pay slippage.
* **Timeout exits.** A trade still open at the horizon is closed at the last
  close inside the horizon, at the bid (long) or ask (short), with slippage.
* **R** is measured against the trade's planned risk: planned entry to stop.
  Gross R uses no costs — a stop is exactly -1R and a target exactly its
  reward/risk. Net R includes costs. The two are separate results with
  separate labels and are never mixed.

Bar-level conservatism, as in ``outcome_tracker``: only CLOSED bars that
opened at or after the decision are read; the stop and the target in the
same bar count as the stop; on a resting order's fill bar only the stop
counts, because its target touch may have come before the fill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence

from costs import NO_COSTS, Costs
from market_data import Candle

BAR = timedelta(minutes=15)

FINAL = ("tp", "sl", "timeout", "not_filled", "no_data", "invalid")


@dataclass(frozen=True)
class TradeResult:
    basis: str                      # "gross" or "net"
    status: str                     # tp / sl / timeout / not_filled / no_data / invalid / open
    r: Optional[float] = None       # realised R; None unless tp / sl / timeout
    planned_entry: Optional[float] = None
    planned_risk: Optional[float] = None
    target: Optional[float] = None
    entry_fill: Optional[float] = None
    exit_price: Optional[float] = None
    filled_at: Optional[datetime] = None
    exit_at: Optional[datetime] = None
    holding_minutes: Optional[float] = None
    costs: Costs = NO_COSTS
    note: str = ""

    @property
    def final(self) -> bool:
        return self.status in FINAL

    def as_row(self) -> Dict[str, Any]:
        iso = lambda d: d.isoformat() if d is not None else None  # noqa: E731
        return {
            "basis": self.basis, "status": self.status, "r": self.r,
            "planned_entry": self.planned_entry, "planned_risk": self.planned_risk,
            "target": self.target, "entry_fill": self.entry_fill, "exit_price": self.exit_price,
            "filled_at": iso(self.filled_at), "exit_at": iso(self.exit_at),
            "holding_minutes": self.holding_minutes, "cost_spread": self.costs.spread,
            "cost_slippage": self.costs.slippage, "cost_source": self.costs.source,
            "note": self.note,
        }


def nearest_target(direction: str, entry: float, targets: Sequence[float]) -> Optional[float]:
    """The first target price must reach: nearest beyond the entry."""
    sign = 1.0 if direction == "long" else -1.0
    ahead = [t for t in targets if t is not None and sign * (t - entry) > 0]
    return min(ahead, key=lambda t: sign * (t - entry)) if ahead else None


def _targets(row: Dict[str, Any]) -> list:
    raw = row.get("take_profits_json")
    try:
        listed = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        listed = []
    return listed or ([row["take_profit"]] if row.get("take_profit") is not None else [])


def account(
    row: Dict[str, Any],
    candles: Sequence[Candle],
    now: datetime,
    *,
    costs: Costs = NO_COSTS,
    fill_window: timedelta = timedelta(hours=24),
    horizon: timedelta = timedelta(days=5),
    bar: timedelta = BAR,
) -> TradeResult:
    """Account one tracked signal. Pure: same inputs, same result.

    ``candles`` are M15 mid-price bars, oldest first. ``costs=NO_COSTS``
    gives the gross result; anything else is the net result.
    """
    basis = "gross" if costs is NO_COSTS or costs.source == "none" else "net"

    def result(status: str, **kw) -> TradeResult:
        return TradeResult(basis=basis, status=status, costs=costs, **kw)

    direction = row.get("direction")
    if direction not in ("long", "short"):
        return result("invalid", note="no trade direction")
    sign = 1.0 if direction == "long" else -1.0
    half, slip = costs.spread / 2.0, costs.slippage

    market = row.get("order_type") == "market"
    live = row.get("price_at_decision")
    planned = live if market and live is not None else row.get("entry")
    stop = row.get("stop_loss")
    if planned is None or stop is None:
        return result("invalid", note="no entry or stop")
    target = nearest_target(direction, planned, _targets(row))
    risk = sign * (planned - stop)
    if target is None or risk <= 0:
        return result("invalid", planned_entry=planned, note="levels are not a valid trade "
                                                             "from the fill price")
    common = dict(planned_entry=planned, planned_risk=risk, target=target)

    decided_at = datetime.fromisoformat(row["recorded_at"])
    closed = [c for c in candles if c.timestamp + bar <= now]
    if not closed:
        return result("open", **common)
    if closed[0].timestamp > decided_at:
        return result("no_data", note="candle history does not reach back to the decision", **common)
    after = [c for c in closed if c.timestamp >= decided_at]

    # ---- fill
    if market:
        fill_index, filled_at = 0, decided_at
        fill_price = planned + sign * (half + slip)
    else:
        entry = row["entry"]
        limit_like = live is None or sign * (entry - live) <= 0
        fill_index = None
        for i, c in enumerate(after):
            if c.timestamp >= decided_at + fill_window:
                break
            if live is None:
                hit = c.low <= entry <= c.high
            elif direction == "long":
                hit = c.low <= entry - half if limit_like else c.high >= entry - half
            else:
                hit = c.high >= entry + half if limit_like else c.low <= entry + half
            if hit:
                fill_index, filled_at = i, c.timestamp
                break
        if fill_index is None:
            window_over = closed[-1].timestamp + bar >= decided_at + fill_window
            return (result("not_filled", note=f"entry not reached within {fill_window}", **common)
                    if window_over else result("open", **common))
        fill_price = entry if limit_like else entry + sign * slip

    if not after:
        return result("open", **common)

    def closed_out(status: str, exit_price: float, exit_at: datetime, note: str = "") -> TradeResult:
        r = sign * (exit_price - fill_price) / risk
        return result(status, r=round(r, 4), entry_fill=round(fill_price, 6),
                      exit_price=round(exit_price, 6), filled_at=filled_at, exit_at=exit_at,
                      holding_minutes=max(0.0, (exit_at - filled_at).total_seconds() / 60.0),
                      note=note, **common)

    def timeout(last_close: Optional[float]) -> TradeResult:
        price = (last_close if last_close is not None else fill_price) - sign * (half + slip)
        return closed_out("timeout", price, filled_at + horizon,
                          f"neither level reached within {horizon}")

    last_close: Optional[float] = None
    for i in range(fill_index, len(after)):
        c = after[i]
        if c.timestamp >= filled_at + horizon:
            return timeout(last_close)
        if direction == "long":
            stop_hit, tp_hit = c.low <= stop + half, c.high >= target + half
        else:
            stop_hit, tp_hit = c.high >= stop - half, c.low <= target - half
        if stop_hit:
            note = "stop and target in the same bar — counted as the stop" if tp_hit else ""
            return closed_out("sl", stop - sign * slip, c.timestamp, note)
        if tp_hit and not (i == fill_index and not market):
            return closed_out("tp", target, c.timestamp)
        last_close = c.close

    if closed[-1].timestamp + bar >= filled_at + horizon:
        return timeout(last_close)
    return result("open", **common)


def account_both(row: Dict[str, Any], candles: Sequence[Candle], now: datetime, *,
                 costs: Optional[Costs], **kw) -> Dict[str, TradeResult]:
    """Gross always; net only when a cost figure exists for the trade."""
    out = {"gross": account(row, candles, now, costs=NO_COSTS, **kw)}
    if costs is not None:
        out["net"] = account(row, candles, now, costs=costs, **kw)
    return out
