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

**Gaps.** When a bar OPENS through the stop — the market reopened, or jumped,
beyond it — no trade happened at the stop price. The stop exits at the
first executable price: that bar's open at the bid (long) / ask (short),
minus slippage. Candles are mid prices, so the bid/ask is the open less/plus
half the spread from the trade's cost figure (real bid/ask when one was
recorded, else the labelled assumption). It is never better than the stop
and can be much worse; a target gapped through still fills at the target
(the conservative side). Before this rule a gapped stop was booked at the
stop price, which overstated every trade held across a gap.

``breakeven_at`` exists for challenger accounting only (experiment E1,
``experiments.py``) and is never set for the live strategy. When set, once a
closed bar has carried the bid (long) / ask (short) ``breakeven_at`` x planned
risk beyond the fill, the stop moves to the fill price from the NEXT bar; an
exit there is a stop order (``breakeven``: the fill minus slippage). A
resting order's fill bar cannot arm it, for the same reason it cannot take
profit. Left unset, nothing in this function behaves differently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence

from costs import NO_COSTS, Costs
from market_data import Candle

BAR = timedelta(minutes=15)

FINAL = ("tp", "sl", "breakeven", "timeout", "not_filled", "no_data", "invalid")


@dataclass(frozen=True)
class TradeResult:
    basis: str                      # "gross" or "net"
    status: str                     # tp / sl / breakeven / timeout / not_filled / no_data / invalid / open
    r: Optional[float] = None       # realised R; None unless tp / sl / breakeven / timeout
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
    breakeven_at: Optional[float] = None,
) -> TradeResult:
    """Account one tracked signal. Pure: same inputs, same result.

    ``candles`` are M15 mid-price bars, oldest first. ``costs=NO_COSTS``
    gives the gross result; anything else is the net result.
    ``breakeven_at`` (in R) is for challenger accounting only — see the
    module docstring.
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
    level, armed = stop, False      # the working stop; moves to the fill once breakeven arms
    for i in range(fill_index, len(after)):
        c = after[i]
        if c.timestamp >= filled_at + horizon:
            return timeout(last_close)
        if direction == "long":
            stop_hit, tp_hit = c.low <= level + half, c.high >= target + half
        else:
            stop_hit, tp_hit = c.high >= level - half, c.low <= target - half
        resting_fill_bar = i == fill_index and not market
        # A bar that OPENED through the working stop (a weekend reopen, a
        # news jump): the stop order became a market order at the first price
        # that traded — the open, at the bid (long) / ask (short), plus
        # slippage. The stop price itself never traded, so it is not a fill.
        # On a resting order's fill bar the open came BEFORE the fill, so
        # that bar is judged the normal way.
        opened_through = not resting_fill_bar and (
            c.open - half <= level if direction == "long" else c.open + half >= level)
        if opened_through:
            note = ("market opened through the stop: filled at the open "
                    f"{'bid' if direction == 'long' else 'ask'} plus slippage, not at the stop")
            exit_price = c.open - sign * (half + slip)
            if armed:
                return closed_out("breakeven", exit_price, c.timestamp,
                                  f"stop moved to the fill after +{breakeven_at:g}R; {note}")
            return closed_out("sl", exit_price, c.timestamp, note)
        if stop_hit:
            note = "stop and target in the same bar — counted as the stop" if tp_hit else ""
            if armed:
                return closed_out("breakeven", level - sign * slip, c.timestamp,
                                  f"stop moved to the fill after +{breakeven_at:g}R. {note}".strip())
            return closed_out("sl", stop - sign * slip, c.timestamp, note)
        if tp_hit and not resting_fill_bar:
            return closed_out("tp", target, c.timestamp)
        if breakeven_at is not None and not armed and not resting_fill_bar:
            # Bid (long) / ask (short) beyond the fill on this CLOSED bar;
            # the moved stop applies from the next bar only.
            moved = (c.high - half) - fill_price if direction == "long" else fill_price - (c.low + half)
            if moved >= breakeven_at * risk:
                level, armed = fill_price, True
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
