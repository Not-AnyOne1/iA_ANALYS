"""Outcome resolution for the trades the shadow analyst recorded.

Every recorded analysis lists the trades to track, fixed at decision time:
the original signal (whatever the verdict — SKIP, WAIT and NO TRADE
included, so false negatives can be counted), defined as the live outcome
accounting defines it, and every READY proposal.
``trade_accounting.account`` resolves each against closed M15 bars that
opened after the decision: market fill at the live price, stop first when a
bar touches both levels, a bar that opens through the stop exits at its open,
timeout after five days. Gross (no costs) and net (assumed spread and
slippage, labelled) are both written. Only final outcomes are written, once.

On the live bot the resolver wakes every ``interval`` (default 30 min) and
makes at most one M15 request per wake-up, and none when nothing is pending.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Sequence

from costs import Costs
from market_data import Candle, MarketDataError, MarketDataService, Timeframe
from market_hours import closure_allowance
from trade_accounting import account_both

from analyst.data import drop_closure_stubs

log = logging.getLogger(__name__)

HORIZON = timedelta(days=5)
FILL_WINDOW = timedelta(hours=24)
BAR = timedelta(minutes=15)
MAX_BARS = 5000


def accounting_row(trade: Dict[str, Any]) -> Dict[str, Any]:
    return {"direction": trade["direction"], "order_type": trade["order_type"],
            "price_at_decision": trade.get("price_at_decision"), "entry": trade.get("entry"),
            "stop_loss": trade.get("stop_loss"),
            "take_profits_json": json.dumps(trade.get("take_profits") or []),
            "recorded_at": trade["recorded_at"]}


def resolve_trade(trade: Dict[str, Any], candles: Sequence[Candle], now: datetime,
                  costs: Optional[Costs]):
    """Final results only. A ``no_data`` result for a trade still inside its
    horizon means the fetched history was too short: retried later."""
    results = account_both(accounting_row(trade), candles, now, costs=costs,
                           horizon=HORIZON, fill_window=FILL_WINDOW)
    recorded = datetime.fromisoformat(trade["recorded_at"])
    young = now - recorded < HORIZON + FILL_WINDOW
    return {basis: r for basis, r in results.items()
            if r.final and not (r.status == "no_data" and young)}


class OutcomeResolver:
    def __init__(self, store, market_data: MarketDataService, *, costs: Optional[Costs],
                 symbol: str = "XAUUSD", interval: float = 1800.0) -> None:
        self._store, self._market_data, self._costs = store, market_data, costs
        self._symbol, self._interval = symbol, interval

    async def resolve_once(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(timezone.utc)
        pending = await self._store.pending(now - HORIZON - FILL_WINDOW - timedelta(days=2))
        due = [t for t in pending if datetime.fromisoformat(t["recorded_at"]) <= now - BAR]
        if not due:
            return 0
        oldest = min(datetime.fromisoformat(t["recorded_at"]) for t in due)
        bars = math.ceil((now - oldest) / BAR) + 4
        limit = min(MAX_BARS, bars + closure_allowance(BAR, bars))
        try:
            series = await self._market_data.get_candles(self._symbol, Timeframe.M15, limit)
        except MarketDataError as exc:
            log.warning("analyst_outcomes_unavailable error=%s", exc)
            return 0
        candles = drop_closure_stubs(series.candles, Timeframe.M15, self._symbol)
        written = 0
        for trade in due:
            final = resolve_trade(trade, candles, now, self._costs)
            if final:
                written += await self._store.record_outcome(trade["analysis_id"], trade["subject"],
                                                            final, now)
        if written:
            log.info("analyst_outcomes_recorded rows=%d pending=%d", written, len(due))
        return written

    async def run_forever(self) -> None:
        while True:
            try:
                await self.resolve_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - outcome tracking must never stop the bot
                log.exception("analyst_outcomes_failed")
            await asyncio.sleep(self._interval)
