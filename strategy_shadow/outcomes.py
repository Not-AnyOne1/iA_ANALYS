"""Outcome of every recorded signal — the same trade whatever the verdicts.

Each signal's hypothetical trade is fixed at decision time (``trade_json``):
the original Entry, SL and TP, never modified, under the P1 definition:

* entry at the stated zone's midpoint (the live price when none is stated),
  the stated order type; a market order fills at the live decision price;
* a pending (limit/stop/unknown with an entry) order must trade through its
  entry within 24 h, else ``not_filled`` — neither a win nor a loss;
* the nearest take-profit beyond the fill; the stop counts first when one
  bar touches both; a bar that opens through the stop exits at its open;
* still open after 5 days: ``timeout`` at the last close;
* gross R (no costs) and net R (assumed spread and slippage, labelled).

``trade_accounting.account`` does all of that unchanged. Only final
outcomes are written: an open trade stays pending, a trade whose bars are
missing is retried until it is older than the horizon plus the fill window,
and only then recorded ``no_data``. Bars come from the UTC-labelled
provider, closed and stub-free.

One wake-up every ``interval`` (30 min) makes at most one M15 request, and
none when nothing is pending.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from costs import Costs
from market_data import MarketDataError, MarketDataService, Timeframe
from market_hours import closure_allowance

from analyst.data import drop_closure_stubs
from analyst.tracker import BAR, resolve_trade

log = logging.getLogger(__name__)


class StrategyOutcomeResolver:
    def __init__(self, store, market_data: MarketDataService, *, costs: Optional[Costs],
                 symbol: str = "XAUUSD", interval: float = 1800.0) -> None:
        self._store, self._market_data, self._costs = store, market_data, costs
        self._symbol, self._interval = symbol, interval
        self.requests = 0

    async def resolve_once(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(timezone.utc)
        pending = await self._store.pending()
        due = [t for t in pending if datetime.fromisoformat(t["recorded_at"]) <= now - BAR]
        if not due:
            return 0
        oldest = min(datetime.fromisoformat(t["recorded_at"]) for t in due)
        bars = math.ceil((now - oldest) / BAR) + 4
        limit = min(5000, bars + closure_allowance(BAR, bars))
        try:
            self.requests += 1
            series = await self._market_data.get_candles(self._symbol, Timeframe.M15, limit)
        except MarketDataError as exc:
            log.warning("strategy_shadow_outcomes_unavailable error=%s", exc)
            return 0
        candles = drop_closure_stubs(series.candles, Timeframe.M15, self._symbol)
        written = 0
        for trade in due:
            final = resolve_trade(trade, candles, now, self._costs)
            if final:
                written += await self._store.record_outcome(trade["signal_id"], final, now)
        if written:
            log.info("strategy_shadow_outcomes_recorded rows=%d pending=%d", written, len(due))
        return written

    async def run_forever(self) -> None:
        while True:
            try:
                await self.resolve_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - outcome tracking must never stop the bot
                log.exception("strategy_shadow_outcomes_failed")
            await asyncio.sleep(self._interval)
