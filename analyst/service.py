"""The shadow hook: runs the independent analyst after the live report.

``pipeline.py`` calls :meth:`ShadowAnalyst.on_signal` once per signal, after
the live verdict has been decided, rendered and sent — so nothing here can
delay or change what the bot already did. It reuses the M15/H1/H4 candles
the live context fetched (no extra requests) and fetches only M5, then
writes one row to the analyst's own write-once table. It sends nothing to
Telegram; the latest card can be pulled with ``/shadow`` from the
configured bot chat.

Analysis time is the live decision's own time (``MarketContext.generated_at``)
and only bars closed by then are used, so the shadow verdict judges exactly
the moment the live verdict judged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from costs import DEFAULT_ASSUMPTIONS, Costs
from market_data import MarketDataError, MarketDataService, Timeframe
from models import TradeSetup
from trade_quality import effective_entry

from analyst.data import TIMEFRAMES
from analyst.engine import Analysis, analyse
from analyst.report import render_card
from analyst.rules import Rules
from analyst.strategies import READY
from analyst.store import AnalystStore

log = logging.getLogger(__name__)

SYMBOL = "XAUUSD"


def assumed_costs(symbol: str = SYMBOL) -> Optional[Costs]:
    """The labelled cost assumption (``COST_SPREAD_<SYMBOL>`` /
    ``COST_SLIPPAGE_<SYMBOL>`` override the defaults in ``costs.py``)."""
    if symbol not in DEFAULT_ASSUMPTIONS:
        return None
    spread, slippage = DEFAULT_ASSUMPTIONS[symbol]
    values = []
    for name, default in ((f"COST_SPREAD_{symbol}", spread), (f"COST_SLIPPAGE_{symbol}", slippage)):
        raw = os.getenv(name, "").strip()
        value = float(raw) if raw else default
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
        values.append(value)
    return Costs(values[0], values[1], "assumed")


def signal_trade(setup: TradeSetup, price: Optional[float], at: str) -> Optional[Dict[str, Any]]:
    """The original signal as a trade, defined exactly as the live outcome
    accounting defines it (P1): entry at the stated zone's midpoint (the live
    price when none is stated), the stated order type, a market order filled
    at the live price. Tracked whatever the verdict."""
    if price is None or not (setup.direction and setup.stop_loss is not None and setup.take_profits):
        return None
    return {"subject": "signal", "direction": setup.direction, "order_type": setup.order_type,
            "entry": effective_entry(setup, price), "price_at_decision": price,
            "stop_loss": setup.stop_loss, "take_profits": list(setup.take_profits),
            "recorded_at": at}


def trades_to_track(a: Analysis, setup: TradeSetup) -> List[Dict[str, Any]]:
    """Fixed at decision time: the signal (any verdict) and each READY proposal."""
    at = a.as_of.isoformat()
    trades = [t for t in (signal_trade(setup, a.price, at),) if t is not None]
    for name, sides in a.proposals.items():
        for direction, p in sides.items():
            if p is not None and p.status == READY:
                trades.append({"subject": f"{name}:{direction}", "direction": direction,
                               "order_type": "market", "entry": p.entry, "price_at_decision": a.price,
                               "stop_loss": p.stop_loss, "take_profits": [p.take_profit],
                               "recorded_at": at})
    return trades


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def analysis_row(a: Analysis, setup: TradeSetup, *, chat_id: int, message_id: int, message_at,
                 live_verdict: Optional[str], live_confidence: Optional[int], card: str,
                 rules: Rules, mode: str = "shadow") -> Dict[str, Any]:
    v = a.verdict
    detail = {
        "context": a.context, "biases": a.biases, "m15_setup": a.m15_setup,
        "m5_confirmation": a.m5_confirmation, "news": a.news, "problems": list(a.view.problems),
        "market": a.view.market,
        "verdict": {"verdict": v.verdict, "reasons": [asdict(r) for r in v.reasons],
                    "weaknesses": [asdict(r) for r in v.weaknesses],
                    "checks": [[name, ok] for name, ok in v.checks],
                    "confirmed_by": list(v.confirmed_by), "plan": v.plan.as_dict()},
        "proposals": {name: {d: (p.as_dict() if p else None) for d, p in sides.items()}
                      for name, sides in a.proposals.items()},
        "headline": [a.headline.strategy, a.headline.direction] if a.headline else None,
        "rules": _jsonable(asdict(rules)),
    }
    return {
        "created_at": a.as_of.isoformat(), "chat_id": chat_id, "message_id": message_id,
        "message_at": message_at.isoformat(), "symbol": a.symbol, "engine_version": a.engine_version,
        "mode": mode, "price_at_decision": a.price, "data_status": a.data_status,
        "live_verdict": live_verdict, "live_confidence": live_confidence, "verdict": v.verdict,
        "reason_codes": ",".join(r.code for r in v.reasons),
        "checks_met": v.checks_met if v.checks else None,
        "checks_total": len(v.checks) if v.checks else None,
        "signal_json": json.dumps({"symbol": setup.symbol, "direction": setup.direction,
                                   "order_type": setup.order_type, "entries": list(setup.entries),
                                   "stop_loss": setup.stop_loss,
                                   "take_profits": list(setup.take_profits)}),
        "analysis_json": json.dumps(_jsonable(detail)),
        "trades_json": json.dumps(trades_to_track(a, setup)),
        "card_text": card,
    }


class ShadowAnalyst:
    def __init__(self, market_data: MarketDataService, store: AnalystStore, *,
                 rules: Optional[Rules] = None, costs: Optional[Costs] = None) -> None:
        self._market_data, self._store = market_data, store
        self._rules = rules or Rules.from_env()
        self._costs = costs or assumed_costs()
        # One analysis at a time, so a duplicate arriving seconds after the
        # first always sees the first one's row.
        self._lock = asyncio.Lock()

    @property
    def costs(self) -> Optional[Costs]:
        return self._costs

    async def on_signal(self, message, analysis, context, live_decision) -> Optional[Analysis]:
        setup = analysis.setup
        if not analysis.is_signal or (setup.symbol or "").strip().upper() != SYMBOL:
            return None
        async with self._lock:
            return await self._analyse(message, setup, context, live_decision)

    async def _analyse(self, message, setup: TradeSetup, context, live_decision) -> Analysis:
        rules, costs = self._rules, self._costs
        as_of = context.generated_at
        candles: Dict[Timeframe, Any] = {
            tf: series.candles for tf, series in getattr(context, "candle_series", {}).items()
            if tf in TIMEFRAMES}
        try:
            m5 = await self._market_data.get_candles(SYMBOL, Timeframe.M5, rules.m5_limit)
            candles[Timeframe.M5] = m5.candles
        except MarketDataError as exc:
            log.info("analyst_m5_unavailable message_id=%s error=%s", message.id, exc)
            candles[Timeframe.M5] = None
        recent = await self._store.recent_signals(
            SYMBOL, as_of - rules.duplicate_window - rules.signal_max_age)
        a = analyse(SYMBOL, setup, message.timestamp, as_of, context.current_price, candles, rules,
                    costs, recent=recent, news=context.news)
        live_verdict = getattr(getattr(live_decision, "verdict", None), "value", None)
        live_confidence = getattr(live_decision, "confidence", None)
        card = render_card(a, setup, live_verdict=live_verdict)
        row = analysis_row(a, setup, chat_id=message.chat_id, message_id=message.id,
                           message_at=message.timestamp, live_verdict=live_verdict,
                           live_confidence=live_confidence, card=card, rules=rules)
        analysis_id = await self._store.record(row)
        log.info("independent_analysis message_id=%s verdict=%s reasons=%s data=%s headline=%s "
                 "live_verdict=%s recorded=%s", message.id, a.verdict.verdict, row["reason_codes"],
                 a.data_status,
                 f"{a.headline.strategy}:{a.headline.direction}:{a.headline.status}" if a.headline else None,
                 live_verdict, analysis_id is not None)
        return a
