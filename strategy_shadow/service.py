"""The pipeline hook for strategy shadow evaluation.

Called by ``pipeline.py`` as a background task once the live verdict has
been decided, rendered and sent. In order:

1. the live decision is copied first (verdict, score, reason) — nothing
   computed later can alter what production did;
2. a signal already recorded (same chat and message id) is left alone —
   restarts and Telegram reconnects do not create a second evaluation;
3. the snapshot's quality is measured (``snapshot.assess``) — the label
   shift comes from a cached calibration, at most one small request every
   6 h;
4. the four strategies are evaluated on the live context itself, in a
   worker thread so the event loop is not blocked;
5. the original signal's trade is fixed for outcome tracking;
6. everything is written in one transaction.

Nothing is sent to Telegram. Any failure is logged by the caller and never
reaches the live path.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional

from market_data import Timeframe
from trade_quality import effective_entry

import strategy_variants
from strategy_shadow.evaluate import baseline_matches, evaluate
from strategy_shadow.snapshot import LabelCalibrator, assess, fingerprint
from strategy_shadow.store import StrategyShadowStore
from strategy_shadow.versions import code_version, rules_hash

log = logging.getLogger(__name__)

SYMBOL = "XAUUSD"
DUPLICATE_WINDOW = timedelta(hours=4)
DUPLICATE_TOLERANCE = 0.5          # USD: same stop and first target within this


def session_of(at: datetime) -> str:
    h = at.hour
    return ("asia" if h < 7 else "london" if h < 12 else "overlap" if h < 16
            else "new_york" if h < 21 else "late")


def regime_labels(context) -> Dict[str, Any]:
    tfs = context.timeframes
    h1 = tfs.get("H1")
    vote = context.scoring.net_score if context.scoring is not None else 0
    try:
        regime = strategy_variants.regime(context)
    except Exception:  # noqa: BLE001 - a label is not worth an evaluation
        regime = None
    return {"d1": tfs["D1"].trend if "D1" in tfs else None,
            "h4": tfs["H4"].trend if "H4" in tfs else None,
            "h1": h1.trend if h1 else None,
            "h1_adx": round(h1.adx.adx, 1) if h1 and h1.adx.adx is not None else None,
            "h1_vote": "long" if vote > 0 else ("short" if vote < 0 else "none"),
            "htf_regime": regime, "session": session_of(context.generated_at)}


def signal_trade(setup, price: Optional[float], decided_at: datetime) -> tuple:
    """(trade, None) or (None, why it cannot be tracked)."""
    if setup.direction not in ("long", "short"):
        return None, "no direction"
    if setup.stop_loss is None or not setup.take_profits:
        return None, "no stop or no take-profit"
    if price is None:
        return None, "no live price at decision"
    return ({"direction": setup.direction, "order_type": setup.order_type,
             "entry": effective_entry(setup, price), "price_at_decision": price,
             "stop_loss": setup.stop_loss, "take_profits": list(setup.take_profits),
             "recorded_at": decided_at.isoformat()}, None)


class StrategyShadow:
    def __init__(self, store: StrategyShadowStore, calibrator: Optional[LabelCalibrator] = None, *,
                 confirmation_start: Optional[date] = None) -> None:
        self._store, self._calibrator = store, calibrator or LabelCalibrator(None)
        self._confirmation_start = confirmation_start
        self._lock = asyncio.Lock()       # one evaluation at a time: duplicates see their twin

    @property
    def store(self) -> StrategyShadowStore:
        return self._store

    def cohort(self, signal_at: datetime) -> str:
        start = self._confirmation_start
        return "confirmation" if start is not None and signal_at.date() >= start else "exploration"

    async def on_signal(self, message, analysis, context, decision) -> Optional[int]:
        setup = analysis.setup
        if not analysis.is_signal or (setup.symbol or "").strip().upper() != SYMBOL:
            return None
        # 1. The live decision, copied before anything experimental runs.
        live_verdict = getattr(getattr(decision, "verdict", None), "value", None)
        live_verdict = live_verdict.upper() if live_verdict else None
        live_score = getattr(decision, "adjusted_score", None)
        live_reason = getattr(decision, "decision_reason", None)
        key = f"{message.chat_id}:{message.id}"
        async with self._lock:
            if await self._store.has(key):                       # 2.
                log.info("strategy_shadow_duplicate_skipped signal_key=%s", key)
                return None
            return await self._evaluate(key, message, setup, context, live_verdict, live_score, live_reason)

    async def _evaluate(self, key, message, setup, context, live_verdict, live_score, live_reason):
        errors = []
        decided_at = context.generated_at
        live_m15 = list(context.candle_series[Timeframe.M15].candles) \
            if Timeframe.M15 in getattr(context, "candle_series", {}) else []
        offset = None
        try:                                                      # 3.
            offset = await self._calibrator.offset(live_m15, decided_at)
        except Exception as exc:  # noqa: BLE001 - quality is then "unverified", not fatal
            errors.append(f"calibration: {type(exc).__name__}: {exc}")
        quality = assess(context, offset, setup)
        decisions = await asyncio.to_thread(evaluate, context)   # 4.
        errors += [f"{d.strategy}: {d.error}" for d in decisions.values() if d.error]
        trade, untracked = signal_trade(setup, context.current_price, decided_at)   # 5.
        duplicate_of = await self._duplicate_of(setup, message.timestamp)
        entries = list(setup.entries or [])
        signal = {
            "signal_key": key, "chat_id": message.chat_id, "message_id": message.id,
            "signal_at": message.timestamp.astimezone(timezone.utc).isoformat(),
            "decided_at": decided_at.isoformat(),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "symbol": SYMBOL, "direction": setup.direction, "order_type": setup.order_type,
            "entries_json": json.dumps(entries),
            "entry_mid": sum(entries) / len(entries) if entries else None,
            "stop_loss": setup.stop_loss, "take_profits_json": json.dumps(list(setup.take_profits)),
            "price_at_decision": context.current_price,
            "snapshot_fingerprint": fingerprint(context),
            "data_status": quality.status, "data_quality_json": json.dumps(quality.as_dict()),
            "regime_json": json.dumps(regime_labels(context)),
            "live_verdict": live_verdict, "live_setup_score": live_score, "live_reason": live_reason,
            "baseline_matches_live": _as_int(baseline_matches(decisions, live_verdict)),
            "duplicate_of": duplicate_of,
            "cohort": self.cohort(message.timestamp.astimezone(timezone.utc)),
            "code_version": code_version(), "rules_hash": rules_hash(),
            "trade_json": json.dumps(trade) if trade else None, "untracked_reason": untracked,
            "errors_json": json.dumps(errors),
        }
        rows = [{**asdict(d), "strategy_version": d.version, "blocked": int(d.blocked),
                 "score_label": d.score_label} for d in decisions.values()]
        signal_id = await self._store.record(signal, rows)          # 6.
        match = signal["baseline_matches_live"]
        level = log.warning if match == 0 else log.info
        level("strategy_shadow_recorded signal_key=%s id=%s data=%s live=%s %s baseline_matches=%s",
              key, signal_id, quality.status, live_verdict,
              " ".join(f"{d.strategy}={d.verdict}" for d in decisions.values()), match)
        return signal_id

    async def _duplicate_of(self, setup, signal_at: datetime) -> Optional[int]:
        """The first earlier signal with the same direction, stop and first
        target within four hours — the same trade idea forwarded again."""
        if setup.stop_loss is None or not setup.take_profits:
            return None
        for row in await self._store.recent_signals(signal_at - DUPLICATE_WINDOW):
            tps = json.loads(row["take_profits_json"])
            if (row["direction"] == setup.direction and row["stop_loss"] is not None and tps
                    and abs(row["stop_loss"] - setup.stop_loss) <= DUPLICATE_TOLERANCE
                    and abs(tps[0] - setup.take_profits[0]) <= DUPLICATE_TOLERANCE):
                return row["duplicate_of"] or row["id"]
        return None


def _as_int(value: Optional[bool]) -> Optional[int]:
    return None if value is None else int(value)
