"""Automatic signal outcome logging — the evaluation dataset.

Every analysed signal is recorded at decision time with everything the
decision was made from, then resolved later against the M15 candles that
FOLLOWED it. The point is a dataset on which the confluence score can one day
be calibrated honestly, so three rules are absolute:

1. **Record before knowing.** The snapshot is written when the decision is
   made, from the decision-time context only. Nothing about the outcome can
   leak into the recorded scores.
2. **No lookahead, no peeking backwards either.** Resolution reads only
   *closed* M15 bars that OPENED at or after the decision time. The bar in
   progress when the signal arrived is skipped, because part of it happened
   before the signal. Candle times are UTC (``market_data`` requests
   ``timezone=UTC`` precisely so this comparison is valid).
3. **Never rewrite history.** A row is resolved once. The storage update
   only touches rows still ``open``; there is no code path that changes a
   resolved outcome, and nothing here edits rows by hand.

Every signal is tracked, SKIP included: whether the trades the system
refused would have worked is exactly what an evaluation needs.

Resolution rules (conservative wherever the bar data is ambiguous):

* **Fill.** A market order fills at the decision price. A resting order fills
  on the first bar that trades through its entry from where price was at the
  decision — ``low <= entry`` if the entry was at or below the price,
  ``high >= entry`` if above. Unfilled after ``fill_window`` -> ``not_filled``.
* **Outcome.** First touch of the stop loss or the FIRST take profit. Both in
  one bar -> loss, because the order inside the bar is unknown. On the bar
  that filled a resting order only the stop counts: its target touch may
  have happened before the fill.
* **MFE / MAE.** Most favourable and most adverse excursion from the entry,
  in price and in R, from the fill to the deciding bar. On a resting order's
  fill bar only the close counts as favourable, for the same reason.
* **Horizon.** Neither level touched within ``horizon`` of the fill ->
  ``expired``, with MFE/MAE kept.
* **Coverage.** If the fetched history does not reach back to the decision
  time the row is ``no_data`` — it cannot be resolved honestly.

Times are bar-granular: a hit "at" a bar means somewhere inside the 15
minutes starting at that bar's open.

These are tracking parameters, not trading thresholds: they decide when a
row stops being open and never feed back into any score.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from costs import CostModel, Costs
from engine_version import engine_version
from experiments import EXPECTED_ROWS, REGISTERED, AccountingExperiment, account_arms
from market_data import Candle, MarketDataError, MarketDataService, Timeframe
from trade_accounting import account_both
from trade_geometry import check_geometry
from trade_quality import effective_entry

log = logging.getLogger(__name__)

BAR = timedelta(minutes=15)
DEFAULT_FILL_WINDOW = timedelta(hours=24)
DEFAULT_HORIZON = timedelta(days=5)
# Twelve Data's maximum page: about 52 days of M15 on a 24/5 market. A fixed
# size keeps one cache entry per symbol in MarketDataService.
TRACKING_CANDLES = 5000
DEFAULT_INTERVAL_SECONDS = 1800.0


# --------------------------------------------------------------- the snapshot

def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


def snapshot(context: Any, decision: Any, *, message: Any = None) -> Dict[str, Any]:
    """Everything the decision was made from, as one flat row.

    Pure: reads the decision-time context and decision, nothing else. A
    setup whose levels are not a valid trade is recorded as ``invalid`` and
    never tracked — there is no outcome for an impossible trade.
    """
    setup = getattr(context, "setup", None)
    primary = getattr(context, "primary", None)
    structure = getattr(context, "structure", None)
    scoring = getattr(context, "scoring", None)
    validation = getattr(context, "validation", None)
    risk = getattr(context, "risk", None)
    entry_q = getattr(context, "entry", None)

    direction = getattr(setup, "direction", None)
    entries = list(getattr(setup, "entries", None) or [])
    take_profits = list(getattr(setup, "take_profits", None) or [])
    stop = getattr(setup, "stop_loss", None)
    entry = effective_entry(setup, getattr(context, "current_price", None)) if setup else None
    geometry = check_geometry(direction, entries or ([entry] if entry is not None else []),
                              stop, take_profits)

    trend = structure.trend.value if structure is not None else None
    volatility = primary.volatility if primary is not None else None
    adx_strength = primary.adx.strength if primary is not None else None

    components = {
        "direction": {
            "deterministic_score": getattr(decision, "deterministic_score", None),
            "scoring_engine_confluence": scoring.confidence if scoring else None,
            "scoring_direction": scoring.direction.value if scoring else None,
            "net_score": scoring.net_score if scoring else None,
            "total_possible": scoring.total_possible if scoring else None,
            "groups": scoring.group_scores() if scoring else {},
            "factors": {f.name: {"vote": f.raw_contribution, "counted": f.contribution,
                                 "group": f.group}
                        for f in (scoring.breakdown if scoring else ())},
        },
        "review": {
            "bonus": getattr(decision, "validation_bonus", None),
            "penalty": getattr(decision, "validation_penalty", None),
            "checks": {c.name: {"severity": c.severity.value, "contribution": c.contribution}
                       for c in (validation.checks if validation else ())},
        },
        "entry": {
            "location_factor": getattr(decision, "location_factor", None),
            "extension_factor": getattr(decision, "extension_factor", None),
            "room_factor": getattr(decision, "room_factor", None),
            "location": entry_q.location.label if entry_q else None,
            "extension": entry_q.extension.label if entry_q else None,
            "room": entry_q.room.label if entry_q else None,
            "room_atr": entry_q.room.room_atr if entry_q else None,
            "obstacle": entry_q.room.obstacle_label if entry_q else None,
        },
        # Where each analysed series stopped: decisions use closed bars only,
        # so this is exactly what a replay must reproduce.
        "closed_through": {tf: ts.isoformat()
                           for tf, ts in (getattr(context, "closed_through", None) or {}).items()},
        "adjusted_score": getattr(decision, "adjusted_score", None),
        "confluence_score": getattr(decision, "confluence_score", None),
        "risk": {
            "approved": risk.approved if risk else None,
            "rejections": [f"{c.name}: {c.detail}" for c in risk.rejections] if risk else [],
            "warnings": [f"{c.name}: {c.detail}" for c in risk.warnings] if risk else [],
        },
    }

    verdict = getattr(decision, "verdict", None)
    return {
        "recorded_at": _iso(getattr(context, "generated_at", None)
                            or datetime.now(timezone.utc)),
        "message_id": getattr(message, "id", None),
        "chat_id": getattr(message, "chat_id", None),
        "message_timestamp": _iso(getattr(message, "timestamp", None)),
        "symbol": getattr(context, "symbol", None),
        "analysis_timeframe": getattr(getattr(context, "primary_timeframe", None), "value", None),
        "signal_timeframe": getattr(setup, "timeframe", None),
        "direction": direction,
        "order_type": getattr(setup, "order_type", None),
        "entry": entry,
        "stop_loss": stop,
        "take_profit": take_profits[0] if take_profits else None,
        "take_profits_json": json.dumps(take_profits),
        "price_at_decision": getattr(context, "current_price", None),
        "risk_reward": geometry.risk_reward,
        "decision": getattr(verdict, "value", None),
        "decision_reason": getattr(decision, "decision_reason", None),
        "confluence_score": getattr(decision, "confluence_score", None),
        "deterministic_score": getattr(decision, "deterministic_score", None),
        "validation_bonus": getattr(decision, "validation_bonus", None),
        "validation_penalty": getattr(decision, "validation_penalty", None),
        "adjusted_score": getattr(decision, "adjusted_score", None),
        "location_factor": getattr(decision, "location_factor", None),
        "extension_factor": getattr(decision, "extension_factor", None),
        "range_position": entry_q.location.position if entry_q else None,
        "range_low": entry_q.location.range_low if entry_q else None,
        "range_high": entry_q.location.range_high if entry_q else None,
        "extension_atr": entry_q.extension.move_atr if entry_q else None,
        "extension_percentile": entry_q.extension.percentile if entry_q else None,
        "atr": primary.atr if primary is not None else None,
        "structure_trend": trend,
        "volatility": volatility,
        "adx": primary.adx.adx if primary is not None else None,
        "market_regime": f"trend={trend} volatility={volatility} adx={adx_strength}",
        "components_json": json.dumps(components, default=str),
        "status": "open" if geometry.valid else "invalid",
        "outcome_note": None if geometry.valid else f"not tracked: {geometry.reason}",
    }


# --------------------------------------------------------------- resolution

@dataclass(frozen=True)
class Resolution:
    status: str                       # tp / sl / not_filled / expired / no_data
    outcome: Optional[str] = None     # win / loss for tp / sl, else None
    filled_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    mfe: Optional[float] = None       # price units, >= 0
    mae: Optional[float] = None
    mfe_r: Optional[float] = None
    mae_r: Optional[float] = None
    minutes_to_tp: Optional[float] = None
    minutes_to_sl: Optional[float] = None
    note: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "status": self.status, "outcome": self.outcome,
            "filled_at": _iso(self.filled_at), "resolved_at": _iso(self.resolved_at),
            "mfe": self.mfe, "mae": self.mae, "mfe_r": self.mfe_r, "mae_r": self.mae_r,
            "minutes_to_tp": self.minutes_to_tp, "minutes_to_sl": self.minutes_to_sl,
            "outcome_note": self.note,
        }


@dataclass
class _Excursion:
    entry: float
    long: bool
    favourable: float = 0.0
    adverse: float = 0.0

    def see(self, high: float, low: float) -> None:
        up, down = high - self.entry, self.entry - low
        fav, adv = (up, down) if self.long else (down, up)
        self.favourable = max(self.favourable, fav)
        self.adverse = max(self.adverse, adv)


def _minutes(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds() / 60.0)


def resolve(
    row: Dict[str, Any],
    candles: Sequence[Candle],
    now: datetime,
    *,
    fill_window: timedelta = DEFAULT_FILL_WINDOW,
    horizon: timedelta = DEFAULT_HORIZON,
    bar: timedelta = BAR,
) -> Optional[Resolution]:
    """Decide a tracked signal's outcome, or ``None`` if it is still open.

    Pure and deterministic: the same row, candles and ``now`` always give
    the same answer. ``candles`` must be oldest-first.
    """
    decided_at = datetime.fromisoformat(row["recorded_at"])
    direction, entry = row["direction"], row["entry"]
    stop, target = row["stop_loss"], row["take_profit"]
    is_long = direction == "long"
    risk = abs(entry - stop)

    closed = [c for c in candles if c.timestamp + bar <= now]
    if not closed:
        return None
    if closed[0].timestamp > decided_at:
        # The history does not reach back to the decision, so a fill or a
        # touch in the gap could have been missed.
        return Resolution("no_data", note="candle history does not reach back to the decision")
    after = [c for c in closed if c.timestamp >= decided_at]

    # ---- fill
    market = row.get("order_type") == "market"
    fill_index: Optional[int] = None
    if market:
        fill_index, filled_at = 0, decided_at
    else:
        reference = row.get("price_at_decision")
        for i, c in enumerate(after):
            if c.timestamp >= decided_at + fill_window:
                break
            if reference is None:
                hit = c.low <= entry <= c.high
            elif entry <= reference:
                hit = c.low <= entry
            else:
                hit = c.high >= entry
            if hit:
                fill_index, filled_at = i, c.timestamp
                break
        if fill_index is None:
            window_over = closed[-1].timestamp + bar >= decided_at + fill_window
            return (Resolution("not_filled", note=f"entry not reached within {fill_window}")
                    if window_over else None)

    if not after:
        return None

    # ---- first touch
    excursion = _Excursion(entry, is_long)

    def result(status: str, at: datetime, note: str = "") -> Resolution:
        win = status == "tp"
        return Resolution(
            status=status, outcome="win" if win else "loss",
            filled_at=filled_at, resolved_at=at,
            mfe=round(excursion.favourable, 6), mae=round(excursion.adverse, 6),
            mfe_r=round(excursion.favourable / risk, 4), mae_r=round(excursion.adverse / risk, 4),
            minutes_to_tp=_minutes(filled_at, at) if win else None,
            minutes_to_sl=None if win else _minutes(filled_at, at),
            note=note,
        )

    for i in range(fill_index, len(after)):
        c = after[i]
        if c.timestamp >= filled_at + horizon:
            return Resolution("expired", filled_at=filled_at, resolved_at=c.timestamp,
                              mfe=round(excursion.favourable, 6), mae=round(excursion.adverse, 6),
                              mfe_r=round(excursion.favourable / risk, 4),
                              mae_r=round(excursion.adverse / risk, 4),
                              note=f"neither level touched within {horizon} of the fill")
        stop_hit = c.low <= stop if is_long else c.high >= stop
        target_hit = c.high >= target if is_long else c.low <= target

        fill_bar = (i == fill_index and not market)
        if fill_bar:
            # Only what is known to be after the fill: the close as the
            # favourable side, the adverse extreme in full (conservative).
            if is_long:
                excursion.see(c.close, c.low)
            else:
                excursion.see(c.high, c.close)
            if stop_hit:
                return result("sl", c.timestamp, "stop touched on the fill bar")
            continue

        excursion.see(c.high, c.low)
        if stop_hit and target_hit:
            return result("sl", c.timestamp, "stop and target in the same bar — counted as a loss")
        if stop_hit:
            return result("sl", c.timestamp)
        if target_hit:
            return result("tp", c.timestamp)

    if closed[-1].timestamp + bar >= filled_at + horizon:
        return Resolution("expired", filled_at=filled_at, resolved_at=closed[-1].timestamp,
                          mfe=round(excursion.favourable, 6), mae=round(excursion.adverse, 6),
                          mfe_r=round(excursion.favourable / risk, 4),
                          mae_r=round(excursion.adverse / risk, 4),
                          note=f"neither level touched within {horizon} of the fill")
    return None


# --------------------------------------------------------------- the tracker

@dataclass
class OutcomeTracker:
    """Records signals as they are decided and resolves them as bars close.

    Each pass also writes the trade ACCOUNTING (``trade_accounting``) — the
    realised R a trader would have made, gross and net of costs — once it is
    final. Accounting is derived data in its own table: it never alters the
    raw outcome columns, and each (signal, basis) result is written once.

    Registered accounting experiments (``experiments.py``) are accounted the
    same way into their own table, for signals recorded since each one's
    start. They read what the live accounting reads and write nothing it reads.
    """

    storage: Any
    market_data: MarketDataService
    timeframe: Timeframe = Timeframe.M15
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    fill_window: timedelta = DEFAULT_FILL_WINDOW
    horizon: timedelta = DEFAULT_HORIZON
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    cost_model: Optional[CostModel] = None
    experiments: Sequence[AccountingExperiment] = REGISTERED

    async def record(self, context: Any, decision: Any, message: Any = None,
                     variants: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[int]:
        """Persist the decision-time snapshot, stamped and costed. Never raises.

        Records the engine version, the costs a trade would pay (real
        bid/ask when available, else the labelled assumption), the original
        signal if this one repeats it, and every variant's verdict.
        """
        try:
            costs = None
            if self.cost_model is not None:
                costs = await self.cost_model.costs_for(getattr(context, "symbol", ""),
                                                        getattr(context, "quote", None))
            row = snapshot(context, decision, message=message)
            row["engine_version"] = engine_version()
            if costs is not None:
                row.update(cost_spread=costs.spread, cost_slippage=costs.slippage,
                           cost_source=costs.source)
            row["duplicate_of"] = await self.storage.find_duplicate(row)
            outcome_id = await self.storage.record_signal_outcome(row)
            if outcome_id is not None and variants:
                await self.storage.record_shadow_decisions(outcome_id, variants)
            if outcome_id is not None:
                await self._register_experiments(row["recorded_at"])
            return outcome_id
        except Exception:  # noqa: BLE001 - logging outcomes must never break the pipeline
            log.exception("outcome_record_failed symbol=%s", getattr(context, "symbol", None))
            return None

    async def _register_experiments(self, recorded_at: str) -> None:
        """The first signal recorded by code carrying an experiment fixes its
        start (write-once); later calls change nothing. Never raises."""
        if not hasattr(self.storage, "register_experiment"):
            return
        for experiment in self.experiments:
            try:
                await self.storage.register_experiment(
                    experiment.name, recorded_at, experiment.registration, experiment.parameters())
            except Exception:  # noqa: BLE001 - an experiment must never affect recording
                log.exception("experiment_register_failed name=%s", experiment.name)

    def _costs_of(self, row: Dict[str, Any]) -> Optional[Costs]:
        """The costs recorded at decision time; for rows recorded before costs
        were, the current assumption — labelled as a backfill."""
        if row.get("cost_spread") is not None:
            return Costs(row["cost_spread"], row.get("cost_slippage") or 0.0,
                         row.get("cost_source") or "assumed")
        assumed = self.cost_model.assumption(row["symbol"]) if self.cost_model else None
        return None if assumed is None else Costs(assumed.spread, assumed.slippage,
                                                  "assumed (backfill)")

    async def resolve_pending(self) -> int:
        """Resolve open outcomes and write final accounting. Never raises.

        One candle fetch per symbol per pass, however many rows need work.
        Returns the number of raw outcomes resolved.
        """
        try:
            rows: List[Dict[str, Any]] = await self.storage.open_signal_outcomes()
            to_account: List[Dict[str, Any]] = (
                await self.storage.outcomes_needing_accounting()
                if hasattr(self.storage, "outcomes_needing_accounting") else [])
        except Exception:  # noqa: BLE001
            log.exception("outcome_fetch_open_failed")
            return 0
        experiment_rows: List[tuple] = []
        for experiment in (self.experiments if hasattr(self.storage, "outcomes_needing_experiment")
                           else ()):
            try:
                experiment_rows += [(experiment, row) for row in
                                    await self.storage.outcomes_needing_experiment(
                                        experiment.name, EXPECTED_ROWS)]
            except Exception:  # noqa: BLE001 - an experiment must never stop tracking
                log.exception("experiment_fetch_failed name=%s", experiment.name)

        resolved = 0
        by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_symbol.setdefault(row["symbol"], []).append(row)
        accounting_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for row in to_account:
            accounting_by_symbol.setdefault(row["symbol"], []).append(row)
        experiments_by_symbol: Dict[str, List[tuple]] = {}
        for experiment, row in experiment_rows:
            experiments_by_symbol.setdefault(row["symbol"], []).append((experiment, row))

        now = self.clock()
        for symbol in sorted(set(by_symbol) | set(accounting_by_symbol) | set(experiments_by_symbol)):
            try:
                series = await self.market_data.get_candles(symbol, self.timeframe,
                                                             TRACKING_CANDLES)
            except MarketDataError as exc:
                log.warning("outcome_candles_unavailable symbol=%s error=%s", symbol, exc)
                continue
            except Exception:  # noqa: BLE001
                log.exception("outcome_candles_failed symbol=%s", symbol)
                continue
            for row in by_symbol.get(symbol, []):
                try:
                    outcome = resolve(row, series.candles, now,
                                      fill_window=self.fill_window, horizon=self.horizon)
                    if outcome is not None and await self.storage.resolve_signal_outcome(
                            row["id"], outcome.as_row()):
                        resolved += 1
                        log.info("outcome_resolved id=%s symbol=%s decision=%s status=%s",
                                 row["id"], symbol, row.get("decision"), outcome.status)
                except Exception:  # noqa: BLE001 - one bad row must not stop the rest
                    log.exception("outcome_resolve_failed id=%s", row.get("id"))
            for row in accounting_by_symbol.get(symbol, []):
                await self._account(row, series.candles, now)
            for experiment, row in experiments_by_symbol.get(symbol, []):
                await self._account_experiment(experiment, row, series.candles, now)
        return resolved

    async def _account(self, row: Dict[str, Any], candles: Sequence[Candle], now: datetime) -> None:
        """Write whichever of the gross/net results is missing and now final."""
        try:
            results = account_both(row, candles, now, costs=self._costs_of(row),
                                   fill_window=self.fill_window, horizon=self.horizon)
            for basis, result in results.items():
                if row.get(f"has_{basis}") or not result.final:
                    continue
                await self.storage.record_accounting(
                    row["id"], result.as_row(), self.horizon.total_seconds() / 60.0)
        except Exception:  # noqa: BLE001 - one bad row must not stop the rest
            log.exception("outcome_accounting_failed id=%s", row.get("id"))

    async def _account_experiment(self, experiment: AccountingExperiment, row: Dict[str, Any],
                                  candles: Sequence[Candle], now: datetime) -> None:
        """Write whichever experiment arm/basis results are missing and now final."""
        try:
            results = account_arms(experiment, row, candles, now, costs=self._costs_of(row),
                                   fill_window=self.fill_window, horizon=self.horizon)
            for key, result in results.items():
                if key in row.get("experiment_done", ()) or not result.final:
                    continue
                arm, basis = key.split("/")
                await self.storage.record_experiment_accounting(
                    row["id"], experiment.name, arm, basis, result.as_row(),
                    self.horizon.total_seconds() / 60.0)
        except Exception:  # noqa: BLE001 - one bad row must not stop the rest
            log.exception("experiment_accounting_failed name=%s id=%s", experiment.name, row.get("id"))

    async def run_forever(self) -> None:
        """Resolve pending rows every ``interval_seconds`` until cancelled."""
        while True:
            await self.resolve_pending()
            await asyncio.sleep(self.interval_seconds)
