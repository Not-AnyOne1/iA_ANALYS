"""SQLite persistence for completed analyses (RFC-003, extended RFC-009).

Scope, by explicit design: this stores **completed analyses only** — the
same "successful analysis" event that already gets appended to JSONL in
``pipeline.py``. It does not have tables for failed analysis attempts or
RFC-001 shadow-mode agreement/disagreement; those remain session-only
runtime metrics on ``pipeline.Stats``, and the bot's own delivery counters
(notifications sent/failed/duplicates skipped) remain session-only on
``telegram_bot.BotStats``. Neither is "trading history" — persisting them
would expand this module well past what ``/latest``, ``/history``, and
``statistics.py`` actually need.

Every write here happens *alongside* the existing JSONL append in
``pipeline.py``, never replacing it — JSONL keeps being the durable,
human-readable audit log; SQLite is what makes ``/latest``/``/history``/
statistics queryable and restart-proof.

RFC-009 additive extension: :meth:`Storage.record` gained optional
``decision``/``structure``/``smc`` parameters so a row can *also* capture
a :class:`decision_engine.DecisionResult`, :class:`structure_engine.
StructureAnalysis`, and :class:`smc_engine.SMCAnalysis`, alongside the
:class:`models.SignalAnalysis` it already stored. This is the *storage*
half of making "statistics derived from stored analyses and decision
results" literally possible — it does not itself wire those engines into
``pipeline.py``'s message processing (that stays exactly as it was; no
previous engine or the pipeline is modified by this file). Every new
column is nullable and every new parameter defaults to ``None``, so
existing callers (``pipeline.py``, unchanged) keep writing exactly the
rows they always have, and :mod:`statistics.py` correctly reports the
decision/structure/SMC-derived metrics as absent (zero/None) until some
future caller actually supplies that data.

Async-safe without a new dependency: every operation opens a short-lived
``sqlite3`` connection and runs it via ``asyncio.to_thread``, so the event
loop is never blocked by disk I/O. SQLite's own file-level locking handles
the resulting access safely; there is no long-lived connection whose
lifecycle needs managing.

Failure isolation matches every other integration point in this project:
:meth:`Storage.record` never raises — a storage outage must never be able
to stop message ingestion or analysis, exactly like a Claude CLI failure or
a Telegram bot failure. Read methods (:meth:`latest`, :meth:`history`,
:meth:`fetch_analyses`) are allowed to raise; callers (``telegram_bot.py``,
``statistics.py``) decide how to degrade for a single failed query.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from market_data import Timeframe
from models import SignalAnalysis

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from decision_engine import DecisionResult
    from smc_engine import SMCAnalysis
    from structure_engine import StructureAnalysis
    from telegram_client import IncomingMessage

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    chat_title TEXT NOT NULL,
    sender TEXT NOT NULL,
    message_timestamp TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    is_signal INTEGER NOT NULL,
    category TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    analysis_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_id ON analyses(id);

-- Bot report recipients. One row per private chat that sent /start; the row
-- is deleted on /stop. Kept here rather than in a side file so a restart
-- (or a redeploy onto the Railway volume) keeps the audience it already had.
CREATE TABLE IF NOT EXISTS subscribers (
    chat_id INTEGER PRIMARY KEY,
    username TEXT,
    subscribed_at TEXT NOT NULL
);

-- The evaluation dataset (outcome_tracker.py). One row per analysed signal:
-- every input to the decision, written at decision time, and the outcome,
-- written exactly once when the M15 bars that followed decide it.
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    message_id INTEGER,
    chat_id INTEGER,
    message_timestamp TEXT,
    symbol TEXT NOT NULL,
    analysis_timeframe TEXT,
    signal_timeframe TEXT,
    direction TEXT,
    order_type TEXT,
    entry REAL,
    stop_loss REAL,
    take_profit REAL,
    take_profits_json TEXT,
    price_at_decision REAL,
    risk_reward REAL,
    decision TEXT,
    decision_reason TEXT,
    confluence_score INTEGER,
    deterministic_score INTEGER,
    validation_bonus INTEGER,
    validation_penalty INTEGER,
    adjusted_score INTEGER,
    location_factor REAL,
    extension_factor REAL,
    range_position REAL,
    range_low REAL,
    range_high REAL,
    extension_atr REAL,
    extension_percentile REAL,
    atr REAL,
    structure_trend TEXT,
    volatility TEXT,
    adx REAL,
    market_regime TEXT,
    components_json TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    outcome TEXT,
    filled_at TEXT,
    resolved_at TEXT,
    mfe REAL,
    mae REAL,
    mfe_r REAL,
    mae_r REAL,
    minutes_to_tp REAL,
    minutes_to_sl REAL,
    outcome_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_signal_outcomes_status ON signal_outcomes(status);

-- What a trade on each signal would actually have made, in R (trade_accounting.py).
-- One row per signal and basis ('gross' or 'net'), written once when final and
-- never updated. Derived from candles after the fact: it never alters the raw
-- outcome columns of signal_outcomes.
CREATE TABLE IF NOT EXISTS trade_accounting (
    outcome_id INTEGER NOT NULL,
    basis TEXT NOT NULL,
    status TEXT NOT NULL,
    r REAL,
    planned_entry REAL,
    planned_risk REAL,
    target REAL,
    entry_fill REAL,
    exit_price REAL,
    filled_at TEXT,
    exit_at TEXT,
    holding_minutes REAL,
    cost_spread REAL,
    cost_slippage REAL,
    cost_source TEXT,
    horizon_minutes REAL,
    note TEXT,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (outcome_id, basis)
);

-- Champion/challenger: every variant's verdict on every signal (shadow.py).
-- Only the champion's verdict is ever sent; the rest are recorded so that
-- variants can be compared on exactly the same signals.
CREATE TABLE IF NOT EXISTS shadow_decisions (
    outcome_id INTEGER NOT NULL,
    variant TEXT NOT NULL,
    engine_version TEXT,
    recorded_at TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confluence_score INTEGER,
    adjusted_score INTEGER,
    reason TEXT,
    details_json TEXT,
    PRIMARY KEY (outcome_id, variant)
);
"""

# Added to signal_outcomes after it first shipped; migrated in place.
_OUTCOME_NEW_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("engine_version", "TEXT"),
    ("duplicate_of", "INTEGER"),
    ("cost_spread", "REAL"),
    ("cost_slippage", "REAL"),
    ("cost_source", "TEXT"),
)

_ACCOUNTING_COLUMNS: Tuple[str, ...] = (
    "status", "r", "planned_entry", "planned_risk", "target", "entry_fill", "exit_price",
    "filled_at", "exit_at", "holding_minutes", "cost_spread", "cost_slippage", "cost_source",
    "note",
)

# Written once, at decision time.
_OUTCOME_SNAPSHOT_COLUMNS: Tuple[str, ...] = (
    "recorded_at", "message_id", "chat_id", "message_timestamp", "symbol",
    "analysis_timeframe", "signal_timeframe", "direction", "order_type", "entry",
    "stop_loss", "take_profit", "take_profits_json", "price_at_decision", "risk_reward",
    "decision", "decision_reason", "confluence_score", "deterministic_score",
    "validation_bonus", "validation_penalty", "adjusted_score", "location_factor",
    "extension_factor", "range_position", "range_low", "range_high", "extension_atr",
    "extension_percentile", "atr", "structure_trend", "volatility", "adx",
    "market_regime", "components_json", "status", "outcome_note",
    "engine_version", "duplicate_of", "cost_spread", "cost_slippage", "cost_source",
)

# Written once, when the outcome is decided — and only onto an open row.
_OUTCOME_RESOLUTION_COLUMNS: Tuple[str, ...] = (
    "status", "outcome", "filled_at", "resolved_at", "mfe", "mae", "mfe_r", "mae_r",
    "minutes_to_tp", "minutes_to_sl", "outcome_note",
)

# RFC-009: additive columns, all nullable. Added via ALTER TABLE (guarded by
# a PRAGMA table_info check) rather than baked into _SCHEMA's CREATE TABLE,
# so upgrading an existing RFC-003-era database file in place is safe and
# idempotent, not just a fresh one.
_NEW_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("symbol", "TEXT"),
    ("timeframe", "TEXT"),
    ("direction", "TEXT"),
    ("decision_verdict", "TEXT"),
    ("decision_confidence", "INTEGER"),
    ("decision_source", "TEXT"),
    ("structure_trend", "TEXT"),
    ("smc_counts_json", "TEXT"),
)

# Indexes covering exactly the columns fetch_analyses() filters on — the
# columns behind every statistics.py query (summary/today/this_week/
# this_month/by_symbol/by_timeframe) and the dashboard. Without these,
# each of those is a full table scan.
#
# Applied AFTER the _NEW_COLUMNS migration above, never inside _SCHEMA:
# symbol/timeframe don't exist yet on a pre-RFC-009 database file, so
# creating their indexes earlier would fail on exactly the upgrade path
# that migration exists to support.
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_analyses_recorded_at ON analyses(recorded_at);
CREATE INDEX IF NOT EXISTS idx_analyses_symbol ON analyses(symbol);
CREATE INDEX IF NOT EXISTS idx_analyses_timeframe ON analyses(timeframe);
"""


@dataclass(frozen=True)
class StoredAnalysis:
    """One persisted analysis, as read back for ``/latest`` and ``/history``.

    Deliberately the same shape ``telegram_bot.py`` used to get from its
    in-memory ``NotificationRecord`` (now removed), so the rendering code
    there didn't need to change — only where the data comes from.
    """

    message_id: int
    chat_title: str
    sender: str
    timestamp: datetime
    analysis: SignalAnalysis


@dataclass(frozen=True)
class AnalysisRecord:
    """One stored row, as read back for :mod:`statistics.py`'s aggregation.

    Deliberately flat and pre-parsed (smc_counts already a ``Dict[str,
    int]``, not raw JSON) so ``statistics.py`` never touches SQL or JSON
    directly — it only ever aggregates over a list of these.

    Every RFC-009 field is ``None`` for a row written the way
    ``pipeline.py`` currently writes them (analysis only, no decision/
    structure/smc) — that is the expected, honest state until some future
    caller supplies that data, not a bug.
    """

    recorded_at: datetime
    is_signal: bool
    category: str
    source: str
    confidence: float
    symbol: Optional[str]
    timeframe: Optional[str]
    direction: Optional[str]
    decision_verdict: Optional[str]
    decision_confidence: Optional[int]
    decision_source: Optional[str]
    structure_trend: Optional[str]
    smc_counts: Dict[str, int] = field(default_factory=dict)


# How long a connection waits for a competing writer before giving up.
# WAL still serialises writers, and with WORKER_COUNT concurrent analyses each
# opening its own connection, the default 5s was measurably too short: a
# 100-writer stress run lost rows to "database is locked", and because
# record() swallows errors by design that loss was silent. 30s is far longer
# than any single INSERT here can legitimately take.
_BUSY_TIMEOUT_SECONDS = 30.0

# A locked database is transient by definition, so a couple of retries on top
# of the busy timeout costs nothing and closes the remaining gap.
_LOCK_RETRIES = 3
_LOCK_RETRY_DELAY = 0.25


class Storage:
    """Owns the SQLite file backing persistent analysis history."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with a write-contention-tolerant timeout.

        Every operation in this module opens its own short-lived connection
        (see the module docstring), so the timeout has to be set here rather
        than once globally.
        """
        conn = sqlite3.connect(str(self._db_path), timeout=_BUSY_TIMEOUT_SECONDS)
        conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_SECONDS * 1000)}")
        return conn

    @staticmethod
    def _is_locked_error(exc: BaseException) -> bool:
        return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()

    async def initialize(self) -> None:
        """Create the schema if it doesn't exist yet, add any RFC-009
        columns missing from an existing (RFC-003-era) database file, enable
        WAL mode, and create the query indexes. Call once at startup."""
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        conn = self._connect()
        try:
            # WAL lets a reader (a separately-running dashboard.py process,
            # or the bot answering /stats) proceed while the monitor is
            # writing, instead of the two blocking each other. It's a
            # persistent property of the database file, so setting it here
            # every startup is idempotent. Best-effort: some filesystems
            # (notably network shares) don't support WAL, and falling back
            # to the default journal mode is far better than refusing to
            # start — every query below works identically either way.
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            if mode is not None and str(mode[0]).lower() != "wal":
                log.info(
                    "storage_wal_unavailable journal_mode=%s path=%s "
                    "(continuing with the default journal mode)",
                    mode[0], self._db_path,
                )

            conn.executescript(_SCHEMA)
            existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
            for name, column_type in _NEW_COLUMNS:
                if name not in existing_columns:
                    conn.execute(f"ALTER TABLE analyses ADD COLUMN {name} {column_type}")
            outcome_columns = {row[1] for row in conn.execute("PRAGMA table_info(signal_outcomes)")}
            for name, column_type in _OUTCOME_NEW_COLUMNS:
                if name not in outcome_columns:
                    conn.execute(f"ALTER TABLE signal_outcomes ADD COLUMN {name} {column_type}")
            # Only safe once the columns above exist — see _INDEXES.
            conn.executescript(_INDEXES)
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------- write

    async def record(
        self,
        message: "IncomingMessage",
        analysis: SignalAnalysis,
        *,
        decision: Optional["DecisionResult"] = None,
        structure: Optional["StructureAnalysis"] = None,
        smc: Optional["SMCAnalysis"] = None,
    ) -> None:
        """Persist one completed analysis. Never raises.

        ``decision``/``structure``/``smc`` are optional and additive
        (RFC-009): supplying them records a verdict/trend/SMC-object-count
        snapshot alongside the analysis for statistics.py to aggregate.
        Omitting them (as pipeline.py currently does) writes exactly the
        same row shape RFC-003 always has, with those columns left NULL.

        Mirrors ``TelegramBot.notify()``'s contract exactly: a storage
        outage is logged and swallowed here, and ``pipeline.py`` wraps this
        call again on top for the same defense-in-depth reason.
        """
        for attempt in range(1, _LOCK_RETRIES + 1):
            try:
                await asyncio.to_thread(
                    self._record_sync, message, analysis, decision, structure, smc
                )
                return
            except Exception as exc:  # noqa: BLE001 - storage must never break the pipeline
                # "database is locked" is transient write contention, not a
                # real failure — retry before giving up, because giving up
                # here loses the analysis silently (record() cannot raise).
                if self._is_locked_error(exc) and attempt < _LOCK_RETRIES:
                    log.warning(
                        "storage_record_locked chat_id=%s message_id=%s attempt=%d/%d",
                        message.chat_id, message.id, attempt, _LOCK_RETRIES,
                    )
                    await asyncio.sleep(_LOCK_RETRY_DELAY * attempt)
                    continue
                log.error(
                    "storage_record_failed chat_id=%s message_id=%s attempts=%d",
                    message.chat_id, message.id, attempt, exc_info=True,
                )
                return

    def _record_sync(
        self,
        message: "IncomingMessage",
        analysis: SignalAnalysis,
        decision: Optional["DecisionResult"],
        structure: Optional["StructureAnalysis"],
        smc: Optional["SMCAnalysis"],
    ) -> None:
        smc_counts_json = None
        if smc is not None:
            counts = {
                "liquidity_pools": len(smc.liquidity_pools),
                "liquidity_sweeps": len(smc.liquidity_sweeps),
                "equal_highs": len(smc.equal_highs),
                "equal_lows": len(smc.equal_lows),
                "fair_value_gaps": len(smc.fair_value_gaps),
                "inverse_fvgs": len(smc.inverse_fvgs),
                "order_blocks": len(smc.order_blocks),
                "breaker_blocks": len(smc.breaker_blocks),
                "mitigation_blocks": len(smc.mitigation_blocks),
                "supply_zones": len(smc.supply_zones),
                "demand_zones": len(smc.demand_zones),
                "premium_zone": 1 if smc.premium_zone is not None else 0,
                "discount_zone": 1 if smc.discount_zone is not None else 0,
                "ote_zone": 1 if smc.ote_zone is not None else 0,
            }
            smc_counts_json = json.dumps(counts)

        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO analyses "
                "(message_id, chat_id, chat_title, sender, message_timestamp, "
                " recorded_at, is_signal, category, source, confidence, analysis_json, "
                " symbol, timeframe, direction, "
                " decision_verdict, decision_confidence, decision_source, "
                " structure_trend, smc_counts_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    message.chat_id,
                    message.chat_title,
                    message.sender,
                    message.timestamp.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    1 if analysis.is_signal else 0,
                    analysis.category,
                    analysis.source,
                    analysis.confidence,
                    json.dumps(analysis.model_dump(mode="json")),
                    analysis.setup.symbol,
                    structure.timeframe.value if structure is not None else None,
                    analysis.setup.direction,
                    decision.verdict.value if decision is not None else None,
                    decision.confidence if decision is not None else None,
                    decision.source.value if decision is not None else None,
                    structure.trend.value if structure is not None else None,
                    smc_counts_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # --------------------------------------------------------------------- read

    async def latest(self) -> Optional[StoredAnalysis]:
        """The most recently recorded analysis, or ``None`` if none yet."""
        rows = await self.history(limit=1)
        return rows[0] if rows else None

    # ------------------------------------------------------------ subscribers

    async def add_subscriber(self, chat_id: int, username: Optional[str] = None) -> bool:
        """Register a chat to receive reports. True if newly added.

        Idempotent: sending /start twice registers once and reports False the
        second time, so the bot can tell the user they were already signed up.
        """
        return await asyncio.to_thread(self._add_subscriber_sync, chat_id, username)

    def _add_subscriber_sync(self, chat_id: int, username: Optional[str]) -> bool:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO subscribers (chat_id, username, subscribed_at) "
                    "VALUES (?, ?, ?)",
                    (chat_id, username, datetime.now(timezone.utc).isoformat()),
                )
            return cur.rowcount > 0
        finally:
            conn.close()

    async def remove_subscriber(self, chat_id: int) -> bool:
        """Unregister a chat. True if it had been registered."""
        return await asyncio.to_thread(self._remove_subscriber_sync, chat_id)

    def _remove_subscriber_sync(self, chat_id: int) -> bool:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute("DELETE FROM subscribers WHERE chat_id = ?", (chat_id,))
            return cur.rowcount > 0
        finally:
            conn.close()

    async def subscribers(self) -> List[int]:
        """Every registered chat id, oldest subscription first."""
        return await asyncio.to_thread(self._subscribers_sync)

    def _subscribers_sync(self) -> List[int]:
        conn = self._connect()
        try:
            return [int(row[0]) for row in conn.execute(
                "SELECT chat_id FROM subscribers ORDER BY subscribed_at, chat_id")]
        finally:
            conn.close()

    # -------------------------------------------------------- signal outcomes

    async def record_signal_outcome(self, row: Dict[str, object]) -> Optional[int]:
        """Insert one decision-time snapshot; the new row id. Never raises.

        Unknown keys are ignored and missing ones stored as NULL, so the
        snapshot can grow without breaking an older database file.
        """
        for attempt in range(1, _LOCK_RETRIES + 1):
            try:
                return await asyncio.to_thread(self._record_signal_outcome_sync, row)
            except Exception as exc:  # noqa: BLE001 - must never break the pipeline
                if self._is_locked_error(exc) and attempt < _LOCK_RETRIES:
                    await asyncio.sleep(_LOCK_RETRY_DELAY * attempt)
                    continue
                log.error("signal_outcome_record_failed symbol=%s attempts=%d",
                          row.get("symbol"), attempt, exc_info=True)
                return None
        return None

    def _record_signal_outcome_sync(self, row: Dict[str, object]) -> int:
        columns = _OUTCOME_SNAPSHOT_COLUMNS
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    f"INSERT INTO signal_outcomes ({', '.join(columns)}) "
                    f"VALUES ({', '.join('?' for _ in columns)})",
                    tuple(row.get(c) for c in columns),
                )
            return int(cur.lastrowid)
        finally:
            conn.close()

    async def open_signal_outcomes(self) -> List[Dict[str, object]]:
        """Every tracked signal whose outcome is not decided yet, oldest first."""
        return await asyncio.to_thread(self._signal_outcomes_sync, "open")

    async def fetch_signal_outcomes(self) -> List[Dict[str, object]]:
        """The whole evaluation dataset, oldest first."""
        return await asyncio.to_thread(self._signal_outcomes_sync, None)

    def _signal_outcomes_sync(self, status: Optional[str]) -> List[Dict[str, object]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            query = "SELECT * FROM signal_outcomes"
            params: Tuple[object, ...] = ()
            if status is not None:
                query += " WHERE status = ?"
                params = (status,)
            return [dict(r) for r in conn.execute(query + " ORDER BY id", params)]
        finally:
            conn.close()

    async def resolve_signal_outcome(self, outcome_id: int, resolution: Dict[str, object]) -> bool:
        """Write an outcome onto an OPEN row. True if this call resolved it.

        The ``status = 'open'`` condition is the guarantee that history is
        never rewritten: a resolved row matches nothing, so a second
        resolution — or any later attempt to change one — is a no-op.
        """
        if resolution.get("status") in (None, "open"):
            raise ValueError("a resolution must close the row")
        return await asyncio.to_thread(self._resolve_signal_outcome_sync, outcome_id, resolution)

    def _resolve_signal_outcome_sync(self, outcome_id: int, resolution: Dict[str, object]) -> bool:
        columns = _OUTCOME_RESOLUTION_COLUMNS
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    f"UPDATE signal_outcomes SET {', '.join(f'{c} = ?' for c in columns)} "
                    f"WHERE id = ? AND status = 'open'",
                    tuple(resolution.get(c) for c in columns) + (outcome_id,),
                )
            return cur.rowcount == 1
        finally:
            conn.close()

    async def find_duplicate(self, row: Dict[str, object], window_hours: float = 6.0) -> Optional[int]:
        """The first earlier signal this one repeats, or None.

        Same symbol, direction, stop and first target, entry within 0.1, and
        recorded within ``window_hours`` before it — a setup posted again (an
        edit, a market and a limit version, a reminder) rather than a new
        trade. Returns the id of the cluster's FIRST signal, so a chain of
        repeats all point at one original. Never raises.
        """
        try:
            return await asyncio.to_thread(self._find_duplicate_sync, row, window_hours)
        except Exception:  # noqa: BLE001 - duplicate tagging must never block recording
            log.warning("duplicate_lookup_failed symbol=%s", row.get("symbol"), exc_info=True)
            return None

    def _find_duplicate_sync(self, row: Dict[str, object], window_hours: float) -> Optional[int]:
        if row.get("entry") is None or row.get("recorded_at") is None:
            return None
        since = (datetime.fromisoformat(str(row["recorded_at"]))
                 - timedelta(hours=window_hours)).isoformat()
        conn = self._connect()
        try:
            found = conn.execute(
                "SELECT id, duplicate_of FROM signal_outcomes WHERE symbol = ? AND direction = ? "
                "AND stop_loss IS ? AND take_profit IS ? AND ABS(entry - ?) <= 0.1 "
                "AND recorded_at >= ? AND recorded_at <= ? ORDER BY id LIMIT 1",
                (row.get("symbol"), row.get("direction"), row.get("stop_loss"),
                 row.get("take_profit"), row["entry"], since, row["recorded_at"]),
            ).fetchone()
        finally:
            conn.close()
        if found is None:
            return None
        return int(found[1]) if found[1] is not None else int(found[0])

    async def record_shadow_decisions(self, outcome_id: int,
                                      decisions: Sequence[Dict[str, object]]) -> None:
        """Store every variant's verdict for one signal. Never raises."""
        try:
            await asyncio.to_thread(self._record_shadow_sync, outcome_id, list(decisions))
        except Exception:  # noqa: BLE001 - shadow records must never break the pipeline
            log.error("shadow_record_failed outcome_id=%s", outcome_id, exc_info=True)

    def _record_shadow_sync(self, outcome_id: int, decisions: List[Dict[str, object]]) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO shadow_decisions (outcome_id, variant, engine_version, "
                    "recorded_at, verdict, confluence_score, adjusted_score, reason, details_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(outcome_id, d["variant"], d.get("engine_version"), d["recorded_at"],
                      d["verdict"], d.get("confluence_score"), d.get("adjusted_score"),
                      d.get("reason"), d.get("details_json")) for d in decisions],
                )
        finally:
            conn.close()

    async def outcomes_needing_accounting(self) -> List[Dict[str, object]]:
        """Tracked signals still missing a final gross or net accounting row."""
        return await asyncio.to_thread(self._needing_accounting_sync)

    def _needing_accounting_sync(self) -> List[Dict[str, object]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(
                "SELECT o.*, "
                " EXISTS(SELECT 1 FROM trade_accounting a WHERE a.outcome_id = o.id "
                "        AND a.basis = 'gross') AS has_gross, "
                " EXISTS(SELECT 1 FROM trade_accounting a WHERE a.outcome_id = o.id "
                "        AND a.basis = 'net') AS has_net "
                "FROM signal_outcomes o WHERE o.status != 'invalid' "
                "AND (NOT EXISTS(SELECT 1 FROM trade_accounting a WHERE a.outcome_id = o.id "
                "                AND a.basis = 'gross') "
                "  OR NOT EXISTS(SELECT 1 FROM trade_accounting a WHERE a.outcome_id = o.id "
                "                AND a.basis = 'net')) "
                "ORDER BY o.id")]
        finally:
            conn.close()

    async def record_accounting(self, outcome_id: int, result: Dict[str, object],
                                horizon_minutes: float) -> bool:
        """Write one FINAL accounting result. True if this call wrote it.

        Write-once: the (signal, basis) key makes a second write a no-op, so
        an accounting result, like an outcome, is never rewritten.
        """
        if result.get("status") not in ("tp", "sl", "timeout", "not_filled", "no_data", "invalid"):
            raise ValueError("only a final accounting result is stored")
        return await asyncio.to_thread(self._record_accounting_sync, outcome_id, result,
                                       horizon_minutes)

    def _record_accounting_sync(self, outcome_id: int, result: Dict[str, object],
                                horizon_minutes: float) -> bool:
        columns = _ACCOUNTING_COLUMNS
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    f"INSERT OR IGNORE INTO trade_accounting (outcome_id, basis, "
                    f"{', '.join(columns)}, horizon_minutes, computed_at) "
                    f"VALUES (?, ?, {', '.join('?' for _ in columns)}, ?, ?)",
                    (outcome_id, result["basis"], *[result.get(c) for c in columns],
                     horizon_minutes, datetime.now(timezone.utc).isoformat()),
                )
            return cur.rowcount == 1
        finally:
            conn.close()

    async def fetch_performance_rows(self) -> List[Dict[str, object]]:
        """Every tracked signal with its accounting and its variant verdicts.

        One dict per signal: the signal_outcomes columns, ``accounting`` keyed
        by basis, and ``variants`` mapping variant name to verdict. The
        champion's verdict is the signal's own ``decision``.
        """
        return await asyncio.to_thread(self._performance_rows_sync)

    def _performance_rows_sync(self) -> List[Dict[str, object]]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = {r["id"]: {**dict(r), "accounting": {}, "variants": {}}
                    for r in conn.execute("SELECT * FROM signal_outcomes ORDER BY id")}
            for a in conn.execute("SELECT * FROM trade_accounting"):
                if a["outcome_id"] in rows:
                    rows[a["outcome_id"]]["accounting"][a["basis"]] = dict(a)
            for s in conn.execute("SELECT * FROM shadow_decisions"):
                if s["outcome_id"] in rows:
                    rows[s["outcome_id"]]["variants"][s["variant"]] = dict(s)
            return list(rows.values())
        finally:
            conn.close()

    # --------------------------------------------------------------- history

    async def history(self, limit: int) -> List[StoredAnalysis]:
        """The ``limit`` most recent analyses, most recent first."""
        return await asyncio.to_thread(self._history_sync, limit)

    def _history_sync(self, limit: int) -> List[StoredAnalysis]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT message_id, chat_title, sender, message_timestamp, analysis_json "
                "FROM analyses ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [
                StoredAnalysis(
                    message_id=row["message_id"],
                    chat_title=row["chat_title"],
                    sender=row["sender"],
                    timestamp=datetime.fromisoformat(row["message_timestamp"]),
                    analysis=SignalAnalysis.model_validate(json.loads(row["analysis_json"])),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    async def fetch_analyses(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[Timeframe] = None,
    ) -> Tuple[AnalysisRecord, ...]:
        """Raw rows for statistics.py to aggregate over, filtered by
        ``recorded_at`` (when the row was written) and/or symbol/timeframe.

        This is the entire "read" surface statistics.py uses — it never
        touches SQL or the database file directly, only this method's
        output, which is what "statistics ... read only from Storage"
        means in practice.
        """
        return await asyncio.to_thread(self._fetch_analyses_sync, since, until, symbol, timeframe)

    def _fetch_analyses_sync(
        self, since: Optional[datetime], until: Optional[datetime],
        symbol: Optional[str], timeframe: Optional[Timeframe],
    ) -> Tuple[AnalysisRecord, ...]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            clauses: List[str] = []
            params: List[object] = []
            if since is not None:
                clauses.append("recorded_at >= ?")
                params.append(since.isoformat())
            if until is not None:
                clauses.append("recorded_at < ?")
                params.append(until.isoformat())
            if symbol is not None:
                clauses.append("symbol = ?")
                params.append(symbol)
            if timeframe is not None:
                clauses.append("timeframe = ?")
                params.append(timeframe.value)

            query = (
                "SELECT recorded_at, is_signal, category, source, confidence, "
                "symbol, timeframe, direction, decision_verdict, decision_confidence, "
                "decision_source, structure_trend, smc_counts_json FROM analyses"
            )
            if clauses:
                query += " WHERE " + " AND ".join(clauses)

            cursor = conn.execute(query, params)
            return tuple(
                AnalysisRecord(
                    recorded_at=datetime.fromisoformat(row["recorded_at"]),
                    is_signal=bool(row["is_signal"]),
                    category=row["category"],
                    source=row["source"],
                    confidence=row["confidence"],
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    direction=row["direction"],
                    decision_verdict=row["decision_verdict"],
                    decision_confidence=row["decision_confidence"],
                    decision_source=row["decision_source"],
                    structure_trend=row["structure_trend"],
                    smc_counts=json.loads(row["smc_counts_json"]) if row["smc_counts_json"] else {},
                )
                for row in cursor.fetchall()
            )
        finally:
            conn.close()

    async def close(self) -> None:
        """No persistent connection is held open — kept for interface parity
        with ``ClaudeAnalyzer.aclose()`` / ``TelegramBot.stop()``."""
        return None
