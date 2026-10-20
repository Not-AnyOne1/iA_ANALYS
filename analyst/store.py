"""Write-once SQLite tables for the shadow analyst, beside the existing ones.

Two tables in the same database file as ``storage.py`` (``STORAGE_DB_PATH``,
on the Railway volume), created only when shadow mode is on:

* ``independent_analyses`` — one row per signal, written at decision time:
  the original signal, the live verdict, the analyst's verdict and reasons,
  every proposal, the trades to track and the rendered card.
  ``UNIQUE(chat_id, message_id)``: a signal analysed twice keeps its first
  row.
* ``independent_outcomes`` — one row per tracked trade and basis
  (gross / net), written once the trade has a final outcome.

Neither table is ever updated or deleted from: triggers abort any UPDATE or
DELETE, so a prediction cannot be rewritten after its outcome is known. The
existing tables are not touched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from trade_accounting import TradeResult

from analyst.verdict import RecentSignal

log = logging.getLogger(__name__)

_BUSY_TIMEOUT_SECONDS = 10.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS independent_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    message_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    mode TEXT NOT NULL,
    price_at_decision REAL,
    data_status TEXT NOT NULL,
    live_verdict TEXT,
    live_confidence INTEGER,
    verdict TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    checks_met INTEGER,
    checks_total INTEGER,
    signal_json TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    trades_json TEXT NOT NULL,
    card_text TEXT NOT NULL,
    UNIQUE (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_independent_analyses_created ON independent_analyses (created_at);

CREATE TABLE IF NOT EXISTS independent_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES independent_analyses (id),
    subject TEXT NOT NULL,
    basis TEXT NOT NULL,
    status TEXT NOT NULL,
    r REAL,
    entry_fill REAL,
    exit_price REAL,
    exit_at TEXT,
    cost_source TEXT,
    note TEXT,
    resolved_at TEXT NOT NULL,
    UNIQUE (analysis_id, subject, basis)
);

CREATE TRIGGER IF NOT EXISTS independent_analyses_no_update BEFORE UPDATE ON independent_analyses
BEGIN SELECT RAISE(ABORT, 'independent_analyses is write-once'); END;
CREATE TRIGGER IF NOT EXISTS independent_analyses_no_delete BEFORE DELETE ON independent_analyses
BEGIN SELECT RAISE(ABORT, 'independent_analyses is write-once'); END;
CREATE TRIGGER IF NOT EXISTS independent_outcomes_no_update BEFORE UPDATE ON independent_outcomes
BEGIN SELECT RAISE(ABORT, 'independent_outcomes is write-once'); END;
CREATE TRIGGER IF NOT EXISTS independent_outcomes_no_delete BEFORE DELETE ON independent_outcomes
BEGIN SELECT RAISE(ABORT, 'independent_outcomes is write-once'); END;
"""

_COLUMNS = ("created_at", "chat_id", "message_id", "message_at", "symbol", "engine_version", "mode",
            "price_at_decision", "data_status", "live_verdict", "live_confidence", "verdict",
            "reason_codes", "checks_met", "checks_total", "signal_json", "analysis_json",
            "trades_json", "card_text")


class AnalystStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=_BUSY_TIMEOUT_SECONDS)
        conn.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_SECONDS * 1000)}")
        conn.row_factory = sqlite3.Row
        return conn

    async def _run(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    # ------------------------------------------------------------- schema

    async def initialize(self) -> None:
        await self._run(self._initialize_sync)

    def _initialize_sync(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------- write

    async def record(self, row: Dict[str, Any]) -> Optional[int]:
        """Insert one analysis. Returns its id, or None when this signal was
        already recorded (the first row is kept) or the write failed."""
        try:
            return await self._run(self._record_sync, row)
        except Exception:  # noqa: BLE001 - shadow storage must never break the pipeline
            log.exception("analyst_record_failed message_id=%s", row.get("message_id"))
            return None

    def _record_sync(self, row: Dict[str, Any]) -> Optional[int]:
        conn = self._connect()
        try:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO independent_analyses ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})", [row[c] for c in _COLUMNS])
            conn.commit()
            return cur.lastrowid if cur.rowcount == 1 else None
        finally:
            conn.close()

    async def record_outcome(self, analysis_id: int, subject: str,
                             results: Dict[str, TradeResult], resolved_at: datetime) -> int:
        """Insert final outcomes (one per basis). Returns rows written."""
        return await self._run(self._record_outcome_sync, analysis_id, subject, results, resolved_at)

    def _record_outcome_sync(self, analysis_id, subject, results, resolved_at) -> int:
        conn = self._connect()
        try:
            written = 0
            for basis, r in results.items():
                if not r.final:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO independent_outcomes (analysis_id, subject, basis, status, "
                    "r, entry_fill, exit_price, exit_at, cost_source, note, resolved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (analysis_id, subject, basis, r.status, r.r, r.entry_fill, r.exit_price,
                     r.exit_at.isoformat() if r.exit_at else None, r.costs.source, r.note,
                     resolved_at.isoformat()))
                written += cur.rowcount
            conn.commit()
            return written
        finally:
            conn.close()

    # --------------------------------------------------------------- read

    async def recent_signals(self, symbol: str, since: datetime) -> List[RecentSignal]:
        try:
            return await self._run(self._recent_sync, symbol, since)
        except Exception:  # noqa: BLE001 - duplicate detection degrades to "none seen"
            log.exception("analyst_recent_failed")
            return []

    def _recent_sync(self, symbol: str, since: datetime) -> List[RecentSignal]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT message_at, signal_json FROM independent_analyses "
                                "WHERE symbol = ? AND message_at >= ? ORDER BY id",
                                (symbol, since.isoformat())).fetchall()
        finally:
            conn.close()
        out = []
        for row in rows:
            s = json.loads(row["signal_json"])
            out.append(RecentSignal(datetime.fromisoformat(row["message_at"]), s.get("direction"),
                                    s.get("stop_loss"), tuple(s.get("take_profits") or ())))
        return out

    async def latest_card(self) -> Optional[str]:
        return await self._run(self._latest_card_sync)

    def _latest_card_sync(self) -> Optional[str]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT card_text FROM independent_analyses "
                               "ORDER BY id DESC LIMIT 1").fetchone()
            return row["card_text"] if row else None
        finally:
            conn.close()

    async def pending(self, since: datetime) -> List[Dict[str, Any]]:
        """Tracked trades recorded since ``since`` with no final net/gross pair yet."""
        return await self._run(self._pending_sync, since)

    def _pending_sync(self, since: datetime) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT id, trades_json FROM independent_analyses "
                                "WHERE created_at >= ? ORDER BY id", (since.isoformat(),)).fetchall()
            done = {(r["analysis_id"], r["subject"]) for r in conn.execute(
                "SELECT analysis_id, subject FROM independent_outcomes GROUP BY analysis_id, subject "
                "HAVING COUNT(*) >= 2").fetchall()}
        finally:
            conn.close()
        out = []
        for row in rows:
            for trade in json.loads(row["trades_json"]):
                if (row["id"], trade["subject"]) not in done:
                    out.append({**trade, "analysis_id": row["id"]})
        return out
