"""Unit tests for storage.py (RFC-003).

Uses a real sqlite3 file per test (pytest's tmp_path), not ":memory:" —
Storage opens a fresh connection per operation, and a ":memory:" database is
private to the connection that created it, so separate connections would
each see an empty database. A real temp file is required for writes made on
one connection to be visible to a later one.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from decision_engine import DecisionResult, DecisionSource, Verdict
from market_data import Timeframe
from models import SignalAnalysis, TradeSetup
from smc_engine import SMCAnalysis
from storage import Storage
from structure_engine import StructureAnalysis, TrendDirection
from telegram_client import IncomingMessage


def _msg(message_id: int = 1, chat_id: int = -100999, chat_title: str = "VIP Signals",
         sender: str = "Analyst") -> IncomingMessage:
    return IncomingMessage(
        id=message_id, chat_id=chat_id, chat_title=chat_title, sender=sender,
        timestamp=datetime.now(timezone.utc), text="hi",
    )


def _analysis(
    is_signal: bool = True, category: str = "signal", source: str = "regex",
    symbol: str = "BTCUSDT",
) -> SignalAnalysis:
    return SignalAnalysis(
        is_signal=is_signal, category=category,
        setup=TradeSetup(symbol=symbol, direction="long", order_type="limit",
                          entries=[61200.0], stop_loss=60350.0, take_profits=[62400.0]),
        summary="Long setup.", confidence=0.9,
        missing_fields=[], notes=None, source=source,
    )


def _decision(verdict: str = "buy", confidence: int = 70, symbol: str = "BTCUSDT") -> DecisionResult:
    return DecisionResult(
        symbol=symbol, timeframe=Timeframe.M1, verdict=Verdict(verdict), confidence=confidence,
        reasoning="test", strengths=[], risks=[], execution_plan="test",
        source=DecisionSource.FALLBACK,
    )


def _structure(trend: TrendDirection = TrendDirection.BULLISH, symbol: str = "BTCUSDT") -> StructureAnalysis:
    return StructureAnalysis(
        symbol=symbol, timeframe=Timeframe.M1, trend=trend,
        swing_points=(), events=(), last_event=None,
    )


def _smc(symbol: str = "BTCUSDT", *, order_blocks=0, fair_value_gaps=0) -> SMCAnalysis:
    return SMCAnalysis(
        symbol=symbol, timeframe=Timeframe.M1,
        liquidity_pools=(), liquidity_sweeps=(), equal_highs=(), equal_lows=(),
        fair_value_gaps=tuple(range(fair_value_gaps)), inverse_fvgs=(),
        order_blocks=tuple(range(order_blocks)), breaker_blocks=(),
        mitigation_blocks=(), supply_zones=(), demand_zones=(),
        premium_zone=None, discount_zone=None, ote_zone=None,
    )


def _storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "signals.db")


def test_initialize_creates_schema(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "signals.db"))
    try:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    finally:
        conn.close()
    assert "analyses" in tables


def test_latest_with_empty_database_returns_none(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    assert asyncio.run(storage.latest()) is None


def test_history_with_empty_database_returns_empty_list(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    assert asyncio.run(storage.history(limit=10)) == []


def test_fetch_analyses_with_empty_database_returns_empty_tuple(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    assert asyncio.run(storage.fetch_analyses()) == ()


def test_record_and_latest_roundtrip(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    message, analysis = _msg(message_id=7), _analysis()

    asyncio.run(storage.record(message, analysis))
    latest = asyncio.run(storage.latest())

    assert latest is not None
    assert latest.message_id == 7
    assert latest.chat_title == "VIP Signals"
    assert latest.sender == "Analyst"
    assert latest.analysis == analysis


def test_history_returns_most_recent_first_and_respects_limit(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    for i in range(1, 6):
        asyncio.run(storage.record(_msg(message_id=i), _analysis()))

    records = asyncio.run(storage.history(limit=2))

    assert [r.message_id for r in records] == [5, 4]


def test_fetch_analyses_returns_analysis_only_fields_when_no_extras_given(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    asyncio.run(storage.record(_msg(1), _analysis(is_signal=True, category="signal", source="regex")))

    records = asyncio.run(storage.fetch_analyses())

    assert len(records) == 1
    r = records[0]
    assert r.is_signal is True
    assert r.category == "signal"
    assert r.source == "regex"
    assert r.symbol == "BTCUSDT"  # extracted from analysis.setup.symbol
    assert r.direction == "long"  # extracted from analysis.setup.direction
    # Nothing supplied decision/structure/smc to record() -> all None/empty,
    # exactly as pipeline.py's current (unmodified) call sites produce.
    assert r.timeframe is None
    assert r.decision_verdict is None
    assert r.decision_confidence is None
    assert r.decision_source is None
    assert r.structure_trend is None
    assert r.smc_counts == {}


def test_record_persists_optional_decision_structure_and_smc(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    asyncio.run(storage.record(
        _msg(1), _analysis(),
        decision=_decision(verdict="buy", confidence=65),
        structure=_structure(trend=TrendDirection.BULLISH),
        smc=_smc(order_blocks=2, fair_value_gaps=1),
    ))

    records = asyncio.run(storage.fetch_analyses())

    assert len(records) == 1
    r = records[0]
    assert r.timeframe == "M1"
    assert r.decision_verdict == "buy"
    assert r.decision_confidence == 65
    assert r.decision_source == "fallback"
    assert r.structure_trend == "bullish"
    assert r.smc_counts["order_blocks"] == 2
    assert r.smc_counts["fair_value_gaps"] == 1


def test_fetch_analyses_filters_by_symbol(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    asyncio.run(storage.record(_msg(1), _analysis(symbol="BTCUSDT")))
    asyncio.run(storage.record(_msg(2), _analysis(symbol="ETHUSDT")))

    records = asyncio.run(storage.fetch_analyses(symbol="ETHUSDT"))

    assert len(records) == 1
    assert records[0].symbol == "ETHUSDT"


def test_fetch_analyses_filters_by_timeframe(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    asyncio.run(storage.record(_msg(1), _analysis(), structure=_structure()))  # M1
    asyncio.run(storage.record(_msg(2), _analysis(), structure=StructureAnalysis(
        symbol="BTCUSDT", timeframe=Timeframe.H4, trend=TrendDirection.BULLISH,
        swing_points=(), events=(), last_event=None,
    )))

    records = asyncio.run(storage.fetch_analyses(timeframe=Timeframe.H4))

    assert len(records) == 1
    assert records[0].timeframe == "H4"


def test_fetch_analyses_filters_by_since_and_until(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    asyncio.run(storage.record(_msg(1), _analysis()))

    now = datetime.now(timezone.utc)
    assert asyncio.run(storage.fetch_analyses(since=now + timedelta(hours=1))) == ()
    assert len(asyncio.run(storage.fetch_analyses(since=now - timedelta(hours=1)))) == 1
    assert asyncio.run(storage.fetch_analyses(until=now - timedelta(hours=1))) == ()
    assert len(asyncio.run(storage.fetch_analyses(until=now + timedelta(hours=1)))) == 1


def test_initialize_adds_new_columns_to_a_pre_rfc009_database(tmp_path):
    # Simulate an existing RFC-003-era database file: only the original
    # columns exist, none of RFC-009's additive ones.
    db_path = tmp_path / "signals.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE analyses (
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
    """)
    conn.commit()
    conn.close()

    storage = Storage(db_path)
    asyncio.run(storage.initialize())  # must not raise, must add the new columns

    conn = sqlite3.connect(str(db_path))
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()}
    finally:
        conn.close()
    for expected in ("symbol", "timeframe", "direction", "decision_verdict",
                     "decision_confidence", "decision_source", "structure_trend", "smc_counts_json"):
        assert expected in columns

    # And the upgraded database is still fully usable afterward.
    asyncio.run(storage.record(_msg(1), _analysis()))
    assert len(asyncio.run(storage.fetch_analyses())) == 1


def test_initialize_is_idempotent_when_columns_already_exist(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    asyncio.run(storage.initialize())  # must not raise ("duplicate column" etc.)
    asyncio.run(storage.record(_msg(1), _analysis()))
    assert len(asyncio.run(storage.fetch_analyses())) == 1


def test_record_failure_is_swallowed_not_raised(tmp_path):
    # Point the "db path" at a directory instead of a file — sqlite3.connect
    # on a path that IS a directory raises, exercising record()'s own
    # internal failure isolation (it must never raise into pipeline.py).
    storage = Storage(tmp_path)

    asyncio.run(storage.record(_msg(), _analysis()))  # must not raise


def test_analysis_round_trips_through_json_without_field_loss(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    analysis = _analysis()
    analysis = analysis.model_copy(update={"notes": "keep an eye on this one"})

    asyncio.run(storage.record(_msg(), analysis))
    latest = asyncio.run(storage.latest())

    assert latest.analysis.notes == "keep an eye on this one"
    assert latest.analysis.setup.symbol == "BTCUSDT"
    assert latest.analysis.confidence == pytest.approx(0.9)


# --------------------------------------------------------------- WAL mode & indexes

def _indexes(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    finally:
        conn.close()


def _journal_mode(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def test_initialize_enables_wal_mode(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    assert _journal_mode(tmp_path / "signals.db") == "wal"


def test_wal_mode_persists_across_reconnects(tmp_path):
    # WAL is a persistent property of the file, so a brand-new Storage
    # instance (i.e. a restart) must still see it.
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    reopened = Storage(tmp_path / "signals.db")
    asyncio.run(reopened.initialize())

    assert _journal_mode(tmp_path / "signals.db") == "wal"


def test_initialize_creates_the_query_indexes(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    names = _indexes(tmp_path / "signals.db")
    for expected in ("idx_analyses_recorded_at", "idx_analyses_symbol", "idx_analyses_timeframe"):
        assert expected in names


def test_indexes_are_created_when_upgrading_a_pre_rfc009_database(tmp_path):
    # The ordering trap: symbol/timeframe don't exist on an old database
    # until the ALTER TABLE migration runs, so their indexes must be created
    # after it — not as part of the base schema.
    db_path = tmp_path / "signals.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE analyses (
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
    """)
    conn.commit()
    conn.close()

    storage = Storage(db_path)
    asyncio.run(storage.initialize())  # must not raise

    names = _indexes(db_path)
    for expected in ("idx_analyses_recorded_at", "idx_analyses_symbol", "idx_analyses_timeframe"):
        assert expected in names

    # ...and the upgraded database still works end to end.
    asyncio.run(storage.record(_msg(1), _analysis()))
    assert len(asyncio.run(storage.fetch_analyses(symbol="BTCUSDT"))) == 1


def test_initialize_remains_idempotent_with_indexes_and_wal(tmp_path):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    asyncio.run(storage.initialize())  # must not raise on existing indexes

    asyncio.run(storage.record(_msg(1), _analysis()))
    assert len(asyncio.run(storage.fetch_analyses())) == 1


def test_filtered_queries_use_an_index_rather_than_scanning(tmp_path):
    # Guards the actual point of the indexes: SQLite's query planner must
    # choose them for the columns fetch_analyses() filters on.
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    asyncio.run(storage.record(_msg(1), _analysis()))

    conn = sqlite3.connect(str(tmp_path / "signals.db"))
    try:
        for column, index in (
            ("symbol = 'BTCUSDT'", "idx_analyses_symbol"),
            ("timeframe = 'H1'", "idx_analyses_timeframe"),
            ("recorded_at >= '2020-01-01'", "idx_analyses_recorded_at"),
        ):
            plan = " ".join(
                str(row) for row in conn.execute(
                    f"EXPLAIN QUERY PLAN SELECT id FROM analyses WHERE {column}"
                ).fetchall()
            )
            assert index in plan, f"query planner did not use {index}: {plan}"
    finally:
        conn.close()


# ------------------------------------------------- write contention (regression)

def test_connections_use_a_long_busy_timeout(tmp_path):
    """A short busy timeout silently lost rows under concurrent writers.

    record() cannot raise (pipeline isolation), so a lock failure is invisible
    unless the timeout is generous enough to never be hit in practice.
    """
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())
    conn = storage._connect()
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 30_000
    finally:
        conn.close()


def test_concurrent_writers_do_not_lose_rows(tmp_path):
    storage = _storage(tmp_path)

    async def run():
        await storage.initialize()
        await asyncio.gather(*(storage.record(_msg(i), _analysis()) for i in range(150)))
        return await storage.fetch_analyses()

    assert len(asyncio.run(run())) == 150


def test_record_retries_a_locked_database(tmp_path, monkeypatch):
    """A transient lock must be retried, not silently dropped."""
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    calls = {"n": 0}
    real = storage._record_sync

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(*args, **kwargs)

    monkeypatch.setattr(storage, "_record_sync", flaky)
    asyncio.run(storage.record(_msg(1), _analysis()))

    assert calls["n"] == 2                                   # retried once
    assert len(asyncio.run(storage.fetch_analyses())) == 1   # and persisted


def test_record_gives_up_on_a_non_lock_error_without_retrying(tmp_path, monkeypatch):
    storage = _storage(tmp_path)
    asyncio.run(storage.initialize())

    calls = {"n": 0}

    def broken(*args, **kwargs):
        calls["n"] += 1
        raise ValueError("not a lock problem")

    monkeypatch.setattr(storage, "_record_sync", broken)
    asyncio.run(storage.record(_msg(1), _analysis()))  # must not raise

    assert calls["n"] == 1
