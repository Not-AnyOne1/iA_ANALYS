"""Write-once shadow storage beside the live tables, and outcome tracking.

All prices are TEST DATA.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from costs import Costs
from market_data import Candle, CandleSeries, MarketDataError, Timeframe
from models import TradeSetup
from storage import Storage

from analyst.engine import analyse
from analyst.rules import Rules
from analyst.service import analysis_row
from analyst.store import AnalystStore
from analyst.tracker import OutcomeResolver, resolve_trade
from analyst_synth import scenario

UTC = timezone.utc
COSTS = Costs(0.30, 0.10, "assumed")


def run(coro):
    return asyncio.run(coro)


def analysed(message_id=1):
    _, frames, as_of, price = scenario("b_long")
    setup = TradeSetup(symbol="XAUUSD", direction="long", order_type="market",
                       stop_loss=2353.0, take_profits=[2372.0])
    a = analyse("XAUUSD", setup, as_of - timedelta(seconds=30), as_of, price, frames, Rules(), COSTS)
    row = analysis_row(a, setup, chat_id=-100, message_id=message_id,
                       message_at=as_of - timedelta(seconds=30), live_verdict="SKIP",
                       live_confidence=0, card="<b>card</b>", rules=Rules())
    return a, row


@pytest.fixture
def db(tmp_path):
    """The live database as the bot creates it, with one live row in it."""
    path = tmp_path / "signals.db"
    run(Storage(path).initialize())
    con = sqlite3.connect(path)
    before = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    return path, before


# ----------------------------------------------------------------- storage

def test_tables_are_added_beside_the_live_ones_and_init_is_idempotent(db):
    path, before = db
    store = AnalystStore(path)
    run(store.initialize())
    run(store.initialize())
    con = sqlite3.connect(path)
    after = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert before <= after
    assert {"independent_analyses", "independent_outcomes"} <= after - before
    assert "analyses" in before and "subscribers" in before


def test_a_record_survives_a_restart_and_a_repeat_keeps_the_first_row(db):
    path, _ = db
    _, row = analysed()
    store = AnalystStore(path)
    run(store.initialize())
    first = run(store.record(row))
    assert first is not None
    again = run(AnalystStore(path).record({**row, "verdict": "ENTER"}))      # a new process
    assert again is None
    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*), verdict FROM independent_analyses").fetchone() == (1, row["verdict"])


def test_rows_cannot_be_updated_or_deleted(db):
    path, _ = db
    store = AnalystStore(path)
    run(store.initialize())
    run(store.record(analysed()[1]))
    con = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        con.execute("UPDATE independent_analyses SET verdict = 'ENTER'")
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        con.execute("DELETE FROM independent_analyses")


def test_the_trades_to_track_are_fixed_at_decision_time():
    a, row = analysed()
    trades = json.loads(row["trades_json"])
    subjects = [t["subject"] for t in trades]
    assert subjects == ["signal", "B:long"]
    assert all(t["recorded_at"] == a.as_of.isoformat() for t in trades)


def test_recent_signals_and_latest_card(db):
    path, _ = db
    store = AnalystStore(path)
    run(store.initialize())
    a, row = analysed()
    run(store.record(row))
    recent = run(store.recent_signals("XAUUSD", a.as_of - timedelta(hours=4)))
    assert len(recent) == 1 and recent[0].direction == "long" and recent[0].stop_loss == 2353.0
    assert run(store.latest_card()) == "<b>card</b>"


# ----------------------------------------------------------------- outcomes

def bars_after(start, path):
    """M15 bars from ``start``: each (high, low, close)."""
    out, ts, prev = [], start, path[0][2]
    for high, low, close in path:
        out.append(Candle(ts, prev, high, low, close))
        ts, prev = ts + timedelta(minutes=15), close
    return out


TRADE = {"subject": "signal", "direction": "long", "order_type": "market", "entry": 100.0,
         "price_at_decision": 100.0, "stop_loss": 95.0, "take_profits": [110.0],
         "recorded_at": "2026-03-10T10:00:00+00:00"}


def test_a_target_hit_resolves_gross_and_net():
    start = datetime(2026, 3, 10, 10, tzinfo=UTC)
    candles = bars_after(start - timedelta(minutes=15), [(100.5, 99.5, 100.0), (104, 99, 103),
                                                        (111, 102, 110.5)])
    final = resolve_trade(TRADE, candles, start + timedelta(hours=1), COSTS)
    assert final["gross"].status == "tp" and final["gross"].r == pytest.approx(2.0)
    assert final["net"].status == "tp" and final["net"].r < 2.0


def test_stop_and_target_in_one_bar_counts_as_the_stop():
    start = datetime(2026, 3, 10, 10, tzinfo=UTC)
    candles = bars_after(start - timedelta(minutes=15), [(100.5, 99.5, 100.0), (111, 94, 100)])
    final = resolve_trade(TRADE, candles, start + timedelta(minutes=30), COSTS)
    assert final["gross"].status == "sl" and final["gross"].r == pytest.approx(-1.0)


def test_an_open_trade_writes_nothing():
    start = datetime(2026, 3, 10, 10, tzinfo=UTC)
    candles = bars_after(start - timedelta(minutes=15), [(100.5, 99.5, 100.0), (102, 98, 101)])
    assert resolve_trade(TRADE, candles, start + timedelta(minutes=30), COSTS) == {}


class _Market:
    def __init__(self, candles=None, error=None):
        self.candles, self.error, self.calls = candles, error, 0

    async def get_candles(self, symbol, timeframe, limit):
        self.calls += 1
        if self.error:
            raise self.error
        assert timeframe is Timeframe.M15 and limit <= 5000
        return CandleSeries(symbol, timeframe, "test", tuple(self.candles))


def test_resolver_writes_each_outcome_once_and_skips_requests_when_idle(db):
    path, _ = db
    store = AnalystStore(path)
    run(store.initialize())
    a, row = analysed()
    analysis_id = run(store.record(row))
    sl = json.loads(row["trades_json"])[0]["stop_loss"]
    start = a.as_of
    candles = bars_after(start - timedelta(minutes=15), [(2360.5, 2359.5, 2360.0),
                                                        (2361, sl - 1.0, sl - 0.5)])
    market = _Market(candles)
    resolver = OutcomeResolver(store, market, costs=COSTS)
    now = start + timedelta(minutes=30)
    written = run(resolver.resolve_once(now))
    assert written == 4                               # signal and B:long, gross and net
    assert run(resolver.resolve_once(now)) == 0 and market.calls == 1   # nothing pending: no request
    con = sqlite3.connect(path)
    rows = con.execute("SELECT subject, basis, status FROM independent_outcomes "
                       "WHERE analysis_id = ? ORDER BY subject, basis", (analysis_id,)).fetchall()
    assert rows == [("B:long", "gross", "sl"), ("B:long", "net", "sl"),
                    ("signal", "gross", "sl"), ("signal", "net", "sl")]
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        con.execute("UPDATE independent_outcomes SET status = 'tp'")


def test_resolver_survives_a_data_outage(db):
    path, _ = db
    store = AnalystStore(path)
    run(store.initialize())
    a, row = analysed()
    run(store.record(row))
    market = _Market(error=MarketDataError("down", retryable=True))
    assert run(OutcomeResolver(store, market, costs=COSTS).resolve_once(a.as_of + timedelta(hours=1))) == 0
    assert len(run(store.pending(a.as_of - timedelta(days=1)))) == 2
