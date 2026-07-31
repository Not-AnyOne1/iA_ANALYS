"""Tests for dashboard.py (RFC-010).

Two layers, matching how the module itself is split:

- Endpoint function tests: call get_*()/_clamp_limit() directly against a
  real Storage+Statistics backed by a temp SQLite file (same pattern as
  test_statistics.py) — these are the "Statistics rendering tests" and
  most of the "Dashboard endpoint tests" the RFC asks for, and they need
  no HTTP server at all.
- A handful of true HTTP-level smoke tests: start a real DashboardServer
  on an OS-assigned port in a background thread and make real HTTP
  requests, to prove the routing/serialisation wiring itself works end to
  end, not just the underlying functions.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dashboard import (
    DashboardServer,
    _clamp_limit,
    get_by_symbol,
    get_by_timeframe,
    get_confidence_chart,
    get_decision_history,
    get_recent_analyses,
    get_signal_timeline,
    get_summary,
    get_this_month,
    get_this_week,
    get_today,
)
from decision_engine import DecisionResult, DecisionSource, Verdict
from market_data import Timeframe
from models import SignalAnalysis, TradeSetup
from smc_engine import SMCAnalysis
from statistics import Statistics
from storage import Storage
from structure_engine import StructureAnalysis, TrendDirection
from telegram_client import IncomingMessage


# --------------------------------------------------------------------------- helpers

def _msg(message_id: int = 1, symbol: str = "BTCUSDT") -> IncomingMessage:
    return IncomingMessage(
        id=message_id, chat_id=-100999, chat_title="VIP Signals", sender="Analyst",
        timestamp=datetime.now(timezone.utc), text="hi",
    )


def _analysis(is_signal: bool = True, confidence: float = 0.9, symbol: str = "BTCUSDT") -> SignalAnalysis:
    return SignalAnalysis(
        is_signal=is_signal, category="signal" if is_signal else "commentary",
        setup=TradeSetup(symbol=symbol, direction="long", order_type="limit",
                          entries=[61200.0], stop_loss=60350.0, take_profits=[62400.0]),
        summary="Long setup.", confidence=confidence,
        missing_fields=[], notes=None, source="regex",
    )


def _decision(verdict: str = "buy", confidence: int = 70) -> DecisionResult:
    return DecisionResult(
        symbol="BTCUSDT", timeframe=Timeframe.M1, verdict=Verdict(verdict), confidence=confidence,
        reasoning="test", strengths=[], risks=[], execution_plan="test",
        source=DecisionSource.FALLBACK,
    )


def _structure(trend: TrendDirection = TrendDirection.BULLISH) -> StructureAnalysis:
    return StructureAnalysis(
        symbol="BTCUSDT", timeframe=Timeframe.M1, trend=trend,
        swing_points=(), events=(), last_event=None,
    )


async def _make_storage(tmp_path: Path) -> Storage:
    storage = Storage(tmp_path / "signals.db")
    await storage.initialize()
    return storage


# --------------------------------------------------------------------------- _clamp_limit

def test_clamp_limit_defaults_to_twenty_when_absent():
    assert _clamp_limit(None) == 20


def test_clamp_limit_passes_through_a_valid_value():
    assert _clamp_limit("5") == 5


def test_clamp_limit_caps_at_the_maximum():
    assert _clamp_limit("10000") == 200


def test_clamp_limit_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        _clamp_limit("0")
    with pytest.raises(ValueError):
        _clamp_limit("-5")


def test_clamp_limit_rejects_non_numeric():
    with pytest.raises(ValueError):
        _clamp_limit("not-a-number")


# --------------------------------------------------------------------------- statistics-backed endpoints

def test_get_summary_reflects_stored_analyses(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(confidence=0.9))
        await storage.record(_msg(2), _analysis(is_signal=False, confidence=0.3))
        return await get_summary(Statistics(storage))

    data = asyncio.run(run())
    assert data["total_analyses"] == 2
    assert data["total_signals"] == 1
    assert data["average_signal_confidence"] == pytest.approx(0.6)


def test_get_summary_on_empty_database_is_json_safe(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        return await get_summary(Statistics(storage))

    data = asyncio.run(run())
    assert data["total_analyses"] == 0
    assert data["wins"] is None
    assert data["losses"] is None
    # Round-trips through real JSON, not just a Python dict.
    json.dumps(data)


def test_get_today_excludes_backdated_rows(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis())
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "signals.db"))
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        conn.execute("UPDATE analyses SET recorded_at = ?", (yesterday,))
        conn.commit()
        conn.close()
        return await get_today(Statistics(storage))

    data = asyncio.run(run())
    assert data["total_analyses"] == 0


def test_get_this_week_and_this_month_are_json_safe(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis())
        stats = Statistics(storage)
        return await get_this_week(stats), await get_this_month(stats)

    week, month = asyncio.run(run())
    assert week["total_analyses"] == 1
    assert month["total_analyses"] == 1


def test_get_by_symbol_filters(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1, symbol="BTCUSDT"), _analysis(symbol="BTCUSDT"))
        await storage.record(_msg(2, symbol="ETHUSDT"), _analysis(symbol="ETHUSDT"))
        return await get_by_symbol(Statistics(storage), "ETHUSDT")

    data = asyncio.run(run())
    assert data["total_analyses"] == 1


def test_get_by_timeframe_filters(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(), structure=_structure())  # M1
        await storage.record(_msg(2), _analysis(), structure=StructureAnalysis(
            symbol="BTCUSDT", timeframe=Timeframe.H4, trend=TrendDirection.BULLISH,
            swing_points=(), events=(), last_event=None,
        ))
        return await get_by_timeframe(Statistics(storage), Timeframe.H4)

    data = asyncio.run(run())
    assert data["total_analyses"] == 1


def test_get_confidence_chart_is_ordered_and_parallel(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(confidence=0.05))
        await storage.record(_msg(2), _analysis(confidence=0.95))
        return await get_confidence_chart(Statistics(storage))

    data = asyncio.run(run())
    assert len(data["labels"]) == len(data["counts"])
    assert data["labels"] == sorted(data["labels"], key=lambda label: int(label.split("-")[0]))
    assert data["labels"][0] == "0-10%"
    assert data["counts"][0] == 1


# --------------------------------------------------------------------------- storage-backed endpoints

def test_get_recent_analyses_shape_and_order(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        for i in range(1, 4):
            await storage.record(_msg(i), _analysis())
        return await get_recent_analyses(storage, limit=2)

    data = asyncio.run(run())
    assert len(data) == 2
    assert data[0]["message_id"] == 3  # most recent first
    assert data[1]["message_id"] == 2
    assert "analysis" in data[0]
    assert data[0]["analysis"]["setup"]["symbol"] == "BTCUSDT"
    json.dumps(data)  # fully JSON-safe, including the nested datetime


def test_get_signal_timeline_excludes_non_signals(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(is_signal=True))
        await storage.record(_msg(2), _analysis(is_signal=False))
        return await get_signal_timeline(storage, limit=20)

    data = asyncio.run(run())
    assert len(data) == 1
    assert data[0]["analysis"]["is_signal"] is True


def test_get_signal_timeline_empty_when_no_analyses(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        return await get_signal_timeline(storage, limit=20)

    assert asyncio.run(run()) == []


def test_get_decision_history_only_includes_decided_rows_most_recent_first(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(), decision=_decision(verdict="buy"))
        await storage.record(_msg(2), _analysis())  # no decision
        await storage.record(_msg(3), _analysis(), decision=_decision(verdict="sell"))
        return await get_decision_history(storage, limit=20)

    data = asyncio.run(run())
    assert len(data) == 2
    assert data[0]["decision_verdict"] == "sell"  # most recent first
    assert data[1]["decision_verdict"] == "buy"


def test_get_decision_history_respects_limit(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        for i in range(1, 6):
            await storage.record(_msg(i), _analysis(), decision=_decision())
        return await get_decision_history(storage, limit=2)

    assert len(asyncio.run(run())) == 2


# --------------------------------------------------------------------------- HTTP-level smoke tests

@pytest.fixture
def running_server(tmp_path):
    async def setup():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(confidence=0.8))
        await storage.record(_msg(2), _analysis(), decision=_decision(verdict="buy", confidence=60))
        return storage

    storage = asyncio.run(setup())
    server = DashboardServer(storage, Statistics(storage), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_root_serves_html(running_server):
    with urllib.request.urlopen(running_server + "/", timeout=5) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/html")
        body = resp.read().decode("utf-8")
    assert "<title>" in body


def test_http_summary_endpoint(running_server):
    status, data = _get(running_server + "/api/summary")
    assert status == 200
    assert data["total_analyses"] == 2


def test_http_recent_endpoint_respects_limit(running_server):
    status, data = _get(running_server + "/api/recent?limit=1")
    assert status == 200
    assert len(data) == 1


def test_http_decisions_endpoint(running_server):
    status, data = _get(running_server + "/api/decisions")
    assert status == 200
    assert len(data) == 1
    assert data[0]["decision_verdict"] == "buy"


def test_http_by_symbol_without_param_returns_400(running_server):
    status, data = _get(running_server + "/api/by-symbol")
    assert status == 400
    assert "symbol" in data["error"]


def test_http_by_timeframe_with_invalid_value_returns_400(running_server):
    status, data = _get(running_server + "/api/by-timeframe?timeframe=NOTATIMEFRAME")
    assert status == 400


def test_http_unknown_path_returns_404(running_server):
    status, data = _get(running_server + "/api/nonexistent")
    assert status == 404


def test_http_bad_limit_returns_400(running_server):
    status, data = _get(running_server + "/api/recent?limit=abc")
    assert status == 400
