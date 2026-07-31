"""Unit tests for statistics.py (RFC-009).

Uses a real sqlite3 file per test (pytest's tmp_path), exactly like
test_storage.py — Statistics never opens its own connection, it always
goes through a real Storage instance, so these tests exercise the actual
read path (Storage.fetch_analyses -> Statistics._aggregate), not a mock.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from decision_engine import DecisionResult, DecisionSource, Verdict
from market_data import Timeframe
from models import SignalAnalysis, TradeSetup
from smc_engine import SMCAnalysis
from statistics import Statistics
from storage import Storage
from structure_engine import StructureAnalysis, TrendDirection
from telegram_client import IncomingMessage


def _msg(message_id: int = 1) -> IncomingMessage:
    return IncomingMessage(
        id=message_id, chat_id=-100999, chat_title="VIP Signals", sender="Analyst",
        timestamp=datetime.now(timezone.utc), text="hi",
    )


def _analysis(
    is_signal: bool = True, category: str = "signal", source: str = "regex",
    confidence: float = 0.9, symbol: str = "BTCUSDT", direction: str = "long",
) -> SignalAnalysis:
    return SignalAnalysis(
        is_signal=is_signal, category=category,
        setup=TradeSetup(symbol=symbol, direction=direction, order_type="limit",
                          entries=[61200.0], stop_loss=60350.0, take_profits=[62400.0]),
        summary="Long setup.", confidence=confidence,
        missing_fields=[], notes=None, source=source,
    )


def _decision(verdict: str = "buy", confidence: int = 70) -> DecisionResult:
    return DecisionResult(
        symbol="BTCUSDT", timeframe=Timeframe.M1, verdict=Verdict(verdict), confidence=confidence,
        reasoning="test", strengths=[], risks=[], execution_plan="test",
        source=DecisionSource.FALLBACK,
    )


def _structure(trend: TrendDirection = TrendDirection.BULLISH, timeframe: Timeframe = Timeframe.M1) -> StructureAnalysis:
    return StructureAnalysis(
        symbol="BTCUSDT", timeframe=timeframe, trend=trend,
        swing_points=(), events=(), last_event=None,
    )


def _smc(*, order_blocks: int = 0, fair_value_gaps: int = 0) -> SMCAnalysis:
    return SMCAnalysis(
        symbol="BTCUSDT", timeframe=Timeframe.M1,
        liquidity_pools=(), liquidity_sweeps=(), equal_highs=(), equal_lows=(),
        fair_value_gaps=tuple(range(fair_value_gaps)), inverse_fvgs=(),
        order_blocks=tuple(range(order_blocks)), breaker_blocks=(),
        mitigation_blocks=(), supply_zones=(), demand_zones=(),
        premium_zone=None, discount_zone=None, ote_zone=None,
    )


async def _make_storage(tmp_path: Path) -> Storage:
    storage = Storage(tmp_path / "signals.db")
    await storage.initialize()
    return storage


# --------------------------------------------------------------------------- summary()

def test_summary_of_empty_database_reports_zeroes_and_no_data(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        return await Statistics(storage).summary()

    summary = asyncio.run(run())

    assert summary.total_analyses == 0
    assert summary.total_signals == 0
    assert summary.verdict_counts == {}
    assert summary.wins is None
    assert summary.losses is None
    assert summary.average_signal_confidence is None
    assert summary.average_decision_confidence is None
    assert summary.direction_distribution == {}
    assert summary.trend_distribution == {}
    assert summary.smc_object_frequencies == {}
    assert summary.source_distribution == {}


def test_summary_counts_total_analyses_and_signals(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(is_signal=True))
        await storage.record(_msg(2), _analysis(is_signal=False, category="commentary"))
        return await Statistics(storage).summary()

    summary = asyncio.run(run())

    assert summary.total_analyses == 2
    assert summary.total_signals == 1


def test_summary_average_signal_confidence(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(confidence=0.8))
        await storage.record(_msg(2), _analysis(confidence=0.4))
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert summary.average_signal_confidence == (0.8 + 0.4) / 2


def test_summary_source_distribution(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(source="regex"))
        await storage.record(_msg(2), _analysis(source="claude"))
        await storage.record(_msg(3), _analysis(source="claude"))
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert summary.source_distribution == {"regex": 1, "claude": 2}


def test_summary_direction_distribution_including_none(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(direction="long"))
        await storage.record(_msg(2), _analysis(is_signal=False, direction=None))
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert summary.direction_distribution == {"long": 1, "none": 1}


def test_summary_confidence_distribution_buckets(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(confidence=0.05))   # 0-10%
        await storage.record(_msg(2), _analysis(confidence=0.95))   # 90-100%
        await storage.record(_msg(3), _analysis(confidence=1.0))    # 90-100%
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert summary.confidence_distribution["0-10%"] == 1
    assert summary.confidence_distribution["90-100%"] == 2


# --------------------------------------------------------------------------- decision/structure/smc-derived stats

def test_summary_verdict_counts_only_from_rows_with_a_decision(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(), decision=_decision(verdict="buy"))
        await storage.record(_msg(2), _analysis(), decision=_decision(verdict="sell"))
        await storage.record(_msg(3), _analysis(), decision=_decision(verdict="buy"))
        await storage.record(_msg(4), _analysis())  # no decision recorded
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert summary.verdict_counts == {"buy": 2, "sell": 1}
    assert summary.total_analyses == 4  # the undecided row still counts as an analysis


def test_summary_average_decision_confidence_ignores_rows_without_a_decision(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(), decision=_decision(confidence=80))
        await storage.record(_msg(2), _analysis(), decision=_decision(confidence=40))
        await storage.record(_msg(3), _analysis())  # no decision -> excluded from the average
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert summary.average_decision_confidence == (80 + 40) / 2


def test_summary_trend_distribution_distinguishes_unknown_from_not_recorded(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(), structure=_structure(trend=TrendDirection.BULLISH))
        await storage.record(_msg(2), _analysis(), structure=_structure(trend=TrendDirection.UNKNOWN))
        await storage.record(_msg(3), _analysis())  # no structure recorded at all
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert summary.trend_distribution == {"bullish": 1, "unknown": 1, "not_recorded": 1}


def test_summary_smc_object_frequencies_sum_across_rows(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(), smc=_smc(order_blocks=2, fair_value_gaps=1))
        await storage.record(_msg(2), _analysis(), smc=_smc(order_blocks=3, fair_value_gaps=0))
        await storage.record(_msg(3), _analysis())  # no smc recorded
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert summary.smc_object_frequencies["order_blocks"] == 5
    assert summary.smc_object_frequencies["fair_value_gaps"] == 1


def test_summary_wins_and_losses_are_always_none():
    # No engine in this project tracks trade outcomes yet — this is a
    # permanent, documented "not available" rather than a transient gap.
    from statistics import _aggregate
    summary = _aggregate(())
    assert summary.wins is None
    assert summary.losses is None


# --------------------------------------------------------------------------- today/this_week/this_month

def test_today_excludes_rows_recorded_before_midnight_utc(tmp_path, monkeypatch):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis())
        # Backdate this row's recorded_at to yesterday, directly in SQLite,
        # since Storage.record() always stamps "now".
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "signals.db"))
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        conn.execute("UPDATE analyses SET recorded_at = ? WHERE message_id = 1", (yesterday,))
        conn.commit()
        conn.close()
        return await Statistics(storage).today()

    summary = asyncio.run(run())
    assert summary.total_analyses == 0


def test_today_includes_rows_recorded_today(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis())
        return await Statistics(storage).today()

    summary = asyncio.run(run())
    assert summary.total_analyses == 1


def test_this_week_includes_a_row_from_two_days_ago(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis())
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "signals.db"))
        two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        conn.execute("UPDATE analyses SET recorded_at = ? WHERE message_id = 1", (two_days_ago,))
        conn.commit()
        conn.close()
        return await Statistics(storage).this_week()

    # This test is only meaningful when "2 days ago" is still within the
    # current ISO week; skip the assertion gracefully right after a week
    # boundary rather than risk a flaky failure.
    now = datetime.now(timezone.utc)
    if now.weekday() < 2:
        return
    summary = asyncio.run(run())
    assert summary.total_analyses == 1


def test_this_month_excludes_a_row_from_last_month(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis())
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "signals.db"))
        last_month = (datetime.now(timezone.utc) - timedelta(days=32)).isoformat()
        conn.execute("UPDATE analyses SET recorded_at = ? WHERE message_id = 1", (last_month,))
        conn.commit()
        conn.close()
        return await Statistics(storage).this_month()

    summary = asyncio.run(run())
    assert summary.total_analyses == 0


# --------------------------------------------------------------------------- by_symbol / by_timeframe

def test_by_symbol_filters_correctly(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(symbol="BTCUSDT"))
        await storage.record(_msg(2), _analysis(symbol="ETHUSDT"))
        return await Statistics(storage).by_symbol("ETHUSDT")

    summary = asyncio.run(run())
    assert summary.total_analyses == 1


def test_by_symbol_is_case_insensitive(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(symbol="BTCUSDT"))
        return await Statistics(storage).by_symbol("btcusdt")

    summary = asyncio.run(run())
    assert summary.total_analyses == 1


def test_by_timeframe_filters_correctly(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(), structure=_structure(timeframe=Timeframe.M1))
        await storage.record(_msg(2), _analysis(), structure=_structure(timeframe=Timeframe.H4))
        return await Statistics(storage).by_timeframe(Timeframe.H4)

    summary = asyncio.run(run())
    assert summary.total_analyses == 1


def test_by_timeframe_excludes_rows_with_no_structure_recorded(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis())  # no structure/timeframe at all
        return await Statistics(storage).by_timeframe(Timeframe.M1)

    summary = asyncio.run(run())
    assert summary.total_analyses == 0


# --------------------------------------------------------------------------- format()

def test_format_reports_not_recorded_yet_when_no_decisions_exist(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis())
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    text = summary.format()
    assert "1 analysed" in text
    assert "not recorded yet" in text
    assert "not tracked yet" in text  # win/loss


def test_format_includes_verdicts_when_present(tmp_path):
    async def run():
        storage = await _make_storage(tmp_path)
        await storage.record(_msg(1), _analysis(), decision=_decision(verdict="buy"))
        return await Statistics(storage).summary()

    summary = asyncio.run(run())
    assert "buy=1" in summary.format()


def test_format_never_includes_the_full_confidence_histogram():
    from statistics import _aggregate
    summary = _aggregate(())
    assert "0-10%" not in summary.format()
