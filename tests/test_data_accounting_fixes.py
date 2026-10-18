"""Regression tests: weekend stub filtering, gap-through-stop fills, persistent safe logging."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import logging_setup
import market_hours as mh
from costs import Costs
from market_data import Candle, CandleSeries, MarketDataError, MarketDataService, Timeframe, clean_candle_series
from trade_accounting import account

UTC = timezone.utc
M15 = timedelta(minutes=15)


def _c(ts, o, h, l, c):  # noqa: E741
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=None)


def _utc(*a):
    return datetime(*a, tzinfo=UTC)


# ------------------------------------------------------------ session hours

@pytest.mark.parametrize("last_open,first_closed,last_closed,reopen", [
    # US summer (EDT): Friday 21:00 .. Sunday 22:00 UTC
    (_utc(2026, 7, 3, 20, 45), _utc(2026, 7, 3, 21, 0), _utc(2026, 7, 5, 21, 45), _utc(2026, 7, 5, 22, 0)),
    # US winter (EST): Friday 22:00 .. Sunday 23:00 UTC
    (_utc(2026, 1, 9, 21, 45), _utc(2026, 1, 9, 22, 0), _utc(2026, 1, 11, 22, 45), _utc(2026, 1, 11, 23, 0)),
    # DST starts Sunday 2026-03-08: closes on winter time, reopens on summer time
    (_utc(2026, 3, 6, 21, 45), _utc(2026, 3, 6, 22, 0), _utc(2026, 3, 8, 21, 45), _utc(2026, 3, 8, 22, 0)),
    # DST ends Sunday 2026-11-01: closes on summer time, reopens on winter time
    (_utc(2026, 10, 30, 20, 45), _utc(2026, 10, 30, 21, 0), _utc(2026, 11, 1, 22, 45), _utc(2026, 11, 1, 23, 0)),
])
def test_weekly_closure_boundaries_follow_new_york_time(last_open, first_closed, last_closed, reopen):
    assert not mh.bar_inside_weekly_closure(last_open, M15)
    assert mh.bar_inside_weekly_closure(first_closed, M15)
    assert mh.bar_inside_weekly_closure(last_closed, M15)
    assert not mh.bar_inside_weekly_closure(reopen, M15)            # the Sunday reopening is kept


def test_a_bar_that_also_covers_live_trading_is_kept():
    h4 = timedelta(hours=4)
    assert not mh.bar_inside_weekly_closure(_utc(2026, 7, 5, 21), h4)    # spans the 22:00 reopen
    assert mh.bar_inside_weekly_closure(_utc(2026, 7, 3, 21), h4)        # wholly inside
    assert not mh.bar_inside_weekly_closure(_utc(2026, 1, 9, 21), h4)    # winter: 21-22 still trades
    assert not mh.bar_inside_weekly_closure(_utc(2026, 7, 4), timedelta(days=1))  # D1: range rule instead


def test_only_metals_follow_the_gold_session():
    assert mh.follows_weekly_session("XAUUSD") and mh.follows_weekly_session("xau/usd")
    assert not mh.follows_weekly_session("EURUSD") and not mh.follows_weekly_session("BTCUSDT")


# ------------------------------------------------------- the cleaning step

def _week_m15():
    """Friday trading, Saturday stubs (one wildly off-market), the summer Sunday reopen."""
    fri = [_c(_utc(2026, 7, 3, 20, 30) + i * M15, 4000, 4001, 3999, 4000) for i in range(2)]
    stubs = [_c(_utc(2026, 7, 4, 10), 4000, 4000.5, 3999.5, 4000),
             _c(_utc(2026, 7, 4, 11), 4000, 4421, 3580, 4100),
             _c(_utc(2026, 7, 5, 21, 45), 4005, 4006, 4004, 4005)]
    sun = [_c(_utc(2026, 7, 5, 22) + i * M15, 4010, 4012, 4008, 4011) for i in range(3)]
    return fri, stubs, sun


def test_clean_candle_series_drops_gold_weekend_stubs_and_keeps_the_reopen():
    fri, stubs, sun = _week_m15()
    series = CandleSeries("XAUUSD", Timeframe.M15, "stub", tuple(fri + stubs + sun))
    assert clean_candle_series(series).candles == tuple(fri + sun)


def test_each_bar_is_judged_on_its_own_timestamp():
    """No lookahead: cleaning a bar alone or inside the series gives the same answer."""
    fri, stubs, sun = _week_m15()
    whole = clean_candle_series(CandleSeries("XAUUSD", Timeframe.M15, "stub", tuple(fri + stubs + sun)))
    alone = [c for c in fri + stubs + sun
             if clean_candle_series(CandleSeries("XAUUSD", Timeframe.M15, "stub", (c,))).candles]
    assert tuple(alone) == whole.candles


class _Provider:
    name = "stub"

    def __init__(self, candles=(), fail=None):
        self.candles, self.fail = tuple(candles), fail

    async def get_quote(self, symbol):
        raise MarketDataError("no quotes here", retryable=False)

    async def get_candles(self, symbol, timeframe, limit):
        if self.fail:
            raise self.fail
        return CandleSeries(symbol, timeframe, self.name, self.candles)


def test_the_live_data_service_serves_cleaned_bars():
    fri, stubs, sun = _week_m15()
    series = asyncio.run(MarketDataService(_Provider(fri + stubs + sun), cache_ttl=0)
                         .get_candles("XAUUSD", Timeframe.M15, 10))
    assert series.candles == tuple(fri + sun)


def test_outcome_tracking_never_sees_a_weekend_stub(tmp_path):
    """A long opened on Friday: the Saturday stub at 3580 would have 'hit' its
    3950 stop. With the stub gone the trade resolves on real Sunday bars."""
    from costs import CostModel
    from outcome_tracker import OutcomeTracker
    from storage import Storage
    fri, stubs, sun = _week_m15()
    sun += [_c(_utc(2026, 7, 5, 23), 4011, 4102, 4010, 4100)]
    s = Storage(tmp_path / "t.db")
    asyncio.run(s.initialize())
    oid = asyncio.run(s.record_signal_outcome({
        "recorded_at": _utc(2026, 7, 3, 20, 29).isoformat(), "symbol": "XAUUSD", "direction": "long",
        "order_type": "market", "entry": 4000.0, "stop_loss": 3950.0, "take_profit": 4100.0,
        "take_profits_json": "[4100.0]", "price_at_decision": 4000.0, "decision": "enter",
        "status": "open", "cost_spread": 0.3, "cost_slippage": 0.1, "cost_source": "assumed"}))
    pre = _c(_utc(2026, 7, 3, 20, 15), 4000, 4001, 3999, 4000)
    tracker = OutcomeTracker(s, MarketDataService(_Provider([pre] + fri + stubs + sun), cache_ttl=0),
                             cost_model=CostModel(), experiments=(), clock=lambda: _utc(2026, 7, 6, 12))
    asyncio.run(tracker.resolve_pending())
    row = [r for r in asyncio.run(s.fetch_performance_rows()) if r["id"] == oid][0]
    assert row["status"] == "tp"
    assert row["accounting"]["net"]["status"] == "tp" and row["accounting"]["gross"]["status"] == "tp"


# ---------------------------------------------------- gap-through stops

T0 = _utc(2026, 7, 3, 20, 7)
COSTS = Costs(spread=0.4, slippage=0.1, source="assumed")      # half spread 0.2


def _row(direction="long", order_type="market", entry=100.0, stop=95.0, target=110.0, live=100.0):
    return {"recorded_at": T0.isoformat(), "direction": direction, "order_type": order_type,
            "entry": entry, "stop_loss": stop, "take_profit": target,
            "take_profits_json": json.dumps([target]), "price_at_decision": live}


def _bars(*specs, start=_utc(2026, 7, 3, 20, 15)):
    pre = _c(_utc(2026, 7, 3, 20, 0), 100, 100.2, 99.8, 100)
    return [pre] + [_c(start + i * M15, *s) for i, s in enumerate(specs)]


LATER = _utc(2026, 7, 10)


def test_a_long_stop_gapped_through_exits_at_the_open_bid_minus_slippage():
    candles = _bars((100, 100.5, 99.6, 100.2)) + [_c(_utc(2026, 7, 5, 22), 92, 93, 91, 92.5)]
    net = account(_row(), candles, LATER, costs=COSTS)
    gross = account(_row(), candles, LATER)
    assert net.status == "sl" and net.exit_price == pytest.approx(92 - 0.2 - 0.1)
    assert net.exit_at == _utc(2026, 7, 5, 22) and "opened through the stop" in net.note
    assert net.r == pytest.approx((91.7 - 100.3) / 5)                    # far worse than -1R
    assert gross.exit_price == pytest.approx(92.0) and gross.r == pytest.approx(-1.6)


def test_a_short_stop_gapped_through_exits_at_the_open_ask_plus_slippage():
    row = _row(direction="short", stop=105.0, target=90.0)
    candles = _bars((100, 100.4, 99.5, 100)) + [_c(_utc(2026, 7, 5, 22), 108, 109, 107, 108)]
    net = account(row, candles, LATER, costs=COSTS)
    assert net.status == "sl" and net.exit_price == pytest.approx(108 + 0.2 + 0.1)


def test_a_normal_stop_is_unchanged():
    candles = _bars((100, 100.5, 94.0, 95))
    net = account(_row(), candles, LATER, costs=COSTS)
    assert net.status == "sl" and net.exit_price == pytest.approx(95 - 0.1) and net.note == ""


def test_an_open_exactly_at_the_stop_costs_what_a_normal_stop_costs():
    candles = _bars((100, 100.5, 99.6, 100)) + [_c(_utc(2026, 7, 5, 22), 95.2, 96, 95, 95.5)]
    net = account(_row(), candles, LATER, costs=COSTS)
    assert net.status == "sl" and net.exit_price == pytest.approx(95 - 0.1)


def test_a_market_orders_first_bar_can_open_through_the_stop():
    candles = _bars((93, 94, 92, 93))
    net = account(_row(), candles, LATER, costs=COSTS)
    assert net.status == "sl" and net.exit_price == pytest.approx(93 - 0.3)


def test_a_resting_orders_fill_bar_is_judged_the_normal_way():
    """The limit filled INSIDE the bar, after its open — the open is not a fill."""
    row = _row(order_type="limit", entry=99.0, stop=95.0, live=100.0)
    candles = _bars((94.9, 99.5, 94.5, 99))       # opens under the stop, then trades up to the limit
    net = account(row, candles, LATER, costs=COSTS)
    assert net.status == "sl" and net.exit_price == pytest.approx(95 - 0.1)


def test_a_gapped_target_still_fills_at_the_target():
    candles = _bars((100, 100.5, 99.6, 100.2)) + [_c(_utc(2026, 7, 5, 22), 115, 116, 114, 115)]
    net = account(_row(), candles, LATER, costs=COSTS)
    assert net.status == "tp" and net.exit_price == pytest.approx(110.0)


def test_the_e1_breakeven_stop_gapped_through_exits_at_the_open():
    candles = _bars((100, 103.5, 100.6, 103)) + [_c(_utc(2026, 7, 5, 22), 97, 98, 96.5, 97)]
    net = account(_row(), candles, LATER, costs=COSTS, breakeven_at=0.5)
    assert net.status == "breakeven" and net.exit_price == pytest.approx(97 - 0.3)
    # The gap only crossed the MOVED stop; the baseline's 95 stop was never reached.
    assert account(_row(), candles, LATER, costs=COSTS).status == "open"


def test_a_short_e1_breakeven_stop_gapped_through_exits_at_the_open_ask():
    row = _row(direction="short", stop=105.0, target=90.0)
    # Short market fill at the bid - slip = 99.7; arms once the ask is 2.5 below it
    # (mid low <= 97.0); then the market reopens at 103, above the moved stop (99.7).
    candles = _bars((100, 100.2, 96.8, 97)) + [_c(_utc(2026, 7, 5, 22), 103, 104, 102.5, 103)]
    net = account(row, candles, LATER, costs=COSTS, breakeven_at=0.5)
    assert net.status == "breakeven" and net.exit_price == pytest.approx(103 + 0.2 + 0.1)
    assert account(row, candles, LATER, costs=COSTS).status == "open"      # 105 stop untouched


# --------------------------------------------------------------- logging

@pytest.fixture
def isolated_logging():
    root = logging.getLogger()
    saved = (root.level, list(root.handlers), sys.excepthook, threading.excepthook)
    yield
    for h in root.handlers:
        if getattr(h, "_owned_by_logging_setup", False):
            h.close()
    root.setLevel(saved[0])
    root.handlers[:] = saved[1]
    sys.excepthook, threading.excepthook = saved[2], saved[3]


SECRET = "s3cr3t-value-123456"
TOKEN = "1234567890:" + "A" * 35
FAKE_ANTHROPIC = "sk-" + "ant-" + "api03-" + "z" * 12      # built at runtime: not a key-shaped literal


def test_secrets_never_reach_the_log_file(tmp_path, monkeypatch, isolated_logging):
    monkeypatch.setenv("TWELVE_DATA_API_KEY", SECRET)
    path = tmp_path / "logs" / "bot.log"
    logging_setup.configure_logging("INFO", path)
    log = logging.getLogger("t")
    log.info("request https://api.twelvedata.com/price?symbol=XAU/USD&apikey=%s failed", SECRET)
    log.warning("telegram https://api.telegram.org/bot%s/sendMessage", TOKEN)
    log.info("Authorization: Bearer abcdef0123456789 %s", FAKE_ANTHROPIC)
    try:
        raise RuntimeError(f"boom with {SECRET}")
    except RuntimeError:
        log.exception("handled")
    for h in logging.getLogger().handlers:
        h.flush()
    text = path.read_text(encoding="utf-8")
    assert SECRET not in text and TOKEN not in text and "abcdef0123456789" not in text
    assert FAKE_ANTHROPIC not in text
    assert "apikey=***" in text and "boom with ***" in text and "Traceback" in text


def test_the_log_file_rotates(tmp_path, isolated_logging):
    path = tmp_path / "bot.log"
    logging_setup.configure_logging("INFO", path, max_bytes=2000, backup_count=2, secrets=())
    for i in range(200):
        logging.getLogger("t").info("line %d %s", i, "x" * 40)
    assert path.exists() and (tmp_path / "bot.log.1").exists() and (tmp_path / "bot.log.2").exists()
    assert not (tmp_path / "bot.log.3").exists()


def _run_main(monkeypatch, tmp_path, runner):
    import main as main_mod
    path = tmp_path / "bot.log"
    monkeypatch.setattr(main_mod.Settings, "load", staticmethod(
        lambda env_file=None: SimpleNamespace(log_level="INFO", log_file=path, color=False, show_json=False)))
    monkeypatch.setattr(main_mod, "run_list_chats", runner)
    return main_mod, path


def test_startup_and_exit_are_logged(tmp_path, monkeypatch, isolated_logging):
    async def runner(settings):
        logging.getLogger("t").info("working")
        return 0
    main_mod, path = _run_main(monkeypatch, tmp_path, runner)
    assert main_mod.main(["--list-chats"]) == 0
    text = path.read_text(encoding="utf-8")
    assert "bot_starting mode=list-chats" in text and "engine=" in text
    assert "bot_exiting mode=list-chats code=0" in text


def test_a_crash_is_logged_before_the_process_dies(tmp_path, monkeypatch, isolated_logging):
    async def runner(settings):
        raise RuntimeError("unexpected")
    main_mod, path = _run_main(monkeypatch, tmp_path, runner)
    with pytest.raises(RuntimeError):
        main_mod.main(["--list-chats"])
    text = path.read_text(encoding="utf-8")
    assert "bot_crashed" in text and "RuntimeError: unexpected" in text and "bot_exiting" in text


def test_uncaught_exceptions_are_logged(tmp_path, isolated_logging):
    path = tmp_path / "bot.log"
    logging_setup.configure_logging("INFO", path, secrets=())
    logging_setup.install_exception_hooks()
    try:
        raise ValueError("escaped")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    logging_setup.asyncio_exception_handler(None, {"message": "Task exception was never retrieved",
                                                  "exception": KeyError("lost")})
    text = path.read_text(encoding="utf-8")
    assert "uncaught_exception type=ValueError" in text and "asyncio_unhandled" in text


def test_a_market_data_failure_is_logged(caplog):
    svc = MarketDataService(_Provider(fail=MarketDataError("Twelve Data error 400: bad symbol",
                                                           retryable=False)), cache_ttl=0)
    with caplog.at_level(logging.WARNING, logger="market_data"):
        with pytest.raises(MarketDataError):
            asyncio.run(svc.get_candles("XAUUSD", Timeframe.M15, 10))
    assert any("market_data_failed" in r.getMessage() for r in caplog.records)
