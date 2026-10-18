"""Merge blockers: full candle depth after stub removal, non-fatal log setup."""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import logging_setup
import market_hours as mh
from market_data import (PROVIDER_MAX_CANDLES, Candle, CandleSeries, MarketDataService, Quote,
                         Timeframe, provider_request_size)

UTC = timezone.utc
START = datetime(2026, 5, 4, tzinfo=UTC)             # a Monday
END = datetime(2026, 7, 6, 12, tzinfo=UTC)           # a Monday, midday (US summer)


def _history(tf: Timeframe, start=START, end=END, trading=None):
    """A bar for every slot from start to end. Inside the weekly closure: an
    off-market stub quote, as Twelve Data serves it. With ``trading``, slots
    outside the closure where ``trading(ts)`` is false have no bar at all (a
    sparse market), so stubs dominate the provider's pages."""
    bars, t, i = [], start, 0
    while t < end:
        if tf is Timeframe.D1:
            closed = t.weekday() >= 5                 # daily bars: weekdays only, no stubs
            if closed:
                t += tf.duration
                continue
        else:
            closed = mh.bar_inside_weekly_closure(t, tf.duration)
        if closed:
            bars.append(Candle(timestamp=t, open=9000.0, high=9001.0, low=8999.0, close=9000.0, volume=None))
        elif trading is None or trading(t):
            p = 4000 + 20 * math.sin(i / 37) + i * 0.01
            bars.append(Candle(timestamp=t, open=p, high=p + 1.5, low=p - 1.5, close=p + 0.3, volume=None))
            i += 1
        t += tf.duration
    return bars


class _Provider:
    """Serves the newest ``limit`` bars closed by ``at`` — what Twelve Data does,
    stub quotes counted towards ``limit``."""
    name = "history"

    def __init__(self, bars, at=END):
        self.bars, self.at, self.requests = bars, at, []

    async def get_quote(self, symbol):
        return Quote(symbol=symbol, price=4000.0, timestamp=self.at, provider=self.name)

    async def get_candles(self, symbol, timeframe, limit):
        self.requests.append((timeframe, limit))
        closed = [b for b in self.bars[timeframe] if b.timestamp + timeframe.duration <= self.at]
        return CandleSeries(symbol, timeframe, self.name, tuple(closed[-limit:]))


def _clean(bars, tf):
    return [b for b in bars if not mh.bar_inside_weekly_closure(b.timestamp, tf.duration)]


INTRADAY = (Timeframe.M15, Timeframe.H1, Timeframe.H4)


@pytest.mark.parametrize("tf", INTRADAY)
def test_the_service_returns_the_full_window_of_real_bars(tf):
    bars = _history(tf, start=datetime(2025, 1, 6, tzinfo=UTC))     # enough history for 300 H4 bars
    provider = _Provider({tf: bars})
    series = asyncio.run(MarketDataService(provider, cache_ttl=0).get_candles("XAUUSD", tf, 300))
    expected = [b for b in _clean(bars, tf) if b.timestamp + tf.duration <= END][-300:]
    assert len(series.candles) == 300
    assert list(series.candles) == expected                   # the newest 300 real bars, nothing else
    assert not any(c.open == 9000.0 for c in series.candles)  # no stub survived
    assert provider.requests == [(tf, provider_request_size("XAUUSD", tf, 300))]
    assert provider.requests[0][1] > 300                      # padded, in ONE request


def test_without_padding_the_window_would_have_shrunk():
    """The defect being fixed: 300 raw H1 bars over two stub weekends."""
    raw = [b for b in _history(Timeframe.H1) if b.timestamp + Timeframe.H1.duration <= END][-300:]
    assert len(_clean(raw, Timeframe.H1)) < 250


def test_the_friday_close_and_sunday_reopen_are_kept():
    tf = Timeframe.M15
    series = asyncio.run(MarketDataService(_Provider({tf: _history(tf)}), cache_ttl=0)
                         .get_candles("XAUUSD", tf, 300))
    times = {c.timestamp for c in series.candles}
    fri, sun = datetime(2026, 7, 3, tzinfo=UTC), datetime(2026, 7, 5, tzinfo=UTC)
    assert fri.replace(hour=20, minute=45) in times                 # last bar before 21:00 UTC close
    assert sun.replace(hour=22) in times                            # the summer reopen
    assert fri.replace(hour=21) not in times and sun.replace(hour=21, minute=45) not in times


def test_with_too_little_history_everything_real_is_returned():
    tf = Timeframe.H1
    short = _history(tf, start=datetime(2026, 6, 29, tzinfo=UTC))   # about one week
    series = asyncio.run(MarketDataService(_Provider({tf: short}), cache_ttl=0).get_candles("XAUUSD", tf, 300))
    assert list(series.candles) == [b for b in _clean(short, tf) if b.timestamp + tf.duration <= END]
    assert len(series.candles) < 300


def test_an_unusually_stub_heavy_window_gets_one_wider_request():
    """Real bars only on Monday 00-02 UTC: the allowance is not enough, more
    history exists, so the service asks once more with a wider page."""
    tf = Timeframe.M15
    sparse = _history(tf, start=datetime(2026, 1, 5, tzinfo=UTC),
                      trading=lambda t: t.weekday() == 0 and t.hour < 2)
    provider = _Provider({tf: sparse})
    asyncio.run(MarketDataService(provider, cache_ttl=0).get_candles("XAUUSD", tf, 300))
    first = provider_request_size("XAUUSD", tf, 300)
    assert provider.requests == [(tf, first), (tf, min(PROVIDER_MAX_CANDLES, first * 2))]


def test_other_symbols_and_the_tracker_window_are_unchanged():
    assert provider_request_size("EURUSD", Timeframe.H1, 300) == 300
    assert provider_request_size("BTCUSDT", Timeframe.M15, 300) == 300
    assert provider_request_size("XAUUSD", Timeframe.M15, 5000) == PROVIDER_MAX_CANDLES
    tf = Timeframe.H1
    bars = _history(tf)
    provider = _Provider({tf: bars})
    series = asyncio.run(MarketDataService(provider, cache_ttl=0).get_candles("EURUSD", tf, 300))
    assert provider.requests == [(tf, 300)] and len(series.candles) == 300   # FX keeps its weekend bars


def test_the_engine_analyses_300_real_bars_on_every_timeframe():
    from market_context import MarketContextBuilder
    tfs = (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)
    provider = _Provider({tf: _history(tf, start=datetime(2025, 1, 6, tzinfo=UTC)) for tf in tfs})
    ctx = asyncio.run(MarketContextBuilder(MarketDataService(provider, cache_ttl=0)).build("XAUUSD", now=END))
    for tf in tfs:
        assert ctx.timeframes[tf.value].candles_analysed == 300, tf


# ----------------------------------------------------------------- logging

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


def _unusable(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the log folder should be")
    return blocker / "logs" / "bot.log"


def test_an_unusable_log_location_falls_back_to_the_console(tmp_path, capsys, isolated_logging):
    active = logging_setup.configure_logging("INFO", _unusable(tmp_path), secrets=())
    logging.getLogger("t").info("still logging")
    err = capsys.readouterr().err
    assert active is None
    assert "log_file_unavailable" in err and "continuing with console logging only" in err
    assert "still logging" in err
    assert not any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)


def test_the_bot_still_starts_without_its_log_file(tmp_path, monkeypatch, capsys, isolated_logging):
    import main as main_mod
    monkeypatch.setattr(main_mod.Settings, "load", staticmethod(lambda env_file=None: SimpleNamespace(
        log_level="INFO", log_file=_unusable(tmp_path), color=False, show_json=False)))

    async def runner(settings):
        return 0
    monkeypatch.setattr(main_mod, "run_list_chats", runner)
    assert main_mod.main(["--list-chats"]) == 0
    err = capsys.readouterr().err
    assert "log_file_unavailable" in err and "log_file=none (console only)" in err
    assert "bot_exiting mode=list-chats code=0" in err


def test_unrelated_logging_errors_still_raise(tmp_path, monkeypatch, isolated_logging):
    def broken(*a, **kw):
        raise ValueError("a programming error, not an unavailable disk")
    monkeypatch.setattr(logging_setup.logging.handlers, "RotatingFileHandler", broken)
    with pytest.raises(ValueError):
        logging_setup.configure_logging("INFO", tmp_path / "bot.log", secrets=())


def test_a_usable_location_is_returned(tmp_path, isolated_logging):
    path = tmp_path / "logs" / "bot.log"
    assert logging_setup.configure_logging("INFO", path, secrets=()) == path and path.exists()
