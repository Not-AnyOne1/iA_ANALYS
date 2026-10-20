"""analyst.data: closed candles, weekend stubs, freshness, grid and alignment.

All prices are TEST DATA.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from market_data import Candle, Timeframe

from analyst.data import (DURATION, MAX_LAG, build_view, closed_bars, drop_closure_stubs,
                          market_state, next_weekly_close, on_grid, prepare_frame, trading_time,
                          weekly_closures)
from analyst.rules import Rules
from analyst_synth import scenario

UTC = timezone.utc


def bar(ts, price=2000.0, spread=1.0):
    return Candle(ts, price, price + spread, price - spread, price)


def run(tf, start, n, price=2000.0):
    return [bar(start + DURATION[tf] * i, price) for i in range(n)]


# ---------------------------------------------------------------- calendar

def test_weekly_closure_is_dst_aware():
    summer = list(weekly_closures(datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 8, tzinfo=UTC)))
    winter = list(weekly_closures(datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 12, tzinfo=UTC)))
    assert summer == [(datetime(2026, 7, 3, 21, tzinfo=UTC), datetime(2026, 7, 5, 22, tzinfo=UTC))]
    assert winter == [(datetime(2026, 1, 9, 22, tzinfo=UTC), datetime(2026, 1, 11, 23, tzinfo=UTC))]


def test_trading_time_excludes_the_weekend():
    friday = datetime(2026, 7, 3, 20, tzinfo=UTC)
    monday = datetime(2026, 7, 6, 0, tzinfo=UTC)
    assert monday - friday == timedelta(hours=52)
    assert trading_time(friday, monday) == timedelta(hours=3)     # 1 h Friday + 2 h Sunday night


def test_market_state_open_closing_closed():
    rules = Rules()
    assert market_state(datetime(2026, 7, 1, 12, tzinfo=UTC), "XAUUSD", rules.close_buffer) == "open"
    assert market_state(datetime(2026, 7, 3, 20, 40, tzinfo=UTC), "XAUUSD", rules.close_buffer) == "closing"
    assert market_state(datetime(2026, 7, 4, 12, tzinfo=UTC), "XAUUSD", rules.close_buffer) == "closed"
    assert next_weekly_close(datetime(2026, 7, 4, 12, tzinfo=UTC)) == datetime(2026, 7, 10, 21, tzinfo=UTC)


# ------------------------------------------------------------------ frames

def test_only_closed_bars_are_used():
    start = datetime(2026, 7, 1, 12, tzinfo=UTC)
    bars = run(Timeframe.M15, start, 4)                  # 12:00 .. 12:45
    kept = closed_bars(bars, Timeframe.M15, datetime(2026, 7, 1, 12, 50, tzinfo=UTC))
    assert [b.timestamp.minute for b in kept] == [0, 15, 30]   # the 12:45 bar is still forming


def test_weekend_stubs_are_dropped_but_a_bar_spanning_the_reopen_is_kept():
    saturday = bar(datetime(2026, 7, 4, 10, tzinfo=UTC))
    sunday_reopen_h4 = bar(datetime(2026, 7, 5, 20, tzinfo=UTC))       # 20:00-24:00, reopen 22:00
    monday = bar(datetime(2026, 7, 6, 0, tzinfo=UTC))
    assert drop_closure_stubs([saturday, monday], Timeframe.M15, "XAUUSD") == (monday,)
    assert drop_closure_stubs([sunday_reopen_h4], Timeframe.H4, "XAUUSD") == (sunday_reopen_h4,)
    assert drop_closure_stubs([saturday], Timeframe.M15, "BTCUSD") == (saturday,)


def test_grid_rules():
    assert on_grid(datetime(2026, 7, 1, 12, 5, tzinfo=UTC), Timeframe.M5)
    assert not on_grid(datetime(2026, 7, 1, 12, 3, tzinfo=UTC), Timeframe.M5)
    assert not on_grid(datetime(2026, 7, 1, 12, 5, tzinfo=UTC), Timeframe.M15)
    assert on_grid(datetime(2026, 7, 1, 1, tzinfo=UTC), Timeframe.H4)      # summer H4 grid
    assert on_grid(datetime(2026, 1, 5, 0, tzinfo=UTC), Timeframe.H4)      # winter H4 grid
    assert not on_grid(datetime(2026, 7, 1, 2, tzinfo=UTC), Timeframe.H4)


def test_missing_short_offgrid_and_stale_frames():
    as_of = datetime(2026, 7, 1, 12, tzinfo=UTC)
    assert prepare_frame(None, Timeframe.M5, as_of, "XAUUSD").status == "unavailable"
    short = run(Timeframe.M5, as_of - timedelta(minutes=5 * 10), 10)
    assert prepare_frame(short, Timeframe.M5, as_of, "XAUUSD").status == "insufficient"
    good = run(Timeframe.M5, as_of - timedelta(minutes=5 * 100), 100)
    assert prepare_frame(good, Timeframe.M5, as_of, "XAUUSD").status == "ok"
    shifted = [replace(b, timestamp=b.timestamp + timedelta(minutes=2)) for b in good]
    assert prepare_frame(shifted, Timeframe.M5, as_of + timedelta(minutes=2), "XAUUSD").status == "off_grid"
    late = as_of + MAX_LAG[Timeframe.M5] + timedelta(minutes=5)
    stale = prepare_frame(good, Timeframe.M5, late, "XAUUSD")
    assert stale.status == "stale" and "trading time" in stale.note


def test_the_weekend_does_not_make_data_stale():
    """Sunday 22:10 UTC after the reopen: Friday's last M15 bar closed 21:00,
    only 10 minutes of trading time ago."""
    friday_close = datetime(2026, 7, 3, 21, tzinfo=UTC)
    bars = run(Timeframe.M15, friday_close - timedelta(minutes=15 * 120), 120)
    frame = prepare_frame(bars, Timeframe.M15, datetime(2026, 7, 5, 22, 10, tzinfo=UTC), "XAUUSD")
    assert frame.status == "ok"


# -------------------------------------------------------------------- view

def test_synthetic_scenario_is_fresh_and_aligned():
    _, frames, as_of, price = scenario("b_long")
    view = build_view("XAUUSD", as_of, price, frames, Rules())
    assert view.status == "fresh", view.problems


def test_a_disagreeing_timeframe_is_misaligned():
    _, frames, as_of, price = scenario("b_long")
    h1 = list(frames[Timeframe.H1])
    closed = [b for b in h1 if b.timestamp + DURATION[Timeframe.H1] <= as_of]
    i = h1.index(closed[-1])
    h1[i] = replace(h1[i], close=h1[i].close + 25.0, high=h1[i].high + 25.0)
    view = build_view("XAUUSD", as_of, price, {**frames, Timeframe.H1: tuple(h1)}, Rules())
    assert view.status == "misaligned" and "disagree" in view.problems[0]


def test_a_live_price_far_from_the_candles_is_misaligned():
    _, frames, as_of, price = scenario("b_long")
    view = build_view("XAUUSD", as_of, price * 1.02, frames, Rules())
    assert view.status == "misaligned" and "live price" in view.problems[0]


def test_missing_m5_or_price_is_unavailable_and_stale_is_stale():
    _, frames, as_of, price = scenario("b_long")
    assert build_view("XAUUSD", as_of, price, {**frames, Timeframe.M5: None}, Rules()).status == "unavailable"
    assert build_view("XAUUSD", as_of, None, frames, Rules()).status == "unavailable"
    later = as_of + timedelta(minutes=40)
    assert build_view("XAUUSD", later, price, frames, Rules()).status == "stale"


def test_weekly_closure_is_market_closed():
    _, frames, _, price = scenario("b_long")
    view = build_view("XAUUSD", datetime(2026, 3, 21, 12, tzinfo=UTC), price, frames, Rules())
    assert view.status == "market_closed"
