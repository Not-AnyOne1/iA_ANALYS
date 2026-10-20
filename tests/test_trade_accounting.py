"""Realised R per trade: live fills, bid/ask, costs, timeouts, no lookahead."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from costs import NO_COSTS, Costs
from market_data import Candle
from trade_accounting import BAR, account, account_both, nearest_target

T0 = datetime(2026, 6, 1, 10, 7, tzinfo=timezone.utc)
BAR0 = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
NEXT = BAR0 + BAR
LATER = NEXT + timedelta(days=10)
COSTS = Costs(spread=0.4, slippage=0.1, source="assumed")


def _bar(start, o, h, l, c):
    return Candle(timestamp=start, open=o, high=h, low=l, close=c, volume=1.0)


def _quiet(start, n, price=100.0):
    return [_bar(start + i * BAR, price, price + 0.5, price - 0.5, price) for i in range(n)]


def _row(**over):
    row = {"recorded_at": T0.isoformat(), "direction": "long", "order_type": "limit",
           "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0,
           "take_profits_json": json.dumps([110.0]), "price_at_decision": 100.0}
    row.update(over)
    return row


# ------------------------------------------------------------------ gross

def test_gross_take_profit_is_exactly_reward_over_risk():
    candles = _quiet(BAR0, 3) + [_bar(NEXT + 2 * BAR, 100, 110.5, 99, 110)]
    r = account(_row(), candles, LATER)
    assert r.basis == "gross" and r.status == "tp"
    assert r.r == pytest.approx(2.0)                      # (110-100)/(100-95)


def test_gross_stop_is_exactly_minus_one():
    candles = _quiet(BAR0, 3) + [_bar(NEXT + 2 * BAR, 100, 100.5, 94, 95)]
    r = account(_row(), candles, LATER)
    assert r.status == "sl" and r.r == pytest.approx(-1.0)


def test_a_market_order_fills_at_the_live_price_not_the_stated_entry():
    """The audit found market signals filling at a price that was never
    available. The fill is the live price at decision time."""
    row = _row(order_type="market", entry=98.0, price_at_decision=100.0)
    candles = _quiet(BAR0, 2) + [_bar(NEXT + BAR, 100, 110.5, 99, 110)]
    r = account(row, candles, LATER)
    assert r.planned_entry == 100.0 and r.entry_fill == 100.0
    assert r.r == pytest.approx(2.0)                      # risk measured from 100, not 98


def test_the_nearest_target_is_tp1_whatever_the_listed_order():
    """A live short listed its far target first (TP1 4023, TP2 4050)."""
    row = _row(direction="short", entry=4113.0, stop_loss=4135.0,
               take_profit=4023.0, take_profits_json=json.dumps([4023.0, 4050.0]),
               price_at_decision=4113.0, order_type="market")
    r = account(row, [], LATER)
    assert r.target == 4050.0
    assert nearest_target("short", 4113.0, [4023.0, 4050.0]) == 4050.0


# ------------------------------------------------------------------ net

def test_net_costs_make_a_winner_smaller_and_a_loser_bigger():
    win = _quiet(BAR0, 2) + [_bar(NEXT + BAR, 100, 111, 99, 110)]
    lose = _quiet(BAR0, 2) + [_bar(NEXT + BAR, 100, 100.5, 94, 95)]
    market = dict(order_type="market")
    gross_w = account(_row(**market), win, LATER)
    net_w = account(_row(**market), win, LATER, costs=COSTS)
    gross_l = account(_row(**market), lose, LATER)
    net_l = account(_row(**market), lose, LATER, costs=COSTS)
    assert net_w.basis == "net" and net_w.r < gross_w.r
    assert net_l.r < gross_l.r == pytest.approx(-1.0)
    # Market long buys at the ask plus slippage: 100 + 0.2 + 0.1.
    assert net_w.entry_fill == pytest.approx(100.3)
    # Stop exit sells at market below the stop: 95 - 0.1.
    assert net_l.exit_price == pytest.approx(94.9)


def test_a_long_target_needs_the_bid_to_reach_it():
    """Mid touches 110 but the bid (mid - 0.2) never does: gross TP, net not."""
    candles = _quiet(BAR0, 2) + [_bar(NEXT + BAR, 100, 110.1, 99, 109)] + _quiet(NEXT + 2 * BAR, 4)
    assert account(_row(), candles, NEXT + 6 * BAR).status == "tp"
    assert account(_row(), candles, NEXT + 6 * BAR, costs=COSTS).status != "tp"


def test_a_long_stop_triggers_when_the_bid_falls_to_it():
    """Mid low 95.1 is above the stop, but the bid (95.1 - 0.2) is below it."""
    candles = _quiet(BAR0, 2) + [_bar(NEXT + BAR, 100, 100.5, 95.1, 96)]
    assert account(_row(order_type="market"), candles, LATER, costs=COSTS).status == "sl"
    assert account(_row(order_type="market"), candles, NEXT + 3 * BAR).status != "sl"


def test_a_buy_limit_fills_only_when_the_ask_trades_down_to_it():
    row = _row(entry=99.0, stop_loss=95.0, take_profit=110.0, take_profits_json="[110.0]")
    touch = _quiet(BAR0, 1) + [_bar(NEXT, 100, 100.5, 98.9, 100)]   # mid 98.9, ask 99.1
    rally = [_bar(NEXT + BAR, 100, 111, 99.5, 110)]
    gross = account(row, touch + rally, NEXT + 2 * BAR)
    net = account(row, touch + rally, NEXT + 2 * BAR, costs=COSTS)
    assert gross.status == "tp" and gross.filled_at == NEXT
    assert net.status == "open" and net.r is None                    # the ask never got to 99


def test_shorts_mirror_longs():
    row = _row(direction="short", order_type="market", stop_loss=105.0, take_profit=90.0,
               take_profits_json="[90.0]")
    candles = _quiet(BAR0, 2) + [_bar(NEXT + BAR, 100, 101, 89.5, 90)]
    gross, net = account(row, candles, LATER), account(row, candles, LATER, costs=COSTS)
    assert gross.status == net.status == "tp"
    assert gross.r == pytest.approx(2.0)
    assert net.entry_fill == pytest.approx(99.7)                     # sells at the bid - slip


# ------------------------------------------------------------------ timeout

def test_a_trade_still_open_at_the_horizon_exits_at_the_last_close():
    candles = _quiet(BAR0, 1) + [_bar(NEXT, 100, 100.5, 99.5, 100)] + \
        [_bar(NEXT + i * BAR, 102, 102.5, 101.5, 102) for i in range(1, 20)]
    gross = account(_row(order_type="market"), candles, NEXT + 20 * BAR, horizon=timedelta(hours=2))
    net = account(_row(order_type="market"), candles, NEXT + 20 * BAR, horizon=timedelta(hours=2),
                  costs=COSTS)
    assert gross.status == net.status == "timeout"
    assert gross.r == pytest.approx((102 - 100) / 5)
    assert net.r < gross.r                                           # exits at the bid - slip


def test_an_unfilled_order_has_no_r():
    row = _row(entry=90.0, stop_loss=85.0, take_profit=100.0, take_profits_json="[100.0]")
    r = account(row, _quiet(BAR0, 200), NEXT + 200 * BAR)
    assert r.status == "not_filled" and r.r is None


# ------------------------------------------------------------------ safety

def test_no_bar_before_the_decision_or_after_now_is_read():
    early = [_bar(BAR0 - BAR, 100, 115, 99, 100), _bar(BAR0, 100, 115, 99, 100)]
    future = [_bar(NEXT + 10 * BAR, 100, 115, 99, 110)]
    candles = early + _quiet(NEXT, 9) + future
    r = account(_row(order_type="market"), candles, NEXT + 9 * BAR)    # future bar not closed
    assert r.status == "open" and r.r is None


def test_the_same_inputs_give_the_same_result():
    candles = _quiet(BAR0, 3) + [_bar(NEXT + 2 * BAR, 100, 110.5, 99, 110)]
    assert account(_row(), candles, LATER, costs=COSTS) == account(_row(), candles, LATER, costs=COSTS)


@pytest.mark.parametrize("over", [
    dict(direction=None),
    dict(stop_loss=None),
    dict(order_type="market", price_at_decision=94.0),              # live price already past the stop
    dict(take_profits_json="[]", take_profit=None),
])
def test_impossible_trades_are_invalid_not_scored(over):
    r = account(_row(**over), _quiet(BAR0, 5), LATER)
    assert r.status == "invalid" and r.r is None


def test_every_r_is_finite():
    for candles in (_quiet(BAR0, 3) + [_bar(NEXT + 2 * BAR, 100, 111, 99, 110)],
                    _quiet(BAR0, 3) + [_bar(NEXT + 2 * BAR, 100, 100.5, 94, 95)]):
        for result in account_both(_row(order_type="market"), candles, LATER, costs=COSTS).values():
            assert math.isfinite(result.r)


def test_net_is_only_produced_when_a_cost_figure_exists():
    both = account_both(_row(), _quiet(BAR0, 3), LATER, costs=None)
    assert set(both) == {"gross"}
    assert account(_row(), _quiet(BAR0, 3), LATER, costs=NO_COSTS).basis == "gross"
