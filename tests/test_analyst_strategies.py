"""analyst.structure and analyst.strategies on synthetic TEST DATA.

The scenarios (``analyst_synth``) are written as M5 paths and aggregated to
M15/H1/H4, so the timeframes agree by construction.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from costs import NO_COSTS, Costs
from market_data import Timeframe

from analyst.data import build_view
from analyst.rules import Rules
from analyst.strategies import INVALID, READY, WAITING, Proposal, evaluate, headline, net_rr, \
    strategy_b, strategy_c
from analyst.structure import LONG, SHORT, find_pullback, find_sweep, is_resting, levels, \
    m5_break, read_frame, read_market
from analyst_synth import scenario

COSTS = Costs(0.30, 0.10, "assumed")


def market(name, rules=Rules(), reader=None):
    _, frames, as_of, price = scenario(name)
    view = build_view("XAUUSD", as_of, price, frames, rules)
    assert view.fresh, view.problems
    return read_market(view, rules, reader)


def with_bias(**biases):
    """A frame reader that overrides the bias of chosen timeframes."""
    def reader(candles, tf):
        frame = read_frame(candles, tf, Rules())
        return replace(frame, bias=biases[tf.value]) if tf.value in biases else frame
    return reader


# --------------------------------------------------------------- structure

def test_bias_needs_a_break_and_the_ema_side():
    read = market("b_long")
    h1 = read.frame(Timeframe.H1)
    assert h1.last_event.direction.value == "bullish" and h1.close > h1.ema
    assert h1.bias == "bullish"
    flat = read_frame(read.frame(Timeframe.H1).candles[:40], Timeframe.H1, Rules())
    assert flat.ema is None and flat.bias == "neutral"          # too few bars for EMA-50


def test_structure_uses_no_future_bar():
    """Reading a prefix gives the same events as the full series up to that
    prefix: nothing later changes an earlier conclusion."""
    read = market("b_long")
    m15 = read.frame(Timeframe.M15).candles
    full = read_frame(m15, Timeframe.M15, Rules())
    prefix = read_frame(m15[:250], Timeframe.M15, Rules())
    early = [(e.timestamp, e.type, e.direction) for e in full.events if e.candle_index < 250 - 2]
    assert early == [(e.timestamp, e.type, e.direction) for e in prefix.events][:len(early)]


def test_a_level_traded_through_is_no_longer_resting():
    read = market("c_long")
    m15 = read.frame(Timeframe.M15)
    swept = [lv for lv in levels(m15, Rules(), 3) if lv.side == "low" and lv.price == pytest.approx(2344.95)]
    assert swept and not is_resting(swept[0], m15.candles, read.view.candles(Timeframe.M5))


def test_pullback_detection():
    pb = find_pullback(market("b_long"), LONG)
    assert pb is not None and 0.382 <= pb.depth < 1.0
    assert pb.origin.price < pb.extreme < pb.impulse.price
    assert find_pullback(market("b_long"), SHORT) is None


def test_sweep_detection_and_its_limits():
    read = market("c_long")
    sw = find_sweep(read, LONG)
    assert sw is not None and sw.bar.low < sw.level.price < sw.bar.close
    assert find_sweep(read, SHORT) is None
    shallow = Rules(sweep_max_depth_atr=0.1)              # the wick is deeper than 0.1 ATR
    assert find_sweep(market("c_long", shallow), LONG) is None
    old = Rules(sweep_max_age_bars=1)                     # the sweep is 4 M15 bars back
    assert find_sweep(market("c_long", old), LONG) is None


def test_a_sweep_that_closes_through_the_level_is_not_a_sweep():
    _, frames, as_of, price = scenario("c_long")
    m15 = list(frames[Timeframe.M15])
    read = market("c_long")
    k = m15.index(find_sweep(read, LONG).bar)
    m15[k] = replace(m15[k], close=m15[k].low + 0.2)      # closes below the swept low
    view = build_view("XAUUSD", as_of, price, {**frames, Timeframe.M15: tuple(m15)}, Rules())
    assert find_sweep(read_market(view, Rules()), LONG) is None


def test_m5_break_must_follow_the_setup_and_be_recent():
    read = market("b_long")
    pb = find_pullback(read, LONG)
    assert m5_break(read, LONG, pb.extreme_at) is not None
    assert m5_break(read, LONG, read.view.as_of) is None        # nothing after the last bar


# ------------------------------------------------------------- arithmetic

def test_net_rr_matches_the_accounting_fills():
    # long market: fill 100 + 0.15 + 0.10 = 100.25; reward 110 - 100.25 = 9.75;
    # loss 100.25 - 95 + 0.10 = 5.35
    assert net_rr(LONG, 100.0, 95.0, 110.0, COSTS) == pytest.approx(9.75 / 5.35, abs=1e-3)
    assert net_rr(SHORT, 100.0, 105.0, 90.0, NO_COSTS) == pytest.approx(2.0)
    assert net_rr(LONG, 100.0, 95.0, 110.0, COSTS, "limit") == pytest.approx(10.0 / 5.1, abs=1e-3)


# ---------------------------------------------------------------- strategy B

def test_b_long_ready_with_structural_levels():
    read = market("b_long")
    p = strategy_b(read, LONG, COSTS)
    pb = find_pullback(read, LONG)
    assert p.status == READY and p.order_type == "market" and p.entry == read.price
    assert p.stop_loss == pytest.approx(pb.extreme - 0.2 * read.atr15, abs=0.01)
    assert p.take_profit > pb.impulse.price                 # beyond the impulse swing
    assert p.rr_net >= Rules().min_rr and p.rr_gross > p.rr_net
    assert p.entry_zone[0] <= p.entry <= p.entry_zone[1]
    assert strategy_b(read, SHORT, COSTS) is None


def test_b_short_mirrors_b_long():
    up, down = strategy_b(market("b_long"), LONG, COSTS), strategy_b(market("b_short"), SHORT, COSTS)
    assert down.status == READY
    assert down.entry + up.entry == pytest.approx(5000.0)
    assert down.stop_loss + up.stop_loss == pytest.approx(5000.0, abs=0.02)
    assert down.rr_net == pytest.approx(up.rr_net)


def test_b_needs_both_h4_and_h1_bias():
    assert strategy_b(market("b_long", reader=with_bias(H4="neutral")), LONG, COSTS) is None
    assert strategy_b(market("b_long", reader=with_bias(H1="bearish")), LONG, COSTS) is None


def test_b_without_the_m5_break_is_waiting_with_a_stop_entry():
    read = market("b_long_unconfirmed")
    p = strategy_b(read, LONG, COSTS)
    assert p.status == WAITING and p.order_type == "stop" and "AWAIT_M5_BREAK" in p.reasons
    assert p.entry > read.price and "M5 close above" in p.confirmation


def test_too_little_room_is_invalid_never_a_moved_target():
    p = strategy_b(market("b_long", Rules(min_rr=5.0)), LONG, COSTS)
    assert p.status == INVALID and p.reasons == ("RR_BELOW_MIN",)
    assert p.take_profit == strategy_b(market("b_long"), LONG, COSTS).take_profit


def test_a_stop_wider_than_the_band_is_invalid_and_a_tight_one_is_widened():
    wide = strategy_b(market("b_long", Rules(max_stop_atr=1.0)), LONG, COSTS)
    assert wide.status == INVALID and wide.reasons == ("STOP_TOO_WIDE",)
    tight = strategy_b(market("b_long", Rules(min_stop_atr=3.0, max_stop_atr=6.0)), LONG, COSTS)
    read = market("b_long")
    assert tight.entry - tight.stop_loss == pytest.approx(3.0 * read.atr15, abs=0.01)
    assert any("widened" in n for n in tight.notes)


def test_price_run_past_the_break_waits_for_a_retest():
    p = strategy_b(market("b_long", Rules(max_chase_r=0.01)), LONG, COSTS)
    assert p.status == WAITING and p.order_type == "limit" and "AWAIT_RETEST" in p.reasons
    assert p.entry < market("b_long").price


# ---------------------------------------------------------------- strategy C

def test_c_long_ready_on_a_reclaimed_sweep():
    read = market("c_long")
    p = strategy_c(read, LONG, COSTS)
    sw = find_sweep(read, LONG)
    assert p.status == READY
    assert p.stop_loss == pytest.approx(sw.extreme - 0.2 * read.atr15, abs=0.01)
    assert p.take_profit > p.entry and p.rr_net >= Rules().min_rr
    assert strategy_c(read, SHORT, COSTS) is None


def test_strategies_are_isolated_and_the_headline_order_is_fixed():
    props = evaluate(market("c_long"), COSTS)
    assert props["B"] == {"long": None, "short": None}
    assert headline(props).strategy == "C"
    a = Proposal("C", LONG, READY)
    b = Proposal("B", SHORT, WAITING)
    assert headline({"B": {"short": b}, "C": {"long": a}}) is a           # READY first
    b_ready = Proposal("B", SHORT, READY)
    assert headline({"B": {"short": b_ready}, "C": {"long": a}}) is b_ready  # then B before C
    assert headline({"B": {}, "C": {}}) is None
