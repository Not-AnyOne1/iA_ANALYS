"""Hard directional validation of a trade's levels.

The motivating defect: a SHORT with entry 4162, stop 4152 and target 4182 —
a long's levels — computed risk/reward 2.0 through ``abs()`` and was approved
at 93. These tests pin that an impossible trade can never be scored, never
be approved, and never become ENTER.
"""

from __future__ import annotations

import math

import pytest

import risk_engine
import trade_quality
from models import TradeSetup
from risk_engine import RiskVerdict
from trade_geometry import check_geometry


def _setup(direction, entries, stop_loss, take_profits, order_type="limit"):
    return TradeSetup(symbol="XAUUSD", direction=direction, order_type=order_type,
                      entries=list(entries), stop_loss=stop_loss,
                      take_profits=list(take_profits))


def _quality(setup, current_price=None):
    return trade_quality.assess(setup, current_price=current_price, atr_value=5.0)


def _geometry_verdict(setup):
    result = risk_engine.evaluate(_quality(setup))
    return next(c for c in result.checks if c.name == "trade_geometry"), result


# ------------------------------------------------------------ the four cases

def test_valid_long():
    g = check_geometry("long", [2700], 2690, [2730])
    assert g.valid
    assert g.risk == 10 and g.reward == 30
    assert g.risk_reward == pytest.approx(3.0)


def test_invalid_long():
    # stop above entry, target below: a short's levels on a long
    g = check_geometry("long", [2700], 2710, [2690])
    assert not g.valid
    assert g.risk_reward is None
    assert "invalid LONG" in g.reason


def test_valid_short():
    g = check_geometry("short", [2700], 2710, [2670])
    assert g.valid
    assert g.risk == 10 and g.reward == 30
    assert g.risk_reward == pytest.approx(3.0)


def test_invalid_short_the_specified_example():
    """SHORT, entry 2700, SL 2690, TP 2710 — MUST be rejected."""
    g = check_geometry("short", [2700], 2690, [2710])
    assert not g.valid
    assert g.risk_reward is None
    assert "invalid SHORT" in g.reason
    assert "stop loss 2690 is below entry 2700" in g.reason
    assert "take profit 2710 is above entry 2700" in g.reason


def test_the_real_invalid_short_that_was_approved_at_93():
    """Signal #57 from the live log: entry 4162, SL 4152, TP 4182."""
    setup = _setup("short", [4162], 4152, [4182])
    assert setup.risk_reward is None            # was 2.0 under abs()
    check, result = _geometry_verdict(setup)
    assert check.verdict is RiskVerdict.REJECT
    assert not result.approved


@pytest.mark.parametrize("direction,stop,tp,which", [
    ("long", 2710, 2730, "stop"),     # stop on the wrong side only
    ("long", 2690, 2680, "target"),   # target on the wrong side only
    ("short", 2690, 2670, "stop"),
    ("short", 2710, 2730, "target"),
])
def test_one_level_on_the_wrong_side_is_enough_to_reject(direction, stop, tp, which):
    g = check_geometry(direction, [2700], stop, [tp])
    assert not g.valid
    assert ("stop loss" if which == "stop" else "take profit") in g.reason


# ------------------------------------------------------------- degenerates

def test_entry_equal_to_stop_is_zero_risk():
    g = check_geometry("long", [2700], 2700, [2730])
    assert not g.valid
    assert "zero risk" in g.reason


def test_entry_equal_to_target_is_zero_reward():
    g = check_geometry("long", [2700], 2690, [2700])
    assert not g.valid
    assert "zero reward" in g.reason


@pytest.mark.parametrize("direction,stop,tp", [
    ("long", 2700, 2730),      # zero risk
    ("short", 2700, 2670),
])
def test_zero_risk_trades_are_rejected_not_abstained(direction, stop, tp):
    """Regression: zero risk used to make R:R ``None``, which the risk engine
    read as ABSTAIN — so a zero-risk trade was approved by default."""
    check, result = _geometry_verdict(_setup(direction, [2700], stop, [tp]))
    assert check.verdict is RiskVerdict.REJECT
    assert not result.approved


@pytest.mark.parametrize("direction,tp", [
    ("long", 2690),     # negative reward: target below a long's entry
    ("long", 2700),     # zero reward
    ("short", 2710),    # negative reward: target above a short's entry
    ("short", 2700),    # zero reward
])
def test_negative_or_zero_reward_is_rejected(direction, tp):
    stop = 2690 if direction == "long" else 2710
    if tp == 2690 and direction == "long":
        stop = 2680                      # keep the stop valid; isolate the target
    g = check_geometry(direction, [2700], stop, [tp])
    assert not g.valid
    assert g.risk_reward is None


@pytest.mark.parametrize("entries,stop,tps", [
    ([float("nan")], 2690, [2730]),
    ([2700], float("inf"), [2730]),
    ([2700], 2690, [float("-inf")]),
    ([0.0], -5.0, [10.0]),
    ([-2700], -2710, [-2690]),
    ([2700], 2690, [True]),               # a bool is not a price
])
def test_malformed_levels_are_rejected(entries, stop, tps):
    g = check_geometry("long", entries, stop, tps)
    assert not g.valid
    assert "malformed" in g.reason


@pytest.mark.parametrize("direction,entries,stop,tps,expected", [
    (None, [2700], 2690, [2730], "direction is unknown"),
    ("long", [], 2690, [2730], "no entry"),
    ("long", [2700], None, [2730], "no stop loss"),
    ("long", [2700], 2690, [], "no take profit"),
])
def test_missing_levels_cannot_be_validated(direction, entries, stop, tps, expected):
    g = check_geometry(direction, entries, stop, tps)
    assert not g.valid
    assert not g.complete
    assert expected in g.reason


def test_missing_levels_are_rejected_by_the_risk_engine():
    """R:R must be validated before a trade can become ENTER; a trade with no
    stop cannot be."""
    check, result = _geometry_verdict(_setup("long", [2700], None, [2730]))
    assert check.verdict is RiskVerdict.REJECT
    assert not result.approved


# ----------------------------------------------------- zones and many targets

def test_every_entry_in_a_zone_is_checked():
    """A zone whose far edge crosses the stop is malformed."""
    assert check_geometry("long", [2700, 2705], 2690, [2730]).valid
    assert not check_geometry("long", [2700, 2685], 2690, [2730]).valid


def test_every_target_is_checked():
    """A second target on the losing side is as invalid as the first."""
    assert check_geometry("long", [2700], 2690, [2730, 2760]).valid
    assert not check_geometry("long", [2700], 2690, [2730, 2650]).valid


def test_a_zone_measures_risk_from_its_midpoint():
    g = check_geometry("long", [2700, 2710], 2690, [2735])
    assert g.entry == 2705
    assert g.risk == 15 and g.reward == 30


# ------------------------------------------------------------- R:R direction

@pytest.mark.parametrize("direction,stop,tp,expected", [
    ("long", 2690, 2720, 2.0),
    ("long", 2680, 2710, 0.5),
    ("short", 2710, 2680, 2.0),
    ("short", 2720, 2690, 0.5),
])
def test_risk_reward_is_directional(direction, stop, tp, expected):
    setup = _setup(direction, [2700], stop, [tp])
    assert setup.risk_reward == pytest.approx(expected)
    assert _quality(setup).risk_reward == pytest.approx(expected)


def test_the_old_absolute_formula_would_have_approved_the_invalid_short():
    """Documents the bug being fixed rather than relying on memory."""
    entry, stop, tp = 2700, 2690, 2710
    old = abs(tp - entry) / abs(entry - stop)
    assert old == 1.0 and math.isfinite(old)               # looked like a trade
    assert _setup("short", [entry], stop, [tp]).risk_reward is None


# ------------------------------------------------------------- market orders

def test_a_market_order_is_validated_against_the_live_price():
    """'BUY NOW' with no stated entry: the live price is the entry, so a stop
    above it is just as invalid."""
    setup = _setup("long", [], 2710, [2730], order_type="market")
    assert not _quality(setup, current_price=2700).geometry.valid
    setup = _setup("long", [], 2690, [2730], order_type="market")
    assert _quality(setup, current_price=2700).geometry.valid


def test_a_valid_setup_still_passes_every_existing_risk_rule():
    """Existing valid signals remain compatible."""
    result = risk_engine.evaluate(_quality(_setup("long", [2700], 2690, [2730])))
    assert next(c for c in result.checks if c.name == "trade_geometry").verdict is RiskVerdict.PASS
    assert result.approved
