"""The original signal's verdict, and the Telegram card. All TEST DATA."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import timedelta

import pytest

from costs import Costs
from market_data import Timeframe
from models import TradeSetup
from news_filter import NewsStatus

from analyst.engine import analyse
from analyst.report import SHADOW_LINE, TEST_LINE, render_card
from analyst.rules import Rules
from analyst.strategies import READY, Proposal
from analyst.structure import read_frame
from analyst.verdict import ENTER, NO_TRADE, SKIP, WAIT, RecentSignal
from analyst_synth import scenario

COSTS = Costs(0.30, 0.10, "assumed")


def run(name, setup, *, rules=Rules(), message_age=timedelta(seconds=30), price=None,
        frames_patch=None, recent=(), news=None, reader=None, as_of_shift=timedelta(0)):
    _, frames, as_of, last = scenario(name)
    if frames_patch:
        frames = {**frames, **frames_patch}
    at = as_of + as_of_shift
    return analyse("XAUUSD", setup, at - message_age, at, last if price is None else price, frames,
                   rules, COSTS, recent=recent, news=news, reader=reader)


def buy(entry=None, sl=2353.0, tps=(2372.0,), order="market"):
    return TradeSetup(symbol="XAUUSD", direction="long", order_type=order,
                      entries=[] if entry is None else list(entry), stop_loss=sl, take_profits=list(tps))


def sell(entry=None, sl=2366.0, tps=(2340.0,), order="market"):
    return TradeSetup(symbol="XAUUSD", direction="short", order_type=order,
                      entries=[] if entry is None else list(entry), stop_loss=sl, take_profits=list(tps))


def codes(a):
    return [r.code for r in a.verdict.reasons]


# ------------------------------------------------------------------ ENTER

def test_a_signal_confirmed_by_strategy_b_is_enter():
    a = run("b_long", buy())
    assert a.verdict.verdict == ENTER and codes(a) == ["CONFIRMED_BY_B"]
    assert a.verdict.confirmed_by == ("B",)
    assert a.verdict.plan.planned_entry == pytest.approx(2360.0)


def test_a_signal_confirmed_by_strategy_c_is_enter():
    a = run("c_long", buy(sl=2341.0, tps=(2366.0,)))
    assert a.verdict.verdict == ENTER and a.verdict.confirmed_by == ("C",)


# ------------------------------------------------------------------- SKIP

def test_a_signal_against_a_ready_setup_is_skip():
    a = run("b_long", sell())
    assert a.verdict.verdict == SKIP and codes(a) == ["OPPOSING_SETUP"]


@pytest.mark.parametrize("setup,code", [
    (TradeSetup(symbol="XAUUSD", direction="long", stop_loss=2350.0), "SIG_INCOMPLETE"),
    (TradeSetup(symbol="XAUUSD", take_profits=[2370.0], stop_loss=2350.0), "SIG_INCOMPLETE"),
    (buy(sl=2361.0), "SIG_STOPPED_OUT"),
    (buy(entry=[2350.0], sl=2345.0, tps=(2358.0, 2380.0)), "SIG_TARGET_REACHED"),
    (buy(tps=(2355.0,)), "SIG_NO_TARGET_AHEAD"),
    (buy(sl=2359.8, tps=(2372.0,)), "SIG_STOP_TOO_TIGHT"),
    (buy(sl=2330.0, tps=(2400.0,)), "SIG_STOP_TOO_WIDE"),
    (buy(sl=2353.0, tps=(2364.0,)), "SIG_RR_BELOW_MIN"),
])
def test_failed_signal_checks_are_skip(setup, code):
    a = run("b_long", setup)
    assert a.verdict.verdict == SKIP and codes(a) == [code]


def test_a_stale_signal_is_skip():
    a = run("b_long", buy(), message_age=timedelta(minutes=20))
    assert codes(a) == ["SIG_STALE"]


def test_a_duplicate_signal_is_skip():
    _, _, as_of, _ = scenario("b_long")
    earlier = RecentSignal(as_of - timedelta(hours=1), "long", 2353.1, (2371.9,))
    assert codes(run("b_long", buy(), recent=[earlier])) == ["SIG_DUPLICATE"]
    other_side = RecentSignal(as_of - timedelta(hours=1), "short", 2353.1, (2371.9,))
    assert run("b_long", buy(), recent=[other_side]).verdict.verdict == ENTER
    too_old = RecentSignal(as_of - timedelta(hours=5), "long", 2353.1, (2371.9,))
    assert run("b_long", buy(), recent=[too_old]).verdict.verdict == ENTER


def test_counter_h1_structure_is_skip():
    def reader(candles, tf):
        frame = read_frame(candles, tf, Rules())
        return replace(frame, bias="bearish") if tf is Timeframe.H1 else frame
    a = run("b_long", buy(), reader=reader)          # B needs H1 bullish, so no setup either
    assert a.verdict.verdict == SKIP and codes(a) == ["COUNTER_H1_STRUCTURE"]


def test_closing_session_is_skip():
    _, frames, as_of, price = scenario("b_long")
    a = run("b_long", buy())
    from analyst.verdict import judge
    view = replace(a.view, market="closing")
    from analyst.structure import read_market
    from analyst.strategies import evaluate
    read = read_market(view, Rules())
    v = judge(buy(), as_of - timedelta(seconds=30), view, read, evaluate(read, COSTS), Rules(), COSTS)
    assert v.verdict == SKIP and v.reasons[0].code == "SESSION_CLOSING"


# ------------------------------------------------------------------- WAIT

def test_an_unconfirmed_setup_is_wait_with_the_confirmation_named():
    a = run("b_long_unconfirmed", buy(sl=2352.0, tps=(2372.0,)))
    assert a.verdict.verdict == WAIT and codes(a) == ["AWAITING_CONFIRMATION"]
    assert "M5 close above" in a.verdict.reasons[0].text


def test_an_entry_not_yet_reached_is_wait_even_when_confirmed():
    a = run("b_long", buy(entry=[2357.0], sl=2352.0, tps=(2372.0,), order="limit"))
    assert a.verdict.verdict == WAIT and "ENTRY_NOT_REACHED" in codes(a)
    assert a.verdict.plan.pending and a.verdict.plan.planned_entry == pytest.approx(2357.0)


def test_an_extended_entry_is_wait():
    a = run("b_long", buy(entry=[2357.5], sl=2353.0, tps=(2372.0,)))
    assert a.verdict.verdict == WAIT and "PRICE_EXTENDED" in codes(a)
    assert any(w.code == "ENTRY_EXTENDED" for w in a.verdict.weaknesses)


def test_a_real_news_block_is_wait_and_no_calendar_is_never_invented():
    blocked = NewsStatus(True, True, (), "high-impact event 'TEST CPI' within the window")
    a = run("b_long", buy(), news=blocked)
    assert a.verdict.verdict == WAIT and "NEWS_WINDOW" in codes(a)
    plain = run("b_long", buy())
    assert plain.news.startswith("not checked")


def test_conflicting_ready_setups_are_wait():
    from analyst.verdict import judge
    a = run("b_long", buy())
    from analyst.structure import read_market
    read = read_market(a.view, Rules())
    props = {"B": {"long": Proposal("B", "long", READY, evidence=("x",)), "short": None},
             "C": {"long": None, "short": Proposal("C", "short", READY, evidence=("y",))}}
    v = judge(buy(), a.as_of - timedelta(seconds=30), a.view, read, props, Rules(), COSTS)
    assert v.verdict == WAIT and v.reasons[0].code == "CONFLICTING_SETUPS"


# --------------------------------------------------------------- NO TRADE

def test_unavailable_or_stale_data_is_no_trade():
    a = run("b_long", buy(), frames_patch={Timeframe.M5: None})
    assert a.verdict.verdict == NO_TRADE and codes(a) == ["DATA_UNAVAILABLE"]
    stale = run("b_long", buy(), as_of_shift=timedelta(minutes=40))
    assert stale.verdict.verdict == NO_TRADE and codes(stale) == ["DATA_STALE"]
    assert stale.headline is None and stale.m15_setup == "not analysed"


def test_no_structure_either_way_is_no_trade():
    def reader(candles, tf):
        frame = read_frame(candles, tf, Rules())
        return replace(frame, bias="neutral") if tf in (Timeframe.H1, Timeframe.H4) else frame
    a = run("b_long", buy(), reader=reader)
    assert a.verdict.verdict == NO_TRADE and codes(a) == ["NO_STRUCTURAL_EVIDENCE"]


# ------------------------------------------------------------- weaknesses

def test_weaknesses_name_the_structure():
    a = run("b_long", buy(sl=2357.0, tps=(2380.0,)))      # stop above the pullback low
    found = {w.code for w in a.verdict.weaknesses}
    assert "SL_INSIDE_STRUCTURE" in found


def test_evidence_checklist_counts_and_never_decides():
    a = run("b_long", buy())
    assert len(a.verdict.checks) == 6 and 0 <= a.verdict.checks_met <= 6
    assert dict(a.verdict.checks)["H1 bias with the signal"] is True


def test_a_skipped_signal_is_still_tracked():
    from analyst.service import trades_to_track
    a = run("b_long", sell())
    assert a.verdict.verdict == SKIP
    signal = [t for t in trades_to_track(a, sell()) if t["subject"] == "signal"]
    assert signal and signal[0]["direction"] == "short" and signal[0]["price_at_decision"] == 2360.0


# -------------------------------------------------------------- the card

def test_card_has_every_section_and_ends_in_shadow_mode():
    a = run("b_long", buy())
    card = render_card(a, buy(), live_verdict="SKIP")
    for heading in ("INDEPENDENT ANALYSIS", "Original signal:", "Market context:", "H4 bias:",
                    "H1 bias:", "M15 setup:", "M5 confirmation:", "Original signal decision: ENTER",
                    "Main reasons:", "Independent trade proposal:", "Entry zone:", "SL:", "TP:",
                    "RR:", "Setup type: B — trend continuation", "Status: READY",
                    "Analysis timestamp:", "Market data status: fresh",
                    "Live bot verdict (unchanged): SKIP"):
        assert heading in card, heading
    assert card.rstrip().endswith(SHADOW_LINE)


def test_card_shows_a_count_never_a_probability():
    card = render_card(run("b_long", buy()), buy())
    assert "uncalibrated score — not a win probability" in card
    assert not re.search(r"\d+\s*%", card.split("M15 setup:")[0] + card.split("Market data")[1])
    assert "Confidence" not in card


def test_card_never_prints_placeholders():
    for a, setup in ((run("b_long", buy(), frames_patch={Timeframe.M5: None}), buy()),
                     (run("b_long", buy()), buy()), (run("c_long", sell()), sell())):
        card = render_card(a, setup)
        assert "None" not in card and "nan" not in card.lower() and "[" not in card


def test_card_with_no_setup_says_so_without_levels():
    a = run("b_long", buy(), frames_patch={Timeframe.M5: None})
    card = render_card(a, buy())
    assert "Direction: NONE" in card and "Status: NO SETUP" in card
    proposal = card.split("Independent trade proposal:")[1]
    assert "SL:" not in proposal and "TP:" not in proposal
    assert "Original signal decision: NO TRADE" in card


def test_test_data_is_labelled_and_the_card_fits_telegram():
    card = render_card(run("b_long", buy()), buy(), test_data=True)
    assert TEST_LINE in card and len(card) < 4096


def test_waiting_card_names_the_confirmation():
    a = run("b_long_unconfirmed", buy(sl=2352.0, tps=(2372.0,)))
    card = render_card(a, buy(sl=2352.0, tps=(2372.0,)))
    assert "Status: WAITING — needs M5 close above" in card and "(stop entry)" in card
