"""Unit tests for the RFC-001 regex-first parser.

No Telegram or Claude involved — pure function tests over signal_parser.parse_signal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from signal_parser import parse_signal
from telegram_client import IncomingMessage


def _msg(text: str, *, reply_to_id: Optional[int] = None, is_edit: bool = False) -> IncomingMessage:
    return IncomingMessage(
        id=1,
        chat_id=-100123,
        chat_title="Test Group",
        sender="Analyst",
        timestamp=datetime.now(timezone.utc),
        text=text,
        reply_to_id=reply_to_id,
        is_edit=is_edit,
    )


# --- clean signal matches ----------------------------------------------------

def test_clean_signal_all_caps_no_separator():
    result = parse_signal(_msg(
        "BTCUSDT LONG\nEntry: 61200\nSL: 60350\nTP1: 62400\nTP2: 63500"
    ))
    assert result.reason == "clean_signal_match"
    a = result.analysis
    assert a is not None
    assert a.is_signal is True
    assert a.category == "signal"
    assert a.source == "regex"
    assert a.setup.symbol == "BTCUSDT"
    assert a.setup.direction == "long"
    assert a.setup.entries == [61200.0]
    assert a.setup.stop_loss == 60350.0
    assert a.setup.take_profits == [62400.0, 63500.0]
    assert a.missing_fields == []


def test_clean_signal_lowercase_separated_symbol():
    result = parse_signal(_msg(
        "short btc/usdt\nentry 61450 - 61200\nstop loss: 62000\ntake profit: 60000"
    ))
    a = result.analysis
    assert a is not None
    assert a.setup.symbol == "BTCUSDT"
    assert a.setup.direction == "short"
    assert a.setup.entries == [61450.0, 61200.0]
    assert a.setup.stop_loss == 62000.0
    assert a.setup.take_profits == [60000.0]


def test_clean_signal_known_base_no_separator_lowercase():
    result = parse_signal(_msg(
        "buy eurusd entry 1.0850 sl 1.0800 tp 1.0950"
    ))
    a = result.analysis
    assert a is not None
    assert a.setup.symbol == "EURUSD"
    assert a.setup.direction == "long"


def test_gold_alias():
    result = parse_signal(_msg(
        "LONG GOLD Entry: 2310 SL: 2295 TP1: 2340"
    ))
    assert result.analysis is not None
    assert result.analysis.setup.symbol == "XAUUSD"


def test_leverage_and_order_type_extracted():
    result = parse_signal(_msg(
        "BTCUSDT LONG limit order\nEntry: 61200\nSL: 60350\nTP1: 62400\n"
        "Leverage: cross 10x"
    ))
    a = result.analysis
    assert a is not None
    assert a.setup.leverage == "cross 10x"
    assert a.setup.order_type == "limit"


def test_stop_order_type_not_confused_with_stop_loss():
    result = parse_signal(_msg(
        "buy stop BTCUSDT\nEntry: 61200\nSL: 60350\nTP1: 62400"
    ))
    assert result.analysis is not None
    assert result.analysis.setup.order_type == "stop"


# --- confidently non-signal ---------------------------------------------------

def test_no_signal_language_at_all():
    result = parse_signal(_msg("gm team, great day today"))
    assert result.reason == "no_signal_markers"
    a = result.analysis
    assert a is not None
    assert a.is_signal is False
    assert a.category == "other"
    assert a.source == "regex"


def test_emoji_only_message():
    result = parse_signal(_msg("🚀🚀🚀"))
    assert result.reason == "no_signal_markers"
    assert result.analysis is not None
    assert result.analysis.is_signal is False


# --- must defer to Claude (ambiguous) -----------------------------------------

def test_update_message_missing_core_fields_defers():
    result = parse_signal(_msg("Move SL to entry on BTCUSDT"))
    assert result.analysis is None
    assert result.reason.startswith("partial_match_missing_")


def test_result_report_defers_even_with_full_fields():
    # Deliberately shaped like a clean signal, but "hit" + "%" mark it as a
    # result report, not a fresh signal.
    result = parse_signal(_msg(
        "BTCUSDT LONG Entry: 61200 SL: 60350 TP1: 62400 hit, +45%"
    ))
    assert result.analysis is None
    assert result.reason == "contains_result_marker"


def test_promotional_message_defers_even_with_full_fields():
    result = parse_signal(_msg(
        "VIP signal! BTCUSDT LONG Entry: 61200 SL: 60350 TP1: 62400 "
        "join https://t.me/vipgroup"
    ))
    assert result.analysis is None
    assert result.reason == "contains_promo_marker"


def test_reply_always_defers_even_if_clean():
    result = parse_signal(_msg(
        "BTCUSDT LONG Entry: 61200 SL: 60350 TP1: 62400", reply_to_id=99,
    ))
    assert result.analysis is None
    assert result.reason == "is_reply"


def test_partial_signal_missing_stop_loss_defers():
    result = parse_signal(_msg("BTCUSDT LONG Entry: 61200 TP1: 62400"))
    assert result.analysis is None
    assert "stop_loss" in result.reason


def test_both_directions_present_is_ambiguous():
    result = parse_signal(_msg(
        "BTCUSDT buy or sell? Entry: 61200 SL: 60350 TP1: 62400"
    ))
    assert result.analysis is None


# --- symbol false-positive guards ----------------------------------------------

def test_the_usd_is_not_a_symbol():
    result = parse_signal(_msg("the USD is strong today against everything"))
    # No separator, not all-caps ticker shape, "the" not a known base.
    assert result.reason == "no_signal_markers"
    assert result.analysis is not None
    assert result.analysis.setup.symbol is None


def test_unrecognised_cross_pair_defers_rather_than_guessing():
    # AUD/NZD-style crosses aren't in the small quote-currency list; this
    # should defer to Claude rather than silently emitting a wrong symbol.
    result = parse_signal(_msg(
        "LONG AUDNZD Entry: 1.0800 SL: 1.0750 TP1: 1.0900"
    ))
    assert result.analysis is None
