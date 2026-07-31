"""Unit tests for sessions.py and levels.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import levels as levels_mod
from market_data import Candle
from sessions import Session, active_sessions, primary_session, session_info
from structure_engine import SwingPoint, SwingType


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 3, 10, hour, minute, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- sessions

@pytest.mark.parametrize("hour,expected", [
    (2, Session.ASIA),        # Asia only
    (9, Session.LONDON),      # London (Asia has closed at 08:00)
    (14, Session.NEW_YORK),   # London/NY overlap -> NY by precedence
    (19, Session.NEW_YORK),   # NY only
    (22, Session.OFF_HOURS),  # after NY close, before Asia open
])
def test_primary_session_at_key_hours(hour, expected):
    assert primary_session(_at(hour)) == expected


def test_asia_window_wraps_midnight():
    assert Session.ASIA in active_sessions(_at(23, 30))
    assert Session.ASIA in active_sessions(_at(1))
    assert Session.ASIA not in active_sessions(_at(10))


def test_london_new_york_overlap_reports_both():
    active = active_sessions(_at(14))
    assert Session.LONDON in active
    assert Session.NEW_YORK in active
    assert session_info(_at(14)).is_overlap is True


def test_asia_london_handover_reports_both():
    active = active_sessions(_at(7, 30))
    assert Session.ASIA in active
    assert Session.LONDON in active


def test_off_hours_has_no_active_session():
    info = session_info(_at(22))
    assert info.active == ()
    assert info.primary == Session.OFF_HOURS
    assert info.is_overlap is False
    assert info.label == "off_hours"


def test_naive_datetime_is_treated_as_utc():
    naive = datetime(2026, 3, 10, 14, 0)
    assert primary_session(naive) == primary_session(_at(14))


def test_non_utc_timezone_is_converted():
    # 14:00 UTC expressed as 16:00 in UTC+2 must give the same answer.
    other = datetime(2026, 3, 10, 16, 0, tzinfo=timezone(timedelta(hours=2)))
    assert primary_session(other) == Session.NEW_YORK


def test_session_label_joins_active_sessions():
    assert session_info(_at(14)).label == "new_york+london"


def test_session_detection_is_deterministic():
    moment = _at(14)
    assert session_info(moment) == session_info(moment)


# --------------------------------------------------------------------------- levels

def _c(day: int, hour: int, high: float, low: float, close: float) -> Candle:
    return Candle(timestamp=datetime(2026, 3, day, hour, tzinfo=timezone.utc),
                  open=close, high=high, low=low, close=close, volume=1.0)


def test_day_range_uses_only_the_latest_day_in_the_series():
    candles = [
        _c(9, 10, 200, 100, 150),   # previous day, much wider
        _c(10, 1, 120, 110, 115),
        _c(10, 2, 130, 105, 125),
    ]
    day = levels_mod.day_range(candles)
    assert day.high == 130
    assert day.low == 105


def test_day_range_is_none_for_empty_input():
    assert levels_mod.day_range([]) is None


def test_position_in_day_range():
    day = levels_mod.DayRange(high=110, low=100)
    assert day.position_of(100) == pytest.approx(0.0)
    assert day.position_of(110) == pytest.approx(1.0)
    assert day.position_of(105) == pytest.approx(0.5)


def test_position_is_none_for_a_zero_width_range():
    assert levels_mod.DayRange(high=100, low=100).position_of(100) is None


def _swing(index: int, price: float, type_: SwingType) -> SwingPoint:
    return SwingPoint(index=index, timestamp=datetime(2026, 3, 10, tzinfo=timezone.utc),
                      price=price, type=type_, strength=1)


def test_support_and_resistance_split_around_price():
    swings = [
        _swing(1, 120, SwingType.HIGH),
        _swing(2, 115, SwingType.HIGH),
        _swing(3, 95, SwingType.LOW),
        _swing(4, 90, SwingType.LOW),
    ]
    support, resistance = levels_mod.support_resistance(swings, price=100)
    assert resistance == [115, 120]     # nearest first, ascending above price
    assert support == [95, 90]          # nearest first, descending below price


def test_swing_high_below_price_is_not_resistance():
    swings = [_swing(1, 90, SwingType.HIGH)]
    _, resistance = levels_mod.support_resistance(swings, price=100)
    assert resistance == []


def test_support_resistance_caps_the_number_of_levels():
    swings = [_swing(i, 100 + i, SwingType.HIGH) for i in range(1, 20)]
    _, resistance = levels_mod.support_resistance(swings, price=100, max_levels=3)
    assert len(resistance) == 3


def test_distance_reports_absolute_percent_and_atr():
    ctx = levels_mod.level_context(
        candles=[_c(10, 1, 110, 90, 100)],
        swings=[_swing(1, 105, SwingType.HIGH)],
        price=100.0,
        atr_value=5.0,
    )
    assert ctx.nearest_resistance.absolute == pytest.approx(5.0)
    assert ctx.nearest_resistance.percent == pytest.approx(5.0)
    assert ctx.nearest_resistance.atr_multiple == pytest.approx(1.0)


def test_distance_atr_multiple_is_none_without_atr():
    ctx = levels_mod.level_context(
        candles=[_c(10, 1, 110, 90, 100)],
        swings=[_swing(1, 105, SwingType.HIGH)],
        price=100.0,
        atr_value=None,
    )
    assert ctx.nearest_resistance.atr_multiple is None
    assert ctx.nearest_resistance.is_near is False   # never guesses from percent


def test_is_near_within_half_an_atr():
    ctx = levels_mod.level_context(
        candles=[_c(10, 1, 110, 90, 100)],
        swings=[_swing(1, 101, SwingType.HIGH)],
        price=100.0,
        atr_value=5.0,
    )
    assert ctx.nearest_resistance.is_near is True


def test_level_context_survives_empty_swings():
    ctx = levels_mod.level_context(candles=[_c(10, 1, 110, 90, 100)],
                                   swings=[], price=100.0, atr_value=1.0)
    assert ctx.nearest_support is None
    assert ctx.nearest_resistance is None
    assert ctx.day is not None
