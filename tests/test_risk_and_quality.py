"""Unit tests for trade_quality.py, news_filter.py and risk_engine.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import risk_engine
import trade_quality
from models import TradeSetup
from news_filter import (
    EconomicEvent,
    Impact,
    NewsFilter,
    StaticCalendar,
    classify_impact,
    currencies_in,
)
from risk_engine import RiskSettings, RiskVerdict
from structure_engine import SwingPoint, SwingType, TrendDirection
from trade_quality import Quality


def _setup(**overrides) -> TradeSetup:
    base = dict(symbol="XAUUSD", direction="long", order_type="limit",
                entries=[100.0], stop_loss=95.0, take_profits=[115.0])
    base.update(overrides)
    return TradeSetup(**base)


def _swing(price: float, type_: SwingType, index: int = 1) -> SwingPoint:
    return SwingPoint(index=index, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                      price=price, type=type_, strength=1)


# --------------------------------------------------------------------------- entry

def test_effective_entry_averages_a_stated_zone():
    setup = _setup(entries=[100.0, 110.0])
    assert trade_quality.effective_entry(setup) == pytest.approx(105.0)


def test_effective_entry_falls_back_to_live_price_for_a_market_order():
    setup = _setup(entries=[])
    assert trade_quality.effective_entry(setup, current_price=123.0) == 123.0


def test_effective_entry_is_none_without_entry_or_price():
    assert trade_quality.effective_entry(_setup(entries=[])) is None


# ------------------------------------------------------------------ stop quality

def test_stop_unknown_without_a_stated_stop():
    q = trade_quality.assess(_setup(stop_loss=None), atr_value=2.0)
    assert q.stop.quality is Quality.UNKNOWN
    assert "no stop loss stated" in " ".join(q.stop.reasons)


def test_stop_poor_when_inside_the_noise_band():
    # 1.0 away with ATR 10 -> 0.1 ATR, far inside normal noise.
    q = trade_quality.assess(_setup(stop_loss=99.0), atr_value=10.0)
    assert q.stop.quality is Quality.POOR
    assert "noise" in " ".join(q.stop.reasons)


def test_stop_poor_when_absurdly_wide():
    q = trade_quality.assess(_setup(stop_loss=40.0), atr_value=2.0)  # 30 ATR
    assert q.stop.quality is Quality.POOR
    assert "wide" in " ".join(q.stop.reasons)


def test_stop_good_in_the_sensible_band():
    q = trade_quality.assess(_setup(stop_loss=95.0), atr_value=2.5)  # 2 ATR
    assert q.stop.quality is Quality.GOOD


def test_stop_unknown_without_atr():
    q = trade_quality.assess(_setup(), atr_value=None)
    assert q.stop.quality is Quality.UNKNOWN


def test_stop_flagged_when_sitting_beyond_liquidity():
    # Long entry 100, stop 95, and a swing low at 97 sits between them.
    q = trade_quality.assess(
        _setup(stop_loss=95.0), atr_value=2.5,
        swings=[_swing(97.0, SwingType.LOW)],
    )
    assert q.stop.inside_liquidity is True
    assert q.stop.quality is Quality.POOR
    assert "sweep" in " ".join(q.stop.reasons)


def test_short_stop_liquidity_check_uses_swing_highs():
    setup = _setup(direction="short", entries=[100.0], stop_loss=105.0, take_profits=[90.0])
    q = trade_quality.assess(setup, atr_value=2.5, swings=[_swing(103.0, SwingType.HIGH)])
    assert q.stop.inside_liquidity is True


# ---------------------------------------------------------------- target quality

def test_target_unknown_without_a_stated_target():
    q = trade_quality.assess(_setup(take_profits=[]), atr_value=2.0)
    assert q.target.quality is Quality.UNKNOWN


def test_target_poor_when_inside_the_noise_band():
    q = trade_quality.assess(_setup(take_profits=[100.5]), atr_value=10.0)
    assert q.target.quality is Quality.POOR


def test_target_good_when_comfortably_beyond_noise():
    q = trade_quality.assess(_setup(take_profits=[115.0]), atr_value=5.0)  # 3 ATR
    assert q.target.quality is Quality.GOOD


def test_target_flagged_when_a_level_blocks_the_path():
    import levels as levels_mod
    ctx = levels_mod.LevelContext(
        day=None, distance_to_day_high=None, distance_to_day_low=None,
        nearest_resistance=None, nearest_support=None,
        resistance_levels=(107.0,), support_levels=(),
    )
    q = trade_quality.assess(_setup(take_profits=[115.0]), atr_value=5.0, levels=ctx)
    assert q.target.blocked_by_level == 107.0
    assert q.target.quality is Quality.POOR


def test_has_complete_levels_reports_missing_pieces():
    assert trade_quality.assess(_setup(), atr_value=2.5).has_complete_levels is True
    assert trade_quality.assess(_setup(stop_loss=None), atr_value=2.5).has_complete_levels is False


# --------------------------------------------------------------------------- news

def test_classify_impact_recognises_major_releases():
    for title in ("FOMC Statement", "US NFP", "CPI y/y", "ECB Rate Decision",
                  "BOE Governor Bailey Speaks", "Non-Farm Payrolls"):
        assert classify_impact(title) is Impact.HIGH


def test_classify_impact_medium_and_low():
    assert classify_impact("Flash Manufacturing PMI") is Impact.MEDIUM
    assert classify_impact("Weekly Bulletin") is Impact.LOW


def test_currencies_in_splits_pairs():
    assert currencies_in("XAUUSD") == ("XAU", "USD")
    assert currencies_in("EUR/USD") == ("EUR", "USD")
    assert currencies_in("BTCUSDT") == ("BTC", "USDT")


def test_news_filter_reports_unavailable_without_a_calendar():
    status = NewsFilter().check("XAUUSD")
    assert status.available is False
    assert status.blocked is False          # unknown is not the same as clear
    assert "no economic calendar" in status.reason


def test_news_filter_blocks_an_imminent_high_impact_event():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    cal = StaticCalendar([EconomicEvent.create("US NFP", now + timedelta(minutes=10), "USD")])
    status = NewsFilter(cal).check("XAUUSD", now)
    assert status.blocked is True
    assert "NFP" in status.reason


def test_news_filter_ignores_an_event_outside_the_window():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    cal = StaticCalendar([EconomicEvent.create("US NFP", now + timedelta(hours=6), "USD")])
    status = NewsFilter(cal).check("XAUUSD", now)
    assert status.blocked is False
    assert status.available is True


def test_news_filter_ignores_an_irrelevant_currency():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    cal = StaticCalendar([EconomicEvent.create("BOJ Rate Decision", now, "JPY")])
    assert NewsFilter(cal).check("EURUSD", now).blocked is False
    assert NewsFilter(cal).check("USDJPY", now).blocked is True


def test_news_filter_ignores_medium_impact_events():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    cal = StaticCalendar([EconomicEvent.create("Flash PMI", now, "USD")])
    assert NewsFilter(cal).check("XAUUSD", now).blocked is False


def test_news_filter_blocks_just_after_a_release_too():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    cal = StaticCalendar([EconomicEvent.create("CPI", now - timedelta(minutes=5), "USD")])
    assert NewsFilter(cal).check("XAUUSD", now).blocked is True


def test_static_calendar_never_invents_events():
    assert StaticCalendar().events_between(
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        datetime(2030, 1, 1, tzinfo=timezone.utc),
    ) == []


# --------------------------------------------------------------------- risk engine

def _good_quality():
    return trade_quality.assess(_setup(take_profits=[115.0]), atr_value=2.5)


def test_risk_rejects_insufficient_risk_reward():
    # entry 100, stop 95 (5 risk), target 102 (2 reward) -> 0.4
    quality = trade_quality.assess(_setup(take_profits=[102.0]), atr_value=2.5)
    result = risk_engine.evaluate(quality)
    assert not result.approved
    assert any(c.name == "risk_reward" for c in result.rejections)


def _rr_verdict(target: float) -> RiskVerdict:
    """Risk verdict for a 5-point stop and the given target (entry 100)."""
    quality = trade_quality.assess(_setup(take_profits=[target]), atr_value=2.5)
    result = risk_engine.evaluate(quality)
    return next(c for c in result.checks if c.name == "risk_reward").verdict


@pytest.mark.parametrize("target,rr,expected", [
    (102.0, 0.40, RiskVerdict.REJECT),   # well below the minimum
    (105.0, 1.00, RiskVerdict.REJECT),   # 1:1 is now below the 1.10 minimum
    (105.4, 1.08, RiskVerdict.REJECT),   # boundary: strictly below rejects
    (105.45, 1.09, RiskVerdict.REJECT),
    (105.5, 1.10, RiskVerdict.PASS),     # boundary: exactly 1.10 must NOT reject
    (105.55, 1.11, RiskVerdict.PASS),
    (107.5, 1.50, RiskVerdict.PASS),     # above the blocker — a preference,
    (109.9, 1.98, RiskVerdict.PASS),     # not an objective blocker
    (110.0, 2.00, RiskVerdict.PASS),
    (115.0, 3.00, RiskVerdict.PASS),
])
def test_risk_reward_rejects_only_strictly_below_the_minimum(target, rr, expected):
    """R:R is the only trade-parameter rule that still gates.

    Anything at or above the minimum is a payoff preference, which the
    validation engine still scores as a weakness below 1:2 — it just no
    longer blocks.
    """
    quality = trade_quality.assess(_setup(take_profits=[target]), atr_value=2.5)
    assert round(quality.risk_reward, 2) == rr, "fixture drifted"
    assert _rr_verdict(target) is expected


def test_a_trade_between_one_and_two_reward_is_no_longer_blocked():
    """Regression for the threshold change: R:R 1.5 used to be rejected."""
    quality = trade_quality.assess(_setup(take_profits=[107.5]), atr_value=2.5)
    result = risk_engine.evaluate(quality, direction="long",
                                  trend=TrendDirection.BULLISH, atr_value=2.5,
                                  spread=0.1, volatility="normal", confidence=70)
    assert result.approved
    assert not any(c.name == "risk_reward" for c in result.rejections)


def test_risk_warns_about_a_stop_inside_the_noise_without_rejecting():
    """Stop placement is a judgement about where price will go, not a fact
    about whether the trade can be placed. Diagnostics are unchanged."""
    quality = trade_quality.assess(_setup(stop_loss=99.9), atr_value=10.0)
    result = risk_engine.evaluate(quality)
    check = next(c for c in result.checks if c.name == "stop_quality")

    assert check.verdict is RiskVerdict.WARN
    assert check.warned and not check.rejected
    assert result.approved
    assert check in result.warnings
    assert check.detail                      # the reasons still come through


def test_risk_warns_about_a_poor_target_without_rejecting():
    quality = trade_quality.assess(_setup(take_profits=[100.05]), atr_value=10.0)
    result = risk_engine.evaluate(quality)
    check = next(c for c in result.checks if c.name == "target_quality")

    assert check.verdict is RiskVerdict.WARN
    assert check.warned and not check.rejected
    assert check in result.warnings
    assert check.detail


def test_neither_stop_nor_target_quality_can_ever_set_approved_false():
    """Both firing at their worst, with nothing else wrong, must still leave
    the gate open."""
    quality = trade_quality.assess(
        _setup(stop_loss=99.9, take_profits=[100.05]), atr_value=10.0)
    result = risk_engine.evaluate(quality, direction="long",
                                  trend=TrendDirection.BULLISH, atr_value=2.0,
                                  spread=0.1, volatility="normal", confidence=70)

    assert {"stop_quality", "target_quality"} <= {c.name for c in result.warnings}
    assert not any(c.name in ("stop_quality", "target_quality")
                   for c in result.rejections)


def test_risk_warns_about_a_wide_spread_without_rejecting():
    """Spread is a cost, and cost is a matter of degree. The ratio and the
    message are unchanged; only the verdict is."""
    result = risk_engine.evaluate(_good_quality(), atr_value=2.0, spread=1.0)  # 0.5 ATR
    check = next(c for c in result.checks if c.name == "spread")

    assert check.verdict is RiskVerdict.WARN
    assert check.warned and not check.rejected
    assert result.approved
    assert check in result.warnings
    assert "0.50 ATR" in check.detail and "max 0.25" in check.detail


def test_spread_can_never_set_approved_false():
    """However wide it gets, with nothing else wrong the gate stays open."""
    for spread in (0.6, 2.0, 20.0):
        result = risk_engine.evaluate(_good_quality(), atr_value=2.0, spread=spread,
                                      direction="long", trend=TrendDirection.BULLISH,
                                      volatility="normal", confidence=70)
        assert result.approved, f"spread {spread} blocked the trade"
        assert "spread" in {c.name for c in result.warnings}


def test_risk_warns_about_a_counter_trend_trade_without_rejecting():
    """Counter-trend is a reported finding, not a gate: it must not set
    ``approved = False``, because that would force SKIP on its own."""
    result = risk_engine.evaluate(_good_quality(), direction="long",
                                  trend=TrendDirection.BEARISH)
    check = next(c for c in result.checks if c.name == "trend_alignment")
    assert check.verdict is RiskVerdict.WARN
    assert check.warned and not check.rejected
    assert result.approved
    assert check in result.warnings
    assert "bearish" in check.detail


def test_risk_allows_a_with_trend_trade():
    result = risk_engine.evaluate(_good_quality(), direction="long",
                                  trend=TrendDirection.BULLISH)
    check = next(c for c in result.checks if c.name == "trend_alignment")
    assert check.verdict is RiskVerdict.PASS


def test_ranging_market_abstains_rather_than_rejecting():
    result = risk_engine.evaluate(_good_quality(), direction="long",
                                  trend=TrendDirection.RANGING)
    check = next(c for c in result.checks if c.name == "trend_alignment")
    assert check.verdict is RiskVerdict.ABSTAIN


def test_risk_warns_about_low_volatility_without_rejecting():
    result = risk_engine.evaluate(_good_quality(), volatility="low")
    check = next(c for c in result.checks if c.name == "volatility")
    assert check.verdict is RiskVerdict.WARN
    assert check.warned and not check.rejected
    assert result.approved
    assert check in result.warnings


def test_the_two_warning_rules_never_block_even_together():
    """Both firing at once still leaves the gate open — the analysis
    continues and the score bands decide."""
    result = risk_engine.evaluate(_good_quality(), direction="long",
                                  trend=TrendDirection.BEARISH, volatility="low")
    assert result.approved
    assert {c.name for c in result.warnings} == {"trend_alignment", "volatility"}
    assert not result.rejections


def test_a_warning_never_hides_a_real_rejection():
    """WARN must not soften the rules that still gate."""
    thin = trade_quality.assess(_setup(take_profits=[102.0]), atr_value=2.5)  # R:R 0.40
    result = risk_engine.evaluate(thin, direction="long",
                                  trend=TrendDirection.BEARISH, volatility="low",
                                  confidence=10, atr_value=2.0, spread=1.0)
    assert not result.approved
    assert {c.name for c in result.rejections} == {"risk_reward"}


def test_risk_warns_about_low_confidence_without_rejecting():
    """A low deterministic score is a market opinion, not an objective
    blocker, and the decision bands in trade_decision.py already send a low
    score to WAIT or SKIP. Rejecting here made the same fact veto twice."""
    result = risk_engine.evaluate(_good_quality(), confidence=10)
    check = next(c for c in result.checks if c.name == "confidence")
    assert check.verdict is RiskVerdict.WARN
    assert check.warned and not check.rejected
    assert result.approved
    assert check in result.warnings


def test_risk_rejects_imminent_news():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    cal = StaticCalendar([EconomicEvent.create("FOMC", now, "USD")])
    news = NewsFilter(cal).check("XAUUSD", now)
    result = risk_engine.evaluate(_good_quality(), news=news)
    assert any(c.name == "news" for c in result.rejections)


def test_missing_inputs_abstain_rather_than_reject():
    """The gate rejects known-bad setups; it must not reject the unmeasured."""
    result = risk_engine.evaluate(_good_quality())   # nothing else supplied
    assert result.approved
    abstained = {c.name for c in result.abstentions}
    assert {"spread", "volatility", "confidence", "news"} <= abstained


def test_a_fully_clean_setup_is_approved():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    result = risk_engine.evaluate(
        _good_quality(), direction="long", trend=TrendDirection.BULLISH,
        atr_value=2.5, spread=0.1, volatility="normal", confidence=80,
        news=NewsFilter(StaticCalendar()).check("XAUUSD", now),
    )
    assert result.approved, result.summary


def test_thresholds_are_configurable():
    quality = trade_quality.assess(_setup(take_profits=[102.0]), atr_value=2.5)  # RR 0.4
    lenient = risk_engine.evaluate(quality, settings=RiskSettings(min_risk_reward=0.1))
    assert not any(c.name == "risk_reward" for c in lenient.rejections)


def test_disabled_checks_pass_instead_of_warning():
    """Disabling a rule silences it entirely — PASS, not WARN."""
    settings = RiskSettings(reject_counter_trend=False, reject_low_volatility=False)
    result = risk_engine.evaluate(_good_quality(), direction="long",
                                  trend=TrendDirection.BEARISH, volatility="low",
                                  settings=settings)
    assert result.approved
    assert not result.warnings
    for name in ("trend_alignment", "volatility"):
        assert next(c for c in result.checks if c.name == name).verdict is RiskVerdict.PASS


def test_summary_lists_every_rejection():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    cal = StaticCalendar([EconomicEvent.create("FOMC", now, "USD")])
    quality = trade_quality.assess(_setup(take_profits=[102.0]), atr_value=2.5)  # RR 0.4
    result = risk_engine.evaluate(quality, news=NewsFilter(cal).check("XAUUSD", now))
    assert "risk_reward" in result.summary
    assert "news" in result.summary


def test_every_remaining_rejection_is_an_objective_blocker():
    """Pins the risk engine's scope: it blocks on facts about whether the
    trade can be placed at all, never on an opinion about the market."""
    objective = {"risk_reward", "news"}
    opinions = {"trend_alignment", "volatility", "confidence",
                "stop_quality", "target_quality", "spread"}

    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    cal = StaticCalendar([EconomicEvent.create("FOMC", now, "USD")])
    worst = risk_engine.evaluate(
        trade_quality.assess(_setup(stop_loss=99.9, take_profits=[102.0]), atr_value=10.0),
        direction="long", trend=TrendDirection.BEARISH, atr_value=2.0, spread=1.0,
        volatility="low", confidence=1, news=NewsFilter(cal).check("XAUUSD", now),
    )
    rejected = {c.name for c in worst.rejections}
    assert rejected <= objective, f"an opinion is blocking: {rejected - objective}"
    assert not (rejected & opinions)
    assert {c.name for c in worst.warnings} == opinions


def test_summary_counts_warnings_when_approved():
    result = risk_engine.evaluate(_good_quality(), volatility="low")
    assert result.approved
    assert "1 warning(s)" in result.summary


def test_risk_evaluation_is_deterministic():
    q = _good_quality()
    assert risk_engine.evaluate(q, confidence=70) == risk_engine.evaluate(q, confidence=70)
