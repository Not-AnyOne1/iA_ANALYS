"""Unit tests for validation_engine.py.

Contexts are built by hand so each check can be isolated: the point is to
prove the engine actually rejects, not that it runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import validation_engine as ve
from indicators import ADXResult, EMASet
from levels import Distance, LevelContext
from market_context import MarketContext, TimeframeAnalysis
from market_data import Candle, Timeframe
from models import TradeSetup
from news_filter import EconomicEvent, NewsFilter, StaticCalendar
from scoring_engine import FactorScore, ScoreDirection, ScoringResult
from sessions import session_info
from smc_engine import SMCAnalysis
from structure_engine import (
    StructureAnalysis,
    StructureEvent,
    StructureEventType,
    SwingPoint,
    SwingType,
    TrendDirection,
)
from trade_quality import assess as assess_quality


def _ts(i: int = 0) -> datetime:
    return datetime(2026, 5, 1, 14, tzinfo=timezone.utc) + timedelta(hours=i)


def _tf(trend: str = "bullish", *, atr: float = 2.0, atr_pct: float = 1.0,
        vol: str = "normal", ema200: float = 90.0) -> TimeframeAnalysis:
    return TimeframeAnalysis(
        timeframe=Timeframe.H1, candles_analysed=300, trend=trend,
        ema=EMASet(100.0, 95.0, ema200), rsi=55.0, atr=atr, atr_percent=atr_pct,
        volatility=vol, adx=ADXResult(30.0, 25.0, 15.0), last_close=100.0,
    )


def _smc(**overrides) -> SMCAnalysis:
    base = dict(
        liquidity_pools=(), liquidity_sweeps=(), equal_highs=(), equal_lows=(),
        fair_value_gaps=(), inverse_fvgs=(), order_blocks=(), breaker_blocks=(),
        mitigation_blocks=(), supply_zones=(), demand_zones=(),
        premium_zone=None, discount_zone=None, ote_zone=None,
    )
    base.update(overrides)
    return SMCAnalysis(symbol="XAUUSD", timeframe=Timeframe.H1, **base)


def _structure(trend=TrendDirection.BULLISH, events=(), swings=()) -> StructureAnalysis:
    events = tuple(events)
    return StructureAnalysis(symbol="XAUUSD", timeframe=Timeframe.H1, trend=trend,
                             swing_points=tuple(swings), events=events,
                             last_event=events[-1] if events else None)


def _event(direction: TrendDirection, type_=StructureEventType.BOS, index: int = 5):
    return StructureEvent(type=type_, direction=direction, timestamp=_ts(index),
                          candle_index=index, break_price=100.0,
                          broken_swing=SwingPoint(index=index - 1, timestamp=_ts(index - 1),
                                                  price=99.0, type=SwingType.HIGH, strength=1))


def _candles(n: int = 260, drift: float = 0.1, spread: float = 1.0):
    out = []
    price = 100.0
    for i in range(n):
        price += drift
        out.append(Candle(timestamp=_ts(i), open=price, high=price + spread,
                          low=price - spread, close=price, volume=1.0))
    return tuple(out)


def _context(**overrides) -> MarketContext:
    ctx = MarketContext(symbol="XAUUSD", generated_at=_ts(),
                        primary_timeframe=Timeframe.H1)
    ctx.timeframes = {"H1": _tf(), "H4": _tf(), "D1": _tf(), "M15": _tf()}
    ctx.current_price = 100.0
    ctx.spread = 0.1
    ctx.structure = _structure()
    ctx.smc = _smc()
    ctx.session = session_info(_ts())
    ctx.news = NewsFilter(StaticCalendar()).check("XAUUSD", _ts())
    ctx.primary_candles = _candles()
    ctx.scoring = ScoringResult(
        symbol="XAUUSD", timeframe=Timeframe.H1, direction=ScoreDirection.BUY,
        confidence=70, net_score=60.0, total_possible=106.0,
        breakdown=(FactorScore("trend", 20.0, 20.0, TrendDirection.BULLISH, "bullish"),),
        reasons=("trend bullish",))
    ctx.quality = assess_quality(
        TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                   entries=[100.0], stop_loss=94.0, take_profits=[120.0]),
        current_price=100.0, atr_value=2.0)
    ctx.levels = LevelContext(
        day=None, distance_to_day_high=None, distance_to_day_low=None,
        nearest_resistance=Distance(level=110.0, absolute=10.0, percent=10.0, atr_multiple=5.0),
        nearest_support=Distance(level=95.0, absolute=5.0, percent=5.0, atr_multiple=2.5),
        resistance_levels=(110.0,), support_levels=(95.0,))
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def _validate(ctx, direction="long"):
    return ve.ValidationEngine().validate(ctx, direction=direction)


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


# --------------------------------------------------------------------------- shape

def test_every_listed_check_runs():
    result = _validate(_context())
    names = {c.name for c in result.checks}
    for expected in ve._WEIGHTS:
        assert expected in names, f"{expected} did not run"
    assert len(result.checks) == len(ve._WEIGHTS)


def test_score_is_bounded():
    """Each side has its own budget: the review may add up to
    MAX_ADJUSTMENT and remove up to MAX_PENALTY, and neither depends on the
    deterministic score."""
    r = _validate(_context())
    assert 0 <= r.bonus <= ve.MAX_ADJUSTMENT
    assert 0 <= r.penalty <= ve.MAX_PENALTY


def test_validation_is_deterministic():
    ctx = _context()
    assert _validate(ctx).adjustment == _validate(ctx).adjustment


# --------------------------------------------------------------- skeptical scoring

def test_an_unverifiable_setup_scores_low_not_neutral():
    """The core of the design: no data must not read as 'fine'."""
    blank = MarketContext(symbol="X", generated_at=_ts(), primary_timeframe=Timeframe.H1)
    result = _validate(blank)
    assert result.bonus <= 3
    assert len(result.unknowns) >= 15


def test_unknown_checks_dilute_the_score():
    full = _validate(_context()).bonus
    ctx = _context()
    ctx.smc = None
    ctx.levels = None
    assert _validate(ctx).bonus < full


def test_any_fatal_problem_zeroes_the_score():
    """The mechanism is retained even though no check emits FATAL any more.

    Blocking moved to its single owner, ``risk_engine``; ``Severity.FATAL``
    stays as the extension point, so its effect on the score must keep
    working. Built directly rather than through a check, because there is no
    longer any input that produces one.
    """
    clean = _validate(_context())
    assert not clean.has_fatal and clean.bonus > 0

    fatal = ve.ValidationResult(
        bonus=0, penalty=0,
        checks=clean.checks + (ve._check("news", ve.Severity.FATAL, "synthetic blocker"),),
    )
    assert fatal.has_fatal
    assert [c.detail for c in fatal.fatal_problems] == ["synthetic blocker"]


def test_no_check_emits_fatal_any_more():
    """Pins the architecture: this module scores, it does not veto.

    Sweeps every input that used to produce a FATAL — an unusable spread, a
    sub-1:1 payoff, an imminent release — plus the clean baseline.
    """
    rr = assess_quality(
        TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                   entries=[100.0], stop_loss=90.0, take_profits=[105.0]),
        current_price=100.0, atr_value=2.0)
    cal = StaticCalendar([EconomicEvent.create("FOMC Statement", _ts(), "USD")])

    wide = _context(); wide.spread = 1.0
    poor = _context(); poor.quality = rr
    news = _context(); news.news = NewsFilter(cal).check("XAUUSD", _ts())

    for label, ctx in (("clean", _context()), ("wide spread", wide),
                       ("R:R below 1", poor), ("news blackout", news)):
        result = _validate(ctx)
        assert not result.has_fatal, f"{label} produced a fatal: {result.summary}"


def test_a_strong_setup_scores_well():
    ctx = _context()
    ctx.smc = _smc(
        liquidity_sweeps=(_sweep_bullish(),),
        order_blocks=(_order_block_bullish(),),
        fair_value_gaps=(_fvg_bullish(),),
        liquidity_pools=(_pool(level=80.0),),
    )
    ctx.structure = _structure(events=(_event(TrendDirection.BULLISH),
                                       _event(TrendDirection.BULLISH,
                                              StructureEventType.CHOCH, 6)))
    result = _validate(ctx)
    assert not result.has_fatal
    assert result.adjustment >= 10, result.summary


def _sweep_bullish():
    from smc_engine import LiquiditySide, LiquiditySweep
    return LiquiditySweep(timeframe=Timeframe.H1, candle_index=9, timestamp=_ts(9),
                          price_high=101.0, price_low=99.0, strength=1.0,
                          side=LiquiditySide.SELL_SIDE, direction=TrendDirection.BULLISH,
                          swept_level=SwingPoint(index=8, timestamp=_ts(8), price=99.0,
                                                 type=SwingType.LOW, strength=1))


def _order_block_bullish():
    from smc_engine import OrderBlock
    return OrderBlock(timeframe=Timeframe.H1, candle_index=7, timestamp=_ts(7),
                      price_high=101.0, price_low=99.0, strength=1.0,
                      direction=TrendDirection.BULLISH,
                      source_event=_event(TrendDirection.BULLISH))


def _fvg_bullish():
    from smc_engine import FairValueGap
    return FairValueGap(timeframe=Timeframe.H1, candle_index=6, timestamp=_ts(6),
                        price_high=101.0, price_low=100.0, strength=1.0,
                        direction=TrendDirection.BULLISH)


def _pool(level: float):
    from smc_engine import LiquidityPool, LiquiditySide
    return LiquidityPool(timeframe=Timeframe.H1, candle_index=3, timestamp=_ts(3),
                         price_high=level, price_low=level, strength=1.0,
                         side=LiquiditySide.SELL_SIDE,
                         swing_points=(SwingPoint(index=3, timestamp=_ts(3), price=level,
                                                  type=SwingType.LOW, strength=1),))


# ------------------------------------------------- objective blockers, scored
#
# These three used to be FATAL here AND a REJECT in risk_engine — the same
# fact reaching the decision twice, by two routes, with two different
# consequences. risk_engine now owns the block; this module scores them at
# the full weight of their check and vetoes nothing.

def test_a_wide_spread_is_scored_not_vetoed():
    ctx = _context()
    ctx.spread = 1.0          # 0.5 ATR
    check = _check(_validate(ctx), "spread")
    assert check.severity is ve.Severity.WEAKNESS
    assert check.contribution == -check.weight
    assert not _validate(ctx).has_fatal


def test_risk_reward_below_one_is_scored_not_vetoed():
    ctx = _context()
    ctx.quality = assess_quality(
        TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                   entries=[100.0], stop_loss=90.0, take_profits=[105.0]),
        current_price=100.0, atr_value=2.0)
    check = _check(_validate(ctx), "risk_reward")
    assert check.severity is ve.Severity.WEAKNESS
    assert check.contribution == -check.weight
    assert not _validate(ctx).has_fatal


def test_imminent_news_is_scored_not_vetoed():
    ctx = _context()
    cal = StaticCalendar([EconomicEvent.create("FOMC Statement", _ts(), "USD")])
    ctx.news = NewsFilter(cal).check("XAUUSD", _ts())
    check = _check(_validate(ctx), "news")
    assert check.severity is ve.Severity.WEAKNESS
    assert check.contribution == -check.weight
    assert not _validate(ctx).has_fatal


def test_execution_findings_are_reported_but_do_not_move_confidence():
    """A wide spread is a real finding and still reaches the report, the
    risk engine and Claude — it just says nothing about which way price is
    going, so it cannot change how confident we are in the direction."""
    clean = _validate(_context())
    wide = _context(); wide.spread = 1.0
    result = _validate(wide)

    assert _check(result, "spread").severity is ve.Severity.WEAKNESS   # still found
    assert "spread" in {c.name for c in result.weaknesses}             # still reported
    assert result.adjustment == clean.adjustment                       # confidence untouched


# ------------------------------------------------- judgement rules (weaknesses)
#
# Each of these was fatal and is now a weakness. They stay fully evaluated and
# still subtract the full weight of their check — the score keeps reflecting
# the risk — but none of them can force SKIP on its own.

def _liquidity_against_context():
    ctx = _context()
    from smc_engine import LiquiditySide, LiquiditySweep
    bearish = LiquiditySweep(timeframe=Timeframe.H1, candle_index=9, timestamp=_ts(9),
                             price_high=101.0, price_low=99.0, strength=1.0,
                             side=LiquiditySide.BUY_SIDE, direction=TrendDirection.BEARISH,
                             swept_level=SwingPoint(index=8, timestamp=_ts(8), price=101.0,
                                                    type=SwingType.HIGH, strength=1))
    ctx.smc = _smc(liquidity_sweeps=(bearish,))
    return ctx


def _low_volatility_context():
    ctx = _context()
    ctx.timeframes["H1"] = _tf(vol="low")
    return ctx


def _opposite_probability_context():
    ctx = _context()
    ctx.scoring = ScoringResult(
        symbol="XAUUSD", timeframe=Timeframe.H1, direction=ScoreDirection.SELL,
        confidence=70, net_score=-60.0, total_possible=106.0,
        breakdown=(), reasons=())
    return ctx


# (context factory, direction, the check the scenario is about)
_DEMOTED = [
    (_context, "short", "higher_timeframe_trend"),          # counter-trend
    (_liquidity_against_context, "long", "liquidity_direction"),
    (_low_volatility_context, "long", "volatility"),
    (_opposite_probability_context, "long", "probability_score"),
]
_DEMOTED_IDS = [name for _, _, name in _DEMOTED]


@pytest.mark.parametrize("factory,direction,name", _DEMOTED, ids=_DEMOTED_IDS)
def test_a_demoted_check_is_a_weakness_not_fatal(factory, direction, name):
    check = _check(_validate(factory(), direction=direction), name)
    assert check.severity is ve.Severity.WEAKNESS
    assert check.severity is not ve.Severity.FATAL


@pytest.mark.parametrize("factory,direction,name", _DEMOTED, ids=_DEMOTED_IDS)
def test_a_demoted_check_is_reported_under_weaknesses(factory, direction, name):
    result = _validate(factory(), direction=direction)
    assert name in {c.name for c in result.weaknesses}
    assert name not in {c.name for c in result.fatal_problems}


@pytest.mark.parametrize("factory,direction,name", _DEMOTED, ids=_DEMOTED_IDS)
def test_a_demoted_check_never_forces_skip_on_its_own(factory, direction, name):
    """No fatal, so the score bands decide — not a veto."""
    result = _validate(factory(), direction=direction)
    assert not result.has_fatal, result.summary
    assert result.adjustment > -ve.MAX_ADJUSTMENT


@pytest.mark.parametrize("factory,direction,name", _DEMOTED, ids=_DEMOTED_IDS)
def test_a_demoted_check_still_costs_the_full_weight_of_its_check(factory, direction, name):
    """Demoted, not softened: the score still reflects the risk."""
    check = _check(_validate(factory(), direction=direction), name)
    assert check.contribution == -check.weight


@pytest.mark.parametrize("factory,direction,name", _DEMOTED, ids=_DEMOTED_IDS)
def test_a_demoted_check_reduces_the_score(factory, direction, name):
    clean = _validate(_context(), direction="long").adjustment
    assert _validate(factory(), direction=direction).adjustment < clean


# ----------------------------------------------------------- adverse excursion

def test_mae_is_none_without_atr():
    assert ve.estimate_mae_atr(_candles(), "long", None) is None


def test_mae_is_none_with_too_little_history():
    assert ve.estimate_mae_atr(_candles(5), "long", 2.0) is None


def test_mae_is_larger_for_the_direction_the_market_moves_against():
    falling = tuple(reversed(_candles(260, drift=0.1)))
    long_mae = ve.estimate_mae_atr(falling, "long", 2.0)
    short_mae = ve.estimate_mae_atr(falling, "short", 2.0)
    # In a falling series a long suffers far more adverse excursion.
    assert long_mae > short_mae


def _tight_stop_context():
    """A context whose stop sits well inside the typical adverse excursion."""
    ctx = _context()
    # Wide swings: MAE will be large. Stop is deliberately tight.
    ctx.primary_candles = _candles(260, drift=0.0, spread=8.0)
    ctx.quality = assess_quality(
        TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                   entries=[100.0], stop_loss=99.0, take_profits=[130.0]),
        current_price=100.0, atr_value=2.0)
    return ctx


def test_a_stop_inside_typical_adverse_excursion_is_a_weakness_not_fatal():
    """MAE is a trade-quality warning, not an objective blocker.

    A stop tighter than the usual heat is a statistical tendency, not an
    impossibility, so it weighs heavily on the score and lets the decision
    bands judge — it must never veto the trade on its own.
    """
    check = _check(_validate(_tight_stop_context()), "max_adverse_excursion")

    assert check.severity is ve.Severity.WEAKNESS
    assert check.severity is not ve.Severity.FATAL
    assert "heat" in check.detail


def test_a_tight_stop_does_not_appear_in_fatal_problems():
    result = _validate(_tight_stop_context())

    assert "max_adverse_excursion" not in {c.name for c in result.fatal_problems}
    assert "max_adverse_excursion" in {c.name for c in result.weaknesses}


def test_a_tight_stop_alone_does_not_zero_the_score():
    """It must reduce the score, not collapse it to 0 the way a fatal does."""
    result = _validate(_tight_stop_context())

    assert not result.has_fatal
    assert result.adjustment > -ve.MAX_ADJUSTMENT


def test_a_tight_stop_still_reduces_the_score():
    """The score must continue to reflect the risk.

    Both contexts share the same candles and a healthy R:R — only the stop
    width differs, sized from the measured excursion so the comparison
    isolates the MAE check rather than tripping some other rule.
    """
    candles = _candles(260, drift=0.0, spread=8.0)
    mae = ve.estimate_mae_atr(candles, "long", 2.0)
    assert mae is not None

    def context_with_stop(stop_atr_multiple: float):
        ctx = _context()
        ctx.primary_candles = candles
        risk = stop_atr_multiple * 2.0            # ATR multiple -> price units
        ctx.quality = assess_quality(
            TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                       entries=[100.0], stop_loss=100.0 - risk,
                       take_profits=[100.0 + risk * 3]),   # R:R fixed at 3
            current_price=100.0, atr_value=2.0)
        return ctx

    tight_ctx = context_with_stop(mae * 0.5)      # stop inside the excursion
    generous_ctx = context_with_stop(mae * 2.0)   # stop comfortably clear
    tight, generous = _validate(tight_ctx), _validate(generous_ctx)

    # The finding is still made, and still says the tight stop is worse...
    assert _check(tight, "max_adverse_excursion").severity is ve.Severity.WEAKNESS
    assert _check(generous, "max_adverse_excursion").severity is ve.Severity.STRENGTH
    # ...but stop width is trade management, so confidence does not move.
    assert tight.adjustment == generous.adjustment


def test_the_severe_mae_case_costs_more_than_the_marginal_one():
    """Full weight for a stop inside the excursion, half for one that
    barely clears it — the score stays proportional to the hazard."""
    severe = _check(_validate(_tight_stop_context()), "max_adverse_excursion")

    marginal = _context()
    marginal.primary_candles = _candles(260, drift=0.0, spread=8.0)
    # Sized to land between mae and mae * 1.3.
    mae = ve.estimate_mae_atr(marginal.primary_candles, "long", 2.0)
    stop_distance = mae * 1.15 * 2.0
    marginal.quality = assess_quality(
        TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                   entries=[100.0], stop_loss=100.0 - stop_distance,
                   take_profits=[130.0]),
        current_price=100.0, atr_value=2.0)
    marginal_check = _check(_validate(marginal), "max_adverse_excursion")

    assert severe.severity is ve.Severity.WEAKNESS
    assert marginal_check.severity is ve.Severity.WEAKNESS
    assert severe.contribution < marginal_check.contribution   # more negative


def test_a_generous_stop_clears_adverse_excursion():
    ctx = _context()
    ctx.primary_candles = _candles(260, drift=0.05, spread=0.2)
    ctx.quality = assess_quality(
        TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                   entries=[100.0], stop_loss=94.0, take_profits=[130.0]),
        current_price=100.0, atr_value=2.0)
    assert _check(_validate(ctx), "max_adverse_excursion").severity is ve.Severity.STRENGTH


# --------------------------------------------------------------------- weaknesses

def test_lower_timeframe_contradiction_is_a_weakness():
    ctx = _context()
    ctx.timeframes["M15"] = _tf(trend="bearish")
    assert _check(_validate(ctx), "lower_timeframe_confirmation").severity is ve.Severity.WEAKNESS


def test_broken_order_block_is_a_weakness():
    from smc_engine import BreakerBlock
    ctx = _context()
    ob = _order_block_bullish()
    breaker = BreakerBlock(timeframe=Timeframe.H1, candle_index=8, timestamp=_ts(8),
                           price_high=101.0, price_low=99.0, strength=1.0,
                           direction=TrendDirection.BEARISH, source_order_block=ob)
    ctx.smc = _smc(order_blocks=(ob,), breaker_blocks=(breaker,))
    assert _check(_validate(ctx), "order_block_quality").severity is ve.Severity.WEAKNESS


def test_inverted_fvg_is_a_weakness():
    from smc_engine import InverseFVG
    ctx = _context()
    fvg = _fvg_bullish()
    inv = InverseFVG(timeframe=Timeframe.H1, candle_index=9, timestamp=_ts(9),
                     price_high=101.0, price_low=100.0, strength=1.0,
                     direction=TrendDirection.BEARISH, source_fvg=fvg)
    ctx.smc = _smc(fair_value_gaps=(fvg,), inverse_fvgs=(inv,))
    assert _check(_validate(ctx), "fvg_quality").severity is ve.Severity.WEAKNESS


def test_entry_on_top_of_liquidity_is_a_weakness():
    ctx = _context()
    ctx.smc = _smc(liquidity_pools=(_pool(level=100.2),))   # 0.1 ATR away
    assert _check(_validate(ctx), "distance_from_liquidity").severity is ve.Severity.WEAKNESS


def test_wrong_side_of_ema200_is_a_weakness():
    ctx = _context()
    ctx.timeframes["H1"] = _tf(ema200=120.0)   # price 100 is below for a long
    assert _check(_validate(ctx), "distance_to_ema200").severity is ve.Severity.WEAKNESS


def test_resistance_immediately_ahead_of_a_long_is_a_weakness():
    ctx = _context()
    ctx.levels = LevelContext(
        day=None, distance_to_day_high=None, distance_to_day_low=None,
        nearest_resistance=Distance(level=100.4, absolute=0.4, percent=0.4, atr_multiple=0.2),
        nearest_support=None, resistance_levels=(100.4,), support_levels=())
    assert _check(_validate(ctx), "distance_to_resistance").severity is ve.Severity.WEAKNESS


def test_off_hours_session_is_a_weakness():
    ctx = _context()
    ctx.session = session_info(datetime(2026, 5, 1, 22, tzinfo=timezone.utc))
    assert _check(_validate(ctx), "session_quality").severity is ve.Severity.WEAKNESS


def test_overlap_session_is_a_strength():
    ctx = _context()
    ctx.session = session_info(datetime(2026, 5, 1, 14, tzinfo=timezone.utc))
    assert _check(_validate(ctx), "session_quality").severity is ve.Severity.STRENGTH


# ---------------------------------------------------------------------- summary

def test_summary_names_fatal_problems():
    clean = _validate(_context())
    result = ve.ValidationResult(
        bonus=0, penalty=0,
        checks=clean.checks + (ve._check("news", ve.Severity.FATAL, "synthetic blocker"),),
    )
    summary = result.summary
    assert "fatal" in summary.lower()
    assert "synthetic blocker" in summary


def test_summary_counts_findings_when_clean():
    summary = _validate(_context()).summary
    assert "strengths" in summary and "weaknesses" in summary


def test_mixed_higher_timeframes_do_not_crash():
    """Regression: `verdicts` holds 3-tuples (tf, aligned, trend) but the
    disagreement message unpacked only two, raising ValueError.

    Reached only when D1 and H4 contradict each other — every other fixture
    sets them to the same trend, which is why nothing caught it.
    """
    for d1, h4 in (("bullish", "bearish"), ("bearish", "bullish")):
        for direction in ("long", "short"):
            ctx = _context()
            ctx.timeframes["D1"] = _tf(d1)
            ctx.timeframes["H4"] = _tf(h4)
            check = ve._check_higher_timeframe_trend(ctx, direction)
            assert check.severity is ve.Severity.WEAKNESS
            assert "higher timeframes disagree" in check.detail
            assert f"D1={d1}" in check.detail and f"H4={h4}" in check.detail


def test_mixed_higher_timeframes_survive_a_full_validation():
    """The whole engine must run, not just the one check."""
    ctx = _context()
    ctx.timeframes["D1"] = _tf("bullish")
    ctx.timeframes["H4"] = _tf("bearish")
    result = _validate(ctx, direction="long")
    assert _check(result, "higher_timeframe_trend").severity is ve.Severity.WEAKNESS
    assert 0 <= result.bonus <= ve.MAX_ADJUSTMENT
    assert 0 <= result.penalty <= ve.MAX_ADJUSTMENT


# ------------------------------------------------- nonlinear review adjustment

def _adj(deterministic, positive_weight, negative_weight, total=106.0):
    """Drive the adjustment directly from weighted evidence.

    ``deterministic`` is accepted only so the call sites read naturally; the
    penalty deliberately does NOT depend on it — see the test below.
    """
    checks = (
        ve.ValidationCheck("pos", ve.Severity.STRENGTH, positive_weight, positive_weight, ""),
        ve.ValidationCheck("neg", ve.Severity.WEAKNESS, -negative_weight, negative_weight, ""),
        ve.ValidationCheck("rest", ve.Severity.UNKNOWN, 0.0,
                           max(0.0, total - positive_weight - negative_weight), ""),
    )
    return ve.ValidationEngine._adjustments(checks)


def _final(deterministic, positive_weight, negative_weight):
    bonus, penalty = _adj(deterministic, positive_weight, negative_weight)
    return max(0, min(100, deterministic + bonus - penalty))


def test_a_strong_score_cannot_outrun_overwhelming_opposition():
    """The reported defect: deterministic 92 with the review 5x against it
    used to land on 79 (ENTER) because the penalty was capped at 25."""
    assert _final(92, 0.12 * 106, 0.64 * 106) < 60


def test_the_response_curve_accelerates_with_opposition():
    """Convexity, measured on the penalty itself.

    Evenly spaced steps of opposition must cost strictly more each time —
    that is what "small disagreement -> tiny, overwhelming -> collapse"
    means as a curve rather than as a table of thresholds.
    """
    penalties = [_adj(80, (1 - f) * 106, f * 106)[1]
                 for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    steps = [b - a for a, b in zip(penalties, penalties[1:])]

    assert penalties[0] == 0                 # no opposition, no penalty
    assert penalties[1] <= 5                 # small disagreement -> tiny
    assert penalties[-1] == 80               # total opposition -> collapse
    assert steps == sorted(steps), f"not convex: {penalties} -> {steps}"
    assert steps[-1] > 2 * steps[0]          # and markedly so


def test_confidence_never_rises_as_opposition_grows():
    for deterministic in (0, 25, 50, 75, 100):
        for pos in range(0, 107, 5):
            scores = [_final(deterministic, pos, neg) for neg in range(0, 107 - pos, 5)]
            assert scores == sorted(scores, reverse=True), (deterministic, pos, scores)


def test_confidence_never_falls_as_support_grows():
    for deterministic in (0, 25, 50, 75, 100):
        for neg in range(0, 107, 5):
            scores = [_final(deterministic, pos, neg) for pos in range(0, 107 - neg, 5)]
            assert scores == sorted(scores), (deterministic, neg, scores)


def test_the_adjustment_is_smooth():
    """No step, no threshold: one unit of weight can never move confidence
    by more than a point."""
    for deterministic in (50, 80, 100):
        vals = [_final(deterministic, 20, neg) for neg in range(0, 87)]
        assert max(abs(b - a) for a, b in zip(vals, vals[1:])) <= 1


def test_no_opposition_means_no_penalty():
    assert _adj(90, 0.8 * 106, 0.0)[1] == 0
    assert _adj(90, 0.0, 0.0)[1] == 0


def test_thin_opposition_cannot_collapse_a_score():
    """One small check firing alone is lopsided (100% of what spoke) but is
    still only a sliver of the evidence base — the N/T factor sees that."""
    assert _final(90, 0.0, 6.0) >= 85


def test_the_bonus_is_unchanged_and_still_bounded():
    for pos in (0, 20, 53, 106):
        assert _adj(80, pos, 0.0)[0] == round(ve.MAX_ADJUSTMENT * pos / 106.0)
        assert _adj(80, pos, 0.0)[0] <= ve.MAX_ADJUSTMENT


def test_the_penalty_does_not_depend_on_the_deterministic_score():
    """The review is a brake, not a second opinion on how strong the setup
    is. Identical evidence must cost identical points at every score —
    scaling by the score made the brake bite hardest exactly where
    conviction was highest, which is backwards.
    """
    penalties = {_adj(D, 0.2 * 106, 0.6 * 106)[1] for D in (0, 25, 50, 75, 100)}
    assert len(penalties) == 1, f"penalty varied with the score: {penalties}"


def test_a_strong_setup_survives_moderate_opposition():
    """The brake must not reject good trades: balanced evidence leaves a
    strong deterministic reading in ENTER, and only marginal ones fall."""
    for D in (75, 80, 85, 92, 100):
        assert _final(D, 0.5 * 106, 0.5 * 106) >= 60, D
    assert _final(65, 0.5 * 106, 0.5 * 106) < 60      # marginal ENTER -> WAIT


def test_the_penalty_is_bounded_by_its_budget():
    for pos in range(0, 107, 5):
        for neg in range(0, 107 - pos, 5):
            assert 0 <= _adj(80, pos, neg)[1] <= ve.MAX_PENALTY


def test_the_adjustment_is_deterministic():
    assert _adj(77, 30.0, 40.0) == _adj(77, 30.0, 40.0)


# ------------------------------------- confidence is directional evidence only

def _setup(stop_loss=94.0, take_profits=(120.0,), entries=(100.0,)):
    return assess_quality(
        TradeSetup(symbol="XAUUSD", direction="long", order_type="limit",
                   entries=list(entries), stop_loss=stop_loss,
                   take_profits=list(take_profits)),
        current_price=100.0, atr_value=2.0)


@pytest.mark.parametrize("stop_loss,take_profits", [
    (99.0, (101.0,)),        # tight stop, poor 1:1 payoff
    (94.0, (120.0,)),        # the baseline
    (80.0, (140.0,)),        # very wide stop, generous payoff
    (99.5, (160.0,)),        # razor stop, enormous payoff
    (60.0, (101.0,)),        # absurd stop, feeble payoff
])
def test_changing_sl_and_tp_never_changes_confidence(stop_loss, take_profits):
    """The stated requirement, directly: same market analysis, different
    trade management, identical confidence."""
    baseline = _validate(_context())
    ctx = _context()
    ctx.quality = _setup(stop_loss, take_profits)

    result = _validate(ctx)

    assert result.bonus == baseline.bonus
    assert result.penalty == baseline.penalty


def test_a_perfect_payoff_cannot_rescue_a_weak_analysis():
    """A 1:3 R:R must never raise confidence when the read is poor."""
    weak = _context()                       # long into bearish higher timeframes
    weak.timeframes["D1"] = _tf("bearish")
    weak.timeframes["H4"] = _tf("bearish")

    thin_payoff = _validate(weak).adjustment
    weak.quality = _setup(stop_loss=99.0, take_profits=(103.0,))   # R:R 1:3
    assert _validate(weak).adjustment == thin_payoff


def test_changing_the_market_analysis_does_change_confidence():
    """The other half of the contract — confidence is not simply inert."""
    bullish = _validate(_context()).adjustment
    bearish = _context()
    bearish.timeframes["D1"] = _tf("bearish")
    bearish.timeframes["H4"] = _tf("bearish")

    assert _validate(bearish).adjustment < bullish


def test_execution_checks_still_run_and_are_still_reported():
    """Excluded from confidence, not from the analysis."""
    ctx = _context()
    ctx.spread = 1.0
    ctx.quality = _setup(stop_loss=99.9, take_profits=(100.1,))
    result = _validate(ctx)

    names = {c.name for c in result.checks}
    for name in ve._EXECUTION_CHECKS:
        assert name in names, f"{name} stopped running"
    reported = {c.name for c in result.weaknesses} | {c.name for c in result.strengths}
    assert ve._EXECUTION_CHECKS & reported, "no execution finding reached the report"


def test_the_execution_set_is_exactly_the_checks_that_read_the_trade():
    """Guards the classification rule itself: a check belongs to the
    execution set iff it reads the trade's own parameters or the cost of
    transacting. Verified by mutating each and watching what moves."""
    baseline = _validate(_context())

    # Mutating ONLY trade parameters must leave confidence untouched.
    trade_only = _context()
    trade_only.quality = _setup(stop_loss=99.0, take_profits=(101.0,))
    trade_only.spread = 1.0
    assert _validate(trade_only).adjustment == baseline.adjustment

    # Mutating market evidence must move it.
    market = _context()
    market.smc = _smc(liquidity_sweeps=(_sweep_bullish(),))
    assert _validate(market).adjustment != baseline.adjustment


def test_every_execution_check_is_a_known_weight():
    assert ve._EXECUTION_CHECKS <= set(ve._WEIGHTS)


def test_directional_checks_are_the_majority_of_the_evidence():
    """Confidence must still rest on a broad base after the exclusion."""
    directional = {n: w for n, w in ve._WEIGHTS.items() if n not in ve._EXECUTION_CHECKS}
    assert len(directional) == len(ve._WEIGHTS) - len(ve._EXECUTION_CHECKS)
    assert sum(directional.values()) > sum(ve._WEIGHTS.values()) / 2
