"""Unit tests for scoring_engine.py (RFC-007).

Entirely offline and deterministic: every test hand-builds a
StructureAnalysis and/or SMCAnalysis directly (constructing the RFC-005/
RFC-006 dataclasses with whatever fields the scoring rule under test
actually reads — no need to run the real detectors) and asserts on the
exact ScoringResult returned.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from market_data import Timeframe
from scoring_engine import FactorScore, ScoreDirection, ScoringEngine
from smc_engine import (
    DemandZone,
    DiscountZone,
    FairValueGap,
    InverseFVG,
    LiquiditySide,
    LiquiditySweep,
    OrderBlock,
    OTEZone,
    PremiumZone,
    SMCAnalysis,
    SupplyZone,
)
from structure_engine import (
    StructureAnalysis,
    StructureEvent,
    StructureEventType,
    SwingPoint,
    SwingType,
    TrendDirection,
)


# --------------------------------------------------------------------------- helpers

def _ts(i: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)


def _swing(index: int, price: float, type_: SwingType) -> SwingPoint:
    return SwingPoint(index=index, timestamp=_ts(index), price=price, type=type_, strength=1)


def _event(candle_index: int, direction: TrendDirection, type_=StructureEventType.BOS) -> StructureEvent:
    return StructureEvent(
        type=type_, direction=direction, timestamp=_ts(candle_index),
        candle_index=candle_index, break_price=100.0,
        broken_swing=_swing(max(candle_index - 1, 0), 100.0, SwingType.HIGH),
    )


def _structure(
    swings=(), events=(), trend=TrendDirection.UNKNOWN,
    symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1,
) -> StructureAnalysis:
    events = tuple(events)
    return StructureAnalysis(
        symbol=symbol, timeframe=timeframe, trend=trend,
        swing_points=tuple(swings), events=events,
        last_event=events[-1] if events else None,
    )


def _sweep(candle_index: int, direction: TrendDirection) -> LiquiditySweep:
    swing_type = SwingType.HIGH if direction == TrendDirection.BEARISH else SwingType.LOW
    side = LiquiditySide.BUY_SIDE if direction == TrendDirection.BEARISH else LiquiditySide.SELL_SIDE
    return LiquiditySweep(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=101.0, price_low=100.0, strength=1.0,
        side=side, direction=direction, swept_level=_swing(0, 100.0, swing_type),
    )


def _fvg(candle_index: int, direction: TrendDirection) -> FairValueGap:
    return FairValueGap(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=101.0, price_low=100.0, strength=1.0, direction=direction,
    )


def _ifvg(candle_index: int, direction: TrendDirection) -> InverseFVG:
    opposite = TrendDirection.BEARISH if direction == TrendDirection.BULLISH else TrendDirection.BULLISH
    source = _fvg(max(candle_index - 1, 0), opposite)
    return InverseFVG(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=101.0, price_low=100.0, strength=1.0, direction=direction, source_fvg=source,
    )


def _order_block(candle_index: int, direction: TrendDirection) -> OrderBlock:
    return OrderBlock(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=101.0, price_low=100.0, strength=1.0,
        direction=direction, source_event=_event(candle_index + 1, direction),
    )


def _demand_zone(candle_index: int = 0) -> DemandZone:
    return DemandZone(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=101.0, price_low=100.0, strength=1.0,
        source_order_block=_order_block(candle_index, TrendDirection.BULLISH),
    )


def _supply_zone(candle_index: int = 0) -> SupplyZone:
    return SupplyZone(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=101.0, price_low=100.0, strength=1.0,
        source_order_block=_order_block(candle_index, TrendDirection.BEARISH),
    )


def _ote(candle_index: int, direction: TrendDirection) -> OTEZone:
    return OTEZone(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=101.0, price_low=100.0, strength=1.0, direction=direction,
    )


def _premium_zone(candle_index: int = 0) -> PremiumZone:
    return PremiumZone(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=110.0, price_low=105.0, strength=10.0,
    )


def _discount_zone(candle_index: int = 0) -> DiscountZone:
    return DiscountZone(
        timeframe=Timeframe.M1, candle_index=candle_index, timestamp=_ts(candle_index),
        price_high=105.0, price_low=100.0, strength=10.0,
    )


def _smc(
    symbol: str = "XAUUSD", timeframe: Timeframe = Timeframe.M1,
    liquidity_pools=(), liquidity_sweeps=(), equal_highs=(), equal_lows=(),
    fair_value_gaps=(), inverse_fvgs=(), order_blocks=(), breaker_blocks=(),
    mitigation_blocks=(), supply_zones=(), demand_zones=(),
    premium_zone=None, discount_zone=None, ote_zone=None,
) -> SMCAnalysis:
    return SMCAnalysis(
        symbol=symbol, timeframe=timeframe,
        liquidity_pools=tuple(liquidity_pools), liquidity_sweeps=tuple(liquidity_sweeps),
        equal_highs=tuple(equal_highs), equal_lows=tuple(equal_lows),
        fair_value_gaps=tuple(fair_value_gaps), inverse_fvgs=tuple(inverse_fvgs),
        order_blocks=tuple(order_blocks), breaker_blocks=tuple(breaker_blocks),
        mitigation_blocks=tuple(mitigation_blocks),
        supply_zones=tuple(supply_zones), demand_zones=tuple(demand_zones),
        premium_zone=premium_zone, discount_zone=discount_zone, ote_zone=ote_zone,
    )


# --------------------------------------------------------------------------- bullish / bearish confluence

def test_fully_bullish_confluence_gives_high_confidence_buy():
    # ote and premium_discount deliberately excluded here — see the
    # dedicated test below documenting why they can't both agree with the
    # rest when built from the same swing pair.
    swings = (_swing(2, 90.0, SwingType.LOW), _swing(5, 110.0, SwingType.HIGH))
    structure = _structure(
        swings=swings, trend=TrendDirection.BULLISH,
        events=(_event(3, TrendDirection.BULLISH, StructureEventType.BOS),
                _event(4, TrendDirection.BULLISH, StructureEventType.CHOCH)),
    )
    smc = _smc(
        liquidity_sweeps=(_sweep(1, TrendDirection.BULLISH),),
        fair_value_gaps=(_fvg(2, TrendDirection.BULLISH),),
        inverse_fvgs=(_ifvg(3, TrendDirection.BULLISH),),
        order_blocks=(_order_block(2, TrendDirection.BULLISH),),
        demand_zones=(_demand_zone(2),),
    )
    engine = ScoringEngine()
    result = engine.score(structure, smc)

    assert result.direction == ScoreDirection.BUY
    # Eight factors voted, unanimously bullish — but they are three pieces of
    # evidence, each worth its strongest member: structure 20, liquidity 10,
    # imbalance 8. Every group agrees, so this is the whole evidence base.
    assert result.net_score == pytest.approx(38.0)
    assert result.total_possible == pytest.approx(38.0)
    assert result.confidence == 100
    assert len(result.reasons) == 8


def test_fully_bearish_confluence_gives_high_confidence_sell():
    swings = (_swing(2, 110.0, SwingType.HIGH), _swing(5, 90.0, SwingType.LOW))
    structure = _structure(
        swings=swings, trend=TrendDirection.BEARISH,
        events=(_event(3, TrendDirection.BEARISH, StructureEventType.BOS),
                _event(4, TrendDirection.BEARISH, StructureEventType.CHOCH)),
    )
    smc = _smc(
        liquidity_sweeps=(_sweep(1, TrendDirection.BEARISH),),
        fair_value_gaps=(_fvg(2, TrendDirection.BEARISH),),
        inverse_fvgs=(_ifvg(3, TrendDirection.BEARISH),),
        order_blocks=(_order_block(2, TrendDirection.BEARISH),),
        supply_zones=(_supply_zone(2),),
    )
    engine = ScoringEngine()
    result = engine.score(structure, smc)

    assert result.direction == ScoreDirection.SELL
    assert result.net_score == pytest.approx(-38.0)
    assert len(result.reasons) == 8


def test_ote_and_premium_discount_naturally_oppose_when_both_fire():
    # Both are derived from the same latest-high/latest-low ordering and vote
    # opposite ways from it — which is exactly why they carry no direction.
    swings = (_swing(2, 90.0, SwingType.LOW), _swing(5, 110.0, SwingType.HIGH))
    structure = _structure(swings=swings)
    smc = _smc(ote_zone=_ote(5, TrendDirection.BULLISH),
               premium_zone=_premium_zone(), discount_zone=_discount_zone())
    result = ScoringEngine().score(structure, smc)

    ote_factor = next(f for f in result.breakdown if f.name == "ote")
    pd_factor = next(f for f in result.breakdown if f.name == "premium_discount")
    assert ote_factor.direction == TrendDirection.BULLISH
    assert pd_factor.direction == TrendDirection.BEARISH
    # The votes are still recorded, and they are exact negations...
    assert ote_factor.raw_contribution == -pd_factor.raw_contribution
    # ...so they no longer count: no direction, and no fake participation.
    assert ote_factor.contribution == pd_factor.contribution == 0.0
    assert result.net_score == 0.0
    assert result.confidence == 0


# --------------------------------------------------------------------------- neutral / conflicting

def test_all_neutral_input_gives_none_direction_and_zero_confidence():
    result = ScoringEngine().score(_structure(), _smc())

    assert result.direction == ScoreDirection.NONE
    assert result.confidence == 0
    assert result.net_score == 0
    assert result.reasons == ()
    assert len(result.breakdown) == 10
    assert all(f.contribution == 0.0 for f in result.breakdown)


def test_conflicting_evidence_reduces_confidence():
    """Conflict costs confidence — but the winning side still wins.

    All three factors are in the market-structure group, so they are scaled
    together (by 20/68) and keep their relative weights: trend +20 against
    BOS -15 and order block -10 is still a narrow bearish verdict.
    """
    structure = _structure(
        trend=TrendDirection.BULLISH,
        events=(_event(3, TrendDirection.BEARISH, StructureEventType.BOS),),
    )
    smc = _smc(order_blocks=(_order_block(2, TrendDirection.BEARISH),))
    result = ScoringEngine().score(structure, smc)

    k = 20.0 / 68.0
    assert result.net_score == pytest.approx((20.0 - 15.0 - 10.0) * k)   # leans bearish
    # One group, contradicting itself, with nothing else behind it: the lean is
    # kept in net_score but is too thin to report as a direction.
    assert result.direction == ScoreDirection.NONE
    assert result.confidence == _confidence(20 * k, 25 * k, 38.0)
    assert _confidence(20 * k, 25 * k, 38.0) < _confidence(45 * k, 0, 38.0)


def test_a_lone_weak_factor_never_produces_a_direction():
    """One factor firing into silence is the participation floor's job.

    The margin term cannot help here — nothing opposes the factor, so the
    vote is perfectly one-sided. Only the evidence base disqualifies it.
    """
    smc = _smc(fair_value_gaps=(_fvg(1, TrendDirection.BULLISH),))   # 8, halved in its group
    result = ScoringEngine().score(_structure(), smc)

    assert result.net_score == pytest.approx(4.0)   # the vote is still cast and reported
    assert result.confidence == 0
    assert result.direction == ScoreDirection.NONE


def test_ote_alone_is_location_and_casts_no_direction_at_all():
    """The example from the original requirement: OTE alone must never
    produce BUY or SELL. It now cannot even vote — it is location, not
    direction."""
    result = ScoringEngine().score(_structure(), _smc(ote_zone=_ote(1, TrendDirection.BULLISH)))
    ote = next(f for f in result.breakdown if f.name == "ote")

    assert ote.raw_contribution == 6.0     # the factor's own reading is kept
    assert result.net_score == 0.0
    assert result.direction == ScoreDirection.NONE


@pytest.mark.parametrize("build", [
    lambda: (_structure(trend=TrendDirection.BULLISH), _smc()),
    lambda: (_structure(events=(_event(3, TrendDirection.BULLISH),)), _smc()),
    lambda: (_structure(events=(_event(3, TrendDirection.BULLISH, StructureEventType.CHOCH),)), _smc()),
    lambda: (_structure(), _smc(order_blocks=(_order_block(2, TrendDirection.BULLISH),))),
    lambda: (_structure(), _smc(demand_zones=(_demand_zone(2),))),
    lambda: (_structure(), _smc(fair_value_gaps=(_fvg(2, TrendDirection.BULLISH),))),
    lambda: (_structure(), _smc(inverse_fvgs=(_ifvg(2, TrendDirection.BULLISH),))),
    lambda: (_structure(), _smc(liquidity_sweeps=(_sweep(2, TrendDirection.BULLISH),))),
], ids=["trend", "bos", "choch", "order_block", "supply_demand", "fvg", "inverse_fvg",
        "liquidity_sweep"])
def test_no_lone_factor_can_reach_the_enter_threshold(build):
    """A single factor firing into silence is thin evidence, whatever it is.

    Every factor that shares a group with others stays under the direction
    gate. The liquidity sweep is the one factor that is a whole group on its
    own: with redundant weight no longer padding the denominator it is a
    quarter of the independent evidence, so it registers a weak lean (17) —
    still far below the ENTER threshold, so it can never put a trade on its own.
    """
    from trade_decision import ENTER_MIN_SCORE
    structure, smc = build()
    result = ScoringEngine().score(structure, smc)
    assert result.confidence < 40 < ENTER_MIN_SCORE
    sole_member_of_its_group = {"liquidity_sweep"}
    voted = [f.name for f in result.breakdown if f.contribution]
    if not set(voted) & sole_member_of_its_group:
        assert result.direction == ScoreDirection.NONE, voted


def test_min_confidence_threshold_still_suppresses_direction():
    """The rule-12 gate is live: reachable states sit below it."""
    engine = ScoringEngine()
    below = [(b, s) for b in range(107) for s in range(107 - b)
             if 0 < _confidence(b, s) < engine._min_confidence_for_direction]
    assert below, "nothing lands under the threshold — the gate would be dead"


def test_lowering_min_confidence_threshold_allows_weak_direction_through():
    smc = _smc(fair_value_gaps=(_fvg(1, TrendDirection.BULLISH),))
    result = ScoringEngine(min_confidence_for_direction=0.0).score(_structure(), smc)
    assert result.direction == ScoreDirection.BUY


# --------------------------------------------------------------------------- configurability

def test_custom_weights_change_relative_influence():
    structure = _structure(trend=TrendDirection.BEARISH)
    smc = _smc(liquidity_sweeps=(_sweep(1, TrendDirection.BULLISH),))
    engine = ScoringEngine(trend_weight=5.0, liquidity_sweep_weight=50.0,
                           min_confidence_for_direction=0.0)
    result = engine.score(structure, smc)

    # trend 5 is the weakest member of a group whose strongest is 15.
    assert result.net_score == pytest.approx(50.0 - 5.0 * 15.0 / 53.0)
    assert result.direction == ScoreDirection.BUY


def test_zero_weight_factor_never_influences_score():
    structure = _structure(trend=TrendDirection.BULLISH)
    engine = ScoringEngine(trend_weight=0.0)
    result = engine.score(structure, _smc())

    trend_factor = next(f for f in result.breakdown if f.name == "trend")
    assert trend_factor.weight == 0.0
    assert trend_factor.contribution == 0.0


def test_negative_weight_raises():
    try:
        ScoringEngine(trend_weight=-1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative weight")


def test_min_confidence_out_of_range_raises():
    try:
        ScoringEngine(min_confidence_for_direction=101.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range min_confidence_for_direction")


# --------------------------------------------------------------------------- input validation

def test_mismatched_symbol_raises():
    structure = _structure(symbol="XAUUSD")
    smc = _smc(symbol="EURUSD")
    try:
        ScoringEngine().score(structure, smc)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched symbol")


def test_mismatched_timeframe_raises():
    structure = _structure(timeframe=Timeframe.M1)
    smc = _smc(timeframe=Timeframe.H1)
    try:
        ScoringEngine().score(structure, smc)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched timeframe")


# --------------------------------------------------------------------------- breakdown / reasons shape

def test_breakdown_always_has_all_ten_factors_in_fixed_order():
    result = ScoringEngine().score(_structure(), _smc())
    names = [f.name for f in result.breakdown]

    assert names == [
        "trend", "bos", "choch", "liquidity_sweep", "fvg", "inverse_fvg",
        "order_block", "supply_demand", "ote", "premium_discount",
    ]


def test_breakdown_weight_matches_configured_weight():
    engine = ScoringEngine(trend_weight=42.0)
    result = engine.score(_structure(), _smc())
    trend_factor = next(f for f in result.breakdown if f.name == "trend")
    assert trend_factor.raw_weight == 42.0
    # ...and what it counted for: the strongest member of its group.
    assert trend_factor.weight == pytest.approx(42.0 * 42.0 / (42 + 15 + 15 + 10 + 8))


def test_reasons_only_include_factors_that_fired():
    structure = _structure(trend=TrendDirection.BULLISH)
    result = ScoringEngine().score(structure, _smc())

    assert len(result.reasons) == 1
    assert "bullish" in result.reasons[0].lower()


# --------------------------------------------------------------------------- individual factor rules

def test_supply_demand_more_demand_votes_bullish():
    smc = _smc(demand_zones=(_demand_zone(1), _demand_zone(2)), supply_zones=(_supply_zone(3),))
    factor = ScoringEngine()._score_supply_demand(_structure(), smc)
    assert factor.direction == TrendDirection.BULLISH
    assert factor.contribution == factor.weight


def test_supply_demand_more_supply_votes_bearish():
    smc = _smc(supply_zones=(_supply_zone(1), _supply_zone(2)), demand_zones=(_demand_zone(3),))
    factor = ScoringEngine()._score_supply_demand(_structure(), smc)
    assert factor.direction == TrendDirection.BEARISH
    assert factor.contribution == -factor.weight


def test_supply_demand_equal_counts_votes_neutral():
    smc = _smc(demand_zones=(_demand_zone(1),), supply_zones=(_supply_zone(3),))
    factor = ScoringEngine()._score_supply_demand(_structure(), smc)
    assert factor.direction is None
    assert factor.contribution == 0.0


def test_premium_discount_recent_low_votes_bullish():
    swings = (_swing(5, 90.0, SwingType.LOW), _swing(2, 110.0, SwingType.HIGH))
    structure = _structure(swings=swings)
    smc = _smc(premium_zone=_premium_zone(), discount_zone=_discount_zone())
    factor = ScoringEngine()._score_premium_discount(structure, smc)
    assert factor.direction == TrendDirection.BULLISH


def test_premium_discount_recent_high_votes_bearish():
    swings = (_swing(2, 90.0, SwingType.LOW), _swing(5, 110.0, SwingType.HIGH))
    structure = _structure(swings=swings)
    smc = _smc(premium_zone=_premium_zone(), discount_zone=_discount_zone())
    factor = ScoringEngine()._score_premium_discount(structure, smc)
    assert factor.direction == TrendDirection.BEARISH


def test_premium_discount_votes_neutral_without_zones():
    factor = ScoringEngine()._score_premium_discount(_structure(), _smc())
    assert factor.direction is None
    assert factor.contribution == 0.0


def test_bos_and_choch_scored_independently():
    structure = _structure(events=(
        _event(2, TrendDirection.BULLISH, StructureEventType.BOS),
        _event(4, TrendDirection.BEARISH, StructureEventType.CHOCH),
    ))
    engine = ScoringEngine()
    bos_factor = engine._score_bos(structure, _smc())
    choch_factor = engine._score_choch(structure, _smc())

    assert bos_factor.direction == TrendDirection.BULLISH
    assert choch_factor.direction == TrendDirection.BEARISH


def test_only_latest_bos_counts_when_multiple_exist():
    structure = _structure(events=(
        _event(2, TrendDirection.BULLISH, StructureEventType.BOS),
        _event(6, TrendDirection.BEARISH, StructureEventType.BOS),
    ))
    factor = ScoringEngine()._score_bos(structure, _smc())
    assert factor.direction == TrendDirection.BEARISH


# --------------------------------------------------------------------------- determinism

def test_scoring_is_deterministic():
    structure = _structure(trend=TrendDirection.BULLISH, events=(_event(2, TrendDirection.BULLISH),))
    smc = _smc(order_blocks=(_order_block(1, TrendDirection.BULLISH),))
    engine = ScoringEngine()

    assert engine.score(structure, smc) == engine.score(structure, smc)


# --------------------------------------------------------------------------- confidence curve

def _confidence(bullish: float, bearish: float, total_possible: float = 106.0) -> int:
    """Drive the confidence curve directly from a bullish/bearish split.

    The curve reads only the signed contributions, so a two-element
    breakdown carrying those sums exercises exactly the real code path.
    """
    breakdown = (
        FactorScore("bull", bullish, bullish, None, ""),
        FactorScore("bear", bearish, -bearish, None, ""),
    )
    return ScoringEngine._confidence(breakdown, total_possible)


def test_confidence_meets_every_specified_requirement():
    """Every constraint the curve was designed against, in one place.

    One point moved. 55-vs-51 used to score 60 because the 51 opposing
    points counted as participation — the same mechanism that let opposition
    RAISE confidence (see the regression test below). Measured on the
    winning side's support alone it scores 51: contested but decided, mid-scale.
    """
    assert _confidence(53, 53) == 0             # dead tie
    assert 40 <= _confidence(55, 51) < 60       # contested but decided: mid-scale
    assert _confidence(80, 20) >= 88            # dominant ~90
    assert _confidence(106, 0) == 100           # unanimous, full slate
    assert _confidence(6, 0) < 10               # lone weak factor: no direction
    assert _confidence(53, 52) <= 30            # tiny margin stays low


def test_the_curve_has_no_step_anywhere():
    """The point of the smooth curves: no threshold to cross, so confidence
    never leaps because one point changed. The previous floor-based version
    jumped 0 -> 55 between adjacent margins; nothing here may come close."""
    jumps = [(_confidence(w + 1, loser) - _confidence(w, loser), w, loser)
             for loser in range(0, 54) for w in range(loser, 106 - loser)]
    worst, w, loser = max(jumps)
    assert worst <= 30, f"confidence leapt {worst} between {w}v{loser} and {w+1}v{loser}"


def test_confidence_rises_gradually_off_a_tie():
    """Each extra point of margin adds something, and the increments shrink
    rather than arriving all at once."""
    ramp = [_confidence((106 + m) / 2, (106 - m) / 2) for m in range(0, 9)]
    assert ramp[0] == 0
    assert ramp == sorted(ramp)
    steps = [b - a for a, b in zip(ramp, ramp[1:])]
    assert all(s > 0 for s in steps), f"a point of margin bought nothing: {steps}"
    # past the initial rise the increments must be decreasing
    assert steps[2:] == sorted(steps[2:], reverse=True), steps


def test_a_thin_evidence_base_earns_almost_nothing():
    assert _confidence(0, 0) == 0
    assert _confidence(20, 0) < 10              # one factor, whatever its weight
    assert _confidence(20, 0) < _confidence(30, 0) < _confidence(45, 0)


def test_confidence_is_symmetric_between_the_two_sides():
    """Nothing about the curve prefers longs to shorts."""
    for bull, bear in ((55, 51), (80, 20), (106, 0), (6, 0), (70, 36), (53, 52)):
        assert _confidence(bull, bear) == _confidence(bear, bull)


def test_confidence_never_falls_as_the_winning_margin_grows():
    for loser in (0, 10, 20, 40):
        scores = [_confidence(w, loser) for w in range(loser, 107 - loser)]
        assert scores == sorted(scores), f"non-monotone against loser={loser}"


def test_participation_separates_unanimous_from_merely_uncontested():
    """The margin term scores all of these identically (nothing opposes
    them); only the evidence base tells them apart."""
    assert _confidence(20, 0) < _confidence(45, 0) < _confidence(75, 0) < _confidence(106, 0)
    assert _confidence(106, 0) == 100


def test_confluence_beats_a_thin_unanimous_verdict():
    """The whole point of the change: a full slate arguing to a decision
    outranks a couple of factors speaking into silence."""
    assert _confidence(70, 36) > _confidence(35, 0)
    assert _confidence(55, 51) > _confidence(30, 0)


def test_confidence_stays_within_bounds_across_every_reachable_split():
    weights = list(ScoringEngine()._weights.values())
    total = float(sum(weights))
    for bullish in range(0, int(total) + 1):
        for bearish in range(0, int(total) - bullish + 1, 7):
            assert 0 <= _confidence(bullish, bearish, total) <= 100


def test_confidence_is_dominance_ordered():
    """A setup with a bigger winning side AND a smaller losing side must not
    score lower. Violating this is how an earlier curve had 70-vs-36 (99)
    outranking 80-vs-20 (91)."""
    ladder = [(106, 0), (94, 0), (85, 12), (80, 20), (70, 36), (60, 46), (55, 51),
              (53, 52), (53, 53)]
    scores = [_confidence(b, s) for b, s in ladder]
    assert scores == sorted(scores, reverse=True), list(zip(ladder, scores))


def test_confidence_does_not_rise_when_the_losing_side_grows():
    """Opposing evidence may only take confidence away — strictly, with no
    tolerance. The old test allowed +3 per point of opposition, which hid a
    26-point rise across ten points."""
    for total in (106.0, 38.0):
        for winner in range(1, int(total) + 1):
            prev = None
            for loser in range(0, min(winner, int(total) - winner) + 1):
                v = _confidence(winner, loser, total)
                if prev is not None:
                    assert v <= prev, f"{winner}v{loser}/{total} rose to {v} from {prev}"
                prev = v


def test_opposition_can_never_buy_confidence():
    """The regression, by its numbers: 30 points bullish alone scored 23, and
    adding 10 points BEARISH lifted it to 49 — contradicting evidence made the
    trade look twice as good. Participation counted the opposing votes."""
    alone = _confidence(30, 0)
    opposed = _confidence(30, 10)
    assert opposed < alone, (alone, opposed)


# --------------------------------------------------------------------------- evidence groups

def _structure_only_bullish():
    """Every market-structure factor bullish, nothing else voting: one strong
    trend, seen five ways."""
    structure = _structure(
        trend=TrendDirection.BULLISH,
        events=(_event(3, TrendDirection.BULLISH, StructureEventType.BOS),
                _event(4, TrendDirection.BULLISH, StructureEventType.CHOCH)),
    )
    smc = _smc(order_blocks=(_order_block(2, TrendDirection.BULLISH),),
               demand_zones=(_demand_zone(2),))
    return structure, smc


def test_a_strong_trend_alone_does_not_reach_100():
    """Trend, BOS, CHoCH, order block and supply/demand all agreeing is ONE
    directional state. Ungrouped it scored 95-96 on its own; grouped it is a
    single piece of evidence and cannot reach the top of the scale."""
    result = ScoringEngine().score(*_structure_only_bullish())
    legacy = ScoringEngine(groups=()).score(*_structure_only_bullish())

    assert result.direction == ScoreDirection.BUY
    assert result.confidence < 95
    assert result.confidence < legacy.confidence


def test_redundant_factors_count_once():
    """Five agreeing structure factors are worth exactly their strongest
    member — the trend's 20 — not the 68 they sum to."""
    result = ScoringEngine().score(*_structure_only_bullish())
    assert result.group_scores()["market_structure"] == pytest.approx(20.0)
    assert result.net_score == pytest.approx(20.0)


def test_each_group_is_worth_its_strongest_member():
    from evidence_groups import SCORING_GROUPS
    engine = ScoringEngine()
    for group in SCORING_GROUPS:
        effective = sum(engine._weights[m] * engine._scale[m] for m in group.members)
        expected = max(engine._weights[m] for m in group.members) if group.in_direction_score else 0
        assert effective == pytest.approx(expected), group.name


def test_every_factor_belongs_to_exactly_one_group():
    from evidence_groups import SCORING_GROUPS
    names = [m for g in SCORING_GROUPS for m in g.members]
    assert sorted(names) == sorted(ScoringEngine()._weights)
    assert all(g.basis for g in SCORING_GROUPS)        # every grouping is justified


def test_grouping_never_changes_the_sign_inside_a_group():
    """Members are scaled together, so a group's internal disagreement is
    preserved in proportion rather than resolved."""
    structure = _structure(trend=TrendDirection.BULLISH,
                           events=(_event(3, TrendDirection.BEARISH),))
    result = ScoringEngine().score(structure, _smc())
    trend = next(f for f in result.breakdown if f.name == "trend")
    bos = next(f for f in result.breakdown if f.name == "bos")
    assert trend.contribution > 0 > bos.contribution
    assert trend.contribution / -bos.contribution == pytest.approx(20.0 / 15.0)


def test_the_ungrouped_engine_is_still_available():
    """Backward compatibility: ``groups=()`` is the original ten independent
    votes over 106 points."""
    result = ScoringEngine(groups=()).score(*_structure_only_bullish())
    assert result.total_possible == 106.0
    assert result.net_score == 68.0


def test_both_curves_are_pinned_at_their_endpoints():
    """The exact 0 and exact 100 are properties of the curves, not clipping."""
    import scoring_engine as se
    assert se._margin_curve(0.0) == 0.0
    assert abs(se._margin_curve(1.0) - 1.0) < 1e-12
    assert abs(se._participation_curve(0.0)) < 1e-12
    assert abs(se._participation_curve(1.0) - 1.0) < 1e-12


def test_both_curves_are_continuous():
    """No jump anywhere: sampling finely, no adjacent pair differs by more
    than a small amount. A floor or threshold would show up here."""
    import scoring_engine as se
    for curve in (se._margin_curve, se._participation_curve):
        xs = [i / 4000 for i in range(4001)]
        vals = [curve(x) for x in xs]
        biggest = max(b - a for a, b in zip(vals, vals[1:]))
        assert biggest < 0.01, f"{curve.__name__} jumped {biggest}"
        assert vals == sorted(vals), f"{curve.__name__} is not monotone"
