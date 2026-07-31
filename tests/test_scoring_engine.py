"""Unit tests for scoring_engine.py (RFC-007).

Entirely offline and deterministic: every test hand-builds a
StructureAnalysis and/or SMCAnalysis directly (constructing the RFC-005/
RFC-006 dataclasses with whatever fields the scoring rule under test
actually reads — no need to run the real detectors) and asserts on the
exact ScoringResult returned.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    expected_net = sum(w for name, w in engine._weights.items() if name not in ("ote", "premium_discount"))
    assert result.net_score == expected_net
    # Eight factors voted, unanimously bullish: the margin term is maxed out,
    # so confidence is set by the evidence base (94 of 106 possible weight).
    assert result.confidence == _confidence(expected_net, 0, result.total_possible)
    assert result.confidence >= 90
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
    expected_net = -sum(w for name, w in engine._weights.items() if name not in ("ote", "premium_discount"))
    assert result.net_score == expected_net
    assert len(result.reasons) == 8


def test_ote_and_premium_discount_naturally_oppose_when_both_fire():
    # Both are derived from the same latest-high/latest-low ordering but
    # represent opposite perspectives (trend-continuation vs. mean-
    # reversion-at-extremes) — see the scoring_engine module docstring.
    swings = (_swing(2, 90.0, SwingType.LOW), _swing(5, 110.0, SwingType.HIGH))
    structure = _structure(swings=swings)
    smc = _smc(ote_zone=_ote(5, TrendDirection.BULLISH),
               premium_zone=_premium_zone(), discount_zone=_discount_zone())
    result = ScoringEngine().score(structure, smc)

    ote_factor = next(f for f in result.breakdown if f.name == "ote")
    pd_factor = next(f for f in result.breakdown if f.name == "premium_discount")
    assert ote_factor.direction == TrendDirection.BULLISH
    assert pd_factor.direction == TrendDirection.BEARISH
    assert result.net_score == ote_factor.contribution + pd_factor.contribution


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

    The comparison holds participation fixed (45 points cast either way)
    so it isolates the margin term: 20-vs-25 must score below 45-vs-0.
    """
    structure = _structure(
        trend=TrendDirection.BULLISH,
        events=(_event(3, TrendDirection.BEARISH, StructureEventType.BOS),),
    )
    smc = _smc(order_blocks=(_order_block(2, TrendDirection.BEARISH),))
    result = ScoringEngine().score(structure, smc)

    assert result.net_score == 20.0 - 15.0 - 10.0  # trend(+20) vs bos(-15) vs order_block(-10)
    assert result.direction == ScoreDirection.SELL
    assert result.confidence == _confidence(20, 25)
    assert _confidence(20, 25) < _confidence(45, 0)


def test_a_lone_weak_factor_never_produces_a_direction():
    """One factor firing into silence is the participation floor's job.

    The margin term cannot help here — nothing opposes the factor, so the
    vote is perfectly one-sided. Only the evidence base disqualifies it.
    """
    smc = _smc(ote_zone=_ote(1, TrendDirection.BULLISH))   # weight 6 of 106
    result = ScoringEngine().score(_structure(), smc)

    assert result.net_score == 6.0        # the vote is still cast and reported
    assert result.confidence == 0
    assert result.direction == ScoreDirection.NONE


def test_no_single_factor_can_ever_produce_a_direction():
    """Not just the weak ones. A lone factor is perfectly one-sided, so the
    margin curve scores it at the maximum; only the participation curve's
    flat start holds it down, and it must do so for every weight."""
    engine = ScoringEngine()
    gate = engine._min_confidence_for_direction
    for weight in sorted(set(engine._weights.values())):
        assert _confidence(weight, 0) < gate, f"lone factor of weight {weight} scored"
        assert _confidence(0, weight) < gate


def test_min_confidence_threshold_still_suppresses_direction():
    """The rule-12 gate is live: reachable states sit below it."""
    engine = ScoringEngine()
    below = [(b, s) for b in range(107) for s in range(107 - b)
             if 0 < _confidence(b, s) < engine._min_confidence_for_direction]
    assert below, "nothing lands under the threshold — the gate would be dead"


def test_lowering_min_confidence_threshold_allows_weak_direction_through():
    smc = _smc(ote_zone=_ote(1, TrendDirection.BULLISH))
    result = ScoringEngine(min_confidence_for_direction=0.0).score(_structure(), smc)
    assert result.direction == ScoreDirection.BUY


# --------------------------------------------------------------------------- configurability

def test_custom_weights_change_relative_influence():
    structure = _structure(trend=TrendDirection.BEARISH)
    smc = _smc(ote_zone=_ote(1, TrendDirection.BULLISH))
    engine = ScoringEngine(trend_weight=5.0, ote_weight=50.0, min_confidence_for_direction=0.0)
    result = engine.score(structure, smc)

    assert result.net_score == 50.0 - 5.0
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
    assert trend_factor.weight == 42.0


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
    """Every constraint the curve was designed against, in one place."""
    assert _confidence(53, 53) == 0             # dead tie
    assert 55 <= _confidence(55, 51) <= 65      # contested but decided ~60
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
    """Adding opposing evidence raises participation but cuts the margin; the
    margin must win, or a losing vote would buy confidence."""
    for winner in range(20, 107, 7):
        prev = None
        for loser in range(0, min(winner, 106 - winner) + 1):
            v = _confidence(winner, loser)
            if prev is not None:
                assert v <= prev + 3, f"{winner}v{loser} rose to {v} from {prev}"
            prev = v


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
