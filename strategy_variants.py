"""Candidate changes to the production verdict — predeclared, evaluated, not live.

The production verdict (``trade_decision.decide_deterministically``) is,
in practice, one question: does the primary-timeframe (H1) scoring vote
agree with the signal's direction? (On the 109 recorded signals it decided
104 verdicts; the validation adjustments flipped 5.) The vote sums ten
"most recent object" factors, none of which checks whether its event still
holds at the current price or how old it is, so in a grinding decline stale
bullish breaks keep voting long. ``research/reports/PS_root_cause.md``
has the evidence.

Each candidate below changes ONE thing and is evaluated separately against
the unchanged baseline (``research/strategy_backtest.py``). Nothing here is
called by the live bot; the baseline stays the live rule until a change is
approved.

``clean_data``
    Data only: closed bars, weekend stubs removed, full 300-bar windows
    (``MarketContextBuilder(clean_candles=True)``). Decision unchanged.
``htf_regime``
    The direction the score must agree with comes from the H4 and H1 regime
    instead of the H1 vote: each timeframe's bias is the direction of its
    last BOS/CHoCH when the close is on that side of EMA-50, else neutral;
    the regime is a direction only when H4 and H1 agree. No regime, no
    favoured direction: every signal scores 0 and is SKIP.
``valid_events``
    The H1 vote, with stale or invalidated evidence removed, in both
    directions alike: the latest BOS and the latest CHoCH vote only while
    price is still beyond the level they broke; the latest order block
    only while price has not crossed its zone; supply/demand counts only
    zones formed in the last 100 bars. Everything else unchanged.
``range_filter``
    The baseline verdict, except that a market with no H1 structure trend
    (ranging/unknown) and an H1 ADX below 20 is SKIP.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from market_data import Timeframe
from scoring_engine import FactorScore, ScoringEngine
from structure_engine import StructureEngine, StructureEventType, TrendDirection
from trade_decision import Action, DecisionTrace, decide_deterministically
from validation_engine import ValidationEngine

VARIANTS = ("baseline", "clean_data", "htf_regime", "valid_events", "range_filter")

ZONE_LOOKBACK_BARS = 100
RANGE_ADX = 20.0


# ---------------------------------------------------------------- htf_regime

def timeframe_bias(context, tf: Timeframe) -> str:
    """bullish / bearish / neutral: last BOS-or-CHoCH direction, kept only
    when the last close is on that side of the timeframe's EMA-50."""
    analysis = context.timeframes.get(tf.value)
    series = getattr(context, "candle_series", {}).get(tf)
    if analysis is None or series is None or analysis.ema.ema_50 is None or analysis.last_close is None:
        return "neutral"
    structure = (context.structure if tf is context.primary_timeframe and context.structure is not None
                 else StructureEngine().analyze(series))
    last = structure.last_event
    if last is None:
        return "neutral"
    if last.direction is TrendDirection.BULLISH and analysis.last_close > analysis.ema.ema_50:
        return "bullish"
    if last.direction is TrendDirection.BEARISH and analysis.last_close < analysis.ema.ema_50:
        return "bearish"
    return "neutral"


def regime(context) -> Optional[str]:
    """"long" / "short" when H4 and H1 bias agree, else None."""
    h4, h1 = timeframe_bias(context, Timeframe.H4), timeframe_bias(context, Timeframe.H1)
    if h4 == h1 == "bullish":
        return "long"
    if h4 == h1 == "bearish":
        return "short"
    return None


def _with_direction(scoring, direction: Optional[str]):
    """The scoring result re-pointed at ``direction`` (None: no direction,
    no confidence), its confidence value otherwise untouched."""
    if direction is None:
        return replace(scoring, net_score=1.0, confidence=0)
    magnitude = max(1.0, abs(scoring.net_score))
    return replace(scoring, net_score=magnitude if direction == "long" else -magnitude)


# -------------------------------------------------------------- valid_events

class ValidEventScoringEngine(ScoringEngine):
    """ScoringEngine whose structure votes must still hold at ``price``."""

    def __init__(self, price: float, bars: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._price, self._bars = price, bars

    def _event_vote(self, name, structure, kind) -> FactorScore:
        weight = self._weights[name]
        events = [e for e in structure.events if e.type == kind]
        if not events:
            return FactorScore(name, weight, 0.0, None, f"No {name.upper()} event detected")
        latest = max(events, key=lambda e: e.candle_index)
        level = latest.broken_swing.price
        holds = self._price > level if latest.direction == TrendDirection.BULLISH else self._price < level
        if not holds:
            return FactorScore(name, weight, 0.0, None,
                               f"Latest {name.upper()} ({latest.direction.value}) no longer holds: "
                               f"price is back through {level:g}")
        sign = 1.0 if latest.direction == TrendDirection.BULLISH else -1.0
        return FactorScore(name, weight, sign * weight, latest.direction,
                           f"Latest {name.upper()} is {latest.direction.value} and still holds")

    def _score_bos(self, structure, smc):
        return self._event_vote("bos", structure, StructureEventType.BOS)

    def _score_choch(self, structure, smc):
        return self._event_vote("choch", structure, StructureEventType.CHOCH)

    def _score_order_block(self, structure, smc):
        weight = self._weights["order_block"]
        if not smc.order_blocks:
            return FactorScore("order_block", weight, 0.0, None, "No order block detected")
        latest = max(smc.order_blocks, key=lambda o: o.candle_index)
        bullish = latest.direction == TrendDirection.BULLISH
        holds = self._price > latest.price_low if bullish else self._price < latest.price_high
        if not holds:
            return FactorScore("order_block", weight, 0.0, None,
                               "Latest order block has been crossed by price")
        return FactorScore("order_block", weight, weight if bullish else -weight, latest.direction,
                           f"Latest order block is {latest.direction.value} and intact")

    def _score_supply_demand(self, structure, smc):
        weight = self._weights["supply_demand"]
        recent = lambda zones: [z for z in zones  # noqa: E731
                                if z.candle_index >= self._bars - ZONE_LOOKBACK_BARS]
        demand, supply = len(recent(smc.demand_zones)), len(recent(smc.supply_zones))
        if demand == supply:
            return FactorScore("supply_demand", weight, 0.0, None,
                               f"Equal recent demand ({demand}) and supply ({supply}) zones")
        if demand > supply:
            return FactorScore("supply_demand", weight, weight, TrendDirection.BULLISH,
                               f"More recent demand ({demand}) than supply ({supply}) zones")
        return FactorScore("supply_demand", weight, -weight, TrendDirection.BEARISH,
                           f"More recent supply ({supply}) than demand ({demand}) zones")


def rescored(context):
    """The context with its H1 score recomputed by the valid-events rules and
    its review re-run on that score (the review reads the score)."""
    if context.structure is None or context.smc is None or context.current_price is None:
        return context
    scoring = ValidEventScoringEngine(context.current_price, len(context.primary_candles)).score(
        context.structure, context.smc)
    updated = replace(context, scoring=scoring)
    direction = getattr(context.setup, "direction", None)
    updated.validation = ValidationEngine().validate(updated, direction=direction)
    return updated


# ------------------------------------------------------------------ decide

def decide(context, variant: str) -> DecisionTrace:
    """The verdict under ``variant``. ``clean_data`` is a data change: pass a
    context built with ``clean_candles=True``; the decision is the baseline's."""
    if variant in ("baseline", "clean_data"):
        return decide_deterministically(context)
    if variant == "htf_regime":
        if context.scoring is None:
            return decide_deterministically(context)
        return decide_deterministically(replace(context, scoring=_with_direction(context.scoring,
                                                                                 regime(context))))
    if variant == "valid_events":
        return decide_deterministically(rescored(context))
    if variant == "range_filter":
        trace = decide_deterministically(context)
        h1 = context.timeframes.get(Timeframe.H1.value)
        if (trace.verdict is not Action.SKIP and h1 is not None and h1.trend in ("ranging", "unknown")
                and h1.adx.adx is not None and h1.adx.adx < RANGE_ADX):
            return replace(trace, verdict=Action.SKIP,
                           reason=f"ranging market: H1 structure {h1.trend}, ADX {h1.adx.adx:.0f} < {RANGE_ADX:g}")
        return trace
    raise ValueError(f"unknown variant {variant!r}")
