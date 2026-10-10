"""Which pieces of evidence are really the same piece of evidence.

The defect this exists to fix: several scoring factors are computed from the
same underlying market state, so one strong trend arrived at the decision as
five or six "independent" confirmations. Trend, BOS, CHoCH, the order block
and supply/demand are all read off the same H1 swing points; a fast trend
lights every one of them at once, and the vote looked like overwhelming
confluence exactly when the move was already extended.

ONE rule, applied everywhere, with no new weights invented:

    A group of redundant factors is worth its strongest member's existing
    weight.

Members keep their relative weights inside the group — each is multiplied by
``strongest / sum(members)`` — so the group can reach exactly the weight of
its strongest member, members still disagree with each other in proportion,
and nothing else about the configured weights changes.

Membership is decided by the CODE, not by outcomes. Each group records the
derivation that makes its members one piece of evidence (``basis``), so the
claim can be checked by reading the engines rather than taken on trust.
Factors that merely tend to agree, but are computed from different objects,
are NOT grouped.

The categories follow the redesign's evidence taxonomy:

    A. DIRECTION          B. MARKET STRUCTURE     C. LOCATION
    D. LIQUIDITY / IMBALANCE                      E. RISK / REWARD
    F. REGIME

On the H1 frame A and B are one measured quantity — the code derives the
trend from the same swings whose breaks are the structure events — so they
form one group. C is not direction evidence at all and is measured with the
live price by ``entry_quality``. E is a hard gate in ``risk_engine`` and
``trade_geometry``, never a vote.

Deliberately free of project imports so both engines can use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EvidenceGroup:
    """A set of factors that measure one underlying market condition."""

    name: str
    category: str                  # A..F label from the evidence taxonomy
    members: Tuple[str, ...]
    basis: str                     # the code-level reason they are one piece
    # False: not direction evidence at all. Members are kept in the
    # breakdown for transparency but carry no weight in the direction score.
    in_direction_score: bool = True
    # False: the group re-measures a state the base score already counts at
    # full weight. Its agreement therefore adds nothing — that would count the
    # same state twice — but its dissent still costs, because a member that
    # contradicts the state is new information. Only meaningful for the
    # validation review, which adjusts the base rather than forming it.
    adds_conviction: bool = True


# --------------------------------------------------------------- scoring engine

SCORING_GROUPS: Tuple[EvidenceGroup, ...] = (
    EvidenceGroup(
        name="market_structure",
        category="A. Direction + B. Market structure",
        members=("trend", "bos", "choch", "order_block", "supply_demand"),
        basis=(
            "structure_engine: the trend is the HH/HL (or LH/LL) pattern of the "
            "swing points; every BOS and CHoCH is a close beyond one of those "
            "same swings, classified by the trend of the swings confirmed so "
            "far. smc_engine: every order block is built from a BOS/CHoCH "
            "event and inherits its direction (direction=event.direction); "
            "supply and demand zones are exactly the unbroken order blocks."
        ),
    ),
    EvidenceGroup(
        name="liquidity",
        category="D. Liquidity",
        members=("liquidity_sweep",),
        basis=(
            "a wick through a swing that closes back inside it — the opposite "
            "outcome to a BOS at the same level, so it is not derived from the "
            "structure events and stands on its own"
        ),
    ),
    EvidenceGroup(
        name="imbalance",
        category="D. Imbalance",
        members=("fvg", "inverse_fvg"),
        basis=(
            "smc_engine: every inverse FVG is a fair value gap that price later "
            "closed through (source_fvg); both are read from the same gap list"
        ),
    ),
    EvidenceGroup(
        name="location",
        category="C. Location",
        members=("ote", "premium_discount"),
        basis=(
            "both read nothing but the ORDER of the latest swing high and low. "
            "OTE votes bullish exactly when premium/discount votes bearish, and "
            "the two exist under the same conditions, so at their equal default "
            "weights they always cancel: zero direction, but twelve points of "
            "apparent participation. Neither sees the price. Location is "
            "measured against the live entry price by entry_quality instead."
        ),
        in_direction_score=False,
    ),
)


# ------------------------------------------------------------- validation review

VALIDATION_GROUPS: Tuple[EvidenceGroup, ...] = (
    EvidenceGroup(
        name="timeframe_direction",
        category="A. Direction",
        members=("higher_timeframe_trend", "lower_timeframe_confirmation",
                 "distance_to_ema200"),
        basis=(
            "market_context: every timeframe's trend is the same "
            "structure_engine swing trend that forms the base's market-"
            "structure group, run on H4/D1/M15 candles; EMA200 side is the "
            "same directional state read from a moving average. Trend + BOS + "
            "CHoCH + EMA + HTF trend are one directional state, counted once — "
            "in the base."
        ),
        adds_conviction=False,
    ),
    EvidenceGroup(
        name="regime",
        category="F. Regime",
        members=("atr_condition", "volatility"),
        basis=(
            "indicators: atr_condition reads ATR as a percent of price and "
            "volatility reads the same Wilder ATR against its own median — one "
            "ATR series, two thresholds"
        ),
    ),
    EvidenceGroup(
        name="levels",
        category="C. Location",
        members=("distance_to_support", "distance_to_resistance",
                 "distance_from_liquidity"),
        basis=(
            "levels.py: support and resistance are the nearest swing low below "
            "and swing high above price; smc_engine: liquidity pools are those "
            "same swing points (plus equal-level clusters). All three are the "
            "distance from price to the same H1 swings."
        ),
    ),
)


def group_of(name: str, groups: Sequence[EvidenceGroup]) -> Optional[EvidenceGroup]:
    """The group ``name`` belongs to, or ``None`` if it stands alone."""
    return next((g for g in groups if name in g.members), None)


def scale_factors(
    weights: Mapping[str, float], groups: Sequence[EvidenceGroup]
) -> Dict[str, float]:
    """The multiplier for each named factor under the one rule.

    Factors in no group keep 1.0 — they are already independent evidence.
    A group outside the direction score scales its members to 0.0. Otherwise
    every present member is scaled by ``strongest / sum``, so the group's
    total weight equals its strongest member's configured weight.

    Only members actually present in ``weights`` take part, so a caller that
    runs a subset of checks still gets a correctly sized group.
    """
    factors = {name: 1.0 for name in weights}
    for group in groups:
        present = [m for m in group.members if m in weights]
        if not present:
            continue
        if not group.in_direction_score:
            for m in present:
                factors[m] = 0.0
            continue
        total = sum(weights[m] for m in present)
        strongest = max(weights[m] for m in present)
        scale = strongest / total if total > 0 else 0.0
        for m in present:
            factors[m] = scale
    return factors
