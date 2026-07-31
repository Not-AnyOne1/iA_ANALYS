"""Scoring Engine (RFC-007): deterministic confluence scoring.

Combines an already-computed ``structure_engine.StructureAnalysis`` and
``smc_engine.SMCAnalysis`` (RFC-005/RFC-006 outputs, unchanged) into a
single directional confidence score. Pure and deterministic — no I/O, no
async, no AI or probabilistic logic, no randomness, no live price input
(this engine consumes only the two analyses named above, nothing else).
Not wired into the analysis pipeline yet (RFC-007 scope) — this is a
reusable engine, not a change to what happens to a Telegram message today.

Scoring model, stated precisely:

Each of the 10 factors below casts a SIGNED vote: +weight for a bullish
reading, -weight for a bearish reading, 0 if the factor has nothing to say
(no relevant data, or genuinely neutral). Every weight is a constructor
parameter with a documented default, so the relative importance of each
factor is explicit and tunable, never hardcoded into the logic itself.

  1. trend (default 20)              - structure.trend: BULLISH/BEARISH
     vote accordingly; RANGING/UNKNOWN votes 0.
  2. bos (default 15)                - the most recent BOS event's
     direction (by candle_index); 0 if no BOS exists.
  3. choch (default 15)              - the most recent CHoCH event's
     direction; 0 if no CHoCH exists. Evaluated independently of BOS, so a
     late CHoCH against an earlier BOS's direction is exactly the kind of
     disagreement rule 11 below is designed to absorb.
  4. liquidity_sweep (default 10)    - the most recent liquidity sweep's
     reaction direction; 0 if none.
  5. fvg (default 8)                 - the most recent Fair Value Gap's
     direction; 0 if none.
  6. inverse_fvg (default 8)         - the most recent Inverse FVG's
     (flipped) direction; 0 if none. Kept separate from plain FVG since an
     inversion is a meaningfully different signal from a fresh gap.
  7. order_block (default 10)        - the most recent Order Block's
     direction; 0 if none.
  8. supply_demand (default 8)       - sign of
     len(demand_zones) - len(supply_zones); 0 if equal (including both
     empty).
  9. ote (default 6)                 - smc.ote_zone's direction; 0 if no
     OTE zone was produced.
  10. premium_discount (default 6)   - this engine has no live price, so
     "is price in premium or discount" is inferred from which swing is
     more recent: a recent swing HIGH means price's last notable move
     ended at the top of the range (premium, bearish bias); a recent
     swing LOW means the opposite (discount, bullish bias). Only voted
     when smc actually produced premium/discount zones (i.e. both a
     confirmed swing high and low exist) - otherwise 0.

11. Combining factors — net_score is the sum of all ten signed votes;
    total_possible is the sum of all ten weights (a fixed constant of the
    engine's configuration, independent of the input). Both are reported
    unchanged.

    Confidence measures CONFLUENCE — the quality of the winning side — not
    the size of the disagreement. Let ``bullish`` and ``bearish`` be the sums
    of the positive and negative votes, ``winner = max(...)``,
    ``loser = min(...)`` and ``cast = winner + loser``. Two raw quantities:

        margin        = (winner - loser) / cast     # how one-sided the vote is
        participation = cast / total_possible       # how much evidence spoke

    Neither is used raw. Each is passed through its own smooth S-curve, and
    the two results multiply:

        confidence = round(100 * margin_curve(margin)
                               * participation_curve(participation))

    Both curves are continuous and infinitely differentiable — no floor, no
    threshold, no branch — and both are pinned so that ``f(0) = 0`` and
    ``f(1) = 1`` exactly. See :func:`_margin_curve` (Richards curve in
    log-margin) and :func:`_participation_curve` (Gompertz).

    Why two curves rather than one equation. A single power law cannot meet
    the requirements, and the reason is arithmetic rather than taste: a
    margin of 4 points out of 106 must score about 60 while a margin of 1
    point out of 105 must stay well below 30. Those inputs differ by a factor
    of 4 and the outputs by a factor of more than 2, which pins any pure
    power law to an exponent near 1/2 — and ``sqrt(4/106)`` is 19, nowhere
    near 60. No exponent satisfies both the ratio and the level.

    An S-curve escapes because its slope is not constant in log-space: it can
    be steep across the region separating "tie" from "genuine lean" and flat
    elsewhere. That is precisely the distinction that matters for trading,
    and a power law spends its dynamic range everywhere except there.

    Why the two curves face opposite ways:

    * **margin** rises fast and then crawls. Most of the information in a
      vote is in whether one side won at all; the difference between winning
      by 40 and by 60 is comparatively minor.
    * **participation** starts almost flat. One factor firing into silence is
      perfectly *one-sided* — nothing contradicts it — so the margin curve
      scores it at the maximum, and only the evidence base can disqualify it.
      The Gompertz's long flat start does that smoothly: every single factor,
      including the heaviest (trend, 20 of 106), scores 1 or less.

    They multiply rather than average because either being zero must be
    disqualifying: a tie is worthless however many factors voted, and a
    unanimous verdict is worthless if almost nothing voted.

    Confidence is 0 exactly at a dead tie or an empty slate, and reaches 100
    only when every factor votes and all agree.

12. Direction — BUY if net_score > 0, SELL if net_score < 0, NONE if
    net_score == 0, EXCEPT that confidence below
    ``min_confidence_for_direction`` (default 10) always forces NONE
    regardless of net_score's sign — a technically-nonzero but tiny net
    score (e.g. one weak factor firing alone) should not be reported as a
    trade direction. Confidence itself is always reported truthfully,
    even when direction is forced to NONE.

13. Reasons — one human-readable string per factor that actually voted
    (contribution != 0), in the fixed factor-evaluation order listed
    above. Factors that didn't fire are omitted from reasons but still
    appear in the breakdown (with contribution 0) for full transparency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp
from typing import Optional, Tuple

from market_data import Timeframe
from smc_engine import SMCAnalysis
from structure_engine import StructureAnalysis, StructureEventType, SwingType, TrendDirection


# Shape of the confidence curve (rule 11). Both curves are smooth S-shapes
# with no floor, no threshold and no branch, so confidence is continuous and
# differentiable in the vote split everywhere.
#
# Margin: a mixture of two log-logistic curves. A_1/B_1 is the steep early
# component that separates "tie" from "genuine lean"; A_2/B_2 the slower one
# that carries the approach to 1. W_ is their mixing weight.
_MARGIN_W = 0.8241
_MARGIN_A1, _MARGIN_B1 = 2.0, 0.025
_MARGIN_A2, _MARGIN_B2 = 3.0, 0.70

# Participation: a Gompertz curve, near-zero until roughly a fifth of the
# weight has voted and saturated once about two thirds has.
_BASE_B = 20.17
_BASE_C = 9.21


def _log_logistic(x: float, alpha: float, beta: float) -> float:
    """``1 / (1 + (beta/x)**alpha)`` — a logistic in ``log x``.

    Reaches 0 at x = 0 exactly (the inner term diverges), rather than merely
    approaching it, which is what lets a dead tie score exactly 0 with no
    special case.
    """
    return 1.0 / (1.0 + (beta / x) ** alpha) if x > 0.0 else 0.0


def _margin_curve(margin: float) -> float:
    """Mixture of two log-logistic curves, on [0, 1].

    Why a mixture rather than one sigmoid. Written in ``u = log(margin)``,
    the required shape has slopes of roughly 0.22, then 0.11, then 0.20 over
    three consecutive stretches — it flattens in the middle and steepens
    again near the top. A single sigmoid has exactly one inflection, so its
    slope rises then falls and can never fall then rise. Two components can:
    the first carries the rise off zero, the second the approach to 1.

    A convex combination of increasing functions is increasing, and each
    component is smooth, so the mixture is monotone and C-infinity on (0, 1].
    Dividing by its value at 1 pins the top at exactly 1, so an unopposed
    vote reaches exactly 100 with no clipping.
    """
    if margin <= 0.0:
        return 0.0
    blend = (lambda x: _MARGIN_W * _log_logistic(x, _MARGIN_A1, _MARGIN_B1)
             + (1.0 - _MARGIN_W) * _log_logistic(x, _MARGIN_A2, _MARGIN_B2))
    return blend(margin) / blend(1.0)


def _participation_curve(participation: float) -> float:
    """Normalised Gompertz curve on [0, 1].

    ``G(q) = exp(-b*exp(-c*q))`` is S-shaped but asymmetric: a long, almost
    flat start, then a rise, then a slow approach to its asymptote. The flat
    start is the point — it is what keeps a lone factor speaking into silence
    from earning any confidence, smoothly, with no threshold to cross.

    The parameters put the rise between roughly a fifth and two thirds of the
    total weight, and that upper end matters as much as the lower. If this
    curve were still climbing at high participation, adding a *losing* vote
    would buy more participation than it costs in margin, and confidence
    would go UP — 70-vs-36 outscoring 80-vs-20. Saturating early keeps
    confidence non-increasing in the losing side.

    Rescaled so ``G(0) -> 0`` and ``G(1) -> 1`` exactly.
    """
    lo = exp(-_BASE_B)
    hi = exp(-_BASE_B * exp(-_BASE_C))
    return (exp(-_BASE_B * exp(-_BASE_C * participation)) - lo) / (hi - lo)


class ScoreDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"
    NONE = "none"


@dataclass(frozen=True)
class FactorScore:
    """One factor's contribution to the overall score."""

    name: str
    weight: float                        # max possible |contribution|
    contribution: float                  # signed: -weight..+weight
    direction: Optional[TrendDirection]  # None if the factor didn't vote
    reason: str


@dataclass(frozen=True)
class ScoringResult:
    """The full scoring outcome for one symbol/timeframe."""

    symbol: str
    timeframe: Timeframe
    direction: ScoreDirection
    confidence: int  # 0-100
    net_score: float
    total_possible: float
    breakdown: Tuple[FactorScore, ...]
    reasons: Tuple[str, ...]


class ScoringEngine:
    """Deterministic confluence scorer. Stateless across calls."""

    def __init__(
        self,
        *,
        trend_weight: float = 20.0,
        bos_weight: float = 15.0,
        choch_weight: float = 15.0,
        liquidity_sweep_weight: float = 10.0,
        fvg_weight: float = 8.0,
        inverse_fvg_weight: float = 8.0,
        order_block_weight: float = 10.0,
        supply_demand_weight: float = 8.0,
        ote_weight: float = 6.0,
        premium_discount_weight: float = 6.0,
        min_confidence_for_direction: float = 10.0,
    ) -> None:
        weights = dict(
            trend=trend_weight, bos=bos_weight, choch=choch_weight,
            liquidity_sweep=liquidity_sweep_weight, fvg=fvg_weight,
            inverse_fvg=inverse_fvg_weight, order_block=order_block_weight,
            supply_demand=supply_demand_weight, ote=ote_weight,
            premium_discount=premium_discount_weight,
        )
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name}_weight must be >= 0, got {value}")
        if not (0 <= min_confidence_for_direction <= 100):
            raise ValueError(
                f"min_confidence_for_direction must be within [0, 100], "
                f"got {min_confidence_for_direction}"
            )

        self._weights = weights
        self._min_confidence_for_direction = min_confidence_for_direction

    def score(self, structure: StructureAnalysis, smc: SMCAnalysis) -> ScoringResult:
        if structure.symbol != smc.symbol or structure.timeframe != smc.timeframe:
            raise ValueError(
                f"structure ({structure.symbol}/{structure.timeframe}) and smc "
                f"({smc.symbol}/{smc.timeframe}) do not describe the same data"
            )

        breakdown = (
            self._score_trend(structure, smc),
            self._score_bos(structure, smc),
            self._score_choch(structure, smc),
            self._score_liquidity_sweep(structure, smc),
            self._score_fvg(structure, smc),
            self._score_inverse_fvg(structure, smc),
            self._score_order_block(structure, smc),
            self._score_supply_demand(structure, smc),
            self._score_ote(structure, smc),
            self._score_premium_discount(structure, smc),
        )

        net_score = sum(f.contribution for f in breakdown)
        total_possible = sum(f.weight for f in breakdown)
        confidence = self._confidence(breakdown, total_possible)

        if confidence < self._min_confidence_for_direction or net_score == 0:
            direction = ScoreDirection.NONE
        elif net_score > 0:
            direction = ScoreDirection.BUY
        else:
            direction = ScoreDirection.SELL

        reasons = tuple(f.reason for f in breakdown if f.contribution != 0)

        return ScoringResult(
            symbol=structure.symbol, timeframe=structure.timeframe,
            direction=direction, confidence=confidence,
            net_score=net_score, total_possible=total_possible,
            breakdown=breakdown, reasons=reasons,
        )

    # ------------------------------------------------------------- confidence

    @staticmethod
    def _confidence(breakdown: Tuple[FactorScore, ...], total_possible: float) -> int:
        """How much confluence is there behind the winning side? (Rule 11.)

        Deliberately not a function of ``net_score`` alone. The margin on its
        own cannot tell "51 points of evidence lost the argument" apart from
        "no evidence existed", because both leave a small numerator over the
        same fixed denominator. Splitting the vote into the two sides keeps
        that distinction, which is what makes the floors below expressible.
        """
        bullish = sum(f.contribution for f in breakdown if f.contribution > 0)
        bearish = -sum(f.contribution for f in breakdown if f.contribution < 0)
        winner, loser = max(bullish, bearish), min(bullish, bearish)
        cast = winner + loser
        if cast <= 0 or total_possible <= 0:
            return 0

        margin = (winner - loser) / cast            # 0 at a tie, 1 unopposed
        participation = cast / total_possible       # share of weight that voted

        return round(min(100.0, 100.0
                         * _margin_curve(margin)
                         * _participation_curve(participation)))

    # ---------------------------------------------------------------- factors

    def _score_trend(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["trend"]
        if structure.trend == TrendDirection.BULLISH:
            return FactorScore("trend", weight, weight, TrendDirection.BULLISH,
                                "Structure trend is bullish")
        if structure.trend == TrendDirection.BEARISH:
            return FactorScore("trend", weight, -weight, TrendDirection.BEARISH,
                                "Structure trend is bearish")
        return FactorScore("trend", weight, 0.0, None,
                            "Structure trend is ranging/unknown - no directional vote")

    def _score_bos(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["bos"]
        events = [e for e in structure.events if e.type == StructureEventType.BOS]
        if not events:
            return FactorScore("bos", weight, 0.0, None, "No BOS event detected")
        latest = max(events, key=lambda e: e.candle_index)
        if latest.direction == TrendDirection.BULLISH:
            return FactorScore("bos", weight, weight, TrendDirection.BULLISH,
                                f"Latest BOS at candle {latest.candle_index} is bullish")
        return FactorScore("bos", weight, -weight, TrendDirection.BEARISH,
                            f"Latest BOS at candle {latest.candle_index} is bearish")

    def _score_choch(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["choch"]
        events = [e for e in structure.events if e.type == StructureEventType.CHOCH]
        if not events:
            return FactorScore("choch", weight, 0.0, None, "No CHoCH event detected")
        latest = max(events, key=lambda e: e.candle_index)
        if latest.direction == TrendDirection.BULLISH:
            return FactorScore("choch", weight, weight, TrendDirection.BULLISH,
                                f"Latest CHoCH at candle {latest.candle_index} is bullish")
        return FactorScore("choch", weight, -weight, TrendDirection.BEARISH,
                            f"Latest CHoCH at candle {latest.candle_index} is bearish")

    def _score_liquidity_sweep(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["liquidity_sweep"]
        if not smc.liquidity_sweeps:
            return FactorScore("liquidity_sweep", weight, 0.0, None, "No liquidity sweep detected")
        latest = max(smc.liquidity_sweeps, key=lambda s: s.candle_index)
        sign = 1.0 if latest.direction == TrendDirection.BULLISH else -1.0
        return FactorScore(
            "liquidity_sweep", weight, sign * weight, latest.direction,
            f"Latest liquidity sweep at candle {latest.candle_index} reacted {latest.direction.value}",
        )

    def _score_fvg(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["fvg"]
        if not smc.fair_value_gaps:
            return FactorScore("fvg", weight, 0.0, None, "No fair value gap detected")
        latest = max(smc.fair_value_gaps, key=lambda f: f.candle_index)
        sign = 1.0 if latest.direction == TrendDirection.BULLISH else -1.0
        return FactorScore(
            "fvg", weight, sign * weight, latest.direction,
            f"Latest FVG at candle {latest.candle_index} is {latest.direction.value}",
        )

    def _score_inverse_fvg(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["inverse_fvg"]
        if not smc.inverse_fvgs:
            return FactorScore("inverse_fvg", weight, 0.0, None, "No inverse FVG detected")
        latest = max(smc.inverse_fvgs, key=lambda f: f.candle_index)
        sign = 1.0 if latest.direction == TrendDirection.BULLISH else -1.0
        return FactorScore(
            "inverse_fvg", weight, sign * weight, latest.direction,
            f"Latest inverse FVG at candle {latest.candle_index} is {latest.direction.value}",
        )

    def _score_order_block(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["order_block"]
        if not smc.order_blocks:
            return FactorScore("order_block", weight, 0.0, None, "No order block detected")
        latest = max(smc.order_blocks, key=lambda o: o.candle_index)
        sign = 1.0 if latest.direction == TrendDirection.BULLISH else -1.0
        return FactorScore(
            "order_block", weight, sign * weight, latest.direction,
            f"Latest order block at candle {latest.candle_index} is {latest.direction.value}",
        )

    def _score_supply_demand(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["supply_demand"]
        demand, supply = len(smc.demand_zones), len(smc.supply_zones)
        if demand == supply:
            return FactorScore(
                "supply_demand", weight, 0.0, None,
                f"Equal demand ({demand}) and supply ({supply}) zones - no bias",
            )
        if demand > supply:
            return FactorScore(
                "supply_demand", weight, weight, TrendDirection.BULLISH,
                f"More demand zones ({demand}) than supply zones ({supply})",
            )
        return FactorScore(
            "supply_demand", weight, -weight, TrendDirection.BEARISH,
            f"More supply zones ({supply}) than demand zones ({demand})",
        )

    def _score_ote(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["ote"]
        if smc.ote_zone is None:
            return FactorScore("ote", weight, 0.0, None, "No OTE zone available")
        direction = smc.ote_zone.direction
        sign = 1.0 if direction == TrendDirection.BULLISH else -1.0
        return FactorScore("ote", weight, sign * weight, direction,
                            f"OTE zone favors {direction.value}")

    def _score_premium_discount(self, structure: StructureAnalysis, smc: SMCAnalysis) -> FactorScore:
        weight = self._weights["premium_discount"]
        if smc.premium_zone is None or smc.discount_zone is None:
            return FactorScore("premium_discount", weight, 0.0, None,
                                "No premium/discount zone available")

        highs = [s for s in structure.swing_points if s.type == SwingType.HIGH]
        lows = [s for s in structure.swing_points if s.type == SwingType.LOW]
        latest_high = max(highs, key=lambda s: s.index)
        latest_low = max(lows, key=lambda s: s.index)

        if latest_high.index == latest_low.index:
            return FactorScore("premium_discount", weight, 0.0, None,
                                "Ambiguous swing ordering for premium/discount bias")
        if latest_high.index > latest_low.index:
            return FactorScore(
                "premium_discount", weight, -weight, TrendDirection.BEARISH,
                "Most recent swing is a high - price sits in premium (bearish bias)",
            )
        return FactorScore(
            "premium_discount", weight, weight, TrendDirection.BULLISH,
            "Most recent swing is a low - price sits in discount (bullish bias)",
        )
