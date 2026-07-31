"""The Validation Engine: a skeptical challenge of every setup.

Sits between the Python Analysis Engine and Claude:

    Python Analysis  ->  THIS MODULE  ->  Claude  ->  Decision

Pure and deterministic. Its purpose is the opposite of the scoring engine's:
where ``scoring_engine.py`` asks *"what supports this trade?"*, this module
asks **"what argues against it?"**

**This module does not produce a confidence.** It used to emit a 0-100 score
that was then min()'d against the deterministic confidence and Claude's own
number — three different quantities (market conviction, setup-quality review,
reviewer self-assessment) combined by an operator that has no meaning across
units. There is now exactly one confidence in the system, the deterministic
score from ``scoring_engine``, and this module emits **adjustments to it**:

    bonus    points of confidence the review earned  (0..MAX_ADJUSTMENT)
    penalty  points of confidence the review charged (0..MAX_ADJUSTMENT)

applied once, in ``trade_decision``, as
``clamp(deterministic + bonus - penalty, 0, 100)``.

Two design choices keep the skepticism structural rather than a matter of
prompt wording:

1. **Unvalidated is not the same as fine.** Both adjustments are shares of
   the *total possible* weight, so a check whose data is missing contributes
   nothing to the numerator while still counting in the denominator. It
   cannot earn bonus. A setup nobody could verify gets almost no credit —
   the burden of proof sits with the trade.

2. **Nothing here can approve or block anything.** This module produces
   adjustments, strengths, weaknesses and warnings. The decision stays with
   ``trade_decision.py``; the hard gate stays with ``risk_engine.py``.

Ownership is strict, and it is why nothing here is fatal any more:

* ``scoring_engine`` owns **confidence**.
* ``risk_engine`` owns **blocking**. Its REJECTs are objective blockers.
* this module owns **review** — the adjustments and the narrative findings.

The three checks that used to be FATAL here — spread, risk/reward and news —
were each an exact duplicate of a risk-engine REJECT on the same input. One
fact reaching the decision by two routes with two different consequences is
not a trading rule, so they are scored and the block is left to its single
owner. ``Severity.FATAL`` is retained as the extension point for a finding
that genuinely cannot be expressed as an adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from market_data import Candle
from news_filter import NewsStatus
from sessions import Session, SessionInfo
from structure_engine import StructureEventType, TrendDirection


class Severity(str, Enum):
    STRENGTH = "strength"
    WEAKNESS = "weakness"
    FATAL = "fatal"
    NEUTRAL = "neutral"    # the check ran and found nothing either way
    UNKNOWN = "unknown"    # the data the check needs was unavailable


@dataclass(frozen=True)
class ValidationCheck:
    """One rule's finding."""

    name: str
    severity: Severity
    contribution: float     # signed, bounded by ±weight
    weight: float           # maximum this check can contribute
    detail: str

    @property
    def is_fatal(self) -> bool:
        return self.severity is Severity.FATAL


@dataclass(frozen=True)
class ValidationResult:
    """The complete challenge of one setup.

    Deliberately NOT a confidence. This module reviews; it does not measure
    market conviction. It emits two adjustments in *points of confidence*,
    which ``trade_decision`` applies to the one and only confidence number
    (the deterministic score from ``scoring_engine``):

        adjusted = clamp(deterministic + bonus - penalty, 0, 100)

    Keeping bonus and penalty separate rather than pre-netting them means the
    report can state what the review added and what it took away, instead of
    a single opaque delta.
    """

    bonus: int                          # 0..MAX_ADJUSTMENT, earned by strengths
    penalty: int                        # 0..MAX_ADJUSTMENT, charged by weaknesses
    checks: Tuple[ValidationCheck, ...]

    @property
    def adjustment(self) -> int:
        """Signed net effect on confidence."""
        return self.bonus - self.penalty

    @property
    def fatal_problems(self) -> Tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.severity is Severity.FATAL)

    @property
    def weaknesses(self) -> Tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.severity is Severity.WEAKNESS)

    @property
    def strengths(self) -> Tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.severity is Severity.STRENGTH)

    @property
    def unknowns(self) -> Tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if c.severity is Severity.UNKNOWN)

    @property
    def has_fatal(self) -> bool:
        return bool(self.fatal_problems)

    @property
    def summary(self) -> str:
        if self.has_fatal:
            return (f"{len(self.fatal_problems)} fatal problem(s): "
                    + "; ".join(c.detail for c in self.fatal_problems))
        return (f"+{self.bonus} / -{self.penalty} confidence — "
                f"{len(self.strengths)} strengths, {len(self.weaknesses)} weaknesses, "
                f"{len(self.unknowns)} unverified")


# The most confidence the review may add, and separately the most it may
# take away. Chosen so a review can carry a setup across one decision band
# (the bands are 20 points wide) but never across both: market conviction
# from the scoring engine remains the base, and this remains an adjustment
# to it. Surfaced in the report so it is never a hidden constant.
MAX_ADJUSTMENT = 25


# Per-check weights. Higher weight = more influence on the score. Structural
# and risk facts outrank cosmetic confluence, which is why trend alignment
# and risk/reward carry more than the presence of an order block.
_WEIGHTS = {
    "higher_timeframe_trend": 12.0,
    "lower_timeframe_confirmation": 6.0,
    "liquidity_direction": 8.0,
    "bos_alignment": 8.0,
    "choch_alignment": 6.0,
    "order_block_quality": 5.0,
    "fvg_quality": 5.0,
    "distance_from_liquidity": 6.0,
    "atr_condition": 5.0,
    "volatility": 5.0,
    "spread": 6.0,
    "session_quality": 5.0,
    "risk_reward": 10.0,
    "news": 8.0,
    "distance_to_ema200": 6.0,
    "distance_to_support": 5.0,
    "distance_to_resistance": 5.0,
    "max_adverse_excursion": 9.0,
    "probability_score": 8.0,
}


def _check(name: str, severity: Severity, detail: str, ratio: float = 1.0) -> ValidationCheck:
    """Build a check, translating severity into a signed contribution.

    ``ratio`` scales the magnitude for checks that are a matter of degree
    (0.5 = "half as convincing as it could be").
    """
    weight = _WEIGHTS[name]
    if severity is Severity.STRENGTH:
        contribution = weight * ratio
    elif severity is Severity.WEAKNESS:
        contribution = -weight * ratio
    elif severity is Severity.FATAL:
        contribution = -weight
    else:
        contribution = 0.0
    return ValidationCheck(name=name, severity=severity, contribution=contribution,
                           weight=weight, detail=detail)


# --------------------------------------------------------- adverse excursion

def estimate_mae_atr(
    candles: Sequence[Candle],
    direction: str,
    atr_value: Optional[float],
    *,
    horizon: int = 20,
    lookback: int = 200,
    percentile: float = 0.8,
) -> Optional[float]:
    """Estimate the adverse excursion a trade of this direction typically
    suffers, in ATR multiples.

    Deterministic and empirical rather than assumed: for every candle in the
    lookback window, measure how far price went *against* ``direction`` over
    the following ``horizon`` candles, then take the ``percentile`` of those
    observations. The result answers "on this instrument, at this
    volatility, how much heat does a position of this direction usually
    take?" — which is exactly what a stop has to survive.

    ``None`` when there is not enough data or no ATR to normalise by; the
    caller reports that as unverified rather than assuming a benign value.
    """
    if atr_value in (None, 0) or direction not in ("long", "short"):
        return None
    window = list(candles[-lookback:]) if len(candles) > lookback else list(candles)
    if len(window) < horizon + 2:
        return None

    excursions: List[float] = []
    for i in range(len(window) - horizon):
        entry = window[i].close
        following = window[i + 1: i + 1 + horizon]
        if direction == "long":
            worst = min(c.low for c in following)
            excursions.append(max(0.0, entry - worst))
        else:
            worst = max(c.high for c in following)
            excursions.append(max(0.0, worst - entry))

    if not excursions:
        return None
    excursions.sort()
    index = min(len(excursions) - 1, int(len(excursions) * percentile))
    return excursions[index] / atr_value


# ------------------------------------------------------------- the checks

def _trend_of(context, name: str) -> Optional[str]:
    tf = context.timeframes.get(name)
    return tf.trend if tf else None


def _aligned(direction: Optional[str], trend: Optional[str]) -> Optional[bool]:
    if direction is None or trend in (None, "ranging", "unknown"):
        return None
    return (direction == "long" and trend == "bullish") or (
        direction == "short" and trend == "bearish")


def _check_higher_timeframe_trend(context, direction) -> ValidationCheck:
    name = "higher_timeframe_trend"
    trends = [(tf, _trend_of(context, tf)) for tf in ("D1", "H4")]
    known = [(tf, t) for tf, t in trends if t not in (None, "unknown")]
    if not known or direction is None:
        return _check(name, Severity.UNKNOWN, "no higher-timeframe trend available")

    verdicts = [(tf, _aligned(direction, t), t) for tf, t in known]
    against = [(tf, t) for tf, a, t in verdicts if a is False]
    withit = [(tf, t) for tf, a, t in verdicts if a is True]

    if against and not withit:
        # Trading against every higher timeframe that has an opinion is the
        # single most common way a technically tidy setup loses. A WEAKNESS at
        # full weight, not fatal: counter-trend trades are a legitimate style
        # (reversals, mean reversion), so this costs the maximum the check can
        # subtract and lets the score bands judge rather than vetoing.
        return _check(name, Severity.WEAKNESS,
                      f"{direction} against the higher-timeframe trend "
                      f"({', '.join(f'{tf}={t}' for tf, t in against)})",
                      ratio=1.0)
    if against:
        return _check(name, Severity.WEAKNESS,
                      f"higher timeframes disagree "
                      f"({', '.join(f'{tf}={t}' for tf, _aligned_flag, t in verdicts)})")
    if withit:
        return _check(name, Severity.STRENGTH,
                      f"aligned with {', '.join(f'{tf}={t}' for tf, t in withit)}",
                      ratio=1.0 if len(withit) > 1 else 0.7)
    return _check(name, Severity.NEUTRAL, "higher timeframes are ranging")


def _check_lower_timeframe_confirmation(context, direction) -> ValidationCheck:
    name = "lower_timeframe_confirmation"
    trend = _trend_of(context, "M15")
    if trend is None or direction is None:
        return _check(name, Severity.UNKNOWN, "no lower-timeframe trend available")
    aligned = _aligned(direction, trend)
    if aligned is None:
        return _check(name, Severity.NEUTRAL, f"M15 is {trend} — no confirmation either way")
    if aligned:
        return _check(name, Severity.STRENGTH, f"M15 confirms ({trend})")
    return _check(name, Severity.WEAKNESS, f"M15 contradicts the trade ({trend})")


def _check_liquidity_direction(context, direction) -> ValidationCheck:
    name = "liquidity_direction"
    smc = context.smc
    if smc is None or not smc.liquidity_sweeps:
        return _check(name, Severity.UNKNOWN, "no liquidity sweep detected")
    last = max(smc.liquidity_sweeps, key=lambda s: s.candle_index)
    reaction = last.direction.value
    if direction is None:
        return _check(name, Severity.UNKNOWN, "trade direction unknown")
    if (direction == "long" and reaction == "bullish") or (
            direction == "short" and reaction == "bearish"):
        return _check(name, Severity.STRENGTH,
                      f"last sweep reacted {reaction}, with the trade")
    # A WEAKNESS at full weight, not fatal: a sweep in the opposite direction
    # is evidence the near-term flow is against the trade, but it is a read on
    # likelihood, not a blocker — the sweep may already be exhausted.
    return _check(name, Severity.WEAKNESS,
                  f"last liquidity sweep reacted {reaction}, against a {direction} trade",
                  ratio=1.0)


def _event_alignment(context, direction, event_type, name) -> ValidationCheck:
    structure = context.structure
    if structure is None:
        return _check(name, Severity.UNKNOWN, "no structure analysis")
    events = [e for e in structure.events if e.type is event_type]
    if not events:
        return _check(name, Severity.UNKNOWN, f"no {event_type.value.upper()} detected")
    last = max(events, key=lambda e: e.candle_index)
    if direction is None:
        return _check(name, Severity.UNKNOWN, "trade direction unknown")
    aligned = ((direction == "long" and last.direction is TrendDirection.BULLISH)
               or (direction == "short" and last.direction is TrendDirection.BEARISH))
    label = event_type.value.upper()
    if aligned:
        return _check(name, Severity.STRENGTH,
                      f"latest {label} is {last.direction.value}, with the trade")
    return _check(name, Severity.WEAKNESS,
                  f"latest {label} is {last.direction.value}, against the trade")


def _check_order_block_quality(context, direction) -> ValidationCheck:
    name = "order_block_quality"
    smc = context.smc
    if smc is None:
        return _check(name, Severity.UNKNOWN, "no SMC analysis")
    if not smc.order_blocks:
        return _check(name, Severity.UNKNOWN, "no order block detected")
    last = max(smc.order_blocks, key=lambda o: o.candle_index)
    # A block that has already been broken (breaker) no longer supports the
    # original direction — it is evidence the other way.
    broken = {b.source_order_block.candle_index for b in smc.breaker_blocks}
    if last.candle_index in broken:
        return _check(name, Severity.WEAKNESS,
                      "the most recent order block has already been broken")
    aligned = ((direction == "long" and last.direction is TrendDirection.BULLISH)
               or (direction == "short" and last.direction is TrendDirection.BEARISH))
    if aligned:
        return _check(name, Severity.STRENGTH, "an unbroken order block supports the trade")
    return _check(name, Severity.WEAKNESS, "the most recent order block opposes the trade")


def _check_fvg_quality(context, direction) -> ValidationCheck:
    name = "fvg_quality"
    smc = context.smc
    if smc is None or not smc.fair_value_gaps:
        return _check(name, Severity.UNKNOWN, "no fair value gap detected")
    last = max(smc.fair_value_gaps, key=lambda f: f.candle_index)
    inverted = {i.source_fvg.candle_index for i in smc.inverse_fvgs}
    if last.candle_index in inverted:
        return _check(name, Severity.WEAKNESS,
                      "the most recent FVG has already inverted")
    aligned = ((direction == "long" and last.direction is TrendDirection.BULLISH)
               or (direction == "short" and last.direction is TrendDirection.BEARISH))
    if aligned:
        return _check(name, Severity.STRENGTH, "an unfilled FVG supports the trade")
    return _check(name, Severity.WEAKNESS, "the most recent FVG opposes the trade")


def _check_distance_from_liquidity(context, direction) -> ValidationCheck:
    name = "distance_from_liquidity"
    smc, primary = context.smc, context.primary
    price = context.current_price
    atr = primary.atr if primary else None
    if smc is None or not smc.liquidity_pools or price is None or not atr:
        return _check(name, Severity.UNKNOWN, "no liquidity pools or no ATR")

    # Entering right on top of resting liquidity invites a sweep through the
    # entry before any move in the intended direction.
    nearest = min(smc.liquidity_pools,
                  key=lambda p: abs(((p.price_high + p.price_low) / 2) - price))
    level = (nearest.price_high + nearest.price_low) / 2
    distance_atr = abs(level - price) / atr
    if distance_atr < 0.25:
        return _check(name, Severity.WEAKNESS,
                      f"entry sits {distance_atr:.2f} ATR from resting liquidity at {level:g}")
    if distance_atr > 1.0:
        return _check(name, Severity.STRENGTH,
                      f"nearest liquidity is {distance_atr:.2f} ATR away")
    return _check(name, Severity.NEUTRAL,
                  f"nearest liquidity is {distance_atr:.2f} ATR away")


def _check_atr_condition(context, direction) -> ValidationCheck:
    name = "atr_condition"
    primary = context.primary
    if primary is None or primary.atr_percent is None:
        return _check(name, Severity.UNKNOWN, "ATR unavailable")
    pct = primary.atr_percent
    if pct < 0.05:
        return _check(name, Severity.WEAKNESS,
                      f"ATR is {pct:.3f}% of price — the instrument is barely moving")
    if pct > 5.0:
        return _check(name, Severity.WEAKNESS,
                      f"ATR is {pct:.2f}% of price — unusually violent")
    return _check(name, Severity.STRENGTH, f"ATR is {pct:.2f}% of price — workable")


def _check_volatility(context, direction) -> ValidationCheck:
    name = "volatility"
    primary = context.primary
    state = primary.volatility if primary else None
    if state in (None, "unknown"):
        return _check(name, Severity.UNKNOWN, "volatility unavailable")
    if state == "low":
        # A WEAKNESS at full weight, not fatal: low volatility makes the target
        # slower and less likely to be reached, but it is a market condition
        # that can change within the life of the trade, not a blocker.
        return _check(name, Severity.WEAKNESS,
                      "volatility is low — the target may never be reached",
                      ratio=1.0)
    if state == "high":
        return _check(name, Severity.WEAKNESS,
                      "volatility is high — stops are more likely to be swept")
    return _check(name, Severity.STRENGTH, "volatility is normal")


def _check_spread(context, direction) -> ValidationCheck:
    name = "spread"
    primary = context.primary
    atr = primary.atr if primary else None
    if context.spread is None or not atr:
        return _check(name, Severity.UNKNOWN, "spread or ATR unavailable")
    ratio = context.spread / atr
    if ratio > 0.25:
        # Scored, not vetoed. ``risk_engine._check_spread`` rejects on exactly
        # this threshold and owns the block; duplicating it here made one fact
        # veto twice, from two modules, with two different consequences.
        return _check(name, Severity.WEAKNESS,
                      f"spread is {ratio:.2f} ATR — too expensive to trade", ratio=1.0)
    if ratio > 0.10:
        return _check(name, Severity.WEAKNESS, f"spread is {ratio:.2f} ATR — costly")
    return _check(name, Severity.STRENGTH, f"spread is {ratio:.2f} ATR")


def _check_session_quality(context, direction) -> ValidationCheck:
    name = "session_quality"
    session: Optional[SessionInfo] = context.session
    if session is None:
        return _check(name, Severity.UNKNOWN, "session unknown")
    if not session.active:
        return _check(name, Severity.WEAKNESS,
                      "off-hours — thin liquidity and wider spreads")
    if session.is_overlap:
        return _check(name, Severity.STRENGTH, f"{session.label} overlap — deepest liquidity")
    if session.primary is Session.ASIA:
        return _check(name, Severity.WEAKNESS,
                      "Asia session — typically the narrowest ranges")
    return _check(name, Severity.STRENGTH, f"{session.primary.value} session")


def _check_risk_reward(context, direction) -> ValidationCheck:
    name = "risk_reward"
    quality = context.quality
    rr = quality.risk_reward if quality else None
    if rr is None:
        return _check(name, Severity.UNKNOWN, "risk/reward not computable")
    if rr < 1.0:
        # Scored, not vetoed — ``risk_engine._check_risk_reward`` owns the
        # block. Its minimum (2.0 by default) is stricter than this 1.0, so
        # every trade this used to veto was already rejected there; the only
        # thing the duplicate added was a second, differently-thresholded
        # opinion about the same number.
        return _check(name, Severity.WEAKNESS,
                      f"R:R {rr:.2f} — risking more than the reward", ratio=1.0)
    if rr < 2.0:
        return _check(name, Severity.WEAKNESS, f"R:R {rr:.2f} — below the 1:2 standard")
    if rr >= 3.0:
        return _check(name, Severity.STRENGTH, f"R:R {rr:.2f}")
    return _check(name, Severity.STRENGTH, f"R:R {rr:.2f}", ratio=0.6)


def _check_news(context, direction) -> ValidationCheck:
    name = "news"
    news: Optional[NewsStatus] = context.news
    if news is None or not news.available:
        # Unknown news risk is a real weakness, not a neutral: trading blind
        # into a possible release is a choice, and this makes it visible.
        return _check(name, Severity.UNKNOWN, "no economic calendar — news risk unverified")
    if news.blocked:
        # Scored, not vetoed — ``risk_engine._check_news`` owns the block on
        # exactly the same ``news.blocked`` flag.
        return _check(name, Severity.WEAKNESS, news.reason, ratio=1.0)
    return _check(name, Severity.STRENGTH, "no high-impact events in the window")


def _check_distance_to_ema200(context, direction) -> ValidationCheck:
    name = "distance_to_ema200"
    primary, price = context.primary, context.current_price
    if primary is None or primary.ema.ema_200 is None or price is None or not primary.atr:
        return _check(name, Severity.UNKNOWN, "EMA200, price or ATR unavailable")
    ema200 = primary.ema.ema_200
    distance_atr = abs(price - ema200) / primary.atr
    on_correct_side = (direction == "long" and price > ema200) or (
        direction == "short" and price < ema200)

    if not on_correct_side and direction in ("long", "short"):
        return _check(name, Severity.WEAKNESS,
                      f"price is on the wrong side of EMA200 ({ema200:g}) for a {direction}")
    if distance_atr > 4.0:
        # Far above/below the mean: entering here is buying extension.
        return _check(name, Severity.WEAKNESS,
                      f"price is {distance_atr:.1f} ATR from EMA200 — extended")
    return _check(name, Severity.STRENGTH,
                  f"price is {distance_atr:.1f} ATR from EMA200, on the correct side")


def _check_distance_to_level(context, direction, *, which: str) -> ValidationCheck:
    """Support and resistance share one implementation; only which level
    blocks which direction differs."""
    name = f"distance_to_{which}"
    levels, primary = context.levels, context.primary
    if levels is None or primary is None or not primary.atr:
        return _check(name, Severity.UNKNOWN, f"{which} or ATR unavailable")
    distance = levels.nearest_resistance if which == "resistance" else levels.nearest_support
    if distance is None or distance.atr_multiple is None:
        return _check(name, Severity.UNKNOWN, f"no {which} level identified")

    # Resistance blocks a long; support blocks a short.
    blocks = (which == "resistance" and direction == "long") or (
        which == "support" and direction == "short")
    if blocks and distance.atr_multiple < 0.5:
        return _check(name, Severity.WEAKNESS,
                      f"{which} at {distance.level:g} is only "
                      f"{distance.atr_multiple:.2f} ATR ahead")
    if blocks:
        return _check(name, Severity.STRENGTH,
                      f"{distance.atr_multiple:.2f} ATR of room before {which}")
    # The level behind the trade is protective rather than obstructive.
    if distance.atr_multiple < 0.5:
        return _check(name, Severity.STRENGTH,
                      f"{which} at {distance.level:g} sits close behind as protection")
    return _check(name, Severity.NEUTRAL,
                  f"{which} is {distance.atr_multiple:.2f} ATR away")


def _check_max_adverse_excursion(context, direction) -> ValidationCheck:
    name = "max_adverse_excursion"
    primary, quality = context.primary, context.quality
    atr = primary.atr if primary else None
    mae = estimate_mae_atr(context.primary_candles, direction or "", atr)
    if mae is None:
        return _check(name, Severity.UNKNOWN, "not enough history to estimate adverse excursion")
    if quality is None or quality.stop.atr_multiple is None:
        return _check(name, Severity.UNKNOWN,
                      f"typical adverse excursion is {mae:.2f} ATR, but no stop to compare")

    stop_atr = quality.stop.atr_multiple

    # Deliberately a WEAKNESS, never FATAL. Adverse excursion is a
    # *statistical* property of recent history, not an objective blocker
    # like a news blackout or a nonsensical trade parameter: a stop tighter
    # than the usual heat is a real hazard, but it is a judgement about
    # likelihood, and the trade can still be valid (a shallower entry, a
    # different regime, a deliberate tight-stop scalp). It therefore weighs
    # on the score — heavily, at full weight for the severe case — and lets
    # the score bands decide, rather than forcing SKIP on its own.
    if stop_atr < mae:
        # The stop sits inside the range price routinely travels against a
        # position of this direction: ordinary noise is likely to take it
        # out before the idea has a chance to work.
        return _check(name, Severity.WEAKNESS,
                      f"stop is only {stop_atr:.2f} ATR but trades of this direction "
                      f"typically take {mae:.2f} ATR of heat — likely to be stopped "
                      f"by normal movement",
                      ratio=1.0)
    if stop_atr < mae * 1.3:
        return _check(name, Severity.WEAKNESS,
                      f"stop ({stop_atr:.2f} ATR) barely clears the typical "
                      f"{mae:.2f} ATR adverse excursion",
                      ratio=0.5)
    return _check(name, Severity.STRENGTH,
                  f"stop ({stop_atr:.2f} ATR) comfortably clears the typical "
                  f"{mae:.2f} ATR adverse excursion")


def _check_probability_score(context, direction) -> ValidationCheck:
    name = "probability_score"
    scoring = context.scoring
    if scoring is None:
        return _check(name, Severity.UNKNOWN, "no probability score")
    confidence = scoring.confidence
    # The scoring engine's own direction must agree with the trade.
    if direction and scoring.direction.value in ("buy", "sell"):
        implied = "long" if scoring.direction.value == "buy" else "short"
        if implied != direction:
            # A WEAKNESS at full weight, not fatal: the scoring engine reading
            # the other way is a serious disagreement, but it is one model's
            # opinion over a fixed lookback — a judgement, not a blocker.
            return _check(name, Severity.WEAKNESS,
                          f"the probability model favours {implied}, not {direction}",
                          ratio=1.0)
    if confidence < 30:
        return _check(name, Severity.WEAKNESS, f"probability score is only {confidence}")
    if confidence >= 60:
        return _check(name, Severity.STRENGTH, f"probability score {confidence}")
    return _check(name, Severity.STRENGTH, f"probability score {confidence}", ratio=0.5)


class ValidationEngine:
    """Runs every check and produces the score. Stateless."""

    def validate(self, context, direction: Optional[str] = None) -> ValidationResult:
        """Challenge one setup.

        ``direction`` defaults to whatever the analysed setup stated; pass it
        explicitly only when validating a hypothetical.
        """
        if direction is None:
            quality = context.quality
            direction = getattr(getattr(quality, "setup", None), "direction", None)
        if direction is None and context.scoring is not None:
            # Fall back to what the deterministic model itself implies, so a
            # market order with no stated side is still challenged.
            mapping = {"buy": "long", "sell": "short"}
            direction = mapping.get(context.scoring.direction.value)

        checks: List[ValidationCheck] = [
            _check_higher_timeframe_trend(context, direction),
            _check_lower_timeframe_confirmation(context, direction),
            _check_liquidity_direction(context, direction),
            _event_alignment(context, direction, StructureEventType.BOS, "bos_alignment"),
            _event_alignment(context, direction, StructureEventType.CHOCH, "choch_alignment"),
            _check_order_block_quality(context, direction),
            _check_fvg_quality(context, direction),
            _check_distance_from_liquidity(context, direction),
            _check_atr_condition(context, direction),
            _check_volatility(context, direction),
            _check_spread(context, direction),
            _check_session_quality(context, direction),
            _check_risk_reward(context, direction),
            _check_news(context, direction),
            _check_distance_to_ema200(context, direction),
            _check_distance_to_level(context, direction, which="support"),
            _check_distance_to_level(context, direction, which="resistance"),
            _check_max_adverse_excursion(context, direction),
            _check_probability_score(context, direction),
        ]

        bonus, penalty = self._adjustments(checks)
        return ValidationResult(bonus=bonus, penalty=penalty, checks=tuple(checks))

    @staticmethod
    def _adjustments(checks: Sequence[ValidationCheck]) -> Tuple[int, int]:
        """Turn the checks into confidence points to add and to subtract.

        Each side is expressed as a share of the *total possible* weight and
        scaled by :data:`MAX_ADJUSTMENT`::

            bonus   = MAX_ADJUSTMENT * (sum of positive contributions) / total
            penalty = MAX_ADJUSTMENT * (sum of negative contributions) / total

        Two properties follow from the fixed denominator, and both are
        deliberate:

        * **Skepticism survives.** A check whose data was unavailable
          contributes nothing to the numerator while still counting in the
          denominator, so it cannot earn bonus. A setup nobody could verify
          gets almost no credit — it is not quietly treated as fine.
        * **The review can never dominate.** Bonus and penalty are each
          bounded by MAX_ADJUSTMENT, so the review can move a verdict across
          a band but never manufacture one on its own. Market conviction, the
          deterministic score, stays the base.

        No clamping happens here; the single clamp is applied once in
        ``trade_decision`` after the arithmetic.
        """
        total = sum(c.weight for c in checks)
        if total == 0:
            return 0, 0
        positive = sum(c.contribution for c in checks if c.contribution > 0)
        negative = -sum(c.contribution for c in checks if c.contribution < 0)
        return (round(MAX_ADJUSTMENT * positive / total),
                round(MAX_ADJUSTMENT * negative / total))
