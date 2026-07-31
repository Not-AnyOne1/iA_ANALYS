"""Deterministic quality assessment of a stated trade setup.

Pure functions over a :class:`models.TradeSetup` plus market facts. No I/O,
no model involvement — Claude reads these verdicts, it never derives them.

Answers three questions a desk would ask before sizing anything:

- **Risk/reward** — computed from the stated levels only, never inferred.
- **Stop-loss quality** — is the stop survivable given current volatility,
  and is it parked somewhere price is likely to sweep before continuing?
- **Take-profit quality** — is the target realistic for the volatility, or
  does it sit behind a level price must break first?

Every verdict is ``unknown`` when the inputs needed for it are absent. A
signal that states no stop loss gets ``unknown``, never a default or an
assumed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from levels import LevelContext
from models import TradeSetup
from structure_engine import SwingPoint, SwingType


class Quality(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNKNOWN = "unknown"


# A stop closer than this many ATRs is inside normal noise: ordinary
# volatility alone is likely to take it out regardless of direction.
_MIN_STOP_ATR = 0.5
# Beyond this, the stop is so wide the position must be tiny to keep risk
# constant, and the R:R maths stops working in the trade's favour.
_MAX_STOP_ATR = 5.0
# A target under this many ATRs away is inside the noise band too.
_MIN_TARGET_ATR = 0.5


@dataclass(frozen=True)
class StopAssessment:
    quality: Quality
    distance: Optional[float]           # price units from entry
    atr_multiple: Optional[float]
    inside_liquidity: Optional[bool]    # sits beyond a swing price may sweep
    reasons: tuple


@dataclass(frozen=True)
class TargetAssessment:
    quality: Quality
    distance: Optional[float]
    atr_multiple: Optional[float]
    blocked_by_level: Optional[float]   # level standing between entry and TP
    reasons: tuple


@dataclass(frozen=True)
class TradeQuality:
    """Everything deterministic that can be said about a stated setup."""

    risk_reward: Optional[float]
    entry_price: Optional[float]
    stop: StopAssessment
    target: TargetAssessment

    @property
    def has_complete_levels(self) -> bool:
        """True only when entry, stop and at least one target are stated."""
        return (
            self.entry_price is not None
            and self.stop.distance is not None
            and self.target.distance is not None
        )


def effective_entry(setup: TradeSetup, current_price: Optional[float] = None) -> Optional[float]:
    """The price a stop/target should be measured from.

    Uses the midpoint of a stated entry zone (matching how
    ``TradeSetup.risk_reward`` already averages entries, so the two never
    disagree). Falls back to the live price for a market order that states
    no entry — which is the honest reading of "BUY NOW".
    """
    if setup.entries:
        return sum(setup.entries) / len(setup.entries)
    return current_price


def assess_stop(
    setup: TradeSetup,
    entry: Optional[float],
    atr_value: Optional[float],
    swings: Sequence[SwingPoint] = (),
) -> StopAssessment:
    """Judge the stop's survivability and placement."""
    reasons: list[str] = []
    if setup.stop_loss is None or entry is None:
        return StopAssessment(Quality.UNKNOWN, None, None, None,
                              ("no stop loss stated" if setup.stop_loss is None
                               else "no entry price to measure from",))

    distance = abs(entry - setup.stop_loss)
    atr_multiple = (distance / atr_value) if atr_value else None

    # Is the stop parked beyond a swing point price commonly sweeps first?
    inside_liquidity: Optional[bool] = None
    if swings:
        if setup.direction == "long":
            # A long's stop sits below entry; a swing low just above it is
            # liquidity that a sweep would run through on the way down.
            inside_liquidity = any(
                s.type == SwingType.LOW and setup.stop_loss < s.price < entry
                for s in swings
            )
        elif setup.direction == "short":
            inside_liquidity = any(
                s.type == SwingType.HIGH and entry < s.price < setup.stop_loss
                for s in swings
            )

    quality = Quality.ACCEPTABLE
    if atr_multiple is None:
        quality = Quality.UNKNOWN
        reasons.append("ATR unavailable, cannot judge stop width")
    elif atr_multiple < _MIN_STOP_ATR:
        quality = Quality.POOR
        reasons.append(f"stop only {atr_multiple:.2f} ATR away — inside normal noise")
    elif atr_multiple > _MAX_STOP_ATR:
        quality = Quality.POOR
        reasons.append(f"stop {atr_multiple:.2f} ATR away — unusually wide")
    else:
        reasons.append(f"stop {atr_multiple:.2f} ATR from entry")
        quality = Quality.GOOD if 1.0 <= atr_multiple <= 3.0 else Quality.ACCEPTABLE

    if inside_liquidity:
        # Downgrade, never upgrade: resting behind obvious liquidity is a
        # real hazard regardless of how well-sized the stop is.
        quality = Quality.POOR if quality != Quality.UNKNOWN else quality
        reasons.append("stop sits beyond a swing level price may sweep first")

    return StopAssessment(quality, distance, atr_multiple, inside_liquidity, tuple(reasons))


def assess_target(
    setup: TradeSetup,
    entry: Optional[float],
    atr_value: Optional[float],
    levels: Optional[LevelContext] = None,
) -> TargetAssessment:
    """Judge the first take-profit's realism."""
    reasons: list[str] = []
    if not setup.take_profits or entry is None:
        return TargetAssessment(Quality.UNKNOWN, None, None, None,
                                ("no take profit stated" if not setup.take_profits
                                 else "no entry price to measure from",))

    target = setup.take_profits[0]
    distance = abs(target - entry)
    atr_multiple = (distance / atr_value) if atr_value else None

    # Does a support/resistance level stand between entry and the target?
    blocked_by: Optional[float] = None
    if levels is not None:
        if setup.direction == "long":
            blocked_by = next(
                (lvl for lvl in sorted(levels.resistance_levels) if entry < lvl < target),
                None,
            )
        elif setup.direction == "short":
            blocked_by = next(
                (lvl for lvl in sorted(levels.support_levels, reverse=True)
                 if target < lvl < entry),
                None,
            )

    quality = Quality.ACCEPTABLE
    if atr_multiple is None:
        quality = Quality.UNKNOWN
        reasons.append("ATR unavailable, cannot judge target distance")
    elif atr_multiple < _MIN_TARGET_ATR:
        quality = Quality.POOR
        reasons.append(f"target only {atr_multiple:.2f} ATR away — inside normal noise")
    else:
        reasons.append(f"target {atr_multiple:.2f} ATR from entry")
        quality = Quality.GOOD if atr_multiple >= 1.5 else Quality.ACCEPTABLE

    if blocked_by is not None:
        quality = Quality.POOR if quality != Quality.UNKNOWN else quality
        reasons.append(f"level at {blocked_by:g} stands between entry and target")

    return TargetAssessment(quality, distance, atr_multiple, blocked_by, tuple(reasons))


def assess(
    setup: TradeSetup,
    *,
    current_price: Optional[float] = None,
    atr_value: Optional[float] = None,
    swings: Sequence[SwingPoint] = (),
    levels: Optional[LevelContext] = None,
) -> TradeQuality:
    """Full deterministic assessment of a stated setup."""
    entry = effective_entry(setup, current_price)
    return TradeQuality(
        risk_reward=setup.risk_reward,   # reuse the existing definition
        entry_price=entry,
        stop=assess_stop(setup, entry, atr_value, swings),
        target=assess_target(setup, entry, atr_value, levels),
    )
