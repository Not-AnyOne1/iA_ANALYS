"""Key price levels and distances to them.

Deterministic and pure: candles (and a price) in, levels out. No I/O, no
model involvement.

Provides the "where is price relative to what matters" facts:

- High and low of the current day, and how far price sits from each
- Major support and resistance, derived from confirmed swing points
- Distances expressed both in price units and — more usefully for
  cross-symbol comparison — in ATR multiples and percent

Support/resistance is deliberately derived from the same confirmed swing
points ``structure_engine`` already produces rather than inventing a second
notion of a "level": one definition of a swing, used everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import List, Optional, Sequence

from market_data import Candle
from structure_engine import SwingPoint, SwingType


@dataclass(frozen=True)
class Distance:
    """How far price is from a level, in three comparable units."""

    level: float
    absolute: float           # price units, always >= 0
    percent: float            # % of the reference price
    atr_multiple: Optional[float]   # None when ATR is unavailable

    @property
    def is_near(self) -> bool:
        """Within half an ATR — close enough that the level realistically
        interacts with an entry or stop. ``False`` when ATR is unknown
        rather than guessing from percent alone."""
        return self.atr_multiple is not None and self.atr_multiple <= 0.5


@dataclass(frozen=True)
class DayRange:
    """High/low of the most recent day present in the series."""

    high: float
    low: float

    @property
    def size(self) -> float:
        return self.high - self.low

    def position_of(self, price: float) -> Optional[float]:
        """Where ``price`` sits in the day's range, 0.0 = low, 1.0 = high.
        ``None`` for a zero-width range (a single flat candle)."""
        if self.size == 0:
            return None
        return (price - self.low) / self.size


@dataclass(frozen=True)
class LevelContext:
    """The complete level picture for one symbol at one moment."""

    day: Optional[DayRange]
    distance_to_day_high: Optional[Distance]
    distance_to_day_low: Optional[Distance]
    nearest_resistance: Optional[Distance]
    nearest_support: Optional[Distance]
    resistance_levels: tuple
    support_levels: tuple


def _distance(price: float, level: float, atr_value: Optional[float]) -> Distance:
    absolute = abs(price - level)
    return Distance(
        level=level,
        absolute=absolute,
        percent=(absolute / price * 100.0) if price else 0.0,
        atr_multiple=(absolute / atr_value) if atr_value else None,
    )


def day_range(candles: Sequence[Candle]) -> Optional[DayRange]:
    """High/low of the latest UTC calendar day in the series.

    Uses the last candle's date as "today" rather than the wall clock, so
    the result describes the data actually provided — important when
    candles arrive delayed or the series ends before the current day.
    """
    if not candles:
        return None
    last = candles[-1].timestamp
    last_utc = (last if last.tzinfo else last.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    today = last_utc.date()

    same_day = [
        c for c in candles
        if (c.timestamp if c.timestamp.tzinfo else c.timestamp.replace(tzinfo=timezone.utc))
        .astimezone(timezone.utc).date() == today
    ]
    if not same_day:
        return None
    return DayRange(high=max(c.high for c in same_day), low=min(c.low for c in same_day))


def support_resistance(
    swings: Sequence[SwingPoint], price: float, *, max_levels: int = 5
) -> tuple[List[float], List[float]]:
    """Split confirmed swing points into resistance (above) and support
    (below) ``price``, nearest first, capped at ``max_levels`` each.

    Swing highs above price act as resistance and swing lows below act as
    support — the conventional reading. A swing high that price has already
    traded above is no longer resistance, so it is simply excluded rather
    than reclassified.
    """
    resistance = sorted(
        {s.price for s in swings if s.type == SwingType.HIGH and s.price > price}
    )[:max_levels]
    support = sorted(
        {s.price for s in swings if s.type == SwingType.LOW and s.price < price},
        reverse=True,
    )[:max_levels]
    return support, resistance


def level_context(
    candles: Sequence[Candle],
    swings: Sequence[SwingPoint],
    price: float,
    atr_value: Optional[float] = None,
) -> LevelContext:
    """Assemble every level fact for one symbol at one price."""
    day = day_range(candles)
    support, resistance = support_resistance(swings, price)

    return LevelContext(
        day=day,
        distance_to_day_high=_distance(price, day.high, atr_value) if day else None,
        distance_to_day_low=_distance(price, day.low, atr_value) if day else None,
        nearest_resistance=_distance(price, resistance[0], atr_value) if resistance else None,
        nearest_support=_distance(price, support[0], atr_value) if support else None,
        resistance_levels=tuple(resistance),
        support_levels=tuple(support),
    )
