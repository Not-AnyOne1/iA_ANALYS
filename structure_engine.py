"""Structure Engine (RFC-005): deterministic market structure analysis.

Turns an already-fetched ``market_data.CandleSeries`` into market
structure: confirmed swing highs/lows, an overall trend classification,
and a chronological list of BOS (Break of Structure) / CHoCH (Change of
Character) events.

Pure and deterministic: no I/O, no async, no AI or probabilistic logic, no
randomness. The exact same input always produces the exact same output.
This is the structural-facts layer later engines (SMC, Scoring, Claude
Decision) will build on — it does not score, rank, or interpret anything,
only detects it. Out of scope entirely, by design: order blocks, FVGs,
liquidity, premium/discount, OTE, scoring, and any wiring into Claude,
the Telegram bot, or the analysis pipeline.

Detection rules, stated precisely (every one is a pure function of price
data already present in the input ``CandleSeries``):

1. Swing points (fractals) — a candle at index i is a *confirmed* swing
   high iff its high is strictly greater than every candle's high within
   ``swing_lookback`` positions on BOTH sides (a candle is a confirmed
   swing low symmetrically, using lows and "strictly less than"). Default
   ``swing_lookback=2`` -> a classic 5-bar fractal (2 left + itself + 2
   right). A swing within ``swing_lookback`` candles of either end of the
   series can never be confirmed, since the candles needed to confirm it
   don't exist in the given data — only *confirmed* fractals are ever
   used anywhere in this module.

2. Swing strength — the largest N (>= swing_lookback) for which the same
   candle still qualifies as an N-bar fractal, capped by the series
   bounds. A larger strength means the swing held up against more
   surrounding candles, i.e. a more pronounced extreme. Purely arithmetic,
   derived from swing_lookback and the price data — no extra configuration.

3. Trend — computed only from confirmed swings (never from raw candles and
   never from BOS/CHoCH events), using the classic higher-high/higher-low
   vs. lower-high/lower-low comparison between the two most recent
   confirmed swings of each type:
     - BULLISH: last swing high > previous swing high, AND
                last swing low  > previous swing low
     - BEARISH: last swing high < previous swing high, AND
                last swing low  < previous swing low
     - RANGING: neither pattern holds (mixed signals)
     - UNKNOWN: fewer than 2 confirmed swing highs or fewer than 2
                confirmed swing lows exist — not enough data to judge
   This one rule, applied to whichever swings are confirmed as of a given
   point in the series, is reused to classify each BOS/CHoCH event below —
   there is exactly one trend rule in this module, evaluated at different
   points in time, never two competing definitions.

4. BOS / CHoCH — detected with a single forward scan over candle CLOSES
   only; wicks (high/low) are never used to detect a break, per RFC-005's
   requirement. At each candle, the most recently confirmed swing high/low
   *so far* (only swings already confirmed by that candle's index are
   visible — no lookahead) is the active reference:
     - close > active swing high -> a bullish break
     - close < active swing low  -> a bearish break
   A break is classified using rule 3 applied to every swing confirmed up
   to and including the breaking candle (confirmation and the break check
   both use exactly the data available at that candle, never future data):
     - bullish break while that trend is BEARISH -> CHoCH
     - bullish break otherwise (BULLISH/RANGING/UNKNOWN) -> BOS
     - bearish break while that trend is BULLISH -> CHoCH
     - bearish break otherwise (BEARISH/RANGING/UNKNOWN) -> BOS
   Rationale for RANGING/UNKNOWN -> BOS: CHoCH means reversing an
   established opposing trend; with no trend established yet there is
   nothing to reverse away from, so the break is simply the market
   asserting a new direction rather than changing an existing one.
   Once a swing has triggered a break event it is retired (cleared) so it
   cannot re-trigger on every subsequent candle that stays beyond it — the
   next confirmed swing of that type becomes the new active reference.
   If a single candle's close would satisfy both the high-break and the
   low-break condition at once (only possible with an internally
   inconsistent active_high < active_low, which valid OHLC data assembled
   via market_data.py's own validation should never produce), the bullish
   check takes priority — a deterministic, documented tie-break, not a
   scenario expected with valid data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from market_data import Candle, CandleSeries, Timeframe


class SwingType(str, Enum):
    HIGH = "high"
    LOW = "low"


class TrendDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class StructureEventType(str, Enum):
    BOS = "bos"
    CHOCH = "choch"


@dataclass(frozen=True)
class SwingPoint:
    """A confirmed fractal swing point (rules 1-2 above)."""

    index: int
    timestamp: datetime
    price: float
    type: SwingType
    strength: int


@dataclass(frozen=True)
class StructureEvent:
    """A confirmed break of a swing point, using candle closes only (rule 4)."""

    type: StructureEventType
    direction: TrendDirection  # always BULLISH or BEARISH, never RANGING/UNKNOWN
    timestamp: datetime
    candle_index: int
    break_price: float
    broken_swing: SwingPoint


@dataclass(frozen=True)
class StructureAnalysis:
    """The full structural picture derived from one CandleSeries."""

    symbol: str
    timeframe: Timeframe
    trend: TrendDirection
    swing_points: Tuple[SwingPoint, ...]
    events: Tuple[StructureEvent, ...]
    last_event: Optional[StructureEvent]


class StructureEngine:
    """Deterministic structure detector.

    Stateless across calls — ``swing_lookback`` is the only configuration,
    fixed at construction, and :meth:`analyze` never mutates anything it's
    given or keeps state between calls.
    """

    def __init__(self, swing_lookback: int = 2) -> None:
        if swing_lookback < 1:
            raise ValueError(f"swing_lookback must be >= 1, got {swing_lookback}")
        self._swing_lookback = swing_lookback

    def analyze(self, series: CandleSeries) -> StructureAnalysis:
        candles = series.candles
        swings = self._detect_swings(candles)
        trend = self._trend_from_swings(swings)
        events = self._detect_events(candles, swings)

        return StructureAnalysis(
            symbol=series.symbol,
            timeframe=series.timeframe,
            trend=trend,
            swing_points=tuple(swings),
            events=tuple(events),
            last_event=events[-1] if events else None,
        )

    # ------------------------------------------------------------- swings

    def _detect_swings(self, candles: Tuple[Candle, ...]) -> List[SwingPoint]:
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        lookback = self._swing_lookback

        swings: List[SwingPoint] = []
        for i in range(len(candles)):
            high_radius = self._fractal_radius(highs, i, is_high=True)
            if high_radius >= lookback:
                swings.append(SwingPoint(
                    index=i, timestamp=candles[i].timestamp, price=highs[i],
                    type=SwingType.HIGH, strength=high_radius,
                ))
            low_radius = self._fractal_radius(lows, i, is_high=False)
            if low_radius >= lookback:
                swings.append(SwingPoint(
                    index=i, timestamp=candles[i].timestamp, price=lows[i],
                    type=SwingType.LOW, strength=low_radius,
                ))

        swings.sort(key=lambda s: s.index)
        return swings

    @staticmethod
    def _fractal_radius(values: Sequence[float], i: int, *, is_high: bool) -> int:
        """Largest N such that ``values[i]`` is strictly more extreme
        (greater for highs, lesser for lows) than every value within N
        positions on both sides, bounded by the sequence length. Returns 0
        if ``values[i]`` isn't even a 1-bar local extreme."""
        n = len(values)
        radius = 0
        while True:
            candidate = radius + 1
            left, right = i - candidate, i + candidate
            if left < 0 or right >= n:
                return radius
            if is_high:
                more_extreme = values[i] > values[left] and values[i] > values[right]
            else:
                more_extreme = values[i] < values[left] and values[i] < values[right]
            if not more_extreme:
                return radius
            radius = candidate

    # -------------------------------------------------------------- trend

    @staticmethod
    def _trend_from_swings(swings: Sequence[SwingPoint]) -> TrendDirection:
        """Rule 3 from the module docstring, applied to whichever swings
        are passed in — all of them for the final trend, a prefix of them
        when classifying a specific break event below."""
        highs = [s for s in swings if s.type == SwingType.HIGH]
        lows = [s for s in swings if s.type == SwingType.LOW]
        if len(highs) < 2 or len(lows) < 2:
            return TrendDirection.UNKNOWN

        higher_high = highs[-1].price > highs[-2].price
        higher_low = lows[-1].price > lows[-2].price
        lower_high = highs[-1].price < highs[-2].price
        lower_low = lows[-1].price < lows[-2].price

        if higher_high and higher_low:
            return TrendDirection.BULLISH
        if lower_high and lower_low:
            return TrendDirection.BEARISH
        return TrendDirection.RANGING

    # ------------------------------------------------------------- events

    def _detect_events(
        self, candles: Tuple[Candle, ...], swings: List[SwingPoint]
    ) -> List[StructureEvent]:
        lookback = self._swing_lookback
        confirmed_at: Dict[int, List[SwingPoint]] = {}
        for swing in swings:
            confirmed_at.setdefault(swing.index + lookback, []).append(swing)

        events: List[StructureEvent] = []
        confirmed_so_far: List[SwingPoint] = []
        active_high: Optional[SwingPoint] = None
        active_low: Optional[SwingPoint] = None

        for j, candle in enumerate(candles):
            for swing in confirmed_at.get(j, []):
                if swing.type == SwingType.HIGH:
                    active_high = swing
                else:
                    active_low = swing
                confirmed_so_far.append(swing)

            close = candle.close
            if active_high is not None and close > active_high.price:
                trend_before = self._trend_from_swings(confirmed_so_far)
                event_type = (
                    StructureEventType.CHOCH
                    if trend_before == TrendDirection.BEARISH
                    else StructureEventType.BOS
                )
                events.append(StructureEvent(
                    type=event_type, direction=TrendDirection.BULLISH,
                    timestamp=candle.timestamp, candle_index=j,
                    break_price=close, broken_swing=active_high,
                ))
                active_high = None
            elif active_low is not None and close < active_low.price:
                trend_before = self._trend_from_swings(confirmed_so_far)
                event_type = (
                    StructureEventType.CHOCH
                    if trend_before == TrendDirection.BULLISH
                    else StructureEventType.BOS
                )
                events.append(StructureEvent(
                    type=event_type, direction=TrendDirection.BEARISH,
                    timestamp=candle.timestamp, candle_index=j,
                    break_price=close, broken_swing=active_low,
                ))
                active_low = None

        return events
