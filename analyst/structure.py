"""Per-timeframe market facts for the independent analyst.

Everything here is read from CLOSED bars (``analyst.data``) with the live
``StructureEngine`` (confirmed 5-bar fractals; BOS/CHoCH from closes only,
each break judged with the swings confirmed by that bar — no future bar).
On top of it, stated precisely:

* **Bias** of a timeframe: the direction of its last BOS/CHoCH, provided the
  last close is on that side of the EMA-50; otherwise ``neutral``. No break
  in the window, or too few bars for the EMA, is ``neutral``.
* **Trend** is the engine's own higher-high/higher-low classification,
  reported as context.
* **Liquidity level**: a confirmed swing. A swing high holds buy-side
  liquidity above it, a swing low sell-side liquidity below it. A level is
  *swept* once any later bar trades beyond it; before that it is *resting*.
  Bars of the level's own timeframe are checked, then M5 bars after the last
  of them, so a level is never called resting after price has been through.
* **Sweep** (strategy C): the first M15 bar to trade beyond a resting level
  closes back on the level's side, no deeper than ``sweep_max_depth_atr`` M15
  ATRs, within the last ``sweep_max_age_bars`` bars; no later bar has traded
  beyond the sweep's extreme and the last close is still on the level's side.
  Levels eligible: H1 swings, and M15 swings of strength >=
  ``major_swing_strength``.
* **Pullback** (strategy B): the last confirmed M15 swing high H and the last
  swing low L before it form the up-leg (mirror for a down-leg); it must be
  at least ``min_leg_atr`` M15 ATRs. Since H, price has retraced between
  ``pullback_min`` and ``pullback_max`` of the leg and the last close is
  still below H.
* **M5 break**: an M5 BOS or CHoCH in the trade's direction whose bar opened
  at or after the setup's extreme and lies within the last
  ``m5_confirm_max_age_bars`` M5 bars.
* **Target**: the nearest resting level beyond the entry — any H4 or H1
  swing, or an M15 swing of strength >= ``target_swing_strength``. A
  continuation (B) targets the nearest one beyond its impulse swing — the
  thesis is that price runs that swing — and, when none rests there, the
  leg's 1.272 extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import indicators
from market_data import Candle, CandleSeries, Timeframe
from structure_engine import StructureEngine, StructureEvent, SwingPoint, SwingType, TrendDirection

from analyst.data import DURATION, MarketView
from analyst.rules import Rules

LONG, SHORT = "long", "short"


def side_sign(direction: str) -> float:
    return 1.0 if direction == LONG else -1.0


@dataclass(frozen=True)
class FrameRead:
    timeframe: Timeframe
    candles: Tuple[Candle, ...]
    trend: str                       # bullish / bearish / ranging / unknown
    bias: str                        # bullish / bearish / neutral
    last_event: Optional[StructureEvent]
    events: Tuple[StructureEvent, ...]
    swings: Tuple[SwingPoint, ...]
    ema: Optional[float]
    atr: Optional[float]

    @property
    def close(self) -> float:
        return self.candles[-1].close

    def describe(self) -> str:
        if self.last_event is None:
            return f"{self.bias} (no break of structure in the window)"
        e = self.last_event
        arrow = "up" if e.direction is TrendDirection.BULLISH else "down"
        ema = f", close {'above' if self.close > (self.ema or 0) else 'below'} EMA50" \
            if self.ema is not None else ""
        return (f"{self.bias} (last {e.type.value.upper()} {arrow} through "
                f"{e.broken_swing.price:.2f} at {e.timestamp:%m-%d %H:%M}{ema}; "
                f"swings {self.trend})")


def read_frame(candles: Sequence[Candle], tf: Timeframe, rules: Rules, symbol: str = "XAUUSD") -> FrameRead:
    candles = tuple(candles)
    analysis = StructureEngine(rules.swing_lookback).analyze(
        CandleSeries(symbol=symbol, timeframe=tf, provider="analyst", candles=candles))
    closes = [c.close for c in candles]
    ema = indicators.ema(closes, rules.bias_ema)
    last = analysis.last_event
    bias = "neutral"
    if last is not None and ema is not None:
        if last.direction is TrendDirection.BULLISH and closes[-1] > ema:
            bias = "bullish"
        elif last.direction is TrendDirection.BEARISH and closes[-1] < ema:
            bias = "bearish"
    return FrameRead(tf, candles, analysis.trend.value, bias, last, analysis.events,
                     analysis.swing_points, ema, indicators.atr(candles, rules.atr_period))


# ------------------------------------------------------------- liquidity

@dataclass(frozen=True)
class Level:
    price: float
    side: str                        # "high" / "low"
    timeframe: Timeframe
    index: int                       # swing bar index in its own timeframe's candles
    formed_at: datetime              # swing bar open
    confirmed_at: datetime           # close of the bar that confirmed the fractal
    strength: int
    kind: str = "swing"              # "swing", or a computed level's name

    def label(self) -> str:
        if self.kind == "swing":
            return f"{self.timeframe.value} swing {self.side} {self.price:.2f}"
        return f"{self.kind} {self.price:.2f}"


def levels(read: FrameRead, rules: Rules, min_strength: int = 0) -> Tuple[Level, ...]:
    dur = DURATION[read.timeframe]
    out = []
    for s in read.swings:
        if s.strength < max(min_strength, rules.swing_lookback):
            continue
        confirm = read.candles[s.index + rules.swing_lookback]
        out.append(Level(s.price, s.type.value, read.timeframe, s.index,
                         read.candles[s.index].timestamp, confirm.timestamp + dur, s.strength))
    return tuple(out)


def _beyond(level: Level, bar: Candle) -> bool:
    return bar.high > level.price if level.side == "high" else bar.low < level.price


def is_resting(level: Level, own: Sequence[Candle], m5: Sequence[Candle]) -> bool:
    """No bar after the swing has traded beyond it (own timeframe, then M5)."""
    if any(_beyond(level, c) for c in own[level.index + 1:]):
        return False
    after = own[-1].timestamp + DURATION[level.timeframe] if own else level.confirmed_at
    return not any(_beyond(level, c) for c in m5 if c.timestamp >= after)


class MarketRead:
    """Every timeframe of one fresh MarketView, each read on first use.

    ``reader`` lets a caller supply its own (for example cached) frame
    reader; it must return exactly what :func:`read_frame` would.
    """

    def __init__(self, view: MarketView, rules: Rules,
                 reader: Optional[Callable[[Tuple[Candle, ...], Timeframe], FrameRead]] = None) -> None:
        self.view, self.rules = view, rules
        self._reader = reader or (lambda candles, tf: read_frame(candles, tf, rules, view.symbol))
        self._frames: Dict[Timeframe, FrameRead] = {}

    def frame(self, tf: Timeframe) -> FrameRead:
        if tf not in self._frames:
            self._frames[tf] = self._reader(self.view.candles(tf), tf)
        return self._frames[tf]

    @property
    def atr15(self) -> float:
        value = self.frame(Timeframe.M15).atr
        assert value is not None
        return value

    @property
    def price(self) -> float:
        assert self.view.price is not None
        return self.view.price

    def target(self, direction: str, entry: float, extra: Iterable[Level] = (),
               beyond: Optional[float] = None) -> Optional[Level]:
        """Nearest resting level beyond ``entry`` (and beyond ``beyond``, when
        given) in the trade's direction; ``extra`` levels join the pool."""
        side = "high" if direction == LONG else "low"
        sign = side_sign(direction)
        floor = entry if beyond is None else (max(entry, beyond) if direction == LONG
                                              else min(entry, beyond))
        m5 = self.view.candles(Timeframe.M5)
        ahead: List[Level] = [lv for lv in extra if lv.side == side and sign * (lv.price - floor) > 0]
        for tf, strength in ((Timeframe.H4, 0), (Timeframe.H1, 0),
                             (Timeframe.M15, self.rules.target_swing_strength)):
            frame = self.frame(tf)
            ahead += [lv for lv in levels(frame, self.rules, strength)
                      if lv.side == side and sign * (lv.price - floor) > 0
                      and is_resting(lv, frame.candles, m5)]
        return min(ahead, key=lambda lv: (sign * (lv.price - entry), lv.timeframe.value))             if ahead else None

    def nearest_swing_beyond(self, direction: str, entry: float) -> Optional[SwingPoint]:
        """The most recent confirmed M15 swing on the stop side of ``entry``:
        the swing low below a long, the swing high above a short."""
        kind = SwingType.LOW if direction == LONG else SwingType.HIGH
        sign = side_sign(direction)
        for s in reversed(self.frame(Timeframe.M15).swings):
            if s.type is kind and sign * (entry - s.price) > 0:
                return s
        return None


def read_market(view: MarketView, rules: Rules, reader=None) -> Optional[MarketRead]:
    """None unless the view is fresh and the M15 ATR is defined."""
    if not view.fresh:
        return None
    read = MarketRead(view, rules, reader)
    atr15 = read.frame(Timeframe.M15).atr
    if not atr15 or atr15 <= 0:
        return None
    return read


# ---------------------------------------------------------------- setups

@dataclass(frozen=True)
class Pullback:
    direction: str
    origin: SwingPoint               # L of an up-leg (H of a down-leg)
    impulse: SwingPoint              # H of an up-leg (L of a down-leg)
    extreme: float                   # deepest price since the impulse swing
    extreme_at: datetime
    depth: float                     # fraction of the leg retraced


def find_pullback(read: MarketRead, direction: str) -> Optional[Pullback]:
    m15, rules = read.frame(Timeframe.M15), read.rules
    impulse_kind = SwingType.HIGH if direction == LONG else SwingType.LOW
    origin_kind = SwingType.LOW if direction == LONG else SwingType.HIGH
    impulse = next((s for s in reversed(m15.swings) if s.type is impulse_kind), None)
    if impulse is None:
        return None
    origin = next((s for s in reversed(m15.swings)
                   if s.type is origin_kind and s.index < impulse.index), None)
    if origin is None:
        return None
    leg = abs(impulse.price - origin.price)
    if leg < rules.min_leg_atr * read.atr15:
        return None
    after = m15.candles[impulse.index + 1:]
    if not after:
        return None
    if direction == LONG:
        k = min(range(len(after)), key=lambda i: (after[i].low, i))
        extreme = after[k].low
        depth = (impulse.price - extreme) / leg
        still_inside = after[-1].close < impulse.price
    else:
        k = max(range(len(after)), key=lambda i: (after[i].high, -i))
        extreme = after[k].high
        depth = (extreme - impulse.price) / leg
        still_inside = after[-1].close > impulse.price
    if not (rules.pullback_min <= depth < rules.pullback_max) or not still_inside:
        return None
    return Pullback(direction, origin, impulse, extreme, after[k].timestamp, round(depth, 4))


@dataclass(frozen=True)
class Sweep:
    direction: str                   # the reversal's direction: a swept low -> long
    level: Level
    bar: Candle
    extreme: float                   # the sweep bar's wick beyond the level


def find_sweep(read: MarketRead, direction: str) -> Optional[Sweep]:
    rules = read.rules
    m15 = read.frame(Timeframe.M15).candles
    m5 = read.view.candles(Timeframe.M5)
    side = "low" if direction == LONG else "high"
    pool = [lv for lv in levels(read.frame(Timeframe.H1), rules)
            + levels(read.frame(Timeframe.M15), rules, rules.major_swing_strength)
            if lv.side == side]
    first_recent = max(0, len(m15) - rules.sweep_max_age_bars)
    recent = m15[first_recent:]
    if not recent:
        return None
    reach = min(c.low for c in recent) if direction == LONG else max(c.high for c in recent)
    found: List[Tuple[int, Level]] = []
    for lv in pool:
        if side_sign(direction) * (lv.price - reach) <= 0:
            continue                         # nothing in the window traded beyond it
        start = lv.formed_at + DURATION[lv.timeframe]
        k = next((i for i, c in enumerate(m15) if c.timestamp >= start and _beyond(lv, c)), None)
        if k is None or k < first_recent or m15[k].timestamp < lv.confirmed_at:
            continue
        bar = m15[k]
        if direction == LONG:
            ok = bar.close > lv.price and lv.price - bar.low <= rules.sweep_max_depth_atr * read.atr15 \
                and all(c.low >= bar.low for c in m15[k + 1:]) and m15[-1].close > lv.price \
                and all(c.low >= bar.low for c in m5 if c.timestamp >= bar.timestamp + DURATION[Timeframe.M15])
        else:
            ok = bar.close < lv.price and bar.high - lv.price <= rules.sweep_max_depth_atr * read.atr15 \
                and all(c.high <= bar.high for c in m15[k + 1:]) and m15[-1].close < lv.price \
                and all(c.high <= bar.high for c in m5 if c.timestamp >= bar.timestamp + DURATION[Timeframe.M15])
        if ok:
            found.append((k, lv))
    if not found:
        return None
    # The most recent sweep; on one bar, the strongest level, then the one
    # furthest beyond (the larger pool of resting orders).
    k, lv = max(found, key=lambda f: (f[0], f[1].strength, -side_sign(direction) * f[1].price))
    bar = m15[k]
    return Sweep(direction, lv, bar, bar.low if direction == LONG else bar.high)


def m5_break(read: MarketRead, direction: str, since: datetime) -> Optional[StructureEvent]:
    m5 = read.frame(Timeframe.M5)
    want = TrendDirection.BULLISH if direction == LONG else TrendDirection.BEARISH
    first_recent = len(m5.candles) - read.rules.m5_confirm_max_age_bars
    for e in reversed(m5.events):
        if e.candle_index < first_recent:
            return None
        if e.direction is want and e.timestamp >= since:
            return e
    return None


def m5_trigger(read: MarketRead, direction: str) -> Optional[float]:
    """The M5 swing an M5 close must break to confirm: the most recent
    confirmed swing high above price (long) / low below it (short) that no
    close has broken since it formed."""
    m5 = read.frame(Timeframe.M5)
    kind = SwingType.HIGH if direction == LONG else SwingType.LOW
    sign = side_sign(direction)
    for s in reversed(m5.swings):
        if s.type is not kind:
            continue
        closes = [c.close for c in m5.candles[s.index + 1:]]
        if sign * (s.price - m5.close) > 0 and not any(sign * (c - s.price) > 0 for c in closes):
            return s.price
    return None
