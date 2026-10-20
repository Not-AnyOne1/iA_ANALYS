"""Synthetic, internally consistent XAUUSD candles for the analyst's tests.

A price path is written as M5 closes; M15, H1 and H4 bars are aggregated
from those M5 bars, so every timeframe agrees with every other by
construction. No bar is generated inside the metals weekly closure. All
prices here are TEST DATA, not market data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence, Tuple

from market_data import Candle, Timeframe
from market_hours import in_weekly_closure

UTC = timezone.utc
M5 = timedelta(minutes=5)
START = datetime(2026, 3, 2, 0, 0, tzinfo=UTC)          # a Monday


def legs(start: float, moves: Sequence[Tuple[int, float]]) -> List[float]:
    """Closes moving linearly: each (bars, delta) adds ``bars`` closes."""
    closes, price = [], start
    for bars, delta in moves:
        for i in range(bars):
            closes.append(round(price + delta * (i + 1) / bars, 4))
        price += delta
    return closes


def trend(start: float, slow_cycles: int, direction: int = 1) -> List[float]:
    """A trending background with swings on every timeframe: 4-day cycles
    of six 12-hour up-cycles then two 12-hour down-cycles (mirrored for a
    downtrend)."""
    up = [(96, 40.0), (48, -15.0)] * 6
    down = [(48, 10.0), (96, -40.0)] * 2
    moves = (up + down) * slow_cycles
    return legs(start, [(n, d * direction) for n, d in moves])


def m5_bars(opening: float, closes: Sequence[float], start: datetime = START,
            wick: float = 0.05) -> Tuple[Candle, ...]:
    bars, ts, prev = [], start, opening
    for close in closes:
        while in_weekly_closure(ts) or in_weekly_closure(ts + M5 - timedelta(microseconds=1)):
            ts += M5
        o, c = prev, close
        # The wick extends the bar's own move only, so the bar after a
        # turning point never ties the extreme and fractals stay strict.
        bars.append(Candle(ts, o, max(o, c) + (wick if c >= o else 0.0),
                           min(o, c) - (wick if c <= o else 0.0), c))
        prev, ts = close, ts + M5
    return tuple(bars)


def aggregate(bars: Sequence[Candle], tf: Timeframe) -> Tuple[Candle, ...]:
    minutes = {Timeframe.M15: 15, Timeframe.H1: 60, Timeframe.H4: 240}[tf]
    groups: Dict[datetime, List[Candle]] = {}
    for b in bars:
        epoch = int(b.timestamp.timestamp() // 60)
        key = datetime.fromtimestamp((epoch - epoch % minutes) * 60, UTC)
        groups.setdefault(key, []).append(b)
    return tuple(Candle(k, g[0].open, max(c.high for c in g), min(c.low for c in g), g[-1].close)
                 for k, g in sorted(groups.items()))


def frames(m5: Sequence[Candle], m5_window: int = 300) -> Dict[Timeframe, Tuple[Candle, ...]]:
    """All four timeframes, each trimmed to the window the live bot fetches."""
    out = {tf: aggregate(m5, tf)[-300:] for tf in (Timeframe.M15, Timeframe.H1, Timeframe.H4)}
    out[Timeframe.M5] = tuple(m5[-m5_window:])
    return out


def as_of(m5: Sequence[Candle]) -> datetime:
    """The instant the last M5 bar closed."""
    return m5[-1].timestamp + M5


# ------------------------------------------------------------------ scenarios
# Each returns M5 closes; the opening price is 2000 (3000 for the mirrored
# downtrend). Verified shapes, used by the strategy and verdict tests:
#
#   b_long            H4/H1 bullish, M15 pullback 48% into a 20-point leg, an
#                     M5 CHoCH up on the last bar -> strategy B long READY
#   b_long_unconfirmed  the same, stopped at the pullback low -> B WAITING
#   c_long            an M15 swing low swept and reclaimed, then an M5 break
#                     up -> strategy C long READY
#   b_short           b_long mirrored -> strategy B short READY

_UP_CYCLES = [(96, 40.0), (48, -15.0)] * 3
_B_TAIL = [(24, 20.0), (8, -7.0), (3, 1.5), (5, -4.0)]
_C_TAIL = [(24, 25.0), (12, -20.0), (6, 3.0), (2, -9.5), (1, 2.5), (3, 2.0), (3, -1.0), (3, 2.5)]


def _scenario(tail, direction: int = 1, start: float = 2000.0) -> List[float]:
    background = trend(start, 3, direction)
    return background + legs(background[-1], [(n, d * direction) for n, d in _UP_CYCLES + tail])


def b_long() -> List[float]:
    return _scenario(_B_TAIL + [(4, 4.5)])


def b_long_unconfirmed() -> List[float]:
    return _scenario(_B_TAIL)


def c_long() -> List[float]:
    return _scenario(_C_TAIL)


def b_short() -> List[float]:
    return _scenario(_B_TAIL + [(4, 4.5)], direction=-1, start=3000.0)


def scenario(name: str, opening: float = None):
    """(m5 bars, frames, as_of, last price) for a named scenario."""
    closes = globals()[name]()
    start = 3000.0 if name.endswith("short") else 2000.0
    bars = m5_bars(opening if opening is not None else start, closes)
    return bars, frames(bars), as_of(bars), bars[-1].close
