"""The analyst's view of the market: closed candles only, checked before use.

Rules, each a pure function of the candles and the analysis time ``as_of``:

1. Closed only. A bar is used once it has closed: ``open + duration <=
   as_of``. The provider's last bar is usually still forming and is dropped.
2. Weekend stubs dropped. For metals, a bar lying entirely inside the weekly
   closure (Friday 17:00 to Sunday 18:00 New York) is a provider stub quote
   that never traded (``market_hours``) and is removed.
3. On the grid. Every bar must open on its timeframe's grid (M5 on a
   multiple of 5 minutes, M15 of 15, H1 on the hour, H4 on the hour at
   00/04/... or 01/05/... UTC — the provider's H4 grid follows New York time
   and moves an hour with daylight saving).
4. Enough history: H4 >= 60 bars, H1 >= 100, M15 >= 100, M5 >= 60.
5. Fresh. The last closed bar must have closed within a lag measured in
   trading time, so the weekly closure does not count against it: M5 15 min,
   M15 45 min, H1 2 h, H4 8 h.
6. Aligned. Where two timeframes have a bar closing at the same instant, the
   two closes must agree within ``Rules.alignment_pct``. The live price must
   be within ``Rules.price_sanity_pct`` of the last M5 close.

The view's status is the first that applies: ``market_closed``,
``unavailable`` (a timeframe or the live price is missing, or too short),
``stale``, ``misaligned``, else ``fresh``. Only ``fresh`` data is analysed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

from market_data import Candle, Timeframe
from market_hours import bar_inside_weekly_closure, follows_weekly_session, in_weekly_closure, \
    new_york_offset

from analyst.rules import Rules

UTC = timezone.utc

TIMEFRAMES: Tuple[Timeframe, ...] = (Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5)
DURATION: Dict[Timeframe, timedelta] = {
    Timeframe.M5: timedelta(minutes=5), Timeframe.M15: timedelta(minutes=15),
    Timeframe.H1: timedelta(hours=1), Timeframe.H4: timedelta(hours=4),
}
MIN_BARS: Dict[Timeframe, int] = {
    Timeframe.H4: 60, Timeframe.H1: 100, Timeframe.M15: 100, Timeframe.M5: 60,
}
MAX_LAG: Dict[Timeframe, timedelta] = {
    Timeframe.M5: timedelta(minutes=15), Timeframe.M15: timedelta(minutes=45),
    Timeframe.H1: timedelta(hours=2), Timeframe.H4: timedelta(hours=8),
}


# ---------------------------------------------------------------- calendar

def _new_york_to_utc(local: datetime) -> datetime:
    """A naive New York wall-clock time as a UTC instant."""
    offset = new_york_offset(local.replace(tzinfo=UTC) + timedelta(hours=5))
    return (local - offset).replace(tzinfo=UTC)


def weekly_closures(start: datetime, end: datetime) -> Iterator[Tuple[datetime, datetime]]:
    """Every metals weekly closure [Friday 17:00, Sunday 18:00 New York) that
    overlaps [start, end), as UTC instants."""
    day: date = (start - timedelta(days=7)).date()
    friday = day + timedelta(days=(4 - day.weekday()) % 7)
    while True:
        closes = _new_york_to_utc(datetime.combine(friday, time(17, 0)))
        if closes >= end:
            return
        reopens = _new_york_to_utc(datetime.combine(friday + timedelta(days=2), time(18, 0)))
        if reopens > start:
            yield closes, reopens
        friday += timedelta(days=7)


def trading_time(start: datetime, end: datetime) -> timedelta:
    """Elapsed time between two instants, less any weekly closure inside it."""
    if end <= start:
        return timedelta(0)
    closed = sum(((min(end, b) - max(start, a)) for a, b in weekly_closures(start, end)),
                 timedelta(0))
    return end - start - closed


def next_weekly_close(after: datetime) -> datetime:
    for closes, _ in weekly_closures(after, after + timedelta(days=8)):
        if closes > after:
            return closes
    raise AssertionError("a weekly close always falls within 8 days")


def market_state(as_of: datetime, symbol: str, close_buffer: timedelta) -> str:
    """``closed`` inside the weekly closure, ``closing`` within ``close_buffer``
    of it, else ``open``. Non-metal symbols are always ``open`` here."""
    if not follows_weekly_session(symbol):
        return "open"
    if in_weekly_closure(as_of):
        return "closed"
    return "closing" if next_weekly_close(as_of) - as_of <= close_buffer else "open"


# ------------------------------------------------------------------ frames

def on_grid(ts: datetime, tf: Timeframe) -> bool:
    if ts.second or ts.microsecond:
        return False
    if tf is Timeframe.M5:
        return ts.minute % 5 == 0
    if tf is Timeframe.M15:
        return ts.minute % 15 == 0
    if tf is Timeframe.H1:
        return ts.minute == 0
    if tf is Timeframe.H4:
        return ts.minute == 0 and ts.hour % 4 in (0, 1)
    return False


def closed_bars(candles: Iterable[Candle], tf: Timeframe, as_of: datetime) -> Tuple[Candle, ...]:
    dur = DURATION[tf]
    return tuple(c for c in candles if c.timestamp + dur <= as_of)


def drop_closure_stubs(candles: Iterable[Candle], tf: Timeframe, symbol: str) -> Tuple[Candle, ...]:
    if not follows_weekly_session(symbol):
        return tuple(candles)
    dur = DURATION[tf]
    return tuple(c for c in candles if not bar_inside_weekly_closure(c.timestamp, dur))


@dataclass(frozen=True)
class Frame:
    timeframe: Timeframe
    candles: Tuple[Candle, ...]
    status: str                    # ok / unavailable / insufficient / off_grid / stale
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def last_close_at(self) -> Optional[datetime]:
        return self.candles[-1].timestamp + DURATION[self.timeframe] if self.candles else None


def prepare_frame(candles: Optional[Sequence[Candle]], tf: Timeframe, as_of: datetime,
                  symbol: str) -> Frame:
    if candles is None:
        return Frame(tf, (), "unavailable", f"{tf.value} candles unavailable")
    bars = drop_closure_stubs(closed_bars(candles, tf, as_of), tf, symbol)
    off = [c.timestamp for c in bars if not on_grid(c.timestamp, tf)]
    if off:
        return Frame(tf, bars, "off_grid", f"{tf.value} bar off its grid at {off[0].isoformat()}")
    if len(bars) < MIN_BARS[tf]:
        return Frame(tf, bars, "insufficient",
                     f"{tf.value}: {len(bars)} closed bars, {MIN_BARS[tf]} needed")
    lag = trading_time(bars[-1].timestamp + DURATION[tf], as_of)
    if lag > MAX_LAG[tf]:
        return Frame(tf, bars, "stale", f"{tf.value} last bar closed {_minutes(lag)} of trading "
                                        f"time ago (limit {_minutes(MAX_LAG[tf])})")
    return Frame(tf, bars, "ok")


def _minutes(delta: timedelta) -> str:
    return f"{delta.total_seconds() / 60:.0f} min"


# -------------------------------------------------------------------- view

@dataclass(frozen=True)
class MarketView:
    symbol: str
    as_of: datetime
    price: Optional[float]
    frames: Dict[Timeframe, Frame]
    market: str                    # open / closing / closed
    status: str                    # fresh / market_closed / unavailable / stale / misaligned
    problems: Tuple[str, ...] = ()

    @property
    def fresh(self) -> bool:
        return self.status == "fresh"

    def candles(self, tf: Timeframe) -> Tuple[Candle, ...]:
        return self.frames[tf].candles


def _close_at(candles: Sequence[Candle], tf: Timeframe, instant: datetime) -> Optional[float]:
    """Close of the bar in ``candles`` that closes exactly at ``instant``."""
    target = instant - DURATION[tf]
    for c in reversed(candles):
        if c.timestamp == target:
            return c.close
        if c.timestamp < target:
            return None
    return None


def alignment_problems(frames: Dict[Timeframe, Frame], price: Optional[float],
                       rules: Rules) -> Tuple[str, ...]:
    problems = []
    pairs = ((Timeframe.H4, Timeframe.H1), (Timeframe.H1, Timeframe.M15),
             (Timeframe.M15, Timeframe.M5))
    for coarse, fine in pairs:
        a, b = frames[coarse].candles, frames[fine].candles
        if not a or not b:
            continue
        instant = a[-1].timestamp + DURATION[coarse]
        fine_close = _close_at(b, fine, instant)
        if fine_close is not None and \
                abs(a[-1].close - fine_close) > a[-1].close * rules.alignment_pct / 100:
            problems.append(f"{coarse.value} close {a[-1].close:.2f} and {fine.value} close "
                            f"{fine_close:.2f} disagree at {instant:%Y-%m-%d %H:%M}")
    m5 = frames[Timeframe.M5].candles
    if price is not None and m5 and abs(price - m5[-1].close) > price * rules.price_sanity_pct / 100:
        problems.append(f"live price {price:.2f} is more than {rules.price_sanity_pct}% from the "
                        f"last M5 close {m5[-1].close:.2f}")
    return tuple(problems)


def build_view(symbol: str, as_of: datetime, price: Optional[float],
               candles_by_tf: Dict[Timeframe, Optional[Sequence[Candle]]], rules: Rules) -> MarketView:
    symbol = symbol.strip().upper()
    frames = {tf: prepare_frame(candles_by_tf.get(tf), tf, as_of, symbol) for tf in TIMEFRAMES}
    market = market_state(as_of, symbol, rules.close_buffer)
    notes = tuple(f.note for f in frames.values() if f.note)

    def view(status: str, problems: Tuple[str, ...] = notes) -> MarketView:
        return MarketView(symbol, as_of, price, frames, market, status, problems)

    if market == "closed":
        return view("market_closed", ("market is in the weekly closure",))
    if any(f.status in ("unavailable", "insufficient", "off_grid") for f in frames.values()):
        return view("unavailable")
    if price is None:
        return view("unavailable", notes + ("no live price",))
    if any(f.status == "stale" for f in frames.values()):
        return view("stale")
    problems = alignment_problems(frames, price, rules)
    if problems:
        return view("misaligned", problems)
    return view("fresh", ())
