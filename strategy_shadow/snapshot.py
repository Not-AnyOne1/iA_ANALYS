"""The market snapshot behind a decision: its fingerprint and its quality.

The decisions are made on the live context's candles exactly as production
saw them (forming bar and weekend stubs included — that is production's
data policy, and the comparison must be on the same inputs). This module
does not change those inputs; it records what they were and how good they
were, so the report can separate decisions made on fresh data from the rest.

**UTC.** The live provider labels candles in the account zone (measured
+11 h) while calling them UTC. To judge freshness the shift is measured
against a small UTC-labelled M15 request (``LabelCalibrator``) — at most one
request per ``refresh`` (6 h), shared by every signal — and undone for the
checks below only.

Recorded per timeframe (M15, H1, H4; D1 bar counts only):

* bars served, whether the last one is still forming;
* weekend stub bars in the window (a bar wholly inside the metals closure);
* weekday gaps: bars missing between the first and last bar, closure excluded;
* age of the last closed bar, in trading time.

Status, first that applies: ``market_closed``; ``incomplete`` (a timeframe
missing, or no live price); ``unverified`` (the label shift could not be
measured); ``stale`` (last closed M15 older than 45 min, H1 2 h, H4 8 h of
trading time, or the live price more than 0.5% from the latest bar); else
``fresh``. Duplicated or out-of-order bars never get this far:
``MarketDataService`` rejects them.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from market_data import Candle, MarketDataError, MarketDataService, Timeframe
from market_hours import bar_inside_weekly_closure, in_weekly_closure

from analyst.data import label_offset, trading_time

log = logging.getLogger(__name__)

UTC = timezone.utc
FRAMES = (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)
DUR = {Timeframe.M15: timedelta(minutes=15), Timeframe.H1: timedelta(hours=1),
       Timeframe.H4: timedelta(hours=4), Timeframe.D1: timedelta(days=1)}
MAX_AGE = {Timeframe.M15: timedelta(minutes=45), Timeframe.H1: timedelta(hours=2),
           Timeframe.H4: timedelta(hours=8)}
PRICE_DEVIATION_PCT = 0.5
UNITS_BAND_PCT = 10.0


def fingerprint(context) -> str:
    """A hash of everything the decisions read from the market: the quote
    and every candle of every timeframe, plus the decision time."""
    digest = hashlib.sha256()
    digest.update(f"{context.symbol}|{context.generated_at.isoformat()}|{context.current_price!r}".encode())
    for tf in sorted(getattr(context, "candle_series", {}), key=lambda t: t.value):
        for c in context.candle_series[tf].candles:
            digest.update(f"|{tf.value}{c.timestamp.isoformat()},{c.open!r},{c.high!r},{c.low!r},{c.close!r}".encode())
    return digest.hexdigest()[:24]


class LabelCalibrator:
    """Measures the live provider's label shift against UTC-labelled M15
    bars; cached for ``refresh``. Never raises."""

    def __init__(self, utc_market_data: Optional[MarketDataService], *,
                 refresh: timedelta = timedelta(hours=6), symbol: str = "XAUUSD",
                 timeout: float = 20.0) -> None:
        self._market_data, self._refresh, self._symbol, self._timeout = (
            utc_market_data, refresh, symbol, timeout)
        self._cached: Optional[timedelta] = None
        self._measured_at: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self.requests = 0

    async def offset(self, live_m15: List[Candle], now: datetime, *, force: bool = False
                     ) -> Optional[timedelta]:
        if self._market_data is None or not live_m15:
            return None
        async with self._lock:
            fresh = self._measured_at is not None and now - self._measured_at < self._refresh
            if self._cached is not None and fresh and not force:
                return self._cached
            try:
                self.requests += 1
                # 300 bars (one credit, like 64) so that real bars survive
                # a weekend: the closure's stub quotes are flat and would
                # match at many offsets, so they are dropped from the reference.
                series = await asyncio.wait_for(
                    self._market_data.get_candles(self._symbol, Timeframe.M15, 300), self._timeout)
            except (MarketDataError, asyncio.TimeoutError) as exc:
                # The shift is the account zone's UTC offset: it changes only
                # with that zone's daylight saving, so an older measurement is
                # still the best answer available.
                log.warning("strategy_shadow_calibration_failed error=%s", exc)
                return self._cached
            reference = [c for c in series.candles
                         if not bar_inside_weekly_closure(c.timestamp, DUR[Timeframe.M15])]
            delta = label_offset(live_m15, Timeframe.M15, reference, reference_tf=Timeframe.M15)
            if delta is None:
                # Ambiguous or too few real bars (just after the reopen): the
                # last measured shift is still the account zone's offset.
                log.warning("strategy_shadow_label_offset_unknown cached=%s", self._cached)
                return self._cached
            self._cached, self._measured_at = delta, now
            return delta


@dataclass
class DataQuality:
    status: str
    label_offset_minutes: Optional[int]
    frames: Dict[str, Dict[str, object]] = field(default_factory=dict)
    quote_vs_last_bar_pct: Optional[float] = None
    problems: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"status": self.status, "label_offset_minutes": self.label_offset_minutes,
                "frames": self.frames, "quote_vs_last_bar_pct": self.quote_vs_last_bar_pct,
                "problems": self.problems}


def _frame_quality(candles, tf: Timeframe, offset: Optional[timedelta], now: datetime) -> Dict[str, object]:
    out: Dict[str, object] = {"bars": len(candles)}
    if not candles or offset is None or tf is Timeframe.D1:
        return out
    dur = DUR[tf]
    utc = [c.timestamp - offset for c in candles]
    out["forming_bar"] = utc[-1] + dur > now
    stubs = sum(1 for t in utc if bar_inside_weekly_closure(t, dur))
    out["weekend_stubs"] = stubs
    expected = round(trading_time(utc[0], utc[-1] + dur) / dur)
    out["weekday_gaps"] = max(0, expected - (len(utc) - stubs))
    closed = [t for t in utc if t + dur <= now and not bar_inside_weekly_closure(t, dur)]
    out["last_closed_age_min"] = (round(trading_time(closed[-1] + dur, now).total_seconds() / 60, 1)
                                  if closed else None)
    return out


def assess(context, offset: Optional[timedelta], setup=None) -> DataQuality:
    now = context.generated_at
    series = getattr(context, "candle_series", {})
    quality = DataQuality(status="fresh",
                          label_offset_minutes=int(offset.total_seconds() // 60) if offset is not None else None)
    for tf in FRAMES:
        candles = series[tf].candles if tf in series else ()
        quality.frames[tf.value] = _frame_quality(candles, tf, offset, now)
    price = context.current_price
    m15 = series[Timeframe.M15].candles if Timeframe.M15 in series else ()
    if price and m15:
        quality.quote_vs_last_bar_pct = round(abs(price - m15[-1].close) / price * 100, 4)
    if setup is not None and price:
        levels = [*(setup.entries or []), setup.stop_loss, *(setup.take_profits or [])]
        off = [lv for lv in levels if lv is not None and abs(lv - price) / price * 100 > UNITS_BAND_PCT]
        if off:
            quality.problems.append(f"signal levels more than {UNITS_BAND_PCT:g}% from price "
                                    f"(units?): {', '.join(f'{x:g}' for x in off)}")

    def mark(status: str, problem: str) -> None:
        if quality.status == "fresh":
            quality.status = status
        quality.problems.append(problem)

    if in_weekly_closure(now):
        mark("market_closed", "decision made during the weekly closure")
    missing = [tf.value for tf in FRAMES if tf not in series]
    if missing or not price:
        mark("incomplete", f"missing: {', '.join(missing) or 'live price'}")
    if offset is None:
        mark("unverified", "label shift unknown: freshness not measurable")
    else:
        for tf, limit in MAX_AGE.items():
            age = quality.frames.get(tf.value, {}).get("last_closed_age_min")
            if age is not None and age > limit.total_seconds() / 60:
                mark("stale", f"{tf.value} last closed bar {age:.0f} min old (trading time)")
    if quality.quote_vs_last_bar_pct is not None and quality.quote_vs_last_bar_pct > PRICE_DEVIATION_PCT:
        mark("stale", f"live price {quality.quote_vs_last_bar_pct:.2f}% from the latest M15 bar")
    return quality
