"""Replay the production decision as the live bot computes it, at any past time.

The live bot asks Twelve Data for the last 300 bars of M15, H1, H4 and D1.
Those include the bar that is still forming (partial OHLC up to the moment of
the request) and every weekend stub quote. This replay serves exactly that:

* closed bars from ``data/research/`` as they are;
* the forming bar of each timeframe rebuilt from the M5 bars that had closed
  by the decision time — open of the first, extremes of all, close of the last.
  Its labels are UTC; the live provider's labels are shifted, but production's
  verdict never reads a label (see ``research/reports/IA_summary.md``, part 3);
* the quote at the last closed M5 close, with no bid/ask, as Twelve Data gives.

The context is built with production's own builder and engines, and the
verdict with ``trade_decision.decide_deterministically`` (Claude never
changes it). The decision clock is the replayed time, so session and news
checks read the right moment.
"""

from __future__ import annotations

import bisect
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market_data import Candle, CandleSeries, MarketDataService, Quote, Timeframe  # noqa: E402
from market_hours import new_york_offset  # noqa: E402

UTC = timezone.utc
DATA = ROOT / "data" / "research"
FILES = {Timeframe.M5: "5min", Timeframe.M15: "15min", Timeframe.H1: "1h",
         Timeframe.H4: "4h", Timeframe.D1: "1day"}
DURATION = {Timeframe.M5: timedelta(minutes=5), Timeframe.M15: timedelta(minutes=15),
            Timeframe.H1: timedelta(hours=1), Timeframe.H4: timedelta(hours=4),
            Timeframe.D1: timedelta(days=1)}


def _ny17_before(day: date) -> datetime:
    """17:00 New York on the day before ``day``, as UTC (the D1 bucket start)."""
    local = datetime.combine(day - timedelta(days=1), time(17, 0))
    offset = new_york_offset(local.replace(tzinfo=UTC) + timedelta(hours=5))
    return (local - offset).replace(tzinfo=UTC)


class Bars:
    """Research bars, with each bar's real bucket [start, end)."""

    def __init__(self, data: Path = DATA) -> None:
        self.bars: Dict[Timeframe, Tuple[Candle, ...]] = {}
        self.starts: Dict[Timeframe, List[float]] = {}
        self.ends: Dict[Timeframe, List[float]] = {}
        for tf, name in FILES.items():
            raw = json.loads((data / f"xauusd_{name}.json").read_text())["bars"]
            bars = tuple(Candle(datetime.fromisoformat(t), o, h, lo, c) for t, o, h, lo, c in raw)
            if tf is Timeframe.D1:
                starts = [_ny17_before(b.timestamp.date()) for b in bars]
            else:
                starts = [b.timestamp for b in bars]
            self.bars[tf] = bars
            self.starts[tf] = [s.timestamp() for s in starts]
            self.ends[tf] = [(s + DURATION[tf]).timestamp() for s in starts]

    def last_close(self, at: datetime) -> Optional[float]:
        """Close of the last M5 bar closed by ``at``."""
        k = bisect.bisect_right(self.ends[Timeframe.M5], at.timestamp())
        return self.bars[Timeframe.M5][k - 1].close if k else None

    def _forming(self, tf: Timeframe, i: int, at: datetime) -> Optional[Candle]:
        lo = bisect.bisect_left(self.starts[Timeframe.M5], self.starts[tf][i])
        hi = bisect.bisect_right(self.ends[Timeframe.M5], at.timestamp())
        part = self.bars[Timeframe.M5][lo:hi]
        if not part:
            return None
        return Candle(self.bars[tf][i].timestamp, part[0].open, max(c.high for c in part),
                      min(c.low for c in part), part[-1].close)

    def live_window(self, tf: Timeframe, at: datetime, limit: int) -> Tuple[Candle, ...]:
        """What the provider returned at ``at``: up to ``limit`` bars that had
        opened, the forming one rebuilt from M5."""
        t = at.timestamp()
        k = bisect.bisect_left(self.starts[tf], t)          # bars that opened before at
        if k == 0:
            return ()
        if self.ends[tf][k - 1] > t:                        # the last one is still forming
            forming = self._forming(tf, k - 1, at)
            closed = self.bars[tf][max(0, k - limit):k - 1]
            return closed + ((forming,) if forming else ())
        return self.bars[tf][max(0, k - limit):k]


class ReplayProvider:
    """A MarketDataProvider serving :class:`Bars` as of a settable time."""

    name = "replay"

    def __init__(self, bars: Bars, window=None) -> None:
        self.bars, self.at, self.price = bars, None, None
        # window(tf, at, limit) -> candles; the live-like window by default.
        self._window = window or bars.live_window

    async def get_quote(self, symbol: str) -> Quote:
        return Quote(symbol=symbol, price=self.price, timestamp=self.at, provider=self.name)

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        return CandleSeries(symbol=symbol, timeframe=timeframe, provider=self.name,
                            candles=tuple(self._window(timeframe, self.at, limit)))


def service(provider: ReplayProvider) -> MarketDataService:
    return MarketDataService(provider, cache_ttl=0, max_retries=0)
