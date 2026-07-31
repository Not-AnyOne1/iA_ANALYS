"""Deterministic technical indicators.

Pure functions over an OHLC series — no I/O, no async, no randomness, no
model involvement. Claude never computes any of these; it only ever reads
the numbers this module produces (see ``decision_engine.py``).

Every indicator here uses the standard textbook definition, and the two
smoothing conventions in use are named explicitly rather than left implicit:

- **Wilder smoothing** (RSI, ATR, ADX): ``prev + (value - prev) / period``,
  equivalent to an EMA with ``alpha = 1/period``. This is what Wilder
  defined in *New Concepts in Technical Trading Systems* and what charting
  platforms report for these three indicators.
- **Standard EMA** (EMA 20/50/200): ``alpha = 2 / (period + 1)``, seeded
  with the simple average of the first ``period`` values.

Insufficient data returns ``None`` rather than a partial or padded number:
an indicator computed over fewer bars than its period is not that
indicator, and silently returning one would be exactly the kind of
fabricated input this project refuses to hand to Claude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from market_data import Candle

# ADX convention: Wilder's original uses 14 for both the DI period and the
# ADX smoothing period.
_DEFAULT_PERIOD = 14


@dataclass(frozen=True)
class EMASet:
    """The three EMAs traders most commonly stack for trend alignment."""

    ema_20: Optional[float]
    ema_50: Optional[float]
    ema_200: Optional[float]

    @property
    def alignment(self) -> str:
        """'bullish' when 20 > 50 > 200, 'bearish' when 20 < 50 < 200.

        Anything else is 'mixed'; 'unknown' when any EMA is unavailable
        (not enough candles). Deliberately not a numeric score — this is a
        factual description of the stack order, nothing more.
        """
        if self.ema_20 is None or self.ema_50 is None or self.ema_200 is None:
            return "unknown"
        if self.ema_20 > self.ema_50 > self.ema_200:
            return "bullish"
        if self.ema_20 < self.ema_50 < self.ema_200:
            return "bearish"
        return "mixed"


@dataclass(frozen=True)
class ADXResult:
    """ADX with its two directional components."""

    adx: Optional[float]
    plus_di: Optional[float]
    minus_di: Optional[float]

    @property
    def strength(self) -> str:
        """Wilder's conventional reading of ADX magnitude."""
        if self.adx is None:
            return "unknown"
        if self.adx < 20:
            return "absent"      # no trend worth trading
        if self.adx < 25:
            return "weak"
        if self.adx < 50:
            return "strong"
        return "very_strong"


# --------------------------------------------------------------------- helpers

def _closes(candles: Sequence[Candle]) -> List[float]:
    return [c.close for c in candles]


def _wilder_smooth(values: Sequence[float], period: int) -> List[float]:
    """Wilder's running average: seed with the simple mean, then
    ``prev + (x - prev) / period``. Returns one value per input from
    ``period - 1`` onwards."""
    if len(values) < period:
        return []
    out = [sum(values[:period]) / period]
    for value in values[period:]:
        out.append(out[-1] + (value - out[-1]) / period)
    return out


# ------------------------------------------------------------------------ EMA

def ema(values: Sequence[float], period: int) -> Optional[float]:
    """Standard EMA (alpha = 2/(period+1)), seeded with the SMA of the
    first ``period`` values. ``None`` if there aren't enough values."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    current = sum(values[:period]) / period
    for value in values[period:]:
        current = value * alpha + current * (1 - alpha)
    return current


def ema_set(candles: Sequence[Candle]) -> EMASet:
    """EMA 20/50/200 over closes. Each is independently ``None`` when the
    series is too short for that particular period."""
    closes = _closes(candles)
    return EMASet(
        ema_20=ema(closes, 20),
        ema_50=ema(closes, 50),
        ema_200=ema(closes, 200),
    )


# ------------------------------------------------------------------------ RSI

def rsi(candles: Sequence[Candle], period: int = _DEFAULT_PERIOD) -> Optional[float]:
    """Wilder's RSI over closes, 0-100. ``None`` if too few candles.

    Needs ``period + 1`` candles because it works on close-to-close
    changes, not on the closes themselves.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    closes = _closes(candles)
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    avg_gain = _wilder_smooth(gains, period)[-1]
    avg_loss = _wilder_smooth(losses, period)[-1]

    # An unbroken run of gains has no downside to divide by; RSI is 100 by
    # definition there (and 0 for the mirror case), not undefined.
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ------------------------------------------------------------------------ ATR

def true_ranges(candles: Sequence[Candle]) -> List[float]:
    """Wilder's True Range per candle, starting from the second one."""
    out: List[float] = []
    for i in range(1, len(candles)):
        current, prev_close = candles[i], candles[i - 1].close
        out.append(max(
            current.high - current.low,
            abs(current.high - prev_close),
            abs(current.low - prev_close),
        ))
    return out


def atr(candles: Sequence[Candle], period: int = _DEFAULT_PERIOD) -> Optional[float]:
    """Wilder's ATR in price units. ``None`` if too few candles."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    tr = true_ranges(candles)
    if len(tr) < period:
        return None
    return _wilder_smooth(tr, period)[-1]


def atr_percent(candles: Sequence[Candle], period: int = _DEFAULT_PERIOD) -> Optional[float]:
    """ATR as a percentage of the last close — comparable across symbols,
    unlike raw ATR which is in the instrument's own price units."""
    value = atr(candles, period)
    if value is None or not candles:
        return None
    last_close = candles[-1].close
    if last_close == 0:
        return None
    return value / last_close * 100.0


# ----------------------------------------------------------------- volatility

def volatility_state(
    candles: Sequence[Candle],
    period: int = _DEFAULT_PERIOD,
    lookback: int = 100,
) -> str:
    """Where current ATR sits against its own recent history.

    Returns 'low' / 'normal' / 'high' / 'unknown'. Compares the latest ATR
    to the median ATR over ``lookback`` candles, so it is self-calibrating
    per symbol and timeframe rather than using a hardcoded threshold that
    would be meaningless across instruments.
    """
    current = atr(candles, period)
    if current is None:
        return "unknown"

    window = candles[-lookback:] if len(candles) > lookback else candles
    tr = true_ranges(window)
    if len(tr) < period:
        return "unknown"
    history = _wilder_smooth(tr, period)
    if len(history) < 2:
        return "unknown"

    ordered = sorted(history)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    if median == 0:
        return "unknown"

    ratio = current / median
    if ratio < 0.7:
        return "low"
    if ratio > 1.4:
        return "high"
    return "normal"


# ------------------------------------------------------------------------ ADX

def adx(candles: Sequence[Candle], period: int = _DEFAULT_PERIOD) -> ADXResult:
    """Wilder's ADX with +DI / -DI.

    Needs roughly ``2 * period`` candles: one ``period`` to smooth the
    directional movement, another to smooth DX into ADX itself.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if len(candles) < period * 2:
        return ADXResult(adx=None, plus_di=None, minus_di=None)

    plus_dm: List[float] = []
    minus_dm: List[float] = []
    for i in range(1, len(candles)):
        up = candles[i].high - candles[i - 1].high
        down = candles[i - 1].low - candles[i].low
        # Only the larger of the two counts, and only when positive —
        # an inside bar contributes no directional movement at all.
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)

    tr = true_ranges(candles)
    smoothed_tr = _wilder_smooth(tr, period)
    smoothed_plus = _wilder_smooth(plus_dm, period)
    smoothed_minus = _wilder_smooth(minus_dm, period)
    if not smoothed_tr:
        return ADXResult(adx=None, plus_di=None, minus_di=None)

    dx: List[float] = []
    plus_di_series: List[float] = []
    minus_di_series: List[float] = []
    for tr_v, plus_v, minus_v in zip(smoothed_tr, smoothed_plus, smoothed_minus):
        if tr_v == 0:
            plus_di_series.append(0.0)
            minus_di_series.append(0.0)
            dx.append(0.0)
            continue
        p = 100.0 * plus_v / tr_v
        m = 100.0 * minus_v / tr_v
        plus_di_series.append(p)
        minus_di_series.append(m)
        total = p + m
        dx.append(0.0 if total == 0 else 100.0 * abs(p - m) / total)

    if len(dx) < period:
        return ADXResult(adx=None, plus_di=plus_di_series[-1], minus_di=minus_di_series[-1])

    return ADXResult(
        adx=_wilder_smooth(dx, period)[-1],
        plus_di=plus_di_series[-1],
        minus_di=minus_di_series[-1],
    )
