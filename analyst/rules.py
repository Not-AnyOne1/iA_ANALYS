"""Every threshold the independent analyst uses, in one place.

The defaults were fixed from common practice before any backtest was run and
have not been tuned against results. Two of them can be overridden from the
environment because they are risk preferences rather than rules:
``ANALYST_MIN_RR`` (an independent proposal) and ``ANALYST_SIGNAL_MIN_RR``
(the original signal). Both are net reward-to-risk ratios: spread and
slippage are taken out before the ratio is compared.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import timedelta


@dataclass(frozen=True)
class Rules:
    # Structure
    swing_lookback: int = 2              # 5-bar fractals, as the live engines use
    bias_ema: int = 50                   # bias needs the close on the break's side of this EMA
    atr_period: int = 14

    # Strategy B: an M15 impulse leg and a pullback into it
    min_leg_atr: float = 1.0             # leg of at least 1 M15 ATR
    pullback_min: float = 0.382          # retraced at least 38.2% of the leg ...
    pullback_max: float = 1.0            # ... and never closed beyond the leg's origin

    # Strategy C: a sweep of resting liquidity
    sweep_max_age_bars: int = 8          # sweep within the last 8 M15 bars (2 h)
    sweep_max_depth_atr: float = 1.5     # deeper than 1.5 M15 ATR is a breakdown, not a sweep
    major_swing_strength: int = 3        # M15 swings this strong (or any H1 swing) hold liquidity
    target_swing_strength: int = 5       # M15 swings this strong (or any H1 swing) are targets

    # M5 confirmation
    m5_confirm_max_age_bars: int = 12    # the M5 break within the last hour

    # Stops and targets (multiples of the M15 ATR)
    sl_buffer_atr: float = 0.2           # stop this far beyond the invalidation extreme
    min_stop_atr: float = 0.5            # tighter than this is noise
    max_stop_atr: float = 4.0            # wider than this is not a defined setup
    max_chase_r: float = 0.5             # price beyond the entry by more than 0.5R: wait for a retest
    min_rr: float = 1.5                  # independent proposal, net of costs
    signal_min_rr: float = 1.0           # original signal, net of costs

    # Signal hygiene
    signal_max_age: timedelta = timedelta(minutes=15)
    duplicate_window: timedelta = timedelta(hours=4)
    duplicate_tolerance_atr: float = 0.25
    close_buffer: timedelta = timedelta(minutes=30)  # no new trade this close to the weekly close

    # Data
    price_sanity_pct: float = 0.5        # live price within 0.5% of the last M5 close
    alignment_pct: float = 0.05          # same-instant closes on two timeframes within 0.05%
    m5_limit: int = 300                  # real M5 bars wanted; the request is padded for stubs

    @classmethod
    def from_env(cls) -> "Rules":
        rules = cls()
        overrides = {}
        for name, field_name in (("ANALYST_MIN_RR", "min_rr"),
                                 ("ANALYST_SIGNAL_MIN_RR", "signal_min_rr")):
            raw = os.getenv(name, "").strip()
            if raw:
                value = float(raw)       # a malformed value is a configuration error
                if value <= 0:
                    raise ValueError(f"{name} must be > 0, got {value}")
                overrides[field_name] = value
        return replace(rules, **overrides) if overrides else rules


DEFAULT_RULES = Rules()
