"""Entry quality: is this a good trade FROM THIS PRICE?

The direction score answers "which way is the evidence pointing?". It says
nothing about whether *now*, at *this* price, is a sensible place to get on.
A long at the top of the range after an outsized run can have perfect
directional confluence and still be a late entry — the failure mode behind
the losing 99-100 trades. This module measures the two things the direction
score cannot see:

1. **Location** — where the entry sits in the current dealing range.
2. **Extension** — how far price has already run in the trade's direction.
3. **Room** — how far price can travel before the first MAJOR level in the
   way (multi-touch liquidity, an unswept H4 swing, the previous day's high
   or low), measured in ATR.

Each produces a factor in ``[0.5, 1.0]``; none can raise the score. Location
and room look at the same thing from two angles — the structure between the
entry and where the trade needs to go — so they are combined by taking the
worse of the two (``space``), never multiplied: an entry under the range high
is not penalised twice because the range high is also a level. Space and
extension then multiply the direction score in ``trade_decision``. Nothing
here justifies rewarding an entry yet — that would need calibrated outcome
data — so a good location, clear room or a normal-sized move costs nothing.

Room has its own shape, anchored at one ATR: an obstacle at least one ATR
away — or beyond the target — costs nothing, and the factor falls linearly
to 0.5 as the obstacle approaches the entry:

    factor = 0.5 + 0.5 * min(1, room_in_ATR)

One ATR is the instrument's ordinary bar of movement; a major level closer
than that is one normal bar away, which is buying straight into it. It is
measured in ATR rather than in units of the trade's risk on purpose: the
stop is the trader's to place and adjust by hand, and must never change the
decision. It is a grade, not a gate: a tight trade can still enter when the
rest of the evidence carries it.

Both factors share one shape, chosen so that it has no free parameter:

    factor = 1 - max(0, depth - 0.5)

where ``depth`` runs from 0 at the trade's most favourable extreme to 1 at
its least favourable one, and 0.5 is the neutral midpoint. The penalty is
exactly how far past neutral the entry sits, in the measure's own units, so
the worst possible reading halves the score and the midpoint costs nothing.

Neither factor is a reversal signal. An extended long is a long with a worse
entry, never a short: these factors scale the score of the trade that was
proposed and nothing else.

No lookahead. Location uses the dealing range of confirmed swings on closed
bars; room uses levels built from closed bars; extension ranks the latest
move — up to the live price — against closes and ATR values from closed bars
only. The live price enters only as "where we are now", never as a bar.

Missing data yields ``unknown`` with a factor of 1.0 and the gap stated —
the measure is reported as absent, not guessed. Whether the entry assessment
ran at all is a separate question: ``trade_decision`` SKIPs when it did not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import indicators
from market_data import Candle

# The ATR's own window: a 14-bar move normalised by a 14-bar ATR measures the
# move in units of the volatility over the same span.
EXTENSION_WINDOW = 14

# History the move is ranked against: the same 200-candle lookback
# ``validation_engine.estimate_mae_atr`` uses for its excursion statistics.
EXTENSION_LOOKBACK = 200

# The fewest past moves that can define "normal" for this instrument. Mirrors
# ``indicators.volatility_state``, which calibrates "normal" volatility
# against 100 candles of the instrument's own history.
EXTENSION_MIN_SAMPLES = 100

# Labels are descriptive only — the factor is continuous and never steps.
# "Normal" is at or below the instrument's own median move; "strong" is its
# top decile.
_MODERATE_FROM = 0.5
_STRONG_FROM = 0.9


def _depth_factor(depth: float) -> float:
    """``1 - max(0, depth - 0.5)`` on depth clamped to [0, 1], 2 decimals.

    Rounded so the arithmetic printed in the report reproduces the score
    exactly.
    """
    return round(1.0 - max(0.0, min(depth, 1.0) - 0.5), 2)


def _finite(*values: Optional[float]) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


@dataclass(frozen=True)
class LocationAssessment:
    factor: float                     # 0.5 .. 1.0
    label: str                        # favourable / neutral / unfavourable / unknown
    zone: str                         # lower / middle / upper / below range / above range
    position: Optional[float]         # entry in the range: 0 = low, 1 = high
    depth: Optional[float]            # 0 = trade's best end, 1 = its worst end
    range_low: Optional[float]
    range_high: Optional[float]
    detail: str

    @property
    def known(self) -> bool:
        return self.label != "unknown"


@dataclass(frozen=True)
class ExtensionAssessment:
    factor: float                     # 0.5 .. 1.0
    label: str                        # normal / moderate / strong / unknown
    move_atr: Optional[float]         # signed: + = moved WITH the trade
    percentile: Optional[float]       # rank among past |moves|, 0..1
    window: int
    samples: int
    detail: str

    @property
    def known(self) -> bool:
        return self.label != "unknown"


@dataclass(frozen=True)
class RoomAssessment:
    factor: float                     # 0.5 .. 1.0
    label: str                        # clear / ample / tight / unknown
    room_atr: Optional[float]         # distance to the first major level, in ATR
    obstacle: Optional[float]         # that level's price
    obstacle_label: Optional[str]     # what kind of level it is
    detail: str

    @property
    def known(self) -> bool:
        return self.label != "unknown"


NO_ROOM_ASSESSMENT = RoomAssessment(1.0, "unknown", None, None, None,
                                    "room unknown: not assessed")


@dataclass(frozen=True)
class EntryQuality:
    """Location, extension and room, and the factor they apply to the score."""

    location: LocationAssessment
    extension: ExtensionAssessment
    room: RoomAssessment = NO_ROOM_ASSESSMENT

    @property
    def space(self) -> float:
        """The worse of location and room — the structure between the entry
        and the target, counted once."""
        return min(self.location.factor, self.room.factor)

    @property
    def factor(self) -> float:
        return round(self.space * self.extension.factor, 4)


# ----------------------------------------------------------------- location

def _unknown_location(reason: str) -> LocationAssessment:
    return LocationAssessment(1.0, "unknown", "unknown", None, None, None, None,
                              f"location unknown: {reason}")


def assess_location(
    direction: Optional[str],
    entry: Optional[float],
    range_low: Optional[float],
    range_high: Optional[float],
) -> LocationAssessment:
    """Grade where ``entry`` sits in the dealing range for this direction.

    The range is the SMC dealing range — the latest confirmed swing high and
    low, the same range ``smc_engine`` splits into premium and discount at
    its 0.5 equilibrium. For a long the low end is favourable; for a short
    the high end. ``depth`` measures from the favourable end, so the two
    sides are exact mirror images.

    An entry beyond the range keeps its true position in the report but its
    depth is clamped: above the high is the worst place for a long, never
    better than the high itself.
    """
    if direction not in ("long", "short"):
        return _unknown_location("trade direction unknown")
    if not _finite(entry, range_low, range_high):
        return _unknown_location("no dealing range or entry price")
    if range_high <= range_low:
        return _unknown_location("the dealing range has no width")

    position = (entry - range_low) / (range_high - range_low)
    depth = position if direction == "long" else 1.0 - position
    factor = _depth_factor(max(0.0, depth))

    if position < 0:
        zone = "below range"
    elif position > 1:
        zone = "above range"
    else:
        zone = "lower" if position < 1 / 3 else "upper" if position > 2 / 3 else "middle"

    label = ("favourable" if depth < 1 / 3 else
             "unfavourable" if depth > 2 / 3 else "neutral")
    side = "buying" if direction == "long" else "selling"
    span = f"the {range_low:g}-{range_high:g} dealing range"
    if zone in ("above range", "below range"):
        where = "above" if zone == "above range" else "below"
        detail = f"{side} {where} {span} — {label}"
    else:
        detail = f"{side} at {position:.0%} of {span} ({zone} part) — {label}"
    return LocationAssessment(factor, label, zone, round(position, 4), round(depth, 4),
                              range_low, range_high, detail)


# ---------------------------------------------------------------- extension

def _unknown_extension(reason: str, samples: int = 0) -> ExtensionAssessment:
    return ExtensionAssessment(1.0, "unknown", None, None, EXTENSION_WINDOW, samples,
                               f"extension unknown: {reason}")


def assess_extension(
    direction: Optional[str],
    candles: Sequence[Candle],
    *,
    current_price: Optional[float] = None,
    window: int = EXTENSION_WINDOW,
    lookback: int = EXTENSION_LOOKBACK,
    min_samples: int = EXTENSION_MIN_SAMPLES,
) -> ExtensionAssessment:
    """How unusual is the move that has just happened, for this instrument?

    ``move = (price_now - close_window_ago) / ATR`` — the run of the last
    ``window`` bars in units of volatility. It is ranked against the absolute
    size of every earlier ``window``-bar move in the lookback, each
    normalised by the ATR *of its own bar*, so a quiet week and a violent one
    are compared like for like. The percentile is the instrument's own
    answer to "is this a big move?" — no threshold is imposed from outside.

    ``candles`` must be closed bars. With ``current_price`` the latest move
    runs up to the live price — the bar in progress counts as "now", so a
    spike that has not closed yet is still seen — and every closed move is
    history. Without it the last closed move is the current one.

    Only a move WITH the trade can be extension. A long after a drop is not
    late (it may be early, or wrong, but that is the direction score's
    business), so a move against the trade ranks at 0 and costs nothing.
    """
    if direction not in ("long", "short"):
        return _unknown_extension("trade direction unknown")
    candles = list(candles)
    if len(candles) < window + 2:
        return _unknown_extension("not enough candles")

    atrs = indicators.atr_series(candles, window)
    closes = [c.close for c in candles]
    start = max(window, len(candles) - lookback)
    moves = []
    for k in range(start, len(candles)):
        a = atrs[k]
        if not _finite(a, closes[k], closes[k - window]) or a <= 0:
            continue
        moves.append((k, (closes[k] - closes[k - window]) / a))

    if _finite(current_price):
        # The bar in progress sits at index len(candles); its window starts
        # ``window`` bars back from there. ATR is the last closed one.
        latest_atr = atrs[-1]
        if not _finite(latest_atr, closes[len(candles) - window]) or latest_atr <= 0:
            return _unknown_extension("no ATR for the latest candle")
        current = (current_price - closes[len(candles) - window]) / latest_atr
        history = [abs(m) for _, m in moves]
    else:
        if not moves or moves[-1][0] != len(candles) - 1:
            return _unknown_extension("no ATR for the latest candle")
        current = moves[-1][1]
        history = [abs(m) for _, m in moves[:-1]]     # strictly earlier moves only
    if len(history) < min_samples:
        return _unknown_extension(f"only {len(history)} past moves to compare "
                                  f"against (need {min_samples})", len(history))

    with_trade = current if direction == "long" else -current
    if with_trade <= 0:
        percentile = 0.0
    else:
        below = sum(1 for h in history if h < with_trade)
        ties = sum(1 for h in history if h == with_trade)
        percentile = (below + 0.5 * ties) / len(history)

    factor = _depth_factor(percentile)
    label = ("strong" if percentile > _STRONG_FROM else
             "moderate" if percentile > _MODERATE_FROM else "normal")
    word = {"strong": "strongly extended", "moderate": "moderately extended",
            "normal": "normal"}[label]
    if with_trade <= 0:
        detail = (f"price moved {abs(with_trade):.1f} ATR against the trade over {window} "
                  f"bars — no run to be late to (normal)")
    else:
        detail = (f"price already moved {with_trade:.1f} ATR with the trade over {window} "
                  f"bars — larger than {percentile:.0%} of the last {len(history)} such "
                  f"moves ({word})")
    return ExtensionAssessment(factor, label, round(with_trade, 3), round(percentile, 4),
                               window, len(history), detail)


# --------------------------------------------------------------------- room

def _unknown_room(reason: str) -> RoomAssessment:
    return RoomAssessment(1.0, "unknown", None, None, None, f"room unknown: {reason}")


def assess_room(
    direction: Optional[str],
    entry: Optional[float],
    targets: Sequence[float],
    levels: Sequence[Tuple[float, str]],
    atr: Optional[float],
) -> RoomAssessment:
    """How far can the trade travel before the first major level in its way?

    Measured from the entry to the nearest major level beyond it in the
    trade's direction, in ATR. Only MAJOR levels count — the caller passes
    multi-touch liquidity, unswept higher-timeframe swings and the previous
    day's extremes — because every minor swing is "in the way" of almost
    every trade and grading against them says nothing.

    The target checked is the nearest one in the trade's direction: that is
    the first one price has to reach, whatever order the signal listed them.
    A level beyond that target is not in the way of it. The stop plays no
    part: it is the trader's to adjust and never moves the decision.
    """
    if direction not in ("long", "short"):
        return _unknown_room("trade direction unknown")
    if not _finite(entry, atr) or atr <= 0:
        return _unknown_room("no entry or ATR")
    sign = 1.0 if direction == "long" else -1.0
    ahead_targets = [t for t in targets if _finite(t) and sign * (t - entry) > 0]
    if not ahead_targets:
        return _unknown_room("no target beyond the entry to measure against")
    target = min(ahead_targets, key=lambda t: sign * (t - entry))
    reward = sign * (target - entry)

    ahead = [(p, label) for p, label in levels if _finite(p) and sign * (p - entry) > 0]
    if not ahead:
        return RoomAssessment(1.0, "clear", None, None, None,
                              "no major level between the entry and the target")
    price, label = min(ahead, key=lambda lv: sign * (lv[0] - entry))
    distance = sign * (price - entry)
    room_atr = distance / atr
    if distance >= reward:
        return RoomAssessment(1.0, "clear", round(room_atr, 3), price, label,
                              f"the target is reached before the next major level "
                              f"({label} at {price:g})")
    factor = round(0.5 + 0.5 * min(1.0, room_atr), 2)
    tag = "ample" if room_atr >= 1.0 else "tight"
    word = "above" if direction == "long" else "below"
    detail = (f"{label} at {price:g} sits {room_atr:.2f} ATR {word} the entry, before the "
              f"target — {'room to move first' if tag == 'ample' else 'entering into a major level'}")
    return RoomAssessment(factor, tag, round(room_atr, 3), price, label, detail)


# ------------------------------------------------------------------- all

def assess(
    direction: Optional[str],
    entry: Optional[float],
    range_low: Optional[float],
    range_high: Optional[float],
    candles: Sequence[Candle],
    *,
    current_price: Optional[float] = None,
    targets: Sequence[float] = (),
    levels: Sequence[Tuple[float, str]] = (),
    atr: Optional[float] = None,
) -> EntryQuality:
    """Full entry assessment. Never raises for missing or malformed data.

    ``candles`` are closed bars; ``current_price`` is the live price, used
    only for the latest move.
    """
    return EntryQuality(
        location=assess_location(direction, entry, range_low, range_high),
        extension=assess_extension(direction, candles, current_price=current_price),
        room=(assess_room(direction, entry, targets, levels, atr)
              if targets else NO_ROOM_ASSESSMENT),
    )
