"""Directional validity of a trade's levels — the one place it is decided.

A trade is only meaningful when its stop is on the losing side of the entry
and its target on the winning side:

    LONG   stop < entry < target
    SHORT  target < entry < stop

Everything downstream (risk/reward, the risk gate, the decision) depends on
that being true, so it is checked here, before anything is computed from the
levels. A setup that fails is *rejected*, never repaired: swapping a stop and
a target, or inferring a side from where the levels happen to sit, would be
the system inventing a trade nobody stated.

Why this exists: risk/reward used to be ``abs(tp - entry) / abs(entry - sl)``,
which is blind to direction. A SHORT with entry 4162, stop 4152 and target
4182 — a long's levels — computed a healthy 2.0 and was approved at 93.

No project imports, so both ``models`` and ``trade_quality`` can use it
without an import cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class TradeGeometry:
    """Whether a setup's levels describe a placeable trade, and its sizes.

    ``risk`` and ``reward`` are positive price distances measured in the
    trade's own direction, and are only populated when ``valid`` is True —
    there is no meaningful risk/reward for an impossible trade.
    """

    valid: bool
    reason: str
    entry: Optional[float] = None
    risk: Optional[float] = None
    reward: Optional[float] = None

    @property
    def risk_reward(self) -> Optional[float]:
        if not self.valid or not self.risk or self.reward is None:
            return None
        return self.reward / self.risk

    @property
    def complete(self) -> bool:
        """False when levels are merely missing rather than contradictory."""
        return not self.reason.startswith("incomplete")


def _usable(price: Optional[float]) -> bool:
    return (isinstance(price, (int, float)) and not isinstance(price, bool)
            and math.isfinite(price) and price > 0)


def check_geometry(
    direction: Optional[str],
    entries: Sequence[float],
    stop_loss: Optional[float],
    take_profits: Sequence[float],
) -> TradeGeometry:
    """Decide whether these levels form a valid trade in ``direction``.

    Every entry and every target is checked, not just the first: a zone whose
    far edge sits beyond the stop, or a second target on the losing side, is
    as malformed as an inverted first target. Risk/reward is reported against
    the midpoint of the entries and the first target, matching how the rest
    of the project already reads a zone.
    """
    if direction not in ("long", "short"):
        return TradeGeometry(False, "incomplete: trade direction is unknown, "
                                    "so the levels cannot be validated")
    if not entries:
        return TradeGeometry(False, "incomplete: no entry price to validate against")
    if stop_loss is None:
        return TradeGeometry(False, "incomplete: no stop loss stated")
    if not take_profits:
        return TradeGeometry(False, "incomplete: no take profit stated")

    for label, value in ([("entry", e) for e in entries] + [("stop loss", stop_loss)]
                         + [("take profit", t) for t in take_profits]):
        if not _usable(value):
            return TradeGeometry(False, f"malformed: {label} {value!r} is not a positive, "
                                        f"finite price")

    side = direction.upper()
    for e in entries:
        if stop_loss == e:
            return TradeGeometry(False, f"zero risk: stop loss equals entry {e:g}")
        for t in take_profits:
            if t == e:
                return TradeGeometry(False, f"zero reward: take profit equals entry {e:g}")

    if direction == "long":
        bad_stop = [e for e in entries if not stop_loss < e]
        bad_tp = [(e, t) for e in entries for t in take_profits if not e < t]
        rule = "stop < entry < target"
    else:
        bad_stop = [e for e in entries if not stop_loss > e]
        bad_tp = [(e, t) for e in entries for t in take_profits if not t < e]
        rule = "target < entry < stop"

    if bad_stop or bad_tp:
        problems = []
        if bad_stop:
            where = "below" if direction == "short" else "above"
            problems.append(f"stop loss {stop_loss:g} is {where} entry {bad_stop[0]:g}")
        if bad_tp:
            e, t = bad_tp[0]
            where = "above" if direction == "short" else "below"
            problems.append(f"take profit {t:g} is {where} entry {e:g}")
        return TradeGeometry(False, f"invalid {side}: requires {rule}, but "
                                    + " and ".join(problems))

    entry = sum(entries) / len(entries)
    if direction == "long":
        risk, reward = entry - stop_loss, take_profits[0] - entry
    else:
        risk, reward = stop_loss - entry, entry - take_profits[0]
    return TradeGeometry(True, f"valid {side}: {rule}", entry=entry, risk=risk, reward=reward)
