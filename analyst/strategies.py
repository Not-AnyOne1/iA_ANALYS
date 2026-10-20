"""Strategies B and C — hypotheses under test, not proven edges.

Each is a pure function of a :class:`MarketRead` and a direction and returns
a :class:`Proposal` or ``None`` (no setup). They share the stop/target/cost
arithmetic below and nothing else: neither reads the other's result, and no
code here picks between them on performance.

B — trend continuation
    H4 and H1 bias both in the trade's direction; an M15 pullback into the
    last impulse leg (``structure.find_pullback``); an M5 break in the trade's
    direction after the pullback's extreme. Invalidation: the pullback's
    extreme.

C — liquidity-sweep reversal
    An M15 sweep of resting liquidity that closed back on the level's side
    (``structure.find_sweep``); an M5 break in the reversal's direction after
    the sweep bar opened. Invalidation: the sweep's extreme. H4/H1 bias is
    reported as context but is not a condition.

Levels, for both:

* stop = invalidation extreme -/+ ``sl_buffer_atr`` M15 ATR; widened to
  ``min_stop_atr`` ATR if closer; wider than ``max_stop_atr`` ATR is INVALID;
* target = the nearest resting liquidity beyond the entry
  (``MarketRead.target``). B's thesis is that price runs its impulse swing,
  so B targets the nearest resting level beyond that swing, or the leg's
  1.272 extension when none rests there. A target is never moved further
  out to make the ratio work: too little room is INVALID;
* the reward-to-risk ratio is net of spread and slippage and must reach
  ``Rules.min_rr``.

Status:

* READY    — every condition met and price no further than ``max_chase_r``
             of the risk beyond the broken M5 level: a market entry now;
* WAITING  — the setup exists but needs the M5 break (a stop entry at the
             M5 trigger) or a retest of the broken level (a limit entry);
* INVALID  — the setup exists but its levels fail the stop or ratio rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from costs import Costs
from market_data import Timeframe

from analyst.structure import LONG, SHORT, Level, MarketRead, find_pullback, find_sweep, m5_break, \
    m5_trigger, side_sign

READY, WAITING, INVALID = "READY", "WAITING", "INVALID"

# How a strategy picks its target: (read, direction, entry) -> Level or None.
TargetRule = Callable[[MarketRead, str, float], Optional[Level]]


def _nearest_target(read: MarketRead, direction: str, entry: float) -> Optional[Level]:
    return read.target(direction, entry)
NAMES = {"B": "trend continuation", "C": "liquidity sweep reversal"}


@dataclass(frozen=True)
class Proposal:
    strategy: str                     # "B" / "C"
    direction: str                    # long / short
    status: str                       # READY / WAITING / INVALID
    order_type: Optional[str] = None  # market (READY) / stop / limit (WAITING)
    entry: Optional[float] = None
    entry_zone: Optional[Tuple[float, float]] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    target: str = ""
    rr_gross: Optional[float] = None
    rr_net: Optional[float] = None
    evidence: Tuple[str, ...] = ()
    invalidation: str = ""
    confirmation: str = ""            # what is still needed; empty when READY
    reasons: Tuple[str, ...] = ()     # reason codes
    notes: Tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return NAMES[self.strategy]

    def as_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy, "direction": self.direction, "status": self.status,
            "order_type": self.order_type, "entry": self.entry,
            "entry_zone": list(self.entry_zone) if self.entry_zone else None,
            "stop_loss": self.stop_loss, "take_profit": self.take_profit, "target": self.target,
            "rr_gross": self.rr_gross, "rr_net": self.rr_net, "evidence": list(self.evidence),
            "invalidation": self.invalidation, "confirmation": self.confirmation,
            "reasons": list(self.reasons), "notes": list(self.notes),
        }


def net_rr(direction: str, entry: float, stop: float, target: float, costs: Costs,
           order_type: str = "market") -> Optional[float]:
    """Reward-to-risk after costs, filled the way ``trade_accounting`` fills:
    a market entry pays half the spread plus slippage, a stop entry pays
    slippage, a limit entry fills at its price; the stop exit pays slippage;
    the target fills at its price."""
    sign = side_sign(direction)
    half, slip = costs.spread / 2.0, costs.slippage
    fill = entry + sign * {"market": half + slip, "stop": slip, "limit": 0.0}[order_type]
    reward = sign * (target - fill)
    loss = sign * (fill - stop) + slip
    if loss <= 0:
        return None
    return round(reward / loss, 3)


def _levels(read: MarketRead, strategy: str, direction: str, entry: float, order_type: str,
            extreme: float, targets: "TargetRule", costs: Costs, **common) -> Proposal:
    rules, atr, sign = read.rules, read.atr15, side_sign(direction)
    notes: List[str] = list(common.pop("notes", ()))
    stop = extreme - sign * rules.sl_buffer_atr * atr
    risk = sign * (entry - stop)
    if risk < rules.min_stop_atr * atr:
        stop = entry - sign * rules.min_stop_atr * atr
        notes.append(f"stop widened to {rules.min_stop_atr:g} M15 ATR beyond the entry")
        risk = sign * (entry - stop)
    base = dict(strategy=strategy, direction=direction, order_type=order_type,
                entry=round(entry, 2), stop_loss=round(stop, 2), **common)
    if risk > rules.max_stop_atr * atr:
        return Proposal(status=INVALID, reasons=("STOP_TOO_WIDE",), notes=tuple(notes + [
            f"stop {risk:.2f} from the entry is wider than {rules.max_stop_atr:g} M15 ATR "
            f"({rules.max_stop_atr * atr:.2f})"]), **base)
    level = targets(read, direction, entry)
    if level is None:
        return Proposal(status=INVALID, reasons=("NO_TARGET",), notes=tuple(notes + [
            "no resting liquidity beyond the entry to target"]), **base)
    rr_gross = round(sign * (level.price - entry) / risk, 3)
    rr = net_rr(direction, entry, stop, level.price, costs, order_type)
    base.update(take_profit=round(level.price, 2), target=level.label(), rr_gross=rr_gross, rr_net=rr)
    if rr is None or rr < rules.min_rr:
        return Proposal(status=INVALID, reasons=("RR_BELOW_MIN",), notes=tuple(notes + [
            f"net reward-to-risk {rr if rr is not None else 'n/a'} to {level.label()} is below "
            f"{rules.min_rr:g}"]), **base)
    return Proposal(status=READY, notes=tuple(notes), **base)


def _finish(read: MarketRead, strategy: str, direction: str, since, extreme: float,
            targets: "TargetRule", costs: Costs, evidence: List[str], invalidation: str) -> Proposal:
    """M5 confirmation, entry availability and levels — shared by B and C."""
    rules, sign, price = read.rules, side_sign(direction), read.price
    common = dict(evidence=tuple(evidence), invalidation=invalidation)
    brk = m5_break(read, direction, since)
    if brk is None:
        trigger = m5_trigger(read, direction)
        word = "above" if direction == LONG else "below"
        if trigger is None:
            return Proposal(strategy, direction, WAITING, confirmation=(
                f"an M5 swing to form and an M5 close {word} it"), reasons=("AWAIT_M5_BREAK",),
                **common)
        p = _levels(read, strategy, direction, trigger, "stop", extreme, targets, costs, **common)
        return _as_waiting(p, f"M5 close {word} {trigger:.2f}", "AWAIT_M5_BREAK")
    broken = brk.broken_swing.price
    evidence = evidence + [f"M5 {brk.type.value.upper()} through {broken:.2f} at {brk.timestamp:%H:%M}"]
    common["evidence"] = tuple(evidence)
    zone = (round(min(broken, price), 2), round(max(broken, price), 2))
    p = _levels(read, strategy, direction, price, "market", extreme, targets, costs,
                entry_zone=zone, **common)
    if p.status != READY:
        return p
    risk = sign * (price - p.stop_loss)
    if sign * (price - broken) > rules.max_chase_r * risk:
        retest = _levels(read, strategy, direction, broken, "limit", extreme, targets, costs,
                         entry_zone=zone, **common)
        return _as_waiting(retest, f"a retest of {broken:.2f} (price is "
                                   f"{sign * (price - broken):.2f} beyond it, over "
                                   f"{rules.max_chase_r:g}R)", "AWAIT_RETEST")
    return p


def _as_waiting(p: Proposal, confirmation: str, code: str) -> Proposal:
    if p.status == INVALID:
        return p
    return Proposal(**{**p.__dict__, "status": WAITING, "confirmation": confirmation,
                       "reasons": p.reasons + (code,)})


def strategy_b(read: MarketRead, direction: str, costs: Costs) -> Optional[Proposal]:
    want = "bullish" if direction == LONG else "bearish"
    h4, h1 = read.frame(Timeframe.H4), read.frame(Timeframe.H1)
    if h4.bias != want or h1.bias != want:
        return None
    pb = find_pullback(read, direction)
    if pb is None:
        return None
    sign = side_sign(direction)
    leg = abs(pb.impulse.price - pb.origin.price)
    extension = Level(pb.impulse.price + sign * 0.272 * leg, "high" if direction == LONG else "low",
                      Timeframe.M15, pb.impulse.index, pb.extreme_at, pb.extreme_at, 0,
                      kind="M15 leg 1.272 extension")

    def targets(r: MarketRead, d: str, entry: float) -> Optional[Level]:
        return r.target(d, entry, beyond=pb.impulse.price) or             (extension if sign * (extension.price - entry) > 0 else None)

    evidence = [f"H4 bias {want}", f"H1 bias {want}",
                f"M15 pullback {pb.depth:.0%} of the leg {pb.origin.price:.2f} -> "
                f"{pb.impulse.price:.2f}, extreme {pb.extreme:.2f}"]
    word = "below" if direction == LONG else "above"
    return _finish(read, "B", direction, pb.extreme_at, pb.extreme, targets, costs, evidence,
                   f"an M15 close {word} {pb.extreme:.2f} (pullback extreme) or H1 bias turning")


def strategy_c(read: MarketRead, direction: str, costs: Costs) -> Optional[Proposal]:
    sw = find_sweep(read, direction)
    if sw is None:
        return None
    back = "above" if direction == LONG else "below"
    evidence = [f"M15 swept {sw.level.label()} at {sw.bar.timestamp:%m-%d %H:%M} "
                f"(wick {sw.extreme:.2f}, closed back {back})",
                f"context: H4 {read.frame(Timeframe.H4).bias}, H1 {read.frame(Timeframe.H1).bias}"]
    word = "below" if direction == LONG else "above"
    return _finish(read, "C", direction, sw.bar.timestamp, sw.extreme, _nearest_target, costs, evidence,
                   f"a trade {word} the sweep extreme {sw.extreme:.2f}")


STRATEGIES = {"B": strategy_b, "C": strategy_c}


def evaluate(read: MarketRead, costs: Costs) -> Dict[str, Dict[str, Optional[Proposal]]]:
    """Every strategy in both directions, independently."""
    return {name: {d: fn(read, d, costs) for d in (LONG, SHORT)} for name, fn in STRATEGIES.items()}


_RANK = {READY: 0, WAITING: 1, INVALID: 2}


def headline(proposals: Dict[str, Dict[str, Optional[Proposal]]]) -> Optional[Proposal]:
    """The proposal shown first on the card: READY before WAITING before
    INVALID, then B before C, then long before short. A fixed display
    order — never a choice by past results."""
    found = [p for name in ("B", "C") for d in (LONG, SHORT)
             if (p := proposals.get(name, {}).get(d)) is not None]
    return min(found, key=lambda p: (_RANK[p.status], p.strategy, p.direction != LONG)) \
        if found else None
