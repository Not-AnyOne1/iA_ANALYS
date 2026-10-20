"""The original signal, judged against the independent analysis.

Checks run in this order; the first group that fails decides the verdict.

1. The signal itself (no market data needed) -> SKIP
   ``SIG_INCOMPLETE``     no direction, no stop or no take-profit
   ``SIG_STALE``          older than ``Rules.signal_max_age`` when analysed
   ``SIG_DUPLICATE``      same direction, stop and first target as a signal
                          seen within ``Rules.duplicate_window``
2. The data -> NO TRADE
   ``DATA_<STATUS>``      the market view is not fresh (``analyst.data``)
3. The levels at the live price -> SKIP
   ``SIG_STOPPED_OUT``        price is already at or beyond the stop
   ``SIG_TARGET_REACHED``     price already passed a target the stated entry
                              was aiming for
   ``SIG_NO_TARGET_AHEAD``    no take-profit beyond the planned entry
   ``SIG_LEVELS_INCONSISTENT`` the stop is on the wrong side of the entry
   ``SESSION_CLOSING``        within ``Rules.close_buffer`` of the weekly close
   ``SIG_STOP_TOO_TIGHT`` / ``SIG_STOP_TOO_WIDE``  outside the M15-ATR band
   ``SIG_RR_BELOW_MIN``       net ratio to the nearest target below
                              ``Rules.signal_min_rr``
4. Timing -> WAIT
   ``ENTRY_NOT_REACHED``  a limit/stop entry price has not traded yet
   ``PRICE_EXTENDED``     price ran more than ``max_chase_r`` of the risk past
                          the stated entry
   ``NEWS_WINDOW``        only when a real calendar is configured and blocks
5. Structure (strategies B and C, in the signal's direction and against it)
   both directions READY              -> WAIT  ``CONFLICTING_SETUPS``
   same direction READY               -> ENTER ``CONFIRMED_BY_<B|C>``
   opposite direction READY           -> SKIP  ``OPPOSING_SETUP``
   H1 bias against, same-way C forming -> WAIT ``REVERSAL_UNCONFIRMED``
   H1 bias against                    -> SKIP  ``COUNTER_H1_STRUCTURE``
   same way forming, or H1 with it    -> WAIT  ``AWAITING_CONFIRMATION``
   H4 with it, H1 neutral             -> WAIT  ``H1_NEUTRAL``
   otherwise                          -> NO TRADE ``NO_STRUCTURAL_EVIDENCE``

When timing and structure disagree the stricter verdict wins:
SKIP, then NO TRADE, then WAIT, then ENTER.

The evidence checklist (``checks``) is a descriptive count — how many of six
conditions hold — and is labelled uncalibrated wherever it is shown. It never
changes the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from costs import Costs
from market_data import Timeframe
from models import TradeSetup
from news_filter import NewsStatus
from trade_accounting import nearest_target

from analyst.data import MarketView
from analyst.rules import Rules
from analyst.strategies import READY, WAITING, Proposal, net_rr
from analyst.structure import LONG, SHORT, MarketRead, find_pullback, find_sweep, m5_break, side_sign

ENTER, WAIT, SKIP, NO_TRADE = "ENTER", "WAIT", "SKIP", "NO TRADE"
SEVERITY = {SKIP: 3, NO_TRADE: 2, WAIT: 1, ENTER: 0}


@dataclass(frozen=True)
class Reason:
    code: str
    text: str


@dataclass(frozen=True)
class SignalPlan:
    """The original signal as the analyst reads it at the live price."""

    direction: Optional[str]
    order_type: str
    entries: Tuple[float, ...]
    stop_loss: Optional[float]
    take_profits: Tuple[float, ...]
    planned_entry: Optional[float] = None    # where a trade taken now would start
    pending: bool = False                    # the stated entry has not traded yet
    target: Optional[float] = None           # nearest take-profit beyond the planned entry
    rr_gross: Optional[float] = None
    rr_net: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {"direction": self.direction, "order_type": self.order_type,
                "entries": list(self.entries), "stop_loss": self.stop_loss,
                "take_profits": list(self.take_profits), "planned_entry": self.planned_entry,
                "pending": self.pending, "target": self.target,
                "rr_gross": self.rr_gross, "rr_net": self.rr_net}


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reasons: Tuple[Reason, ...]
    plan: SignalPlan
    weaknesses: Tuple[Reason, ...] = ()
    checks: Tuple[Tuple[str, bool], ...] = ()
    confirmed_by: Tuple[str, ...] = ()

    @property
    def checks_met(self) -> int:
        return sum(1 for _, ok in self.checks if ok)


@dataclass(frozen=True)
class RecentSignal:
    """A signal already analysed, for duplicate detection."""

    at: datetime
    direction: Optional[str]
    stop_loss: Optional[float]
    take_profits: Tuple[float, ...]


def plan_from_setup(setup: TradeSetup) -> SignalPlan:
    return SignalPlan(direction=setup.direction, order_type=setup.order_type or "unknown",
                      entries=tuple(setup.entries), stop_loss=setup.stop_loss,
                      take_profits=tuple(setup.take_profits))


def _is_duplicate(plan: SignalPlan, at: datetime, recent: Sequence[RecentSignal], rules: Rules,
                  tolerance: float) -> Optional[RecentSignal]:
    tp1 = plan.take_profits[0]
    for r in recent:
        if not (at - rules.duplicate_window <= r.at <= at):
            continue
        if r.direction != plan.direction or r.stop_loss is None or not r.take_profits:
            continue
        if abs(r.stop_loss - plan.stop_loss) <= tolerance and abs(r.take_profits[0] - tp1) <= tolerance:
            return r
    return None


def plan_at_price(plan: SignalPlan, price: float, atr: float, costs: Costs) -> SignalPlan:
    sign = side_sign(plan.direction)
    entry, pending = price, False
    if plan.entries and plan.order_type != "market":
        lo, hi = min(plan.entries), max(plan.entries)
        slack = 0.1 * atr
        if not (lo - slack <= price <= hi + slack):
            # Not traded yet: a trade would start at the zone edge price reaches first.
            entry, pending = (hi if price > hi else lo), True
    target = nearest_target(plan.direction, entry, plan.take_profits)
    rr_gross = rr = None
    if target is not None and plan.stop_loss is not None and sign * (entry - plan.stop_loss) > 0:
        rr_gross = round(sign * (target - entry) / (sign * (entry - plan.stop_loss)), 3)
        order = "limit" if pending else "market"
        rr = net_rr(plan.direction, entry, plan.stop_loss, target, costs, order)
    return SignalPlan(**{**plan.__dict__, "planned_entry": round(entry, 2), "pending": pending,
                         "target": target, "rr_gross": rr_gross, "rr_net": rr})


def judge(setup: TradeSetup, message_at: datetime, view: MarketView, read: Optional[MarketRead],
          proposals: Dict[str, Dict[str, Optional[Proposal]]], rules: Rules, costs: Costs, *,
          recent: Sequence[RecentSignal] = (), news: Optional[NewsStatus] = None) -> Verdict:
    plan = plan_from_setup(setup)
    as_of = view.as_of

    def verdict(v: str, *reasons: Reason, **kw) -> Verdict:
        return Verdict(v, tuple(reasons), kw.pop("plan", plan), **kw)

    # 1. the signal itself
    missing = [name for name, value in (("direction", plan.direction), ("stop loss", plan.stop_loss),
                                        ("take-profit", plan.take_profits)) if not value]
    if missing:
        return verdict(SKIP, Reason("SIG_INCOMPLETE", f"the signal gives no {', no '.join(missing)}"))
    age = as_of - message_at
    if age > rules.signal_max_age:
        return verdict(SKIP, Reason("SIG_STALE", f"signal is {age.total_seconds() / 60:.0f} min old "
                                                 f"(limit {rules.signal_max_age.total_seconds() / 60:.0f})"))
    tolerance = max(0.5, rules.duplicate_tolerance_atr * read.atr15) if read is not None else 0.5
    dup = _is_duplicate(plan, message_at, recent, rules, tolerance)
    if dup is not None:
        return verdict(SKIP, Reason("SIG_DUPLICATE", f"same direction, stop and first target as the "
                                                     f"signal at {dup.at:%m-%d %H:%M}"))

    # 2. the data
    if not view.fresh or read is None or view.price is None:
        status = view.status if not view.fresh else "unavailable"
        detail = "; ".join(view.problems) or "the M15 ATR could not be computed"
        return verdict(NO_TRADE, Reason(f"DATA_{status.upper()}", f"market data {status}: {detail}"))

    # 3. the levels at the live price
    price, atr, sign = view.price, read.atr15, side_sign(plan.direction)
    plan = plan_at_price(plan, price, atr, costs)
    if sign * (price - plan.stop_loss) <= 0:
        return verdict(SKIP, Reason("SIG_STOPPED_OUT", f"price {price:.2f} is already through the "
                                                       f"stop {plan.stop_loss:.2f}"), plan=plan)
    if plan.entries:
        stated = sum(plan.entries) / len(plan.entries)
        passed = [tp for tp in plan.take_profits if sign * (tp - stated) > 0 and sign * (tp - price) <= 0]
        if passed:
            return verdict(SKIP, Reason("SIG_TARGET_REACHED", f"price {price:.2f} already passed the "
                                                              f"target {passed[0]:.2f}"), plan=plan)
    if plan.target is None:
        return verdict(SKIP, Reason("SIG_NO_TARGET_AHEAD", "no take-profit lies beyond the entry"),
                       plan=plan)
    risk = sign * (plan.planned_entry - plan.stop_loss)
    if risk <= 0:
        return verdict(SKIP, Reason("SIG_LEVELS_INCONSISTENT", "the stop is on the wrong side of "
                                                               "the entry"), plan=plan)
    if view.market == "closing":
        return verdict(SKIP, Reason("SESSION_CLOSING", "the weekly close is less than "
                                                       f"{rules.close_buffer.total_seconds() / 60:.0f} "
                                                       "min away"), plan=plan)
    if risk < rules.min_stop_atr * atr:
        return verdict(SKIP, Reason("SIG_STOP_TOO_TIGHT", f"stop {risk:.2f} from the entry is under "
                                                          f"{rules.min_stop_atr:g} M15 ATR "
                                                          f"({rules.min_stop_atr * atr:.2f})"), plan=plan)
    if risk > rules.max_stop_atr * atr:
        return verdict(SKIP, Reason("SIG_STOP_TOO_WIDE", f"stop {risk:.2f} from the entry is over "
                                                         f"{rules.max_stop_atr:g} M15 ATR "
                                                         f"({rules.max_stop_atr * atr:.2f})"), plan=plan)
    if plan.rr_net is None or plan.rr_net < rules.signal_min_rr:
        return verdict(SKIP, Reason("SIG_RR_BELOW_MIN", f"net reward-to-risk {plan.rr_net} to "
                                                        f"{plan.target:.2f} is below "
                                                        f"{rules.signal_min_rr:g}"), plan=plan)

    checks, weaknesses = _checks(plan, read, rules)

    # 4. timing
    timing: List[Reason] = []
    if plan.pending:
        timing.append(Reason("ENTRY_NOT_REACHED", f"entry {plan.planned_entry:.2f} has not traded "
                                                  f"(price {price:.2f})"))
    elif plan.entries:
        far = max(plan.entries) if plan.direction == LONG else min(plan.entries)
        stated_risk = sign * (far - plan.stop_loss)
        run = sign * (price - far)
        if stated_risk > 0 and run > rules.max_chase_r * stated_risk:
            timing.append(Reason("PRICE_EXTENDED", f"price is {run:.2f} past the stated entry "
                                                   f"{far:.2f}, over {rules.max_chase_r:g}R"))
            weaknesses.append(Reason("ENTRY_EXTENDED", f"entering now is {run:.2f} worse than the "
                                                       f"stated entry {far:.2f}"))
    if news is not None and news.available and news.blocked:
        timing.append(Reason("NEWS_WINDOW", news.reason))

    # 5. structure
    other = SHORT if plan.direction == LONG else LONG
    same = [p for p in (proposals["B"][plan.direction], proposals["C"][plan.direction]) if p]
    against = [p for p in (proposals["B"][other], proposals["C"][other]) if p]
    same_ready = [p for p in same if p.status == READY]
    against_ready = [p for p in against if p.status == READY]
    want = "bullish" if plan.direction == LONG else "bearish"
    h1, h4 = read.frame(Timeframe.H1).bias, read.frame(Timeframe.H4).bias
    confirmed: Tuple[str, ...] = ()
    if same_ready and against_ready:
        structure = (WAIT, Reason("CONFLICTING_SETUPS", "setups are READY in both directions: " +
                                  ", ".join(f"{p.strategy} {p.direction}" for p in same_ready + against_ready)))
    elif same_ready:
        confirmed = tuple(p.strategy for p in same_ready)
        structure = (ENTER, Reason("CONFIRMED_BY_" + "_".join(confirmed),
                                   "; ".join(f"{p.strategy} ({p.name}): " + "; ".join(p.evidence)
                                             for p in same_ready)))
    elif against_ready:
        structure = (SKIP, Reason("OPPOSING_SETUP", "; ".join(
            f"{p.strategy} {p.direction} is READY: " + "; ".join(p.evidence) for p in against_ready)))
    elif h1 not in (want, "neutral"):
        forming = [p for p in same if p.strategy == "C" and p.status == WAITING]
        structure = (WAIT, Reason("REVERSAL_UNCONFIRMED", f"H1 bias is {h1}; a sweep reversal is "
                                                          f"forming — needs {forming[0].confirmation}")) \
            if forming else (SKIP, Reason("COUNTER_H1_STRUCTURE", f"H1 bias is {h1}, against the signal"))
    elif any(p.status == WAITING for p in same) or h1 == want:
        waiting = [p for p in same if p.status == WAITING]
        need = waiting[0].confirmation if waiting else "an M15 setup and an M5 break in its direction"
        structure = (WAIT, Reason("AWAITING_CONFIRMATION", f"H1 bias {h1}; needs {need}"))
    elif h4 == want:
        structure = (WAIT, Reason("H1_NEUTRAL", f"H4 bias {h4} but H1 has no directional structure"))
    else:
        structure = (NO_TRADE, Reason("NO_STRUCTURAL_EVIDENCE",
                                      f"H4 {h4}, H1 {h1}: no structure supports a {plan.direction}"))

    final = structure[0]
    if timing and SEVERITY[WAIT] > SEVERITY[final]:
        final = WAIT
    reasons = (structure[1], *timing) if final != ENTER else (structure[1],)
    return Verdict(final, tuple(reasons), plan, tuple(weaknesses), checks,
                   confirmed if final == ENTER else ())


def _checks(plan: SignalPlan, read: MarketRead, rules: Rules
            ) -> Tuple[Tuple[Tuple[str, bool], ...], List[Reason]]:
    d, sign = plan.direction, side_sign(plan.direction)
    want = "bullish" if d == LONG else "bearish"
    setup_found = find_pullback(read, d) is not None or find_sweep(read, d) is not None
    m5 = m5_break(read, d, read.view.as_of - timedelta(days=1))   # recency is m5_break's own rule
    weaknesses: List[Reason] = []
    swing = read.nearest_swing_beyond(d, plan.planned_entry)
    stop_ok = swing is not None and sign * (swing.price - plan.stop_loss) > 0
    if swing is not None and not stop_ok:
        weaknesses.append(Reason("SL_INSIDE_STRUCTURE", f"stop {plan.stop_loss:.2f} sits "
                                 f"{'above' if d == LONG else 'below'} the M15 swing "
                                 f"{'low' if d == LONG else 'high'} {swing.price:.2f} — a sweep of "
                                 f"it would stop the trade first"))
    level = read.target(d, plan.planned_entry)
    if level is not None and plan.target is not None and sign * (plan.target - level.price) > 0:
        weaknesses.append(Reason("TP_BEYOND_LIQUIDITY", f"target {plan.target:.2f} lies beyond the "
                                                        f"resting {level.label()}"))
    checks = (
        ("H4 bias with the signal", read.frame(Timeframe.H4).bias == want),
        ("H1 bias with the signal", read.frame(Timeframe.H1).bias == want),
        ("M15 setup in its direction", setup_found),
        ("M5 break in its direction", m5 is not None),
        ("stop beyond M15 structure", stop_ok),
        (f"net R:R >= {rules.signal_min_rr:g}", plan.rr_net is not None and plan.rr_net >= rules.signal_min_rr),
    )
    return checks, weaknesses
