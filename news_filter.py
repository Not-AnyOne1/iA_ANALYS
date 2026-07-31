"""High-impact economic event filter.

Trading into a scheduled high-impact release is a known way to get stopped
out by a spread widening rather than by being wrong, so the risk engine
treats an imminent event as a hard rejection.

Two deliberate design points:

**Pluggable calendar, no fabricated schedule.** :class:`EconomicCalendar` is
a Protocol. The only implementation shipped here is
:class:`StaticCalendar`, which holds events an operator supplies. This
module will *never* invent event times — an empty calendar reports "no
known events" and says so, rather than guessing that (say) NFP is the first
Friday of the month. A wrong guess here silently blocks or allows real
trades.

**Fails open, loudly.** If no calendar is configured,
:meth:`NewsFilter.check` returns ``available=False``. The risk engine can
then decide; it does not silently behave as though the window is clear.
Callers can tell "no events" apart from "no data".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional, Protocol, Sequence, Tuple


class Impact(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Event names that are high-impact by convention. Used by
# :func:`classify_impact` so an operator can add a calendar entry without
# having to label its severity by hand.
_HIGH_IMPACT_KEYWORDS: Tuple[str, ...] = (
    "fomc", "nfp", "non-farm", "non farm", "nonfarm", "payroll",
    "cpi", "ppi", "inflation",
    "interest rate", "rate decision", "rate statement",
    "fed ", "federal reserve", "powell",
    "ecb", "lagarde",
    "boe", "bank of england", "bailey",
    "boj", "bank of japan",
    "gdp", "unemployment rate", "retail sales",
    "jackson hole", "press conference",
)

_MEDIUM_IMPACT_KEYWORDS: Tuple[str, ...] = (
    "pmi", "ism", "consumer confidence", "durable goods",
    "trade balance", "jobless claims", "housing",
)


def classify_impact(title: str) -> Impact:
    """Best-effort severity from an event title, by keyword."""
    lowered = title.lower()
    if any(k in lowered for k in _HIGH_IMPACT_KEYWORDS):
        return Impact.HIGH
    if any(k in lowered for k in _MEDIUM_IMPACT_KEYWORDS):
        return Impact.MEDIUM
    return Impact.LOW


@dataclass(frozen=True)
class EconomicEvent:
    """One scheduled release."""

    title: str
    scheduled_at: datetime
    impact: Impact = Impact.HIGH
    currency: Optional[str] = None     # e.g. "USD"; None = affects everything

    @classmethod
    def create(cls, title: str, scheduled_at: datetime,
               currency: Optional[str] = None) -> "EconomicEvent":
        """Build an event, classifying its impact from the title."""
        return cls(title=title, scheduled_at=scheduled_at,
                   impact=classify_impact(title), currency=currency)


class EconomicCalendar(Protocol):
    """Source of scheduled events. Implement this to plug in a real feed."""

    name: str

    def events_between(self, start: datetime, end: datetime) -> Sequence[EconomicEvent]: ...


@dataclass
class StaticCalendar:
    """A calendar of operator-supplied events.

    The only implementation shipped. Deliberately dumb: it returns exactly
    what it was given, so nothing here can invent a release time.
    """

    events: List[EconomicEvent] = field(default_factory=list)
    name: str = "static"

    def events_between(self, start: datetime, end: datetime) -> Sequence[EconomicEvent]:
        return [e for e in self.events if start <= e.scheduled_at <= end]


@dataclass(frozen=True)
class NewsStatus:
    """Verdict for one symbol at one moment."""

    available: bool                      # False = no calendar configured
    blocked: bool                        # True = an imminent high-impact event
    events: Tuple[EconomicEvent, ...]    # what was found in the window
    reason: str

    @property
    def summary(self) -> str:
        if not self.available:
            return "no economic calendar configured"
        if not self.events:
            return "no high-impact events in window"
        soonest = min(self.events, key=lambda e: e.scheduled_at)
        return f"{soonest.title} at {soonest.scheduled_at:%Y-%m-%d %H:%M} UTC"


def currencies_in(symbol: str) -> Tuple[str, ...]:
    """Split a symbol into the currencies it is exposed to.

    XAUUSD -> ("XAU", "USD"); BTCUSDT -> ("BTC", "USDT"). Used to decide
    whether a USD-specific release is even relevant to this instrument.
    """
    s = symbol.upper().replace("/", "")
    if len(s) == 6:
        return (s[:3], s[3:])
    for quote in ("USDT", "USDC", "USD", "EUR", "GBP", "JPY"):
        if s.endswith(quote) and len(s) > len(quote):
            return (s[: -len(quote)], quote)
    return (s,)


class NewsFilter:
    """Answers 'is a high-impact release imminent for this symbol?'"""

    def __init__(
        self,
        calendar: Optional[EconomicCalendar] = None,
        *,
        block_before: timedelta = timedelta(minutes=30),
        block_after: timedelta = timedelta(minutes=15),
    ) -> None:
        self._calendar = calendar
        self._before = block_before
        self._after = block_after

    @property
    def configured(self) -> bool:
        return self._calendar is not None

    def check(self, symbol: str, when: Optional[datetime] = None) -> NewsStatus:
        """Is trading ``symbol`` blocked by news at ``when``?"""
        now = when or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        if self._calendar is None:
            return NewsStatus(
                available=False, blocked=False, events=(),
                reason="no economic calendar configured — news risk unknown",
            )

        window = self._calendar.events_between(now - self._after, now + self._before)
        exposed = set(currencies_in(symbol))
        relevant = tuple(
            e for e in window
            if e.impact is Impact.HIGH
            and (e.currency is None or e.currency.upper() in exposed)
        )

        if not relevant:
            return NewsStatus(True, False, (), "no high-impact events in window")

        soonest = min(relevant, key=lambda e: e.scheduled_at)
        return NewsStatus(
            available=True, blocked=True, events=relevant,
            reason=(
                f"high-impact event '{soonest.title}' at "
                f"{soonest.scheduled_at:%Y-%m-%d %H:%M} UTC is within the "
                f"blackout window"
            ),
        )
