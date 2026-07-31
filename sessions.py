"""Trading session detection (Asia / London / New York).

Deterministic and pure: a timestamp in, a session out. No I/O, no model
involvement.

Session windows are defined in **UTC** and are the widely used retail
definitions:

    Asia    23:00 - 08:00 UTC
    London  07:00 - 16:00 UTC
    New York 12:00 - 21:00 UTC

These deliberately overlap, because the real market does: 07:00-08:00 is
the Asia/London handover and 12:00-16:00 is the London/New York overlap —
the highest-liquidity window of the day. :func:`active_sessions` reports
every session that is open; :func:`primary_session` picks one for display
using a fixed precedence.

Caveat worth stating plainly: these are fixed UTC windows, so they do not
shift with daylight saving. Real exchange hours move by an hour twice a
year. That is a deliberate simplicity trade-off — the windows are wide and
overlapping, so a one-hour DST shift changes the answer only at the very
edges, and the alternative (a full tz database with per-market DST rules)
adds a dependency and far more failure surface than the precision is worth
for a session label.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from enum import Enum
from typing import Tuple


class Session(str, Enum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    # No session open: the gap between the New York close and the Asia open.
    OFF_HOURS = "off_hours"


# (session, open, close). A window whose open > close wraps past midnight.
_WINDOWS: Tuple[Tuple[Session, time, time], ...] = (
    (Session.ASIA, time(23, 0), time(8, 0)),
    (Session.LONDON, time(7, 0), time(16, 0)),
    (Session.NEW_YORK, time(12, 0), time(21, 0)),
)

# Precedence when several are open at once: the more liquid session wins,
# so the London/NY overlap reports as New York and the Asia/London handover
# reports as London.
_PRECEDENCE = (Session.NEW_YORK, Session.LONDON, Session.ASIA)


@dataclass(frozen=True)
class SessionInfo:
    """Which sessions are open at a given instant."""

    primary: Session
    active: Tuple[Session, ...]

    @property
    def is_overlap(self) -> bool:
        """True during a two-session overlap — typically the highest
        liquidity and the widest ranges of the day."""
        return len(self.active) > 1

    @property
    def label(self) -> str:
        if not self.active:
            return Session.OFF_HOURS.value
        return "+".join(s.value for s in self.active)


def _in_window(moment: time, opens: time, closes: time) -> bool:
    if opens <= closes:
        return opens <= moment < closes
    # Wraps midnight (e.g. Asia 23:00 -> 08:00).
    return moment >= opens or moment < closes


def active_sessions(when: datetime) -> Tuple[Session, ...]:
    """Every session open at ``when``, in precedence order.

    Naive datetimes are treated as UTC — the rest of this project works in
    timezone-aware UTC throughout, so a naive value here means the caller
    lost the tzinfo rather than that it meant local time.
    """
    moment = (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).astimezone(
        timezone.utc
    ).timetz().replace(tzinfo=None)

    open_now = {
        session for session, opens, closes in _WINDOWS
        if _in_window(moment, opens, closes)
    }
    return tuple(s for s in _PRECEDENCE if s in open_now)


def primary_session(when: datetime) -> Session:
    """The single most significant session open at ``when``."""
    active = active_sessions(when)
    return active[0] if active else Session.OFF_HOURS


def session_info(when: datetime) -> SessionInfo:
    active = active_sessions(when)
    return SessionInfo(
        primary=active[0] if active else Session.OFF_HOURS,
        active=active,
    )
