"""Weekly trading-session boundaries for spot metals (XAUUSD, XAGUSD).

Spot gold trades from **Sunday 18:00 to Friday 17:00 New York time**. New
York observes US daylight saving (second Sunday of March 02:00 to first
Sunday of November 02:00, local), so in UTC:

* summer (EDT, UTC-4): closes Friday 21:00, reopens Sunday 22:00;
* winter (EST, UTC-5): closes Friday 22:00, reopens Sunday 23:00.

Verified against the provider's own timestamps (``research/session_boundaries.py``):
in every DST state, the hour just inside each boundary has bars in almost
every week, and the hour just outside has bars only in the weeks Twelve Data
also serves weekend stub quotes.

Inside the closure Twelve Data still serves sparse quotes — measured up to
$421 away from Friday's close — that no order could have traded at. Those
bars are dropped by ``market_data.clean_candle_series``. Only bars lying
ENTIRELY inside a closure are dropped: a bar that also covers live trading
(an H4 bar spanning the Sunday reopen, say) is real data and is kept. The
decision uses each bar's own timestamp only — nothing later.

The US DST rule is coded here rather than read from a time-zone database:
Windows has no IANA database and the deployed image need not carry one, and
the rule has been fixed since 2007.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc

# Symbols on the metals weekly session. Anything else is left untouched: FX
# reopens an hour earlier and crypto trades through the weekend.
METALS = frozenset({"XAUUSD", "XAGUSD"})

CLOSE_LOCAL_MINUTES = 17 * 60          # Friday 17:00 New York
REOPEN_LOCAL_MINUTES = 18 * 60         # Sunday 18:00 New York


def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def new_york_offset(ts: datetime) -> timedelta:
    """New York's UTC offset at the UTC instant ``ts``: -4 h in DST, else -5 h."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    year = ts.astimezone(UTC).year
    start = datetime.combine(_nth_sunday(year, 3, 2), datetime.min.time(), UTC) + timedelta(hours=7)
    end = datetime.combine(_nth_sunday(year, 11, 1), datetime.min.time(), UTC) + timedelta(hours=6)
    return timedelta(hours=-4) if start <= ts.astimezone(UTC) < end else timedelta(hours=-5)


def in_weekly_closure(ts: datetime) -> bool:
    """True when the instant ``ts`` falls inside the metals weekly closure."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    local = ts.astimezone(UTC) + new_york_offset(ts)
    minutes = local.hour * 60 + local.minute
    wd = local.weekday()
    return (wd == 4 and minutes >= CLOSE_LOCAL_MINUTES) or wd == 5 \
        or (wd == 6 and minutes < REOPEN_LOCAL_MINUTES)


def bar_inside_weekly_closure(open_ts: datetime, duration: timedelta) -> bool:
    """True when the whole bar ``[open_ts, open_ts + duration)`` lies inside one
    weekly closure — no part of it could contain a real trade."""
    if duration >= timedelta(days=1):
        return False                       # daily bars are handled by the range rule
    last_instant = open_ts + duration - timedelta(microseconds=1)
    return in_weekly_closure(open_ts) and in_weekly_closure(last_instant)


def follows_weekly_session(symbol: str) -> bool:
    return symbol.strip().upper().replace("/", "") in METALS
