"""Statistics layer (RFC-009): read-only aggregation over persisted analyses.

Every number here is derived fresh, on every call, from rows already
written by :class:`storage.Storage` (RFC-003, extended RFC-009) — never
recalculated by Claude, never cached, never tracked as a separate mutable
in-memory counter. A call to any method below is a single round trip:
:meth:`storage.Storage.fetch_analyses` for the raw rows, then a pure,
stateless aggregation pass over them. There is exactly one source of
truth (the SQLite table) and exactly one way any number here can change:
a new row being written to it.

This intentionally does NOT duplicate storage: it holds a reference to an
existing :class:`storage.Storage` instance and never opens its own
database connection, never keeps a running total between calls, and adds
no new tables.

Two categories of the requested metrics are honestly reported as absent
rather than fabricated, because nothing in this project currently produces
the data they'd need:

- Win/Loss: no engine built so far tracks trade *outcomes* (whether a
  stop-loss or take-profit was actually hit after the fact) — that would
  require a future price-tracking mechanism this project doesn't have yet.
  ``StatisticsSummary.wins``/``losses`` are always ``None`` today.
- verdict/trend/SMC-object metrics are only ever populated for rows where
  the optional ``decision``/``structure``/``smc`` arguments were supplied
  to ``Storage.record()`` — which nothing in ``pipeline.py`` currently
  does (RFC-004 through RFC-008 all deliberately stopped short of wiring
  those engines into the message pipeline). Until a future RFC does that
  wiring, these distributions will correctly show as empty/zero in real
  usage — this module reports that state accurately rather than inventing
  numbers, exactly like every other "not yet available" case in this
  project (see e.g. market_data.py's own "report unavailable" requirement).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from market_data import Timeframe
from storage import AnalysisRecord, Storage

_CONFIDENCE_BUCKET_LABELS = tuple(f"{i * 10}-{i * 10 + 10}%" for i in range(10))


def _confidence_bucket(confidence: float) -> str:
    """Which 10-point bucket (0-10%, 10-20%, ..., 90-100%) a 0.0-1.0
    confidence value falls into. Clamped defensively; SignalAnalysis's
    own confidence is meant to be 0.0-1.0 but isn't schema-constrained."""
    pct = max(0.0, min(1.0, confidence)) * 100
    index = min(9, int(pct // 10))
    return _CONFIDENCE_BUCKET_LABELS[index]


@dataclass(frozen=True)
class StatisticsSummary:
    """One aggregation result — the same shape for summary()/today()/
    this_week()/this_month()/by_symbol()/by_timeframe(), only the
    underlying row set differs."""

    total_analyses: int
    total_signals: int
    verdict_counts: Dict[str, int]                  # buy/sell/wait/reject -> count
    wins: Optional[int]                              # always None today — see module docstring
    losses: Optional[int]                            # always None today — see module docstring
    average_signal_confidence: Optional[float]        # 0.0-1.0, None if no rows
    average_decision_confidence: Optional[float]      # 0-100, None if no row has a decision
    confidence_distribution: Dict[str, int]           # bucket label -> count
    direction_distribution: Dict[str, int]            # long/short/none -> count
    trend_distribution: Dict[str, int]                # bullish/bearish/ranging/unknown/not_recorded -> count
    smc_object_frequencies: Dict[str, int]             # category -> total count across all rows
    source_distribution: Dict[str, int]                # regex/claude -> count

    def format(self) -> str:
        """A concise, human-readable summary for /stats and /status.

        Deliberately omits the full confidence_distribution histogram
        (10 buckets) to stay Telegram-message-sized — the raw dataclass
        field is still available to any caller that wants the full detail.
        """
        lines = [f"{self.total_analyses} analysed · {self.total_signals} signals"]

        if self.average_signal_confidence is not None:
            lines.append(f"Avg signal confidence: {self.average_signal_confidence * 100:.0f}%")
        if self.average_decision_confidence is not None:
            lines.append(f"Avg decision confidence: {self.average_decision_confidence:.0f}%")

        lines.append(
            "Verdicts: " + (
                ", ".join(f"{k}={v}" for k, v in sorted(self.verdict_counts.items()))
                if self.verdict_counts else "not recorded yet"
            )
        )
        if self.direction_distribution:
            lines.append("Direction: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.direction_distribution.items())
            ))
        if self.trend_distribution:
            lines.append("Trend: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.trend_distribution.items())
            ))
        if self.source_distribution:
            lines.append("Source: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.source_distribution.items())
            ))
        if self.smc_object_frequencies:
            lines.append("SMC objects: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.smc_object_frequencies.items())
            ))

        lines.append(
            "Win/Loss: not tracked yet" if self.wins is None and self.losses is None
            else f"Win/Loss: {self.wins}/{self.losses}"
        )
        return "\n".join(lines)


def _aggregate(records: Tuple[AnalysisRecord, ...]) -> StatisticsSummary:
    """Pure function: row list in, summary out. No I/O, no state."""
    total = len(records)
    signals = sum(1 for r in records if r.is_signal)

    verdict_counts: Dict[str, int] = {}
    direction_distribution: Dict[str, int] = {}
    trend_distribution: Dict[str, int] = {}
    confidence_distribution: Dict[str, int] = {}
    smc_object_frequencies: Dict[str, int] = {}
    source_distribution: Dict[str, int] = {}
    decision_confidences = []

    for r in records:
        if r.decision_verdict is not None:
            verdict_counts[r.decision_verdict] = verdict_counts.get(r.decision_verdict, 0) + 1
        if r.decision_confidence is not None:
            decision_confidences.append(r.decision_confidence)

        direction_key = r.direction or "none"
        direction_distribution[direction_key] = direction_distribution.get(direction_key, 0) + 1

        trend_key = r.structure_trend or "not_recorded"
        trend_distribution[trend_key] = trend_distribution.get(trend_key, 0) + 1

        bucket = _confidence_bucket(r.confidence)
        confidence_distribution[bucket] = confidence_distribution.get(bucket, 0) + 1

        for category, count in r.smc_counts.items():
            smc_object_frequencies[category] = smc_object_frequencies.get(category, 0) + count

        source_distribution[r.source] = source_distribution.get(r.source, 0) + 1

    return StatisticsSummary(
        total_analyses=total,
        total_signals=signals,
        verdict_counts=verdict_counts,
        wins=None,
        losses=None,
        average_signal_confidence=(sum(r.confidence for r in records) / total) if total else None,
        average_decision_confidence=(
            sum(decision_confidences) / len(decision_confidences) if decision_confidences else None
        ),
        confidence_distribution=confidence_distribution,
        direction_distribution=direction_distribution,
        trend_distribution=trend_distribution,
        smc_object_frequencies=smc_object_frequencies,
        source_distribution=source_distribution,
    )


class Statistics:
    """Read-only statistics over a Storage instance. Holds no state of its
    own beyond the Storage reference — every method is a fresh query."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    async def summary(self) -> StatisticsSummary:
        """All-time statistics over every stored analysis."""
        records = await self._storage.fetch_analyses()
        return _aggregate(records)

    async def today(self) -> StatisticsSummary:
        """Statistics for analyses recorded since midnight UTC today."""
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        records = await self._storage.fetch_analyses(since=start)
        return _aggregate(records)

    async def this_week(self) -> StatisticsSummary:
        """Statistics since Monday 00:00 UTC of the current ISO week."""
        now = datetime.now(timezone.utc)
        start_of_today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        start = start_of_today - timedelta(days=start_of_today.weekday())
        records = await self._storage.fetch_analyses(since=start)
        return _aggregate(records)

    async def this_month(self) -> StatisticsSummary:
        """Statistics since the 1st of the current UTC month, 00:00."""
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        records = await self._storage.fetch_analyses(since=start)
        return _aggregate(records)

    async def by_symbol(self, symbol: str) -> StatisticsSummary:
        """Statistics for one symbol (normalised to uppercase, matching
        how signal_parser.py/models.py normalise setup.symbol)."""
        records = await self._storage.fetch_analyses(symbol=symbol.strip().upper())
        return _aggregate(records)

    async def by_timeframe(self, timeframe: Timeframe) -> StatisticsSummary:
        """Statistics for one Timeframe. Only ever non-trivial for rows
        where structure was supplied to Storage.record() — see the module
        docstring."""
        records = await self._storage.fetch_analyses(timeframe=timeframe)
        return _aggregate(records)
