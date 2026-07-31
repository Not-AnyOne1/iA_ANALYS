"""The analysis pipeline: queue -> Claude -> terminal (and optional JSONL file).

Keeping analysis off the Telethon update loop means a slow API call can never
stall message reception; bursts are absorbed by the queue instead.

Integration phase (post-RFC-010): every completed signal is additionally
enriched, best-effort, through the engines built in RFC-004 through RFC-008,
in this order:

    MarketDataService -> StructureEngine -> SMCEngine -> ScoringEngine -> DecisionEngine

Enrichment only runs for confirmed signals with a known symbol
(``analysis.is_signal and analysis.setup.symbol``) — there is nothing for
these engines to analyse otherwise. It is entirely optional at every step:
if ``market_data`` wasn't configured, or the symbol/timeframe isn't
available from the provider, or any enrichment engine raises unexpectedly,
the analysis/notification/storage of the plain ``SignalAnalysis`` proceeds
exactly as it always has — enrichment can only ever ADD fields to what
gets stored, never block or alter the base pipeline. Nothing is fabricated:
an unavailable step simply leaves the corresponding field absent (``None``)
in what's passed to ``storage.record()``, which already treats those
fields as optional (RFC-009) and stores old-shape rows identically to
before this phase existed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TextIO

from claude_client import AnalysisError, ClaudeAnalyzer
from config import Settings
from decision_engine import DecisionEngine, DecisionResult
from formatter import Formatter
from market_context import MarketContextBuilder
from market_data import MarketDataError, MarketDataService
from models import SignalAnalysis
from report import render_telegram_report, render_text as render_report
from scoring_engine import ScoringEngine
from signal_parser import ParseResult, parse_signal
from smc_engine import SMCAnalysis, SMCEngine
from storage import Storage
from structure_engine import StructureAnalysis, StructureEngine
from telegram_bot import TelegramBot
from telegram_client import IncomingMessage
from trade_decision import TradeDecisionEngine

log = logging.getLogger(__name__)


@dataclass
class Stats:
    """Counters reported on shutdown."""

    received: int = 0
    signals: int = 0
    non_signals: int = 0
    failures: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    # RFC-001 shadow-mode calibration: how often the regex parser's verdict
    # would have matched Claude's, had it been trusted to answer directly.
    shadow_agree: int = 0
    shadow_disagree: int = 0

    def summary(self) -> str:
        categories = ", ".join(
            f"{name}={count}" for name, count in sorted(self.by_category.items())
        )
        sources = ", ".join(
            f"{name}={count}" for name, count in sorted(self.by_source.items())
        )
        text = (
            f"{self.received} analysed · {self.signals} signals · "
            f"{self.non_signals} non-signals · {self.failures} failures"
            + (f" · [{categories}]" if categories else "")
            + (f" · source: [{sources}]" if sources else "")
        )
        if self.shadow_agree or self.shadow_disagree:
            total = self.shadow_agree + self.shadow_disagree
            text += (
                f" · shadow agreement: {self.shadow_agree}/{total} "
                f"({self.shadow_agree / total:.0%})"
            )
        return text


class AnalysisPipeline:
    """Runs ``worker_count`` concurrent consumers over the message queue."""

    def __init__(
        self,
        settings: Settings,
        queue: "asyncio.Queue[IncomingMessage]",
        analyzer: ClaudeAnalyzer,
        formatter: Formatter,
        bot: Optional[TelegramBot] = None,
        storage: Optional[Storage] = None,
        market_data: Optional[MarketDataService] = None,
        structure_engine: Optional[StructureEngine] = None,
        smc_engine: Optional[SMCEngine] = None,
        scoring_engine: Optional[ScoringEngine] = None,
        decision_engine: Optional[DecisionEngine] = None,
        context_builder: Optional["MarketContextBuilder"] = None,
        trade_decision_engine: Optional["TradeDecisionEngine"] = None,
        emit_reports: bool = True,
    ) -> None:
        self._settings = settings
        self._queue = queue
        self._analyzer = analyzer
        self._formatter = formatter
        self._bot = bot
        self._storage = storage
        self._market_data = market_data
        self._structure_engine = structure_engine
        self._smc_engine = smc_engine
        self._scoring_engine = scoring_engine
        self._decision_engine = decision_engine
        self._context_builder = context_builder
        self._trade_decision_engine = trade_decision_engine
        self._emit_reports = emit_reports
        self._print_lock = asyncio.Lock()
        self._jsonl: TextIO | None = None
        self.stats = Stats()

    async def __aenter__(self) -> "AnalysisPipeline":
        if self._settings.jsonl_output:
            path: Path = self._settings.jsonl_output
            path.parent.mkdir(parents=True, exist_ok=True)
            # errors="replace" mirrors what main.py already does for stdout.
            # Telegram text can contain lone surrogates (malformed UTF-16 from
            # some clients); json.dumps(ensure_ascii=False) passes them
            # through, and a plain utf-8 handle then raises UnicodeEncodeError
            # mid-write — which cost the whole message, since the JSONL write
            # sits before storage/notification in _process.
            self._jsonl = path.open("a", encoding="utf-8", errors="replace")
            log.info("Appending structured results to %s", path)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None

    async def run(self) -> None:
        """Start the workers and block until cancelled."""
        workers = [
            asyncio.create_task(self._worker(i + 1), name=f"analysis-worker-{i + 1}")
            for i in range(self._settings.worker_count)
        ]
        log.info("Started %d analysis worker(s)", len(workers))
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

    async def drain(self, timeout: float = 30.0) -> None:
        """Wait for in-flight and queued messages, up to ``timeout`` seconds."""
        if self._queue.empty():
            return
        log.info("Draining %d queued message(s)...", self._queue.qsize())
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("Timed out draining the queue; %d left", self._queue.qsize())

    # ------------------------------------------------------------------ worker

    async def _worker(self, index: int) -> None:
        while True:
            message = await self._queue.get()
            try:
                await self._process(message)
            except asyncio.CancelledError:
                self._queue.task_done()
                raise
            except Exception:  # noqa: BLE001 - one bad message must not kill a worker
                log.exception("Worker %d crashed on message %s", index, message.id)
                self.stats.failures += 1
                self._queue.task_done()
            else:
                self._queue.task_done()

    async def _process(self, message: IncomingMessage) -> None:
        analysis = await self._get_analysis(message)
        if analysis is None:
            return  # failure already logged/rendered inside _analyze_via_claude

        self.stats.received += 1
        if analysis.is_signal:
            self.stats.signals += 1
        else:
            self.stats.non_signals += 1
        self.stats.by_category[analysis.category] = (
            self.stats.by_category.get(analysis.category, 0) + 1
        )
        self.stats.by_source[analysis.source] = (
            self.stats.by_source.get(analysis.source, 0) + 1
        )

        await self._emit(self._formatter.render(message, analysis))

        if self._jsonl is not None:
            record = analysis.to_record(message=message)
            self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl.flush()

        structure, smc, decision = await self._enrich(message, analysis)

        if self._storage is not None:
            # Belt-and-suspenders, same reasoning as the bot.notify() wrapper
            # below: Storage.record() already guarantees it never raises, but
            # a storage outage must never be able to stop ingestion/analysis
            # regardless of what future changes happen inside storage.py.
            try:
                await self._storage.record(
                    message, analysis, decision=decision, structure=structure, smc=smc
                )
            except Exception:  # noqa: BLE001 - see comment above
                log.exception(
                    "Unexpected error recording analysis %s to storage", message.id
                )

        # Telegram delivery deliberately does NOT happen here. This point in
        # _process only has the parser's extraction — an intermediate guess
        # made before any market data was consulted. Sending it would mean
        # two messages per signal, the first of them premature. The single
        # Telegram message is the finished report, sent from
        # _validate_with_market_data() once the whole pipeline has run.
        # The terminal render and the JSONL/SQLite writes above are unchanged.
        if self._bot is not None and self._context_builder is None:
            log.debug(
                "telegram_report_skipped message_id=%s reason=market_validation_disabled "
                "(no final report is produced without a market data provider)",
                message.id,
            )

    async def _validate_with_market_data(
        self, message: IncomingMessage, analysis: SignalAnalysis
    ) -> tuple[Optional[StructureAnalysis], Optional[SMCAnalysis], object]:
        """Market-validated path: full context -> risk gate -> ENTER/WAIT/SKIP.

        Used when a ``MarketContextBuilder`` is configured. Builds the
        complete factual picture (multi-timeframe indicators, structure,
        SMC, levels, session, news, trade quality, risk gate), asks
        ``TradeDecisionEngine`` for a verdict over those facts alone, and
        emits the desk report.

        Returns the same ``(structure, smc, decision)`` triple as
        :meth:`_enrich` so the storage call is unchanged. ``storage.record``
        reads only ``verdict.value``/``confidence``/``source.value`` from the
        decision, which ``TradeDecision`` exposes exactly like
        ``DecisionResult`` — so both paths persist identically.

        Never raises: any failure falls back to the plain analysis, exactly
        like the pre-existing enrichment path.
        """
        builder = self._context_builder
        assert builder is not None  # guarded by the caller

        try:
            context = await builder.build(analysis.setup.symbol, analysis.setup)
        except Exception:  # noqa: BLE001 - market validation must never break ingestion
            log.exception("Market context build failed for message %s", message.id)
            return (None, None, None)

        decision = None
        if self._trade_decision_engine is not None:
            try:
                decision = await self._trade_decision_engine.decide(context)
            except Exception:  # noqa: BLE001
                log.exception("Trade decision failed for message %s", message.id)

        if self._emit_reports:
            try:
                await self._emit(render_report(context, decision))
            except Exception:  # noqa: BLE001 - a rendering bug must not lose the analysis
                log.exception("Report rendering failed for message %s", message.id)

        # The single Telegram message for this signal: the same finished
        # report just written to the terminal, nothing before it. Wrapped
        # even though send_report() already guarantees it never raises —
        # delivery must never be able to stop the pipeline.
        if self._bot is not None:
            try:
                await self._bot.send_report(
                    message, render_telegram_report(context, decision)
                )
            except Exception:  # noqa: BLE001 - see comment above
                log.exception(
                    "Unexpected error sending the Telegram report for message %s", message.id
                )

        if decision is not None:
            log.info(
                "market_validated_decision message_id=%s symbol=%s verdict=%s "
                "confidence=%s source=%s risk_approved=%s",
                message.id, context.symbol, decision.verdict.value,
                decision.confidence, decision.source.value,
                context.risk.approved if context.risk else None,
            )

        return context.structure, context.smc, decision

    async def _enrich(
        self, message: IncomingMessage, analysis: SignalAnalysis
    ) -> tuple[Optional[StructureAnalysis], Optional[SMCAnalysis], Optional[DecisionResult]]:
        """Best-effort MarketDataService -> StructureEngine -> SMCEngine ->
        ScoringEngine -> DecisionEngine chain for one confirmed signal.

        Returns (None, None, None) whenever enrichment doesn't apply or any
        step is unavailable/fails — never raises, and never fabricates a
        field: an absent step is simply left out of what gets stored,
        exactly as RFC-009's optional Storage.record() parameters expect.
        """
        no_enrichment = (None, None, None)

        if not (analysis.is_signal and analysis.setup.symbol):
            return no_enrichment  # nothing for these engines to analyse

        # Market-validated path takes precedence when configured. It is a
        # superset of the chain below (same engines, plus indicators,
        # levels, session, news and the risk gate), so running both would
        # duplicate every provider call for no benefit.
        if self._context_builder is not None:
            return await self._validate_with_market_data(message, analysis)

        if self._market_data is None:
            return no_enrichment  # market data not configured — degrade silently

        try:
            candles = await self._market_data.get_candles(
                analysis.setup.symbol, self._settings.market_data_timeframe
            )
        except MarketDataError as exc:
            log.info(
                "Market data unavailable for %s (message %s): %s — "
                "storing the analysis without structure/SMC/decision enrichment",
                analysis.setup.symbol, message.id, exc,
            )
            return no_enrichment

        if self._structure_engine is None or self._smc_engine is None or self._scoring_engine is None:
            return no_enrichment  # deterministic engines not configured

        try:
            structure = self._structure_engine.analyze(candles)
            smc = self._smc_engine.analyze(candles, structure)
            scoring = self._scoring_engine.score(structure, smc)
            decision: Optional[DecisionResult] = None
            if self._decision_engine is not None:
                decision = await self._decision_engine.decide(analysis, structure, smc, scoring)
        except Exception:  # noqa: BLE001 - an enrichment bug must never break the base pipeline
            log.exception(
                "Unexpected error enriching message %s with structure/SMC/scoring/decision",
                message.id,
            )
            return no_enrichment

        return structure, smc, decision

    async def _get_analysis(self, message: IncomingMessage) -> Optional[SignalAnalysis]:
        """Route a message per RFC-001's ``SIGNAL_PARSER_MODE``.

        - ``off``: unchanged pre-RFC-001 behaviour — Claude analyses every
          message.
        - ``shadow``: Claude still analyses every message (identical output
          to ``off``); the regex parser's verdict is only compared and
          counted, never used, so this mode is safe to leave running
          indefinitely while calibrating.
        - ``active``: a confident regex verdict is used directly and Claude
          is skipped; an ambiguous one still falls back to Claude exactly as
          today.
        """
        mode = self._settings.signal_parser_mode

        if mode == "off":
            return await self._analyze_via_claude(message)

        result = parse_signal(message)

        if mode == "shadow":
            claude_analysis = await self._analyze_via_claude(message)
            if claude_analysis is not None:
                self._record_shadow_comparison(message, result, claude_analysis)
            return claude_analysis

        # mode == "active"
        if result.analysis is not None:
            log.debug(
                "Regex parser matched message %s (%s) — skipping Claude",
                message.id, result.reason,
            )
            return result.analysis
        log.debug(
            "Regex parser deferred message %s to Claude (%s)",
            message.id, result.reason,
        )
        return await self._analyze_via_claude(message)

    async def _analyze_via_claude(self, message: IncomingMessage) -> Optional[SignalAnalysis]:
        try:
            return await self._analyzer.analyze(message)
        except AnalysisError as exc:
            self.stats.failures += 1
            log.error("Analysis failed for message %s: %s", message.id, exc)
            await self._emit(self._formatter.render_error(message, exc))
            return None

    def _record_shadow_comparison(
        self,
        message: IncomingMessage,
        result: ParseResult,
        claude_analysis: SignalAnalysis,
    ) -> None:
        if result.analysis is None:
            log.info(
                "[shadow] message %s: regex deferred (%s) — claude said "
                "category=%s is_signal=%s",
                message.id, result.reason, claude_analysis.category, claude_analysis.is_signal,
            )
            return

        agree = (
            result.analysis.is_signal == claude_analysis.is_signal
            and result.analysis.category == claude_analysis.category
            and result.analysis.setup == claude_analysis.setup
        )
        if agree:
            self.stats.shadow_agree += 1
        else:
            self.stats.shadow_disagree += 1

        log_fn = log.info if agree else log.warning
        log_fn(
            "[shadow] message %s: regex/claude %s (%s) — regex=%s/%s claude=%s/%s",
            message.id, "agree" if agree else "DISAGREE", result.reason,
            result.analysis.category, result.analysis.is_signal,
            claude_analysis.category, claude_analysis.is_signal,
        )

    async def _emit(self, text: str) -> None:
        """Serialise console writes so concurrent workers never interleave."""
        async with self._print_lock:
            print(text, flush=True)
