"""Entry point: monitor one Telegram group and analyse every message with Claude.

Analysis runs entirely through the local Claude Code CLI (subscription auth
via ``claude auth login``) — no ANTHROPIC_API_KEY is read or required.

Usage:
    python main.py                 # run the monitor
    python main.py --check         # verify Telegram access and the Claude Code CLI
    python main.py --list-chats    # log in and list every accessible chat
    python main.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from claude_client import AnalysisError, ClaudeAnalyzer
from config import ConfigError, Settings
from decision_engine import DecisionEngine
from formatter import Formatter
from logging_setup import configure_logging
from market_context import MarketContextBuilder
from market_data import MarketDataService, TwelveDataProvider
from news_filter import NewsFilter
from pipeline import AnalysisPipeline
from scoring_engine import ScoringEngine
from smc_engine import SMCEngine
from statistics import Statistics
from storage import Storage
from structure_engine import StructureEngine
from telegram_bot import BotSettings, TelegramBot
from trade_decision import TradeDecisionEngine
from telegram_client import IncomingMessage, TelegramError, TelegramMonitor

# formatter.py prints box-drawing characters (e.g. "─"). On Windows, stdout
# falls back to the system codepage (commonly cp1252, which can't represent
# them) whenever it isn't a real console — including the `python main.py >
# signals.txt` redirect this README itself recommends — and print() then
# raises UnicodeEncodeError. Force UTF-8 unconditionally so it can't crash;
# on an already-UTF-8 console this is a no-op.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger("monitor")

EXIT_OK = 0
EXIT_CONFIG = 1
EXIT_RUNTIME = 2

_SAMPLE = """\
🔥 BTC/USDT LONG 🔥
Entry: 61200 - 61450
SL: 60350
TP1: 62400
TP2: 63500
TP3: 65000
Leverage: cross 10x
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="telegram-signal-monitor",
        description="Monitor a Telegram trading-signals group and analyse it with Claude.",
    )
    parser.add_argument(
        "--env-file", default=".env", help="path to the .env file (default: .env)"
    )
    parser.add_argument("--log-level", help="override LOG_LEVEL (DEBUG, INFO, ...)")
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="suppress the JSON block in the terminal output",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate config, Telegram access and the Claude Code CLI, then exit",
    )
    parser.add_argument(
        "--list-chats",
        action="store_true",
        help="log in with the existing session and list every accessible chat, then exit",
    )
    return parser.parse_args(argv)


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    """Ask for a graceful shutdown on Ctrl-C / SIGTERM where supported."""
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):
            # Windows event loops don't support this; KeyboardInterrupt is caught
            # in main() instead.
            pass


async def run_check(settings: Settings, formatter: Formatter) -> int:
    """One-shot verification of both integrations."""
    queue: asyncio.Queue[IncomingMessage] = asyncio.Queue(maxsize=1)
    monitor = TelegramMonitor(settings, queue)

    try:
        analyzer = ClaudeAnalyzer(settings)
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG

    try:
        await monitor.start()
        print("\n✓ Telegram: signed in and the target group is reachable.")

        await analyzer.verify_auth()
        print("✓ Claude Code CLI: signed in (no Anthropic API key used).")

        sample = IncomingMessage(
            id=0,
            chat_id=0,
            chat_title="self-test",
            sender="self-test",
            timestamp=datetime.now(timezone.utc),
            text=_SAMPLE,
        )
        analysis = await analyzer.analyze(sample)
        print("✓ Claude: structured analysis returned for the sample signal.")
        print(formatter.render(sample, analysis))
        return EXIT_OK
    except (TelegramError, AnalysisError) as exc:
        log.error("%s", exc)
        return EXIT_RUNTIME
    finally:
        await monitor.stop()
        await analyzer.aclose()


async def run_list_chats(settings: Settings) -> int:
    """Log in with the existing Telethon session and print every accessible chat.

    Independent of the Claude Code CLI — useful for finding a group's id
    before TELEGRAM_TARGET_CHAT is even set to a real value.
    """
    queue: asyncio.Queue[IncomingMessage] = asyncio.Queue(maxsize=1)
    monitor = TelegramMonitor(settings, queue)
    try:
        chats = await monitor.list_chats()
    except TelegramError as exc:
        log.error("%s", exc)
        return EXIT_RUNTIME
    finally:
        await monitor.stop()

    separator = "-" * 43
    print(separator)
    for chat in chats:
        print(f"Title: {chat.title}")
        print(f"ID: {chat.id}")
        print(f"Type: {chat.type}")
        print(f"Username: {chat.username if chat.username else 'None'}")
        print(separator)

    return EXIT_OK


async def _shutdown(
    *,
    monitor: TelegramMonitor,
    bot: TelegramBot,
    storage: Optional[Storage],
    market_data_provider: Optional[TwelveDataProvider],
    decision_engine: Optional[DecisionEngine],
    trade_decision_engine: Optional[TradeDecisionEngine],
    analyzer: ClaudeAnalyzer,
) -> None:
    """Release every resource ``run_monitor`` acquired, on any exit path.

    Each step is isolated: one component failing to shut down must never
    prevent the others from being released, and shutdown must never raise
    into ``main()`` and mask the real exit code.
    """
    async def _safe(what: str, coro) -> None:
        try:
            await coro
        except Exception:  # noqa: BLE001 - shutdown is best-effort by definition
            log.exception("Error while shutting down %s", what)

    await _safe("the Telegram monitor", monitor.stop())
    await _safe("the Telegram bot", bot.stop())
    if storage is not None:
        await _safe("storage", storage.close())
    if market_data_provider is not None:
        await _safe("the market data provider", market_data_provider.aclose())
    if decision_engine is not None:
        await _safe("the decision engine", decision_engine.aclose())
    if trade_decision_engine is not None:
        await _safe("the trade decision engine", trade_decision_engine.aclose())
    await _safe("the Claude analyzer", analyzer.aclose())


async def run_monitor(settings: Settings, formatter: Formatter) -> int:
    queue: asyncio.Queue[IncomingMessage] = asyncio.Queue(maxsize=settings.queue_maxsize)

    try:
        analyzer = ClaudeAnalyzer(settings)
    except ConfigError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG

    monitor = TelegramMonitor(settings, queue)
    # RFC-002: entirely optional and independent of the Telethon session
    # above — a separate Bot API credential. If TELEGRAM_BOT_TOKEN/
    # TELEGRAM_BOT_CHAT_ID aren't set, bot.start() below is a no-op and
    # nothing about the rest of this function changes.
    bot = TelegramBot(BotSettings.from_env())
    # RFC-003: persists completed analyses so /latest, /history and /stats
    # survive a restart. A storage failure must never take down the monitor
    # any more than a bot misconfiguration can — if initialize() fails,
    # storage stays unattached and those three commands just degrade.
    storage: Optional[Storage] = Storage(settings.storage_db_path)
    try:
        await storage.initialize()
    except Exception:  # noqa: BLE001 - storage must never take down the monitor
        log.exception("Failed to initialize storage — continuing without it")
        storage = None
    else:
        bot.attach_storage(storage)
        # RFC-009: read-only aggregation over the same Storage instance —
        # no new database, no in-memory counters. Degrades independently of
        # /latest and /history if it somehow failed to construct (it can't
        # currently fail — Statistics.__init__ does no I/O — but attaching
        # it this way keeps /stats and /status on the same "never take down
        # the monitor" footing as everything else here).
        bot.attach_statistics(Statistics(storage))

    # RFC-004: empty TWELVE_DATA_API_KEY disables it the same way an empty
    # bot token disables the bot itself; provider=None makes MarketDataService
    # report "not configured" rather than raising. Backs /price directly, and
    # (integration phase) is the first step of the pipeline's optional
    # enrichment chain below.
    market_data_provider: Optional[TwelveDataProvider] = None
    if settings.twelve_data_api_key:
        market_data_provider = TwelveDataProvider(settings.twelve_data_api_key)
    market_data = MarketDataService(
        market_data_provider, cache_ttl=settings.market_data_cache_ttl
    )
    bot.attach_market_data(market_data)

    # Integration phase: StructureEngine/SMCEngine/ScoringEngine are pure,
    # stateless, config-only classes — no I/O, so construction can't fail
    # the way storage/market-data/Claude can. DecisionEngine needs the same
    # Claude CLI path resolution ClaudeAnalyzer already succeeded at just
    # above, so it's very unlikely to fail here — but it's still wrapped,
    # matching every other optional integration in this function: a failure
    # disables decision-making, never the monitor.
    structure_engine = StructureEngine()
    smc_engine = SMCEngine()
    scoring_engine = ScoringEngine()
    decision_engine: Optional[DecisionEngine] = None
    try:
        decision_engine = DecisionEngine(settings)
    except Exception:  # noqa: BLE001 - a decision-engine misconfiguration must never take down the monitor
        log.exception("Failed to initialize the decision engine — continuing without it")

    # Market-validated path: every signal is checked against live multi-
    # timeframe data before any verdict. Only enabled when a market data
    # provider is actually configured — without one there is nothing to
    # validate against, and the pipeline falls back to the signal-only
    # chain above rather than reporting empty facts.
    context_builder: Optional[MarketContextBuilder] = None
    trade_decision_engine: Optional[TradeDecisionEngine] = None
    if market_data_provider is not None:
        try:
            context_builder = MarketContextBuilder(
                market_data,
                structure_engine=structure_engine,
                smc_engine=smc_engine,
                scoring_engine=scoring_engine,
                news_filter=NewsFilter(),   # no calendar wired yet: reports "unknown"
                primary_timeframe=settings.market_data_timeframe,
            )
            trade_decision_engine = TradeDecisionEngine(settings)
            log.info(
                "market_validation_enabled primary_timeframe=%s provider=%s",
                settings.market_data_timeframe.value, market_data.provider_name,
            )
        except Exception:  # noqa: BLE001 - must never take down the monitor
            log.exception("Failed to initialize market validation — continuing without it")
            context_builder = None
            trade_decision_engine = None
    else:
        log.info(
            "market_validation_disabled reason=no_provider "
            "(set TWELVE_DATA_API_KEY to enable live validation)"
        )

    stop = asyncio.Event()
    _install_signal_handlers(asyncio.get_running_loop(), stop)

    # Everything from here on runs under a single `finally` so that EVERY
    # exit path — including the two early returns below — releases the same
    # resources. market_data_provider in particular owns a real httpx
    # connection pool, which an early return would otherwise leave open.
    # Each cleanup call is individually idempotent and safe to invoke even
    # if the corresponding component never started: monitor.stop() checks
    # is_connected(), bot.stop() returns early when there's no application,
    # httpx's aclose() tolerates repeat calls, and the remaining aclose()/
    # close() methods are no-ops kept for interface parity.
    try:
        try:
            await analyzer.verify_auth()
        except AnalysisError as exc:
            log.error("%s", exc)
            return EXIT_RUNTIME

        try:
            await monitor.start()
        except TelegramError as exc:
            log.error("%s", exc)
            return EXIT_RUNTIME

        try:
            await bot.start()
        except Exception:  # noqa: BLE001 - a bot misconfiguration must never take down the monitor
            log.exception("Failed to start the Telegram bot — continuing without it")

        exit_code = EXIT_OK
        async with AnalysisPipeline(
            settings, queue, analyzer, formatter, bot=bot, storage=storage,
            market_data=market_data, structure_engine=structure_engine, smc_engine=smc_engine,
            scoring_engine=scoring_engine, decision_engine=decision_engine,
            context_builder=context_builder, trade_decision_engine=trade_decision_engine,
        ) as pipeline:
            tasks = {
                asyncio.create_task(pipeline.run(), name="pipeline"),
                asyncio.create_task(monitor.run_forever(), name="telegram"),
                asyncio.create_task(stop.wait(), name="shutdown-signal"),
            }
            log.info("Listening for new messages — press Ctrl-C to stop")

            try:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                done, pending = set(), tasks

            # Surface whichever task ended first, if it ended badly.
            for task in done:
                if task.get_name() == "shutdown-signal":
                    log.info("Shutdown requested")
                    continue
                exc = task.exception()
                if exc is not None:
                    log.error("%s task failed: %s", task.get_name(), exc)
                    exit_code = EXIT_RUNTIME

            # Stop accepting new messages, then finish what is already queued.
            await monitor.stop()
            await pipeline.drain()

            for task in pending:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*pending, return_exceptions=True)

            log.info("Session summary: %s", pipeline.stats.summary())

        return exit_code
    finally:
        await _shutdown(
            monitor=monitor, bot=bot, storage=storage,
            market_data_provider=market_data_provider,
            decision_engine=decision_engine,
            trade_decision_engine=trade_decision_engine, analyzer=analyzer,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        configure_logging("INFO")
        log.error("Configuration error: %s", exc)
        return EXIT_CONFIG

    configure_logging(args.log_level or settings.log_level, settings.log_file)
    formatter = Formatter(
        color=settings.color, show_json=settings.show_json and not args.no_json
    )

    try:
        if args.list_chats:
            return asyncio.run(run_list_chats(settings))
        runner = run_check if args.check else run_monitor
        return asyncio.run(runner(settings, formatter))
    except KeyboardInterrupt:
        # Fallback path for platforms without asyncio signal handlers.
        log.info("Interrupted")
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
