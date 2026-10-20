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
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
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


def persistence_report(settings: Settings) -> dict:
    """Where the database and the Telegram session actually live, for one
    startup log line. Paths and booleans only — never file contents."""
    db = Path(settings.storage_db_path).resolve()
    session = Path(f"{settings.session_path}.session").resolve()
    data = Path("/data")
    return {
        "storage_db": str(db), "storage_db_exists": db.exists(),
        "storage_on_volume": storage_on_volume(db),
        "session_file": str(session), "session_exists": session.exists(),
        "session_on_volume": storage_on_volume(session),
        "volume_mount": os.getenv("RAILWAY_VOLUME_MOUNT_PATH") or None,
        "data_dir_is_mount": os.path.ismount(data) if data.is_dir() else None,
    }


def storage_on_volume(path: Path) -> Optional[bool]:
    """True/False when running on Railway with a volume attached (is the
    database on it?); None when no volume is declared (local runs)."""
    mount = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if not mount:
        return None
    try:
        Path(path).resolve().relative_to(Path(mount).resolve())
        return True
    except ValueError:
        return False


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

    # RFC-002: entirely optional and independent of the Telethon session
    # above — a separate Bot API credential. If TELEGRAM_BOT_TOKEN/
    # TELEGRAM_BOT_CHAT_ID aren't set, bot.start() below is a no-op and
    # nothing about the rest of this function changes.
    bot_settings = BotSettings.from_env()
    bot = TelegramBot(bot_settings)

    # The monitor must never analyse what this bot posted. When the bot's
    # destination is the chat being monitored, its report comes straight
    # back as an ordinary message; analysing it produces another report, and
    # so on without end. The id comes from the bot token, so no extra
    # configuration is needed; TELEGRAM_BOT_USERNAME is an optional fallback
    # for a token this cannot parse.
    own_bot_ids = [i for i in (bot_settings.bot_user_id,) if i is not None]
    own_bot_usernames = [u for u in (os.getenv("TELEGRAM_BOT_USERNAME", "").strip(),) if u]
    if own_bot_ids or own_bot_usernames:
        log.info("self_filter_enabled bot_user_id=%s bot_username=%s",
                 own_bot_ids or None, own_bot_usernames or None)
    elif bot_settings.enabled:
        log.warning(
            "self_filter_unavailable — the bot token could not be parsed for a "
            "user id and TELEGRAM_BOT_USERNAME is unset; if the bot posts into "
            "the monitored chat its own reports will be analysed"
        )
    monitor = TelegramMonitor(
        settings, queue,
        ignore_sender_ids=own_bot_ids,
        ignore_usernames=own_bot_usernames,
    )
    # RFC-003: persists completed analyses so /latest, /history and /stats
    # survive a restart. A storage failure must never take down the monitor
    # any more than a bot misconfiguration can — if initialize() fails,
    # storage stays unattached and those three commands just degrade.
    # One line that answers "is anything persisted?" without opening the
    # dashboard: resolved paths, whether /data is a real mount, and whether
    # the session and database files exist. Logged before anything is
    # created, so "exists" describes what the previous run left behind.
    log.info("persistence_check %s", " ".join(f"{k}={v}" for k, v in persistence_report(settings).items()))
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

    # Shadow work never touches the live provider: anything that compares a
    # candle with the clock goes through one shared provider that asks for
    # UTC-labelled candles. Created only if some shadow mode is on.
    analyst_provider: Optional[TwelveDataProvider] = None
    utc_market_data: Optional[MarketDataService] = None

    def shared_utc_market_data() -> MarketDataService:
        nonlocal analyst_provider, utc_market_data
        if utc_market_data is None:
            analyst_provider = TwelveDataProvider(settings.twelve_data_api_key, label_timezone="UTC")
            utc_market_data = MarketDataService(analyst_provider, cache_ttl=settings.market_data_cache_ttl)
        return utc_market_data

    # Independent analyst (analyst/), INDEPENDENT_ANALYSIS_MODE=shadow only.
    # Off by default: nothing below is constructed, no table is created and
    # no request is made. In shadow it records its own verdicts in its own
    # write-once tables, after the live report, and never changes the live
    # verdict or what the bot sends.
    independent_analyst = None
    outcome_resolver = None
    if settings.independent_analysis_mode == "shadow":
        if storage is None or context_builder is None:
            log.warning(
                "independent_analysis_unavailable mode=shadow reason=%s",
                "storage unavailable" if storage is None else "no market data provider",
            )
        else:
            try:
                from analyst.service import ShadowAnalyst
                from analyst.store import AnalystStore
                from analyst.tracker import OutcomeResolver

                analyst_store = AnalystStore(settings.storage_db_path)
                await analyst_store.initialize()
                independent_analyst = ShadowAnalyst(shared_utc_market_data(), analyst_store)
                outcome_resolver = OutcomeResolver(
                    analyst_store, shared_utc_market_data(), costs=independent_analyst.costs)
                bot.attach_shadow_store(analyst_store)
                log.info("independent_analysis_enabled mode=shadow "
                         "(records only; live verdicts and reports unchanged)")
            except Exception:  # noqa: BLE001 - the shadow analyst must never take down the monitor
                log.exception("Failed to initialize the independent analyst — continuing without it")
                independent_analyst = outcome_resolver = None

    # Strategy shadow evaluation (strategy_shadow/), STRATEGY_SHADOW_MODE=on
    # only. Off by default: nothing constructed, no table, no request. On, it
    # judges each signal's live snapshot with the baseline replay and three
    # candidates, records them write-once and tracks the signal's outcome;
    # the live verdict and every message sent are unchanged.
    strategy_shadow = None
    strategy_resolver = None
    strategy_store = None
    on_volume = storage_on_volume(settings.storage_db_path)
    if on_volume is False:
        log.warning("storage_not_on_volume path=%s volume=%s — history will not survive a redeploy",
                    settings.storage_db_path, os.getenv("RAILWAY_VOLUME_MOUNT_PATH"))
    if settings.strategy_shadow_mode == "on" or settings.strategy_shadow_reporting == "on":
        if storage is None or (settings.strategy_shadow_mode == "on" and context_builder is None):
            log.warning("strategy_shadow_unavailable reason=%s",
                        "storage unavailable" if storage is None else "no market data provider")
        elif on_volume is False:
            # Records written to container storage vanish on redeploy, which
            # would silently bias the evaluation toward recent signals.
            log.error("strategy_shadow_disabled reason=storage_not_on_volume path=%s",
                      settings.storage_db_path)
        else:
            try:
                from analyst.service import assumed_costs
                from strategy_shadow.outcomes import StrategyOutcomeResolver
                from strategy_shadow.service import StrategyShadow
                from strategy_shadow.snapshot import LabelCalibrator
                from strategy_shadow.store import StrategyShadowStore

                strategy_store = StrategyShadowStore(settings.storage_db_path)
                await strategy_store.initialize()
                if settings.strategy_shadow_mode == "on":
                    strategy_shadow = StrategyShadow(
                        strategy_store, LabelCalibrator(shared_utc_market_data()),
                        confirmation_start=settings.strategy_shadow_confirmation_start)
                    strategy_resolver = StrategyOutcomeResolver(
                        strategy_store, shared_utc_market_data(), costs=assumed_costs())
                if settings.strategy_shadow_reporting == "on":
                    bot.attach_strategy_shadow(strategy_store)
                log.info("strategy_shadow mode=%s reporting=%s confirmation_start=%s "
                         "(records only; live verdicts and reports unchanged)",
                         settings.strategy_shadow_mode, settings.strategy_shadow_reporting,
                         settings.strategy_shadow_confirmation_start)
            except Exception:  # noqa: BLE001 - shadow evaluation must never take down the monitor
                log.exception("Failed to initialize strategy shadow — continuing without it")
                strategy_shadow = strategy_resolver = None

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
            independent_analyst=independent_analyst, strategy_shadow=strategy_shadow,
        ) as pipeline:
            # Outside ``tasks`` on purpose: their ending must never be mistaken
            # for the pipeline or the monitor ending.
            resolver_tasks = [
                asyncio.create_task(resolver.run_forever(), name=name)
                for name, resolver in (("independent-outcomes", outcome_resolver),
                                       ("strategy-shadow-outcomes", strategy_resolver))
                if resolver is not None
            ]
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
            for resolver_task in resolver_tasks:
                resolver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*resolver_tasks, return_exceptions=True)

            for task in pending:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*pending, return_exceptions=True)

            log.info("Session summary: %s", pipeline.stats.summary())

        return exit_code
    finally:
        if analyst_provider is not None:
            with contextlib.suppress(Exception):
                await analyst_provider.aclose()
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
