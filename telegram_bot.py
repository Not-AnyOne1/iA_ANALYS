"""Telegram Bot notifications and status commands (RFC-002).

Fully separate from ``telegram_client.py``: this uses the Bot API
(python-telegram-bot) under its own bot token, not the Telethon user-account
MTProto session. It is entirely optional — if ``TELEGRAM_BOT_TOKEN`` and
``TELEGRAM_BOT_CHAT_ID`` aren't both set, :meth:`TelegramBot.start` is a no-op
and the rest of the application behaves exactly as it did before this
module existed.

Wiring (see ``main.py`` / ``pipeline.py``):

- ``pipeline.py`` calls :meth:`TelegramBot.notify` once per completed
  analysis (regex or Claude), after the existing terminal render/JSONL
  write. ``notify`` never raises — a failure is logged and swallowed, so a
  Telegram outage can never stop message ingestion or analysis.
- ``main.py`` calls :meth:`TelegramBot.start` / :meth:`TelegramBot.stop`
  alongside the Telethon monitor's own start/stop, and
  :meth:`TelegramBot.attach_storage` (RFC-003) to let ``/status``,
  ``/stats``, ``/latest`` and ``/history`` read from the persistent
  ``storage.Storage`` instead of in-memory state, so they survive a
  restart.
- ``main.py`` also calls :meth:`TelegramBot.attach_market_data` (RFC-004) to
  let ``/price`` answer via ``market_data.MarketDataService`` — a thin,
  optional consumer of that service for manual testing. The service itself
  isn't wired into the analysis pipeline yet.
- ``main.py`` also calls :meth:`TelegramBot.attach_statistics` (RFC-009) to
  let ``/stats`` and ``/status`` read from ``statistics.Statistics`` instead
  of the narrower ``storage.Storage`` aggregate they used before — same
  persistent, restart-proof data source, richer breakdown.

Logging convention ("structured logging" without a new dependency): every
log line uses a stable ``event_name key=value key=value`` shape so it stays
greppable and machine-parseable through the project's existing plain-text
formatter (see ``logging_setup.py``, intentionally untouched by this module).
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING, Tuple, Union

from pydantic import BaseModel, Field
from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, ChatMigrated, Forbidden, NetworkError, RetryAfter, TelegramError
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from market_data import MarketDataError, MarketDataService
from models import SignalAnalysis
from statistics import Statistics
from storage import Storage, StoredAnalysis

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from telegram_client import IncomingMessage

log = logging.getLogger(__name__)

_MAX_MESSAGE_CHARS = 4000  # Telegram's hard cap is 4096; leave headroom.
ChatRef = Union[int, str]


class BotSettings(BaseModel):
    """Pydantic configuration for the Telegram Bot, independent of ``config.Settings``.

    Reads ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_BOT_CHAT_ID`` and a handful of
    tuning knobs from the environment. Both the token and chat id are
    optional at the type level — the bot is opt-in; see :attr:`enabled`.
    """

    bot_token: Optional[str] = None
    chat_id: Optional[ChatRef] = None

    max_retries: int = Field(default=5, ge=0)
    retry_base_delay: float = Field(default=2.0, gt=0)
    retry_max_delay: float = Field(default=60.0, gt=0)

    history_display_default: int = Field(default=5, ge=1)
    dedup_cache_size: int = Field(default=2000, ge=1)

    @property
    def bot_user_id(self) -> Optional[int]:
        """This bot's own Telegram user id, read from the token.

        A bot token is ``<user_id>:<secret>``, so the id is available at
        startup with no API call and no extra configuration. The monitor
        uses it to refuse messages this bot itself posted — without that,
        a bot whose destination is the monitored chat reads its own report
        back and analyses it forever.

        Returns ``None`` for a malformed or absent token rather than
        raising: a filter that cannot be built must not stop the bot.
        """
        if not self.bot_token:
            return None
        head, _, _ = self.bot_token.partition(":")
        return int(head) if head.isdigit() else None

    @property
    def enabled(self) -> bool:
        """True as soon as a token is configured.

        A target chat is no longer required: reports go to whoever sent
        /start, and that audience is built at runtime. ``chat_id`` remains
        only as a backward-compatible fallback for deployments that set it
        before subscriptions existed — see :meth:`TelegramBot._destinations`.
        """
        return bool(self.bot_token)

    @classmethod
    def from_env(cls) -> "BotSettings":
        """Build settings from the process environment (after ``.env`` is loaded).

        Does not call ``load_dotenv`` itself — ``config.Settings.load()``
        already does that once, earlier in startup; by the time this runs,
        ``.env`` values are already in ``os.environ``.
        """
        raw_chat_id = os.getenv("TELEGRAM_BOT_CHAT_ID", "").strip()
        chat_id: Optional[ChatRef] = None
        if raw_chat_id:
            chat_id = int(raw_chat_id) if re.fullmatch(r"-?\d+", raw_chat_id) else raw_chat_id

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name, "").strip()
            return float(raw) if raw else default

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name, "").strip()
            return int(raw) if raw else default

        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None,
            chat_id=chat_id,
            max_retries=_int("TELEGRAM_BOT_MAX_RETRIES", 5),
            retry_base_delay=_float("TELEGRAM_BOT_RETRY_BASE_DELAY", 2.0),
            retry_max_delay=_float("TELEGRAM_BOT_RETRY_MAX_DELAY", 60.0),
            history_display_default=_int("TELEGRAM_BOT_HISTORY_DISPLAY_DEFAULT", 5),
            dedup_cache_size=_int("TELEGRAM_BOT_DEDUP_CACHE_SIZE", 2000),
        )


@dataclass
class BotStats:
    """Counters for ``/status`` — the bot's own send-side view, separate
    from (but commonly shown alongside) ``pipeline.Stats``."""

    notifications_sent: int = 0
    notifications_failed: int = 0
    duplicates_skipped: int = 0


class _BoundedSet:
    """A set with a maximum size; oldest entries are evicted first (LRU-ish).

    Used for de-duplication: bounded so a very long-running process can't
    grow this without limit.
    """

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: "OrderedDict[Tuple[int, int], None]" = OrderedDict()

    def add(self, key: Tuple[int, int]) -> None:
        self._data[key] = None
        self._data.move_to_end(key)
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return key in self._data


# Telegram's wording for "the message you tried to reply to isn't here".
# Matched loosely because the exact phrasing has varied across Bot API
# versions, and the consequence of a miss is only that a recoverable send is
# treated as a hard failure.
_MISSING_REPLY_MARKERS = (
    "replied message not found",
    "message to be replied not found",
    "message to reply not found",
    "reply message not found",
    "message_id_invalid",
    "message to reply not found",
)


def _is_missing_reply_target(exc: BaseException) -> bool:
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _MISSING_REPLY_MARKERS)


def _esc(text: str) -> str:
    """HTML-escape dynamic text before interpolating into a formatted message.

    Every field here ultimately traces back to untrusted Telegram group
    content (chat title, sender name, summary text) — same threat model as
    claude_client.py's prompt handling. Never skip this for a field that
    might contain ``<``, ``>``, or ``&``, or Telegram's HTML parser will
    either mangle the message or reject it outright with a 400.
    """
    return html.escape(str(text))


class TelegramBot:
    """Sends analysis notifications and answers status commands.

    Construct once, call :meth:`start` during startup and :meth:`stop` during
    shutdown (both are no-ops if :attr:`BotSettings.enabled` is false), and
    call :meth:`notify` once per completed analysis.
    """

    def __init__(self, settings: BotSettings) -> None:
        self._settings = settings
        self._bot: Optional[Bot] = Bot(token=settings.bot_token) if settings.enabled else None
        self._application: Optional[Application] = None
        self._started_at: Optional[datetime] = None
        self._sent_ids = _BoundedSet(settings.dedup_cache_size)
        self._stats = BotStats()
        self._storage: Optional[Storage] = None
        # Report recipients, mirrored from storage so the sync
        # authorization check does not need a database round-trip.
        self._subscribers: set[int] = set()
        self._market_data: Optional[MarketDataService] = None
        self._statistics: Optional[Statistics] = None

    def attach_storage(self, storage: Storage) -> None:
        """Give ``/latest`` and ``/history`` a persistent data source
        (RFC-003).

        Without this, those commands report "not available" rather than
        raising — a missing/failed storage layer must never be able to stop
        the bot from answering other commands or from sending
        notifications.
        """
        self._storage = storage

    def attach_statistics(self, statistics: Statistics) -> None:
        """Give ``/stats`` and ``/status`` a persistent data source
        (RFC-009), superseding the narrower ``storage.Storage``-only
        aggregate those commands used before.

        Without this, those commands report "not available" rather than
        raising — same degradation contract as :meth:`attach_storage`.
        """
        self._statistics = statistics

    def attach_market_data(self, service: MarketDataService) -> None:
        """Give ``/price`` a market data source (RFC-004).

        Without this, ``/price`` reports "not configured" rather than
        raising — a missing/failed market data layer must never be able to
        stop the bot from answering other commands or from sending
        notifications.
        """
        self._market_data = service

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Register command handlers and begin polling. No-op if not enabled."""
        if not self._settings.enabled:
            log.info(
                "bot_disabled reason=missing_token_or_chat_id "
                "(set TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_CHAT_ID to enable)"
            )
            return

        assert self._settings.bot_token is not None
        self._application = ApplicationBuilder().token(self._settings.bot_token).build()
        for command, handler in (
            ("start", self._cmd_start),
            ("stop", self._cmd_stop),
            ("help", self._cmd_help),
            ("status", self._cmd_status),
            ("stats", self._cmd_stats),
            ("latest", self._cmd_latest),
            ("history", self._cmd_history),
            ("price", self._cmd_price),
        ):
            self._application.add_handler(CommandHandler(command, handler))

        await self._application.initialize()
        await self._application.start()
        assert self._application.updater is not None
        await self._application.updater.start_polling()
        self._started_at = datetime.now(timezone.utc)
        await self._load_subscribers()
        log.info("bot_started subscribers=%d fallback_chat_id=%s",
                 len(self._subscribers), self._settings.chat_id)

    async def _load_subscribers(self) -> None:
        """Restore the audience persisted by previous runs.

        Best-effort: a storage fault leaves the set empty, which falls
        back to TELEGRAM_BOT_CHAT_ID rather than stopping the bot.
        """
        if self._storage is None:
            return
        try:
            self._subscribers = set(await self._storage.subscribers())
        except Exception:  # noqa: BLE001 - never block startup
            log.exception("bot_subscriber_load_failed")

    async def stop(self) -> None:
        """Stop polling and release resources. No-op if never started."""
        if self._application is None:
            return
        try:
            if self._application.updater is not None:
                await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
        except Exception:  # noqa: BLE001 - shutdown must never raise into main()
            log.exception("bot_stop_failed")
        else:
            log.info("bot_stopped")

    # ------------------------------------------------------------------- notify

    async def notify(self, message: "IncomingMessage", analysis: SignalAnalysis) -> None:
        """Send a notification for one completed analysis. Never raises.

        Duplicate-safe: the same (chat, message id) is sent at most once per
        process lifetime, regardless of how many times ``notify`` is called
        for it (e.g. a retried/re-queued message).
        """
        if not self._settings.enabled:
            return

        key = (message.chat_id, message.id)
        if key in self._sent_ids:
            self._stats.duplicates_skipped += 1
            log.debug("bot_notify_skip_duplicate chat_id=%s message_id=%s", *key)
            return

        text = self._format_notification(message, analysis)
        try:
            for destination in self._destinations():
                await self._send_with_retry(text, destination)
        except Exception as exc:  # noqa: BLE001 - the one rule that must never break
            self._stats.notifications_failed += 1
            log.error(
                "bot_notify_failed chat_id=%s message_id=%s error=%s: %s",
                message.chat_id, message.id, type(exc).__name__, exc,
            )
            return

        self._sent_ids.add(key)
        self._stats.notifications_sent += 1
        log.info(
            "bot_notify_sent chat_id=%s message_id=%s is_signal=%s category=%s source=%s",
            message.chat_id, message.id, analysis.is_signal, analysis.category, analysis.source,
        )

    async def send_report(self, message: "IncomingMessage", html_text: str) -> None:
        """Send the final, fully-analysed report for one signal. Never raises.

        This is the *only* thing the pipeline sends to Telegram. The parser's
        extraction summary is deliberately not sent — it is an intermediate
        result, and publishing it would mean two messages per signal, the
        first of them a guess made before any market data was consulted.
        That output remains in the terminal and the logs.

        Replies to the original signal message when Telegram allows it. The
        bot and the Telethon monitor are separate accounts, and the bot's
        target chat is often *not* the monitored group, in which case the
        group's message id does not exist there and Telegram rejects the
        reply. That specific rejection is detected and the report is re-sent
        as a normal message rather than being lost.

        Shares the duplicate guard with :meth:`notify`, so a given signal can
        produce at most one Telegram message however many times this is
        called.
        """
        if not self._settings.enabled:
            return

        key = (message.chat_id, message.id)
        if key in self._sent_ids:
            self._stats.duplicates_skipped += 1
            log.debug("bot_report_skip_duplicate chat_id=%s message_id=%s", *key)
            return

        destinations = self._destinations()
        if not destinations:
            log.info(
                "bot_report_no_subscribers message_id=%s — nobody has sent /start "
                "and no TELEGRAM_BOT_CHAT_ID fallback is configured",
                message.id,
            )
            return

        # One failing subscriber (blocked the bot, deleted the chat) must not
        # cost the others their report, so each destination is attempted
        # independently and the failures are counted, not raised.
        delivered = 0
        for destination in destinations:
            try:
                await self._send_to(destination, message, html_text)
            except Exception as exc:  # noqa: BLE001 - delivery never breaks the pipeline
                self._stats.notifications_failed += 1
                log.error(
                    "bot_report_failed destination=%s message_id=%s error=%s: %s",
                    destination, message.id, type(exc).__name__, exc,
                )
            else:
                delivered += 1

        if not delivered:
            return

        self._sent_ids.add(key)
        self._stats.notifications_sent += 1
        log.info("bot_report_sent message_id=%s delivered=%d/%d chars=%d",
                 message.id, delivered, len(destinations), len(html_text))

    def _destinations(self) -> List[ChatRef]:
        """Who the next report goes to.

        Registered subscribers are the destination. ``TELEGRAM_BOT_CHAT_ID``
        is a *fallback*, used only when nobody has subscribed — that keeps a
        deployment configured before subscriptions existed working unchanged,
        without making a hard-coded chat the primary target again.
        """
        if self._subscribers:
            return sorted(self._subscribers)
        return [self._settings.chat_id] if self._settings.chat_id is not None else []

    async def _send_to(
        self, destination: ChatRef, message: "IncomingMessage", html_text: str
    ) -> None:
        """Deliver one report to one chat, replying where that is meaningful.

        A reply only makes sense in the chat the signal came from. For a
        subscriber's private chat the group's message id does not exist, so
        no reply is attempted rather than provoking a rejection and retrying.
        """
        reply_to = message.id if destination == message.chat_id else None
        try:
            await self._send_with_retry(html_text, destination, reply_to_message_id=reply_to)
        except BadRequest as exc:
            if reply_to is None or not _is_missing_reply_target(exc):
                raise
            log.info(
                "bot_report_reply_unavailable destination=%s message_id=%s (%s) "
                "— sending as a normal message",
                destination, message.id, exc,
            )
            await self._send_with_retry(html_text, destination)

    async def _send_with_retry(
        self, text: str, chat_id: ChatRef, *, reply_to_message_id: Optional[int] = None
    ) -> None:
        """Send ``text`` to ``chat_id``, retrying transient failures with
        exponential backoff (and honouring Telegram's own requested delay on
        rate limits). Raises on the final failed attempt or on any permanent
        (non-retryable) error — callers must catch."""
        assert self._bot is not None
        attempts = self._settings.max_retries + 1
        extra = {"reply_to_message_id": reply_to_message_id} if reply_to_message_id else {}

        for attempt in range(1, attempts + 1):
            try:
                await self._bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    **extra,
                )
                return
            except RetryAfter as exc:
                if attempt == attempts:
                    raise
                cause = f"RetryAfter: {exc}"
                delay = exc.retry_after + random.uniform(0, 1)
            except (BadRequest, Forbidden, ChatMigrated):
                # Permanent: malformed request, bot blocked/kicked, or the
                # chat moved to a new id. Retrying identically won't help.
                raise
            except NetworkError as exc:
                if attempt == attempts:
                    raise
                cause = f"{type(exc).__name__}: {exc}"
                delay = min(
                    self._settings.retry_base_delay * (2 ** (attempt - 1)),
                    self._settings.retry_max_delay,
                ) + random.uniform(0, 1)
            except TelegramError:
                # Anything else Telegram-specific and unclassified: be
                # conservative rather than blindly retrying an unknown error.
                raise

            # Name the cause: a silent retry loop is undiagnosable from logs.
            log.warning(
                "bot_notify_retry attempt=%d/%d delay=%.1fs cause=%s",
                attempt, attempts, delay, cause,
            )
            await asyncio.sleep(delay)

    def _format_notification(self, message: "IncomingMessage", analysis: SignalAnalysis) -> str:
        setup = analysis.setup
        badge = "SIGNAL" if analysis.is_signal else analysis.category.upper()
        lines = [
            f"<b>{_esc(badge)}</b> — {_esc(message.chat_title)}",
            f"#{message.id} · {_esc(message.sender)} · "
            f"{message.timestamp.astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if analysis.is_signal:
            direction = _esc((setup.direction or "?").upper())
            lines.append(f"<b>Symbol:</b> {_esc(setup.symbol or '—')}  <b>{direction}</b>")
            entries = ", ".join(f"{v:g}" for v in setup.entries) or "—"
            lines.append(f"<b>Entry:</b> {entries}")
            sl = f"{setup.stop_loss:g}" if setup.stop_loss is not None else "—"
            lines.append(f"<b>Stop loss:</b> {sl}")
            tps = ", ".join(f"{v:g}" for v in setup.take_profits) or "—"
            lines.append(f"<b>Take profit:</b> {tps}")
            if setup.leverage:
                lines.append(f"<b>Leverage:</b> {_esc(setup.leverage)}")
            rr = setup.risk_reward
            if rr is not None:
                lines.append(f"<b>R:R:</b> {rr:.2f}")

        lines.append(f"<b>Summary:</b> {_esc(analysis.summary)}")
        lines.append(f"<b>Confidence:</b> {analysis.confidence_pct}%")
        if analysis.missing_fields:
            lines.append(f"<b>Missing:</b> {_esc(', '.join(analysis.missing_fields))}")
        if analysis.notes:
            lines.append(f"<b>Notes:</b> {_esc(analysis.notes)}")
        lines.append(f"<i>source: {_esc(analysis.source)}</i>")

        text = "\n".join(lines)
        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[:_MAX_MESSAGE_CHARS] + "\n[... truncated]"
        return text

    # ------------------------------------------------------------------ commands

    def _is_configured_chat(self, update: Update) -> bool:
        """True for the legacy ``TELEGRAM_BOT_CHAT_ID``, if one is set."""
        chat = update.effective_chat
        configured = self._settings.chat_id
        if chat is None or configured is None:
            return False
        if isinstance(configured, int):
            return chat.id == configured
        return (chat.username or "").lower() == str(configured).lstrip("@").lower()

    def _is_authorized_chat(self, update: Update) -> bool:
        """Only answer informational commands to a subscriber.

        ``/status``/``/stats`` reveal operational details about a private
        pipeline, so they stay behind a check. The check is now membership
        rather than a single hard-coded chat: subscribing with /start is what
        grants access, and /stop revokes it. The legacy configured chat still
        counts, so an existing deployment keeps working untouched.

        ``/start`` and ``/stop`` deliberately do NOT go through here — you
        cannot subscribe if subscribing is what unlocks the door.
        """
        chat = update.effective_chat
        if chat is None:
            return False
        return chat.id in self._subscribers or self._is_configured_chat(update)

    async def _reply(self, update: Update, text: str) -> None:
        if update.message is not None:
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Register the sender to receive every final report.

        Open by design — this is how the audience is built. Private chats
        only: registering a group would send one report to every member of
        it, which is a different feature and not what was asked for.
        """
        chat = update.effective_chat
        if chat is None:
            return
        if chat.type != "private":
            await self._reply(
                update,
                "Send /start to me in a private chat to subscribe to reports.",
            )
            return

        # Coerce to a real string or None: this goes straight into SQLite,
        # which refuses to bind anything else, and the column is decorative.
        raw = getattr(chat, "username", None) or getattr(chat, "first_name", None)
        username = raw if isinstance(raw, str) else None
        added = True
        if self._storage is not None:
            try:
                added = await self._storage.add_subscriber(chat.id, username)
            except Exception:  # noqa: BLE001 - a storage fault must not lose the user
                log.exception("bot_subscribe_persist_failed chat_id=%s", chat.id)
        else:
            added = chat.id not in self._subscribers
        self._subscribers.add(chat.id)

        log.info("bot_subscribed chat_id=%s username=%s new=%s total=%d",
                 chat.id, username, added, len(self._subscribers))
        await self._reply(
            update,
            ("You are subscribed — every completed analysis will be sent here.\n\n"
             if added else
             "You were already subscribed.\n\n")
            + "Use /stop to unsubscribe, or /help to see the other commands.",
        )

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Unregister the sender. Open for the same reason as /start: a
        subscriber must always be able to leave."""
        chat = update.effective_chat
        if chat is None:
            return

        removed = chat.id in self._subscribers
        if self._storage is not None:
            try:
                removed = await self._storage.remove_subscriber(chat.id)
            except Exception:  # noqa: BLE001
                log.exception("bot_unsubscribe_persist_failed chat_id=%s", chat.id)
        self._subscribers.discard(chat.id)

        log.info("bot_unsubscribed chat_id=%s was_subscribed=%s total=%d",
                 chat.id, removed, len(self._subscribers))
        await self._reply(
            update,
            "You are unsubscribed — no further reports will be sent here. "
            "Send /start to resubscribe."
            if removed else
            "You were not subscribed. Send /start to subscribe.",
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized_chat(update):
            return
        await self._reply(
            update,
            "<b>Commands</b>\n"
            "/start — subscribe to reports\n"
            "/stop — unsubscribe\n"
            "/help — this message\n"
            "/status — bot uptime and pipeline health\n"
            "/stats — persistent analysis statistics (signals, confidence, "
            "verdicts, sources)\n"
            "/latest — the most recently sent notification\n"
            "/history [n] — the last n notifications (default "
            f"{self._settings.history_display_default})\n"
            "/price SYMBOL — current market price",
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized_chat(update):
            return
        uptime = "unknown"
        if self._started_at is not None:
            seconds = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
            uptime = f"{seconds // 3600}h {(seconds % 3600) // 60}m {seconds % 60}s"
        lines = [
            "<b>Bot status</b>: running",
            f"<b>Uptime</b>: {uptime}",
            f"<b>Notifications sent</b>: {self._stats.notifications_sent}",
            f"<b>Notifications failed</b>: {self._stats.notifications_failed}",
            f"<b>Duplicates skipped</b>: {self._stats.duplicates_skipped}",
        ]
        if self._statistics is None:
            lines.append("<b>Analyses stored</b>: statistics not attached")
        else:
            try:
                summary = await self._statistics.summary()
            except Exception:  # noqa: BLE001 - a query failure must not break /status
                log.exception("bot_status_statistics_query_failed")
                lines.append("<b>Analyses stored</b>: statistics error")
            else:
                lines.append(f"<b>Analyses stored</b>: {_esc(summary.format())}")
        await self._reply(update, "\n".join(lines))

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized_chat(update):
            return
        if self._statistics is None:
            await self._reply(update, "Stats are not available yet.")
            return
        try:
            summary = await self._statistics.summary()
        except Exception:  # noqa: BLE001 - a query failure must not break /stats
            log.exception("bot_stats_statistics_query_failed")
            await self._reply(update, "Stats are temporarily unavailable.")
            return
        await self._reply(update, _esc(summary.format()))

    async def _cmd_latest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized_chat(update):
            return
        if self._storage is None:
            await self._reply(update, "No analyses recorded yet.")
            return
        try:
            record = await self._storage.latest()
        except Exception:  # noqa: BLE001 - a query failure must not break /latest
            log.exception("bot_latest_storage_query_failed")
            await self._reply(update, "Latest analysis is temporarily unavailable.")
            return
        if record is None:
            await self._reply(update, "No analyses recorded yet.")
            return
        await self._reply(update, self._render_record(record))

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized_chat(update):
            return
        if self._storage is None:
            await self._reply(update, "No analyses recorded yet.")
            return

        count = self._settings.history_display_default
        if context.args:
            try:
                count = int(context.args[0])
            except ValueError:
                await self._reply(update, "Usage: /history [number]")
                return
        count = max(1, count)

        try:
            records = await self._storage.history(limit=count)
        except Exception:  # noqa: BLE001 - a query failure must not break /history
            log.exception("bot_history_storage_query_failed")
            await self._reply(update, "History is temporarily unavailable.")
            return

        if not records:
            await self._reply(update, "No analyses recorded yet.")
            return

        # storage.history() already returns most-recent-first.
        rendered = "\n\n".join(self._render_record(r) for r in records)
        await self._reply(update, rendered)

    async def _cmd_price(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Manual, on-demand consumer of market_data.MarketDataService —
        exists to exercise the service for testing; later RFCs (Structure,
        SMC, Scoring, Claude Decision engines) call the same service
        directly, not through this command."""
        if not self._is_authorized_chat(update):
            return
        if not context.args:
            await self._reply(update, "Usage: /price SYMBOL")
            return
        if self._market_data is None:
            await self._reply(update, "Market data is not configured.")
            return

        symbol = context.args[0].strip().upper()
        try:
            quote = await self._market_data.get_quote(symbol)
        except MarketDataError as exc:
            log.warning("bot_price_query_failed symbol=%s error=%s", symbol, exc)
            await self._reply(update, f"Price unavailable for {_esc(symbol)}: {_esc(str(exc))}")
            return

        await self._reply(
            update,
            f"<b>{_esc(quote.symbol)}</b>: {quote.price:g}\n"
            f"<i>{quote.timestamp.astimezone().strftime('%Y-%m-%d %H:%M:%S')} · "
            f"source: {_esc(quote.provider)}</i>",
        )

    def _render_record(self, record: StoredAnalysis) -> str:
        a = record.analysis
        badge = "SIGNAL" if a.is_signal else a.category.upper()
        summary = (
            f"<b>{_esc(badge)}</b> #{record.message_id} · {_esc(record.chat_title)} · "
            f"{record.timestamp.astimezone().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{_esc(a.summary)}"
        )
        if a.is_signal and a.setup.symbol:
            summary += f"\n{_esc(a.setup.symbol)} {_esc((a.setup.direction or '').upper())}"
        return summary
