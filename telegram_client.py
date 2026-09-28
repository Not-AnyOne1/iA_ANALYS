"""Telegram monitoring via Telethon (MTProto, user account — not a bot).

Listens to exactly one group and pushes new messages onto an ``asyncio.Queue``
for the analysis workers. Handlers stay non-blocking: the Claude round-trip
happens in the worker, never inside the update loop.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from telethon import TelegramClient, events, utils
from telethon.errors import (
    ApiIdInvalidError,
    AuthKeyUnregisteredError,
    ChannelPrivateError,
    FloodWaitError,
    SessionPasswordNeededError,
)
from telethon.tl.types import Channel, Chat, User

from config import Settings

log = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    """Raised for unrecoverable Telegram problems (bad credentials, no access)."""


@dataclass(frozen=True)
class IncomingMessage:
    """A single message captured from the monitored group."""

    id: int
    chat_id: int
    chat_title: str
    sender: str
    timestamp: datetime
    text: str
    is_edit: bool = False
    reply_to_id: Optional[int] = None
    link: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(frozen=True)
class ChatSummary:
    """One entry in the account's dialog list, for ``--list-chats``."""

    title: str
    id: int
    type: str
    username: Optional[str]


def _classify_chat_type(entity: Any) -> str:
    """Map a Telethon entity to one of the four types ``--list-chats`` reports.

    Telethon's own ``Dialog.is_group``/``is_channel`` helpers don't cleanly
    separate these — a supergroup is a ``Channel`` with ``megagroup=True``,
    which also satisfies ``is_channel`` — so this checks the concrete type
    and flags directly instead.
    """
    if isinstance(entity, User):
        return "Private"
    if isinstance(entity, Chat):
        return "Group"
    if isinstance(entity, Channel):
        return "Supergroup" if entity.megagroup else "Channel"
    return "Unknown"


class TelegramMonitor:
    """Owns the Telethon client and feeds messages into ``queue``."""

    def __init__(
        self,
        settings: Settings,
        queue: "asyncio.Queue[IncomingMessage]",
        *,
        ignore_sender_ids: Iterable[int] = (),
        ignore_usernames: Iterable[str] = (),
    ) -> None:
        self._settings = settings
        self._queue = queue
        self._stopping = asyncio.Event()
        self._chat_id: int | None = None
        self._chat_title: str = "unknown"
        self._chat_username: str | None = None
        self._dropped = 0
        # Authors whose messages never enter the pipeline. Populated in
        # main.py with our own bot's identity: when the bot posts into the
        # chat this monitor watches, its report arrives back here as an
        # ordinary message and would be analysed, producing another report,
        # forever. Ids are preferred (exact, no API call); usernames are a
        # fallback for the case where the id could not be determined.
        self._ignored_sender_ids = frozenset(ignore_sender_ids)
        self._ignored_usernames = frozenset(
            u.lstrip("@").lower() for u in ignore_usernames if u
        )
        self._ignored_messages = 0

        self._client = TelegramClient(
            settings.session_path,
            settings.api_id,
            settings.api_hash,
            # Reconnect forever with a fixed delay; Telethon handles the retries
            # internally, so short network blips never surface here.
            connection_retries=None,
            retry_delay=5,
            auto_reconnect=True,
            request_retries=5,
            # Sleep through short floods instead of raising.
            flood_sleep_threshold=120,
        )

    # ------------------------------------------------------------------ setup

    async def start(self) -> None:
        """Authorise, resolve the target group, and register event handlers."""
        try:
            await self._authorize()
        except ApiIdInvalidError as exc:
            raise TelegramError(
                "TELEGRAM_API_ID / TELEGRAM_API_HASH rejected by Telegram. "
                "Re-check the values from https://my.telegram.org/apps."
            ) from exc

        me = await self._client.get_me()
        log.info("Signed in as %s (id=%s)", utils.get_display_name(me), me.id)

        await self._resolve_target()
        self._register_handlers()

    async def list_chats(self) -> list[ChatSummary]:
        """Authorise with the existing session and return every accessible chat.

        Independent of ``start()`` — it never touches ``target_chat`` or
        registers message handlers, so it works even before
        TELEGRAM_TARGET_CHAT is set to a real group.
        """
        try:
            await self._authorize()
        except ApiIdInvalidError as exc:
            raise TelegramError(
                "TELEGRAM_API_ID / TELEGRAM_API_HASH rejected by Telegram. "
                "Re-check the values from https://my.telegram.org/apps."
            ) from exc

        me = await self._client.get_me()
        log.info("Signed in as %s (id=%s)", utils.get_display_name(me), me.id)

        chats: list[ChatSummary] = []
        try:
            async for dialog in self._client.iter_dialogs():
                entity = dialog.entity
                chats.append(
                    ChatSummary(
                        title=dialog.name or utils.get_display_name(entity) or "(no title)",
                        id=utils.get_peer_id(entity),
                        type=_classify_chat_type(entity),
                        username=getattr(entity, "username", None),
                    )
                )
        except FloodWaitError as exc:
            raise TelegramError(
                f"Telegram asked us to wait {exc.seconds}s while listing chats."
            ) from exc

        return chats

    async def _authorize(self) -> None:
        await self._client.connect()
        if await self._client.is_user_authorized():
            return

        if not sys.stdin.isatty():
            raise TelegramError(
                "No authorised session found and stdin is not a terminal. "
                "Run the app once interactively to log in — Telethon will store "
                f"the session in '{self._settings.session_path}.session'."
            )

        log.info("No existing session — starting interactive login")
        try:
            await self._client.start(phone=self._settings.phone or None)
        except SessionPasswordNeededError:  # pragma: no cover - handled by start()
            raise TelegramError(
                "Two-factor authentication is enabled; rerun and enter your password."
            )

    async def _resolve_target(self) -> None:
        """Look up the configured group once, so filtering is id-based."""
        target = self._settings.target_chat
        try:
            entity = await self._client.get_entity(target)
        except ChannelPrivateError as exc:
            raise TelegramError(
                f"Your account cannot access {target!r}. Join the group first."
            ) from exc
        except (ValueError, TypeError) as exc:
            raise TelegramError(
                f"Could not resolve TELEGRAM_TARGET_CHAT={target!r}. "
                "See the README section 'Finding the group id'."
            ) from exc
        except FloodWaitError as exc:
            raise TelegramError(
                f"Telegram asked us to wait {exc.seconds}s before resolving the chat."
            ) from exc

        self._chat_id = utils.get_peer_id(entity)
        self._chat_title = utils.get_display_name(entity) or str(self._chat_id)
        self._chat_username = getattr(entity, "username", None)
        log.info("Monitoring '%s' (id=%s)", self._chat_title, self._chat_id)

    def _register_handlers(self) -> None:
        # `chats=` makes Telethon filter server-side updates for us; the explicit
        # id check in _handle() is a second line of defence so a resolution quirk
        # can never leak messages from another chat into the pipeline.
        self._client.add_event_handler(
            self._on_new_message, events.NewMessage(chats=self._chat_id)
        )
        if self._settings.analyse_edits:
            self._client.add_event_handler(
                self._on_edited_message, events.MessageEdited(chats=self._chat_id)
            )

    # --------------------------------------------------------------- handlers

    async def _on_new_message(self, event: events.NewMessage.Event) -> None:
        await self._handle(event, is_edit=False)

    async def _on_edited_message(self, event: events.MessageEdited.Event) -> None:
        await self._handle(event, is_edit=True)

    async def _handle(self, event: Any, *, is_edit: bool) -> None:
        try:
            if event.chat_id != self._chat_id:
                log.debug("Ignoring message from unrelated chat %s", event.chat_id)
                return

            text = (event.message.message or "").strip()
            if not text:
                log.debug("Skipping message %s with no text", event.message.id)
                return

            # Before the queue, deliberately: anything that reaches the
            # pipeline is analysed, and an analysis of our own report is
            # what starts the loop.
            if await self._is_own_bot(event):
                self._ignored_messages += 1
                log.debug(
                    "Ignoring message %s from our own bot (%d ignored total)",
                    event.message.id, self._ignored_messages,
                )
                return

            message = IncomingMessage(
                id=event.message.id,
                chat_id=event.chat_id,
                chat_title=self._chat_title,
                sender=await self._describe_sender(event),
                timestamp=event.message.date or datetime.now(timezone.utc),
                text=text,
                is_edit=is_edit,
                reply_to_id=event.message.reply_to_msg_id,
                link=self._message_link(event.message.id),
            )

            try:
                self._queue.put_nowait(message)
            except asyncio.QueueFull:
                self._dropped += 1
                log.error(
                    "Analysis queue full (%d) — dropped message %s (%d dropped total). "
                    "Raise WORKER_COUNT or QUEUE_MAXSIZE.",
                    self._queue.maxsize, message.id, self._dropped,
                )
        except Exception:  # noqa: BLE001 - a handler must never kill the update loop
            log.exception("Failed to enqueue an incoming message")

    async def _is_own_bot(self, event: Any) -> bool:
        """True when this message was posted by our own Telegram bot.

        Checks the sender id first: it is exact, needs no API call, and is
        derived from the bot token at startup. The username check is only a
        fallback for a deployment whose token could not be parsed, and is
        skipped entirely when no usernames are configured so the common path
        stays free of an extra sender lookup.

        Never raises — a filter that fails open would restart the loop, so a
        lookup failure is treated as "cannot confirm it is ours" only after
        the cheap id check has already said no.
        """
        if not self._ignored_sender_ids and not self._ignored_usernames:
            return False

        sender_id = getattr(event.message, "sender_id", None)
        if sender_id is not None and sender_id in self._ignored_sender_ids:
            return True

        if not self._ignored_usernames:
            return False
        try:
            sender = await event.get_sender()
        except Exception:  # noqa: BLE001 - treated as "not ours", see above
            return False
        username = getattr(sender, "username", None)
        return bool(username) and username.lower() in self._ignored_usernames

    async def _describe_sender(self, event: Any) -> str:
        """Best-effort human-readable author, tolerant of anonymous posts."""
        try:
            sender = await event.get_sender()
        except Exception:  # noqa: BLE001 - sender lookup is decorative only
            sender = None

        if sender is not None:
            name = utils.get_display_name(sender)
            username = getattr(sender, "username", None)
            if name and username:
                return f"{name} (@{username})"
            if name:
                return name
        # Channel posts and anonymous admins have no user sender.
        return getattr(event.message, "post_author", None) or self._chat_title

    def _message_link(self, message_id: int) -> str | None:
        if self._chat_username:
            return f"https://t.me/{self._chat_username}/{message_id}"
        if self._chat_id is not None and str(self._chat_id).startswith("-100"):
            return f"https://t.me/c/{str(self._chat_id)[4:]}/{message_id}"
        return None

    # ------------------------------------------------------------------- loop

    async def run_forever(self) -> None:
        """Stay connected until :meth:`stop` is called.

        Telethon reconnects on its own; this supervisor only covers the case
        where the client gives up entirely and drops the connection.
        """
        backoff = 5
        while not self._stopping.is_set():
            try:
                await self._client.run_until_disconnected()
            except AuthKeyUnregisteredError as exc:
                raise TelegramError(
                    "The session was revoked from another device. Delete "
                    f"'{self._settings.session_path}.session' and log in again."
                ) from exc
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep the monitor alive
                log.exception("Telegram connection failed")

            if self._stopping.is_set():
                break

            log.warning("Disconnected from Telegram — reconnecting in %ds", backoff)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=backoff)
                break  # stop() was called while we waited
            except asyncio.TimeoutError:
                pass

            backoff = min(backoff * 2, 300)
            try:
                await self._client.connect()
                if await self._client.is_user_authorized():
                    log.info("Reconnected to Telegram")
                    backoff = 5
            except Exception:  # noqa: BLE001 - retried on the next iteration
                log.exception("Reconnection attempt failed")

    async def stop(self) -> None:
        """Signal the supervisor to exit and close the connection."""
        self._stopping.set()
        if self._client.is_connected():
            await self._client.disconnect()
