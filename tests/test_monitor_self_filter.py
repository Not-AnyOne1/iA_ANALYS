"""The monitor must never analyse messages our own bot posted.

When the bot's destination is the chat the monitor watches, its report
arrives back as an ordinary message. Analysing it produces another report,
which arrives back again — an unbounded loop that burns the Claude quota and
floods the chat. These tests pin the filter that stops it, and equally pin
that ordinary traffic still gets through.

The Telethon boundary is faked: ``_handle`` is called directly with an event
shaped like the real one, which is exactly the seam the filter sits on.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config import Settings
from market_data import Timeframe
from telegram_bot import BotSettings
from telegram_client import TelegramMonitor

CHAT_ID = -100999
BOT_ID = 8899684070


def _settings() -> Settings:
    return Settings(
        api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
        claude_cli_path="claude", model="", claude_max_turns=3,
        signal_parser_mode="off", worker_count=1, queue_maxsize=50,
        max_message_chars=8000, analyse_edits=False, max_retries=0,
        request_timeout=5.0, log_level="CRITICAL", log_file=None, jsonl_output=None,
        color=False, show_json=False, storage_db_path=Path("data/signals.db"),
        twelve_data_api_key="", market_data_cache_ttl=30.0,
        market_data_timeframe=Timeframe.H1)


def _monitor(*, ignore_ids=(), ignore_usernames=()) -> TelegramMonitor:
    queue: asyncio.Queue = asyncio.Queue()
    monitor = TelegramMonitor(_settings(), queue,
                              ignore_sender_ids=ignore_ids,
                              ignore_usernames=ignore_usernames)
    monitor._chat_id = CHAT_ID
    monitor._chat_title = "VIP Signals"
    return monitor


def _event(*, sender_id: int, text: str = "BUY GOLD NOW", message_id: int = 1,
           username: str | None = None):
    """An event shaped like Telethon's, for one message in the watched chat."""
    message = SimpleNamespace(
        id=message_id, message=text, date=datetime.now(timezone.utc),
        reply_to_msg_id=None, sender_id=sender_id)
    sender = SimpleNamespace(username=username, first_name="Someone", title=None)
    return SimpleNamespace(
        chat_id=CHAT_ID, message=message,
        get_sender=AsyncMock(return_value=sender))


def _drain(monitor) -> list:
    out = []
    while not monitor._queue.empty():
        out.append(monitor._queue.get_nowait())
    return out


# --------------------------------------------------------------- the loop

def test_a_message_from_our_own_bot_is_never_queued():
    monitor = _monitor(ignore_ids=[BOT_ID])

    asyncio.run(monitor._handle(_event(sender_id=BOT_ID), is_edit=False))

    assert _drain(monitor) == []
    assert monitor._ignored_messages == 1


def test_a_user_signal_is_queued_exactly_once():
    monitor = _monitor(ignore_ids=[BOT_ID])

    asyncio.run(monitor._handle(_event(sender_id=555), is_edit=False))

    queued = _drain(monitor)
    assert len(queued) == 1
    assert queued[0].text == "BUY GOLD NOW"


def test_the_loop_cannot_start():
    """The exact reported cycle: user signal in, bot report back, repeat.

    However many times the bot's own output is re-delivered, only the
    original user message is ever analysed.
    """
    monitor = _monitor(ignore_ids=[BOT_ID])

    asyncio.run(monitor._handle(
        _event(sender_id=555, text="BUY GOLD NOW", message_id=1), is_edit=False))
    for i in range(2, 12):          # ten generations of the bot's own reports
        asyncio.run(monitor._handle(
            _event(sender_id=BOT_ID, text="MARKET REPORT XAUUSD [ENTER]", message_id=i),
            is_edit=False))

    queued = _drain(monitor)
    assert len(queued) == 1, [m.text for m in queued]
    assert queued[0].id == 1
    assert monitor._ignored_messages == 10


def test_the_filter_also_covers_edits():
    """An edited bot message must not re-enter either."""
    monitor = _monitor(ignore_ids=[BOT_ID])

    asyncio.run(monitor._handle(_event(sender_id=BOT_ID), is_edit=True))

    assert _drain(monitor) == []


# ------------------------------------------------------- ordinary traffic

@pytest.mark.parametrize("sender_id", [555, -1001234567890, 0, 999999999999])
def test_everything_that_is_not_our_bot_still_gets_through(sender_id):
    """Users, groups, channels and anonymous admins are unaffected."""
    monitor = _monitor(ignore_ids=[BOT_ID])

    asyncio.run(monitor._handle(_event(sender_id=sender_id), is_edit=False))

    assert len(_drain(monitor)) == 1


def test_a_message_with_no_sender_id_still_gets_through():
    """Channel posts can arrive without a user sender; they are not ours."""
    monitor = _monitor(ignore_ids=[BOT_ID])

    asyncio.run(monitor._handle(_event(sender_id=None), is_edit=False))

    assert len(_drain(monitor)) == 1


def test_nothing_is_filtered_when_no_identity_is_configured():
    """A deployment that never sets a bot token behaves exactly as before."""
    monitor = _monitor()

    asyncio.run(monitor._handle(_event(sender_id=BOT_ID), is_edit=False))

    assert len(_drain(monitor)) == 1


# ------------------------------------------------------- username fallback

def test_username_fallback_filters_when_the_id_is_unknown():
    monitor = _monitor(ignore_usernames=["MyTradingBot"])

    asyncio.run(monitor._handle(
        _event(sender_id=BOT_ID, username="mytradingbot"), is_edit=False))

    assert _drain(monitor) == []


def test_username_fallback_is_case_and_at_insensitive():
    monitor = _monitor(ignore_usernames=["@MyTradingBot"])

    asyncio.run(monitor._handle(
        _event(sender_id=1, username="MYTRADINGBOT"), is_edit=False))

    assert _drain(monitor) == []


def test_a_different_username_is_not_filtered():
    monitor = _monitor(ignore_usernames=["mytradingbot"])

    asyncio.run(monitor._handle(
        _event(sender_id=1, username="someone_else"), is_edit=False))

    assert len(_drain(monitor)) == 1


def test_a_failing_sender_lookup_does_not_drop_a_real_message():
    """The username path needs an API call; if it fails, a user's signal
    must still be analysed rather than silently discarded."""
    monitor = _monitor(ignore_usernames=["mytradingbot"])
    event = _event(sender_id=555)
    event.get_sender = AsyncMock(side_effect=RuntimeError("network"))

    asyncio.run(monitor._handle(event, is_edit=False))

    assert len(_drain(monitor)) == 1


def test_the_id_check_needs_no_sender_lookup():
    """The common path must not pay for an extra API round-trip."""
    monitor = _monitor(ignore_ids=[BOT_ID])
    event = _event(sender_id=BOT_ID)

    asyncio.run(monitor._handle(event, is_edit=False))

    event.get_sender.assert_not_awaited()


# ------------------------------------------------------------ the bot id

def test_the_bot_user_id_comes_from_the_token():
    assert BotSettings(bot_token="8899684070:AAF-xyz").bot_user_id == 8899684070


@pytest.mark.parametrize("token", [None, "", "no-colon", "notdigits:AAF", ":AAF"])
def test_a_token_that_cannot_be_parsed_yields_no_id(token):
    """Never raise on a malformed token — a filter that cannot be built must
    not stop the bot from starting."""
    assert BotSettings(bot_token=token).bot_user_id is None
