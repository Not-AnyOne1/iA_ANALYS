"""Unit tests for telegram_bot.py.

No real Telegram network calls or bot token needed anywhere: the underlying
telegram.Bot is replaced with an AsyncMock at the boundary where this module
calls send_message, and command handlers are exercised with mocked
Update/context objects (the standard python-telegram-bot testing pattern).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TimedOut

from market_data import MarketDataError, Quote
from models import SignalAnalysis, TradeSetup
from storage import StoredAnalysis
from telegram_bot import BotSettings, TelegramBot, _BoundedSet
from telegram_client import IncomingMessage


# --------------------------------------------------------------------------- helpers

def _settings(**overrides) -> BotSettings:
    base = dict(bot_token="test-token", chat_id=12345, max_retries=3,
                retry_base_delay=0.01, retry_max_delay=0.05)
    base.update(overrides)
    return BotSettings(**base)


def _msg(
    chat_id: int = -100999, message_id: int = 1, text: str = "hi",
    chat_title: str = "VIP Signals", sender: str = "Analyst",
) -> IncomingMessage:
    return IncomingMessage(
        id=message_id, chat_id=chat_id, chat_title=chat_title, sender=sender,
        timestamp=datetime.now(timezone.utc), text=text,
    )


def _signal_analysis(**overrides) -> SignalAnalysis:
    base = dict(
        is_signal=True, category="signal",
        setup=TradeSetup(symbol="BTCUSDT", direction="long", order_type="limit",
                          entries=[61200.0], stop_loss=60350.0, take_profits=[62400.0]),
        summary="Long setup on BTCUSDT.", confidence=0.9,
        missing_fields=[], notes=None, source="regex",
    )
    base.update(overrides)
    return SignalAnalysis(**base)


def _bot_with_mock_client(**settings_overrides) -> TelegramBot:
    """A TelegramBot whose underlying telegram.Bot is an AsyncMock."""
    bot = TelegramBot(_settings(**settings_overrides))
    bot._bot.send_message = AsyncMock()
    return bot


def _stored_record(message_id: int = 7, **overrides) -> StoredAnalysis:
    return StoredAnalysis(
        message_id=message_id, chat_title="VIP Signals", sender="Analyst",
        timestamp=datetime.now(timezone.utc), analysis=_signal_analysis(**overrides),
    )


class _StubStorage:
    """Fakes storage.Storage's read interface for /status,/stats,/latest,/history.

    Write-path behaviour and real SQLite semantics are covered by
    test_storage.py — telegram_bot.py's own tests only need to verify it
    reads through this interface correctly and degrades gracefully when a
    query raises.
    """

    def __init__(self, *, latest_record=None, history_records=None, raise_on_query: bool = False):
        self._latest_record = latest_record
        self._history_records = history_records or []
        self._raise_on_query = raise_on_query

    async def latest(self):
        if self._raise_on_query:
            raise RuntimeError("storage unavailable")
        return self._latest_record

    async def history(self, limit: int):
        if self._raise_on_query:
            raise RuntimeError("storage unavailable")
        return self._history_records[:limit]


class _StubStatistics:
    """Fakes statistics.Statistics's summary() for /status and /stats tests.

    Real aggregation correctness is covered by test_statistics.py —
    telegram_bot.py's own tests only need to verify it calls through this
    interface and handles a query failure gracefully.
    """

    def __init__(self, *, formatted: str = "0 analysed", raise_on_query: bool = False):
        self._formatted = formatted
        self._raise_on_query = raise_on_query

    async def summary(self):
        if self._raise_on_query:
            raise RuntimeError("statistics unavailable")
        return SimpleNamespace(format=lambda: self._formatted)


class _StubMarketData:
    """Fakes market_data.MarketDataService's get_quote() for /price tests.

    telegram_bot.py's own tests only need to verify it calls through this
    interface and handles MarketDataError correctly — provider/caching/retry
    behavior is covered by test_market_data.py.
    """

    def __init__(self, *, quote=None, error=None):
        self._quote = quote
        self._error = error

    async def get_quote(self, symbol: str):
        if self._error is not None:
            raise self._error
        return self._quote


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retry loops sleep between attempts; keep tests instant."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())


@pytest.fixture(autouse=True)
def _fake_bot_class(monkeypatch):
    """telegram.Bot(token=...) does real (~1s) client/SSL-context setup on
    construction that unit tests never need — replace the class TelegramBot
    constructs so every test stays fast, without touching telegram_bot.py."""
    def _factory(token: str) -> MagicMock:
        fake = MagicMock()
        fake.send_message = AsyncMock()
        return fake

    monkeypatch.setattr("telegram_bot.Bot", _factory)


# --------------------------------------------------------------------------- BotSettings

def test_disabled_without_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_CHAT_ID", "12345")
    assert BotSettings.from_env().enabled is False


def test_enabled_with_only_a_token(monkeypatch):
    """A target chat is no longer required: the audience is built at runtime
    from /start, so a token alone is enough to run."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.delenv("TELEGRAM_BOT_CHAT_ID", raising=False)
    settings = BotSettings.from_env()
    assert settings.enabled is True
    assert settings.chat_id is None


def test_enabled_with_numeric_chat_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT_ID", "-1001234567890")
    settings = BotSettings.from_env()
    assert settings.enabled is True
    assert settings.chat_id == -1001234567890
    assert isinstance(settings.chat_id, int)


def test_enabled_with_username_chat_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT_ID", "@mychannel")
    settings = BotSettings.from_env()
    assert settings.enabled is True
    assert settings.chat_id == "@mychannel"


def test_tuning_defaults(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT_ID", "12345")
    for var in (
        "TELEGRAM_BOT_MAX_RETRIES", "TELEGRAM_BOT_RETRY_BASE_DELAY",
        "TELEGRAM_BOT_RETRY_MAX_DELAY",
        "TELEGRAM_BOT_HISTORY_DISPLAY_DEFAULT", "TELEGRAM_BOT_DEDUP_CACHE_SIZE",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = BotSettings.from_env()
    assert settings.max_retries == 5
    assert settings.retry_base_delay == 2.0
    assert settings.retry_max_delay == 60.0
    assert settings.history_display_default == 5
    assert settings.dedup_cache_size == 2000


# --------------------------------------------------------------------------- _BoundedSet

def test_bounded_set_evicts_oldest():
    s = _BoundedSet(max_size=2)
    s.add((0, 1))
    s.add((0, 2))
    s.add((0, 3))  # evicts (0, 1)
    assert (0, 1) not in s
    assert (0, 2) in s
    assert (0, 3) in s


# --------------------------------------------------------------------------- notify()

def test_disabled_bot_notify_is_noop():
    bot = TelegramBot(BotSettings(bot_token=None, chat_id=None))
    asyncio.run(bot.notify(_msg(), _signal_analysis()))
    assert bot._stats.notifications_sent == 0  # nothing attempted, nothing raised


def test_successful_notify_records_stats_and_dedup_key():
    # notify() itself no longer keeps analysis history (storage.py + the
    # pipeline.py -> storage.record() call do, tested in test_storage.py and
    # test_pipeline_routing.py) — only its own send-side stats/dedup remain.
    bot = _bot_with_mock_client()
    message, analysis = _msg(), _signal_analysis()

    asyncio.run(bot.notify(message, analysis))

    bot._bot.send_message.assert_awaited_once()
    assert bot._stats.notifications_sent == 1
    assert (message.chat_id, message.id) in bot._sent_ids


def test_duplicate_notify_is_skipped():
    bot = _bot_with_mock_client()
    message, analysis = _msg(), _signal_analysis()

    asyncio.run(bot.notify(message, analysis))
    asyncio.run(bot.notify(message, analysis))  # same message id again

    assert bot._bot.send_message.await_count == 1
    assert bot._stats.notifications_sent == 1
    assert bot._stats.duplicates_skipped == 1


def test_retry_after_is_honoured_then_succeeds():
    bot = _bot_with_mock_client()
    bot._bot.send_message = AsyncMock(side_effect=[RetryAfter(1), None])

    asyncio.run(bot.notify(_msg(), _signal_analysis()))

    assert bot._bot.send_message.await_count == 2
    assert bot._stats.notifications_sent == 1


def test_network_error_retried_then_succeeds():
    bot = _bot_with_mock_client(max_retries=3)
    bot._bot.send_message = AsyncMock(side_effect=[TimedOut(), TimedOut(), None])

    asyncio.run(bot.notify(_msg(), _signal_analysis()))

    assert bot._bot.send_message.await_count == 3
    assert bot._stats.notifications_sent == 1


def test_permanent_forbidden_error_is_not_retried():
    bot = _bot_with_mock_client(max_retries=5)
    bot._bot.send_message = AsyncMock(side_effect=Forbidden("bot was blocked"))

    asyncio.run(bot.notify(_msg(), _signal_analysis()))  # must not raise

    assert bot._bot.send_message.await_count == 1  # no retry attempted
    assert bot._stats.notifications_failed == 1
    assert bot._stats.notifications_sent == 0


def test_permanent_bad_request_is_not_retried():
    bot = _bot_with_mock_client(max_retries=5)
    bot._bot.send_message = AsyncMock(side_effect=BadRequest("chat not found"))

    asyncio.run(bot.notify(_msg(), _signal_analysis()))

    assert bot._bot.send_message.await_count == 1
    assert bot._stats.notifications_failed == 1


def test_retries_exhausted_never_raises_and_is_counted():
    bot = _bot_with_mock_client(max_retries=2)
    bot._bot.send_message = AsyncMock(side_effect=TimedOut())  # always fails

    asyncio.run(bot.notify(_msg(), _signal_analysis()))  # must not raise

    assert bot._bot.send_message.await_count == 3  # 1 + max_retries
    assert bot._stats.notifications_failed == 1
    # A failed send must not be marked as sent/deduped, so a later retry of
    # the same message (if the caller chooses to re-notify) is still possible.
    assert (-100999, 1) not in bot._sent_ids


def test_unexpected_exception_type_is_still_swallowed():
    bot = _bot_with_mock_client()
    bot._bot.send_message = AsyncMock(side_effect=ValueError("something weird"))

    asyncio.run(bot.notify(_msg(), _signal_analysis()))  # must not raise

    assert bot._stats.notifications_failed == 1


# --------------------------------------------------------------------------- formatting

def test_html_escaping_of_dynamic_content():
    bot = _bot_with_mock_client()
    message = _msg(chat_title="<script>alert(1)</script> & Co")
    analysis = _signal_analysis(summary="Contains <b>raw</b> & unsafe text")

    text = bot._format_notification(message, analysis)

    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "Contains &lt;b&gt;raw&lt;/b&gt;" in text


def test_signal_formatting_includes_setup_fields():
    bot = _bot_with_mock_client()
    text = bot._format_notification(_msg(), _signal_analysis())
    assert "BTCUSDT" in text
    assert "LONG" in text
    assert "61200" in text
    assert "60350" in text
    assert "source: regex" in text


def test_non_signal_formatting_omits_setup_fields():
    bot = _bot_with_mock_client()
    analysis = _signal_analysis(
        is_signal=False, category="commentary", setup=TradeSetup(),
        summary="Just chatting.",
    )
    text = bot._format_notification(_msg(), analysis)
    assert "Symbol" not in text
    assert "Entry" not in text
    assert "COMMENTARY" in text


def test_overly_long_message_is_truncated():
    bot = _bot_with_mock_client()
    analysis = _signal_analysis(summary="x" * 5000)
    text = bot._format_notification(_msg(), analysis)
    assert len(text) <= 4000 + len("\n[... truncated]")
    assert text.endswith("[... truncated]")


# --------------------------------------------------------------------------- commands

def _update(chat_id: int = 12345, username: str | None = None,
            chat_type: str = "private") -> MagicMock:
    update = MagicMock()
    update.effective_chat = MagicMock(id=chat_id, username=username,
                                  type=chat_type, first_name=None)
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


def _context(args: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(args=args or [])


def test_start_command_replies_for_authorized_chat():
    bot = _bot_with_mock_client()
    update = _update(chat_id=12345)

    asyncio.run(bot._cmd_start(update, _context()))

    update.message.reply_text.assert_awaited_once()


def test_informational_commands_ignore_a_non_subscriber():
    """/status and /history still reveal pipeline internals, so they stay
    behind a check — membership now, rather than one hard-coded chat."""
    bot = _bot_with_mock_client()
    update = _update(chat_id=99999)  # not subscribed, not settings.chat_id

    asyncio.run(bot._cmd_status(update, _context()))
    asyncio.run(bot._cmd_history(update, _context()))

    update.message.reply_text.assert_not_awaited()


def test_start_is_open_to_anyone():
    """Subscribing cannot require being subscribed."""
    bot = _bot_with_mock_client()
    update = _update(chat_id=99999)

    asyncio.run(bot._cmd_start(update, _context()))

    update.message.reply_text.assert_awaited_once()
    assert 99999 in bot._subscribers


def test_subscribing_unlocks_the_informational_commands():
    bot = _bot_with_mock_client()
    update = _update(chat_id=99999)

    asyncio.run(bot._cmd_status(update, _context()))
    assert update.message.reply_text.await_count == 0

    asyncio.run(bot._cmd_start(update, _context()))
    asyncio.run(bot._cmd_status(update, _context()))
    assert update.message.reply_text.await_count == 2


def test_username_based_authorization_still_works_for_the_legacy_chat():
    """The configured @channel keeps its access without subscribing."""
    bot = TelegramBot(_settings(chat_id="@mychannel"))
    authorized = _update(chat_id=555, username="mychannel")
    unauthorized = _update(chat_id=555, username="someoneelse")

    asyncio.run(bot._cmd_status(authorized, _context()))
    asyncio.run(bot._cmd_status(unauthorized, _context()))

    assert authorized.message.reply_text.await_count == 1
    assert unauthorized.message.reply_text.await_count == 0


def test_help_command_lists_all_commands():
    bot = _bot_with_mock_client()
    update = _update()

    asyncio.run(bot._cmd_help(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    for command in ("/start", "/help", "/status", "/stats", "/latest", "/history", "/price"):
        assert command in text


def test_status_command_without_attached_statistics():
    bot = _bot_with_mock_client()
    update = _update()

    asyncio.run(bot._cmd_status(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "statistics not attached" in text


def test_status_command_with_attached_statistics():
    bot = _bot_with_mock_client()
    bot.attach_statistics(_StubStatistics(formatted="42 analysed"))
    update = _update()

    asyncio.run(bot._cmd_status(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "42 analysed" in text


def test_status_command_reports_statistics_query_failure():
    bot = _bot_with_mock_client()
    bot.attach_statistics(_StubStatistics(raise_on_query=True))
    update = _update()

    asyncio.run(bot._cmd_status(update, _context()))  # must not raise

    text = update.message.reply_text.await_args.args[0]
    assert "statistics error" in text


def test_stats_command_without_attached_statistics():
    bot = _bot_with_mock_client()
    update = _update()

    asyncio.run(bot._cmd_stats(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "not available yet" in text


def test_stats_command_with_attached_statistics():
    bot = _bot_with_mock_client()
    bot.attach_statistics(_StubStatistics(formatted="3 analysed"))
    update = _update()

    asyncio.run(bot._cmd_stats(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "3 analysed" in text


def test_stats_command_reports_statistics_query_failure():
    bot = _bot_with_mock_client()
    bot.attach_statistics(_StubStatistics(raise_on_query=True))
    update = _update()

    asyncio.run(bot._cmd_stats(update, _context()))  # must not raise

    text = update.message.reply_text.await_args.args[0]
    assert "temporarily unavailable" in text


def test_latest_command_without_attached_storage():
    bot = _bot_with_mock_client()
    update = _update()

    asyncio.run(bot._cmd_latest(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "No analyses recorded yet." in text


def test_latest_command_with_no_stored_analyses():
    bot = _bot_with_mock_client()
    bot.attach_storage(_StubStorage(latest_record=None))
    update = _update()

    asyncio.run(bot._cmd_latest(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "No analyses recorded yet." in text


def test_latest_command_returns_stored_record():
    bot = _bot_with_mock_client()
    bot.attach_storage(_StubStorage(latest_record=_stored_record(message_id=7)))
    update = _update()

    asyncio.run(bot._cmd_latest(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "#7" in text
    assert "BTCUSDT" in text


def test_latest_command_reports_storage_query_failure():
    bot = _bot_with_mock_client()
    bot.attach_storage(_StubStorage(raise_on_query=True))
    update = _update()

    asyncio.run(bot._cmd_latest(update, _context()))  # must not raise

    text = update.message.reply_text.await_args.args[0]
    assert "temporarily unavailable" in text


def test_history_command_without_attached_storage():
    bot = _bot_with_mock_client()
    update = _update()

    asyncio.run(bot._cmd_history(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "No analyses recorded yet." in text


def test_history_command_respects_count_argument():
    bot = _bot_with_mock_client()
    # storage.history() already returns most-recent-first — the stub mirrors
    # that contract exactly as the real Storage would.
    records = [_stored_record(message_id=i) for i in (5, 4, 3, 2, 1)]
    bot.attach_storage(_StubStorage(history_records=records))
    update = _update()

    asyncio.run(bot._cmd_history(update, _context(args=["2"])))

    text = update.message.reply_text.await_args.args[0]
    assert "#5" in text and "#4" in text
    assert "#3" not in text


def test_history_command_invalid_argument_shows_usage():
    bot = _bot_with_mock_client()
    bot.attach_storage(_StubStorage(history_records=[_stored_record()]))
    update = _update()

    asyncio.run(bot._cmd_history(update, _context(args=["not-a-number"])))

    text = update.message.reply_text.await_args.args[0]
    assert "Usage" in text


def test_history_command_reports_storage_query_failure():
    bot = _bot_with_mock_client()
    bot.attach_storage(_StubStorage(raise_on_query=True))
    update = _update()

    asyncio.run(bot._cmd_history(update, _context()))  # must not raise

    text = update.message.reply_text.await_args.args[0]
    assert "temporarily unavailable" in text


# --------------------------------------------------------------------------- /price (RFC-004)

def test_price_command_without_argument_shows_usage():
    bot = _bot_with_mock_client()
    update = _update()

    asyncio.run(bot._cmd_price(update, _context()))

    text = update.message.reply_text.await_args.args[0]
    assert "Usage" in text


def test_price_command_without_attached_market_data():
    bot = _bot_with_mock_client()
    update = _update()

    asyncio.run(bot._cmd_price(update, _context(args=["XAUUSD"])))

    text = update.message.reply_text.await_args.args[0]
    assert "not configured" in text


def test_price_command_returns_quote():
    bot = _bot_with_mock_client()
    quote = Quote(symbol="XAUUSD", price=2400.5, timestamp=datetime.now(timezone.utc), provider="twelve_data")
    bot.attach_market_data(_StubMarketData(quote=quote))
    update = _update()

    asyncio.run(bot._cmd_price(update, _context(args=["xauusd"])))

    text = update.message.reply_text.await_args.args[0]
    assert "XAUUSD" in text
    assert "2400.5" in text
    assert "twelve_data" in text


def test_price_command_reports_market_data_error():
    bot = _bot_with_mock_client()
    bot.attach_market_data(_StubMarketData(error=MarketDataError("symbol not found", retryable=False)))
    update = _update()

    asyncio.run(bot._cmd_price(update, _context(args=["NOTREAL"])))  # must not raise

    text = update.message.reply_text.await_args.args[0]
    assert "unavailable" in text
    assert "NOTREAL" in text


# ------------------------------------------------------ dynamic subscribers

def _bot_with_storage(tmp_path, **overrides):
    from storage import Storage
    bot = _bot_with_mock_client(**overrides)
    storage = Storage(tmp_path / "subs.db")
    asyncio.run(storage.initialize())
    bot.attach_storage(storage)
    return bot, storage


def test_start_registers_and_stop_unregisters(tmp_path):
    bot, storage = _bot_with_storage(tmp_path)
    update = _update(chat_id=555, username="alice")

    asyncio.run(bot._cmd_start(update, _context()))
    assert bot._subscribers == {555}
    assert asyncio.run(storage.subscribers()) == [555]

    asyncio.run(bot._cmd_stop(update, _context()))
    assert bot._subscribers == set()
    assert asyncio.run(storage.subscribers()) == []


def test_start_is_idempotent(tmp_path):
    bot, storage = _bot_with_storage(tmp_path)
    update = _update(chat_id=555)

    asyncio.run(bot._cmd_start(update, _context()))
    asyncio.run(bot._cmd_start(update, _context()))

    assert asyncio.run(storage.subscribers()) == [555]
    assert "already subscribed" in update.message.reply_text.await_args_list[-1].args[0].lower()


def test_subscribers_survive_a_restart(tmp_path):
    bot, storage = _bot_with_storage(tmp_path)
    asyncio.run(bot._cmd_start(_update(chat_id=111), _context()))
    asyncio.run(bot._cmd_start(_update(chat_id=222), _context()))

    fresh = _bot_with_mock_client()          # a new process
    fresh.attach_storage(storage)
    assert fresh._subscribers == set()
    asyncio.run(fresh._load_subscribers())
    assert fresh._subscribers == {111, 222}


def test_start_in_a_group_does_not_subscribe(tmp_path):
    """Registering a group would broadcast every report to all its members."""
    bot, storage = _bot_with_storage(tmp_path)
    update = _update(chat_id=-100999, chat_type="supergroup")

    asyncio.run(bot._cmd_start(update, _context()))

    assert bot._subscribers == set()
    assert asyncio.run(storage.subscribers()) == []
    assert "private chat" in update.message.reply_text.await_args.args[0]


def test_stop_when_not_subscribed_is_harmless(tmp_path):
    bot, _ = _bot_with_storage(tmp_path)
    update = _update(chat_id=555)

    asyncio.run(bot._cmd_stop(update, _context()))

    assert bot._subscribers == set()
    assert "not subscribed" in update.message.reply_text.await_args.args[0].lower()


def test_subscribing_works_without_storage():
    """Persistence is unavailable, but the bot must still function."""
    bot = _bot_with_mock_client()
    assert bot._storage is None

    asyncio.run(bot._cmd_start(_update(chat_id=555), _context()))
    assert bot._subscribers == {555}


def test_destinations_prefer_subscribers_over_the_configured_chat():
    bot = _bot_with_mock_client(chat_id=12345)
    assert bot._destinations() == [12345]         # fallback while empty
    bot._subscribers = {777, 888}
    assert bot._destinations() == [777, 888]      # subscribers take over


def test_destinations_are_empty_without_subscribers_or_fallback():
    assert _bot_with_mock_client(chat_id=None)._destinations() == []


def test_help_lists_start_and_stop():
    bot = _bot_with_mock_client()
    update = _update(chat_id=12345)
    asyncio.run(bot._cmd_help(update, _context()))
    text = update.message.reply_text.await_args.args[0]
    assert "/start" in text and "/stop" in text
