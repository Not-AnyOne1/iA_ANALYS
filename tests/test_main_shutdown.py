"""Regression tests for run_monitor()'s resource cleanup (audit finding #4).

Both early-return paths in run_monitor() used to skip
``market_data_provider.aclose()`` (a real httpx connection pool) and
``decision_engine.aclose()``. These tests pin the fix: every exit path
must release every resource, and a failure in one cleanup step must not
prevent the rest from running.

Everything run_monitor() constructs is patched at the main.py namespace,
so no Telegram session, Claude CLI, SQLite file, or network call is
involved.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import main as main_mod
from claude_client import AnalysisError
from config import Settings
from market_data import Timeframe
from telegram_client import TelegramError


def _settings() -> Settings:
    return Settings(
        api_id=1, api_hash="x", session_name="t", phone="", target_chat="t",
        claude_cli_path="claude", model="", claude_max_turns=3,
        signal_parser_mode="off", worker_count=1, queue_maxsize=10,
        max_message_chars=8000, analyse_edits=False, max_retries=0,
        request_timeout=5.0, log_level="INFO", log_file=None,
        jsonl_output=None, color=False, show_json=False,
        storage_db_path=Path("data/signals.db"),
        # Non-empty so run_monitor builds a real (patched) TwelveDataProvider —
        # the resource that was actually being leaked.
        twelve_data_api_key="fake-key",
        market_data_cache_ttl=30.0, market_data_timeframe=Timeframe.H1,
    )


def _run(*, fail_verify: bool = False, fail_monitor: bool = False,
         storage_close_raises: bool = False) -> tuple[int, list[str]]:
    """Run run_monitor() with every dependency patched; return (exit_code, closed)."""
    closed: list[str] = []

    analyzer = MagicMock()
    analyzer.verify_auth = AsyncMock(
        side_effect=AnalysisError("auth failed") if fail_verify else None
    )
    analyzer.aclose = AsyncMock(side_effect=lambda: closed.append("analyzer"))

    monitor = MagicMock()
    monitor.start = AsyncMock(
        side_effect=TelegramError("telegram failed") if fail_monitor else None
    )
    monitor.stop = AsyncMock(side_effect=lambda: closed.append("monitor"))

    bot = MagicMock()
    bot.stop = AsyncMock(side_effect=lambda: closed.append("bot"))
    bot.attach_storage = MagicMock()
    bot.attach_statistics = MagicMock()
    bot.attach_market_data = MagicMock()

    storage = MagicMock()
    storage.initialize = AsyncMock()
    storage.close = AsyncMock(
        side_effect=RuntimeError("storage close failed") if storage_close_raises
        else (lambda: closed.append("storage"))
    )

    provider = MagicMock()
    provider.aclose = AsyncMock(side_effect=lambda: closed.append("market_data_provider"))

    decision_engine = MagicMock()
    decision_engine.aclose = AsyncMock(side_effect=lambda: closed.append("decision_engine"))

    with patch.object(main_mod, "ClaudeAnalyzer", return_value=analyzer), \
         patch.object(main_mod, "TelegramMonitor", return_value=monitor), \
         patch.object(main_mod, "TelegramBot", return_value=bot), \
         patch.object(main_mod, "Storage", return_value=storage), \
         patch.object(main_mod, "TwelveDataProvider", return_value=provider), \
         patch.object(main_mod, "DecisionEngine", return_value=decision_engine), \
         patch.object(main_mod, "BotSettings", MagicMock()), \
         patch.object(main_mod, "Statistics", MagicMock()), \
         patch.object(main_mod, "MarketDataService", MagicMock()):
        exit_code = asyncio.run(main_mod.run_monitor(_settings(), MagicMock()))

    return exit_code, closed


_ALL_RESOURCES = ("analyzer", "bot", "decision_engine", "market_data_provider", "monitor", "storage")


def test_verify_auth_failure_still_releases_every_resource():
    exit_code, closed = _run(fail_verify=True)

    assert exit_code == main_mod.EXIT_RUNTIME
    for resource in _ALL_RESOURCES:
        assert resource in closed, f"{resource} was not released"


def test_monitor_start_failure_still_releases_every_resource():
    exit_code, closed = _run(fail_monitor=True)

    assert exit_code == main_mod.EXIT_RUNTIME
    for resource in _ALL_RESOURCES:
        assert resource in closed, f"{resource} was not released"


def test_market_data_provider_is_closed_on_an_early_return():
    # The specific leak audit finding #4 identified: an httpx connection
    # pool left open when verify_auth() fails.
    _, closed = _run(fail_verify=True)
    assert "market_data_provider" in closed


def test_one_failing_cleanup_step_does_not_block_the_others():
    exit_code, closed = _run(fail_verify=True, storage_close_raises=True)

    assert exit_code == main_mod.EXIT_RUNTIME  # shutdown never masks the exit code
    assert "storage" not in closed  # it raised
    for resource in ("analyzer", "bot", "decision_engine", "market_data_provider", "monitor"):
        assert resource in closed, f"{resource} was skipped after an earlier cleanup failure"
