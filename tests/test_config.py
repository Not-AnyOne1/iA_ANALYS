"""Unit tests for config.py.

Every test drives the real ``Settings.load()`` / ``parse_chat_ref()`` code
paths through the process environment (monkeypatch-scoped, so nothing
leaks between tests) with ``env_file=None`` — that skips ``load_dotenv``
entirely, so the developer's real ``.env`` can never influence a result
and these stay parallel-safe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ConfigError, Settings, parse_chat_ref
from market_data import Timeframe


_REQUIRED = {
    "TELEGRAM_API_ID": "123456",
    "TELEGRAM_API_HASH": "deadbeef",
    "TELEGRAM_TARGET_CHAT": "-1001234567890",
}

# Every optional variable Settings.load() reads. Cleared before each test so
# a value set in the developer's real environment can't change an outcome.
_OPTIONAL = (
    "TELEGRAM_SESSION", "TELEGRAM_PHONE", "CLAUDE_CLI_PATH", "CLAUDE_MODEL",
    "CLAUDE_MAX_TURNS", "CLAUDE_MAX_RETRIES", "CLAUDE_TIMEOUT_SECONDS",
    "SIGNAL_PARSER_MODE", "WORKER_COUNT", "QUEUE_MAXSIZE", "MAX_MESSAGE_CHARS",
    "ANALYSE_EDITS", "LOG_LEVEL", "LOG_FILE", "JSONL_OUTPUT", "COLOR_OUTPUT",
    "SHOW_JSON", "NO_COLOR", "STORAGE_DB_PATH", "TWELVE_DATA_API_KEY",
    "MARKET_DATA_CACHE_TTL", "MARKET_DATA_TIMEFRAME",
)


@pytest.fixture
def env(monkeypatch):
    """A clean environment with only the required variables set."""
    for name in _OPTIONAL:
        monkeypatch.delenv(name, raising=False)
    for name, value in _REQUIRED.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def _load() -> Settings:
    # env_file=None: never read the real .env from disk.
    return Settings.load(env_file=None)


# --------------------------------------------------------------------------- happy path

def test_load_with_only_required_variables_uses_documented_defaults(env):
    settings = _load()

    assert settings.api_id == 123456
    assert settings.api_hash == "deadbeef"
    assert settings.target_chat == -1001234567890
    assert settings.session_name == "signal_monitor"
    assert settings.claude_cli_path == "claude"
    assert settings.model == ""
    assert settings.claude_max_turns == 3
    assert settings.signal_parser_mode == "shadow"
    assert settings.worker_count == 2
    assert settings.queue_maxsize == 200
    assert settings.max_message_chars == 8000
    assert settings.analyse_edits is False
    assert settings.max_retries == 3
    assert settings.request_timeout == 180.0
    assert settings.log_level == "INFO"
    assert settings.log_file is None
    assert settings.jsonl_output is None
    assert settings.show_json is True
    assert settings.storage_db_path == Path("data/signals.db")
    assert settings.twelve_data_api_key == ""
    assert settings.market_data_cache_ttl == 30.0
    assert settings.market_data_timeframe is Timeframe.H1


def test_load_returns_a_frozen_settings_object(env):
    settings = _load()
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        settings.worker_count = 99  # type: ignore[misc]


def test_optional_values_override_defaults(env):
    env.setenv("TELEGRAM_SESSION", "custom_session")
    env.setenv("CLAUDE_MODEL", "sonnet")
    env.setenv("WORKER_COUNT", "8")
    env.setenv("STORAGE_DB_PATH", "custom/path.db")
    env.setenv("LOG_LEVEL", "debug")  # lower-cased on purpose

    settings = _load()

    assert settings.session_name == "custom_session"
    assert settings.model == "sonnet"
    assert settings.worker_count == 8
    assert settings.storage_db_path == Path("custom/path.db")
    assert settings.log_level == "DEBUG"  # upper-cased by load()


def test_whitespace_around_values_is_stripped(env):
    env.setenv("CLAUDE_MODEL", "  sonnet  ")
    assert _load().model == "sonnet"


def test_log_file_and_jsonl_output_become_paths_when_set(env):
    env.setenv("LOG_FILE", "logs/monitor.log")
    env.setenv("JSONL_OUTPUT", "data/signals.jsonl")

    settings = _load()

    assert settings.log_file == Path("logs/monitor.log")
    assert settings.jsonl_output == Path("data/signals.jsonl")


# --------------------------------------------------------------------------- required-variable errors

@pytest.mark.parametrize("missing", sorted(_REQUIRED))
def test_missing_required_variable_raises_config_error(env, missing):
    env.delenv(missing, raising=False)
    with pytest.raises(ConfigError) as exc_info:
        _load()
    assert missing in str(exc_info.value)


@pytest.mark.parametrize("missing", sorted(_REQUIRED))
def test_empty_required_variable_is_treated_as_missing(env, missing):
    env.setenv(missing, "   ")  # whitespace only
    with pytest.raises(ConfigError):
        _load()


def test_non_integer_api_id_raises_config_error(env):
    env.setenv("TELEGRAM_API_ID", "not-a-number")
    with pytest.raises(ConfigError) as exc_info:
        _load()
    assert "TELEGRAM_API_ID" in str(exc_info.value)


# --------------------------------------------------------------------------- _int parsing

def test_int_parsing_accepts_a_valid_value(env):
    env.setenv("WORKER_COUNT", "5")
    assert _load().worker_count == 5


def test_int_parsing_rejects_a_non_numeric_value(env):
    env.setenv("WORKER_COUNT", "many")
    with pytest.raises(ConfigError) as exc_info:
        _load()
    assert "WORKER_COUNT" in str(exc_info.value)
    assert "integer" in str(exc_info.value)


def test_int_parsing_enforces_the_minimum(env):
    env.setenv("WORKER_COUNT", "0")  # minimum is 1
    with pytest.raises(ConfigError) as exc_info:
        _load()
    assert ">= 1" in str(exc_info.value)


def test_int_parsing_allows_zero_where_the_minimum_is_zero(env):
    env.setenv("CLAUDE_MAX_RETRIES", "0")  # minimum=0 for this one
    assert _load().max_retries == 0


def test_int_parsing_enforces_a_higher_minimum(env):
    env.setenv("MAX_MESSAGE_CHARS", "50")  # minimum is 100
    with pytest.raises(ConfigError) as exc_info:
        _load()
    assert ">= 100" in str(exc_info.value)


def test_empty_int_value_falls_back_to_the_default(env):
    env.setenv("WORKER_COUNT", "")
    assert _load().worker_count == 2


# --------------------------------------------------------------------------- _bool parsing

@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "  true  "])
def test_bool_parsing_accepts_truthy_spellings(env, raw):
    env.setenv("ANALYSE_EDITS", raw)
    assert _load().analyse_edits is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
def test_bool_parsing_accepts_falsy_spellings(env, raw):
    env.setenv("ANALYSE_EDITS", raw)
    assert _load().analyse_edits is False


def test_bool_parsing_rejects_an_unrecognised_value(env):
    env.setenv("ANALYSE_EDITS", "maybe")
    with pytest.raises(ConfigError) as exc_info:
        _load()
    assert "ANALYSE_EDITS" in str(exc_info.value)


def test_empty_bool_value_falls_back_to_the_default(env):
    env.setenv("ANALYSE_EDITS", "")
    assert _load().analyse_edits is False


def test_no_color_disables_colour_output(env):
    env.setenv("COLOR_OUTPUT", "true")
    env.setenv("NO_COLOR", "1")
    assert _load().color is False


# --------------------------------------------------------------------------- SIGNAL_PARSER_MODE

@pytest.mark.parametrize("mode", ["shadow", "active", "off"])
def test_valid_signal_parser_modes_are_accepted(env, mode):
    env.setenv("SIGNAL_PARSER_MODE", mode)
    assert _load().signal_parser_mode == mode


def test_signal_parser_mode_is_case_insensitive(env):
    env.setenv("SIGNAL_PARSER_MODE", "ACTIVE")
    assert _load().signal_parser_mode == "active"


def test_invalid_signal_parser_mode_raises_config_error(env):
    env.setenv("SIGNAL_PARSER_MODE", "aggressive")
    with pytest.raises(ConfigError) as exc_info:
        _load()
    assert "SIGNAL_PARSER_MODE" in str(exc_info.value)


# --------------------------------------------------------------------------- MARKET_DATA_TIMEFRAME

@pytest.mark.parametrize("name", sorted(Timeframe.__members__))
def test_every_timeframe_member_is_accepted(env, name):
    env.setenv("MARKET_DATA_TIMEFRAME", name)
    assert _load().market_data_timeframe is Timeframe[name]


def test_market_data_timeframe_is_case_insensitive(env):
    env.setenv("MARKET_DATA_TIMEFRAME", "h4")
    assert _load().market_data_timeframe is Timeframe.H4


def test_invalid_market_data_timeframe_raises_config_error(env):
    env.setenv("MARKET_DATA_TIMEFRAME", "M7")
    with pytest.raises(ConfigError) as exc_info:
        _load()
    assert "MARKET_DATA_TIMEFRAME" in str(exc_info.value)


def test_market_data_timeframe_rejects_a_provider_specific_string(env):
    # "4h" is Twelve Data's own interval spelling — provider-specific
    # strings must never be accepted here (see market_data.py's Timeframe).
    env.setenv("MARKET_DATA_TIMEFRAME", "4h")
    with pytest.raises(ConfigError):
        _load()


# --------------------------------------------------------------------------- parse_chat_ref

@pytest.mark.parametrize("raw,expected", [
    ("-1001234567890", -1001234567890),
    ("123456", 123456),
    ("  -1001234567890  ", -1001234567890),
])
def test_parse_chat_ref_accepts_numeric_ids(raw, expected):
    assert parse_chat_ref(raw) == expected


@pytest.mark.parametrize("raw", [
    "@my_signals_group",
    "my_signals_group",
    "t.me/my_signals_group",
    "https://t.me/my_signals_group",
    "telegram.me/my_signals_group",
])
def test_parse_chat_ref_accepts_username_forms(raw):
    assert parse_chat_ref(raw) == "my_signals_group"


def test_parse_chat_ref_rejects_an_empty_value():
    with pytest.raises(ConfigError):
        parse_chat_ref("   ")


@pytest.mark.parametrize("raw", [
    "https://t.me/joinchat/AAAAAEHbEkejzxUjAUCfYg",
    "https://t.me/+AAAAAEHbEkejzxUj",
])
def test_parse_chat_ref_rejects_private_invite_links_with_a_helpful_message(raw):
    with pytest.raises(ConfigError) as exc_info:
        parse_chat_ref(raw)
    assert "invite link" in str(exc_info.value).lower()


@pytest.mark.parametrize("raw", ["!!!", "a b c", "ab"])
def test_parse_chat_ref_rejects_uninterpretable_values(raw):
    with pytest.raises(ConfigError):
        parse_chat_ref(raw)


def test_load_uses_parse_chat_ref_for_the_target_chat(env):
    env.setenv("TELEGRAM_TARGET_CHAT", "@my_signals_group")
    assert _load().target_chat == "my_signals_group"


def test_load_propagates_a_chat_ref_error(env):
    env.setenv("TELEGRAM_TARGET_CHAT", "!!!")
    with pytest.raises(ConfigError):
        _load()


# --------------------------------------------------------------------------- session_path

def test_session_path_property_returns_the_session_name(env):
    env.setenv("TELEGRAM_SESSION", "my_session")
    assert _load().session_path == "my_session"
