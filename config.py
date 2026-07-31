"""Application configuration.

All secrets and tunables are read from environment variables, which are loaded
from a local ``.env`` file (never committed — see ``.env.example``).

The module exposes a single immutable :class:`Settings` object built by
:meth:`Settings.load`, which fails fast with a descriptive
:class:`ConfigError` when a required value is missing or malformed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

from market_data import Timeframe

# A Telegram chat can be addressed by numeric id or by username.
ChatRef = Union[int, str]

# Matches "https://t.me/foo", "t.me/foo", "@foo" and bare "foo".
_USERNAME_RE = re.compile(
    r"^(?:https?://)?(?:t\.me/|telegram\.me/)?@?(?P<name>[A-Za-z][A-Za-z0-9_]{3,})$"
)

# RFC-001: how the regex-first parser interacts with the Claude fallback.
_VALID_PARSER_MODES = {"shadow", "active", "off"}


class ConfigError(RuntimeError):
    """Raised when the environment is missing or contains invalid settings."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable {name!r}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = _optional(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = _optional(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean-ish value, got {raw!r}")


def parse_chat_ref(raw: str) -> ChatRef:
    """Normalise a user-supplied group identifier into something Telethon accepts.

    Accepts numeric ids (``-1001234567890``), usernames (``@mygroup``),
    and invite-style links (``https://t.me/mygroup``).
    """
    raw = raw.strip()
    if not raw:
        raise ConfigError("TELEGRAM_TARGET_CHAT must not be empty")

    # Numeric id, possibly negative for supergroups/channels.
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)

    match = _USERNAME_RE.match(raw)
    if match:
        return match.group("name")

    if "joinchat" in raw or "/+" in raw:
        raise ConfigError(
            "Private invite links are not supported as a target. Join the group "
            "first, then use its numeric id (see README: 'Finding the group id')."
        )

    raise ConfigError(
        f"Could not interpret TELEGRAM_TARGET_CHAT={raw!r} as a chat id or username"
    )


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration."""

    # --- Telegram (MTProto / user account) ---------------------------------
    api_id: int
    api_hash: str
    session_name: str
    phone: str
    target_chat: ChatRef

    # --- Claude Code CLI -----------------------------------------------------
    # No Anthropic API key involved: analysis runs through the locally
    # installed, already-authenticated `claude` CLI (subscription auth via
    # `claude auth login`), invoked as a subprocess per message.
    claude_cli_path: str
    model: str
    claude_max_turns: int

    # --- Pipeline behaviour ------------------------------------------------
    # RFC-001: "shadow" runs the regex parser alongside Claude (which still
    # analyses every message) and only logs agreement/disagreement — zero
    # behaviour change. "active" lets a confident regex verdict skip Claude
    # entirely. "off" restores the pre-RFC-001 all-Claude behaviour.
    signal_parser_mode: str
    worker_count: int
    queue_maxsize: int
    max_message_chars: int
    analyse_edits: bool
    max_retries: int
    request_timeout: float

    # --- Output ------------------------------------------------------------
    log_level: str
    log_file: Path | None
    jsonl_output: Path | None
    color: bool
    show_json: bool

    # --- Storage (RFC-003) ---------------------------------------------------
    # SQLite database of completed analyses, backing /latest, /history and
    # /stats so they survive a restart. Always set (has a default) — unlike
    # jsonl_output, storage isn't optional once RFC-003 is wired into main.py.
    storage_db_path: Path

    # --- Market Data (RFC-004, wired into the pipeline in the integration phase) ---
    # Empty = disabled (MarketDataService is constructed with provider=None
    # and reports "not configured" rather than fabricating data).
    twelve_data_api_key: str
    market_data_cache_ttl: float
    # The single timeframe pipeline.py requests candles for when enriching a
    # signal with StructureEngine/SMCEngine/ScoringEngine/DecisionEngine.
    # Deliberately NOT parsed from the free-text SignalAnalysis.setup.
    # timeframe the message itself stated (e.g. "15m", "4H") — that field
    # isn't guaranteed to match this project's Timeframe enum, and guessing
    # a mapping would risk exactly the kind of fabricated/misattributed data
    # this project avoids everywhere else. One configured timeframe applies
    # to every enriched analysis instead.
    market_data_timeframe: Timeframe

    # Populated for display purposes only; filled in by the Telegram client.
    _resolved: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def load(cls, env_file: str | os.PathLike[str] | None = ".env") -> "Settings":
        """Read ``.env`` (if present) plus the process environment."""
        if env_file is not None:
            # Real environment variables win over .env entries.
            load_dotenv(env_file, override=False)

        try:
            api_id = int(_require("TELEGRAM_API_ID"))
        except ValueError as exc:
            raise ConfigError("TELEGRAM_API_ID must be an integer") from exc

        log_file = _optional("LOG_FILE")
        jsonl_output = _optional("JSONL_OUTPUT")

        signal_parser_mode = _optional("SIGNAL_PARSER_MODE", "shadow").lower()
        if signal_parser_mode not in _VALID_PARSER_MODES:
            raise ConfigError(
                f"SIGNAL_PARSER_MODE must be one of {sorted(_VALID_PARSER_MODES)}, "
                f"got {signal_parser_mode!r}"
            )

        market_data_timeframe_raw = _optional("MARKET_DATA_TIMEFRAME", "H1").upper()
        if market_data_timeframe_raw not in Timeframe.__members__:
            raise ConfigError(
                f"MARKET_DATA_TIMEFRAME must be one of {sorted(Timeframe.__members__)}, "
                f"got {market_data_timeframe_raw!r}"
            )

        return cls(
            api_id=api_id,
            api_hash=_require("TELEGRAM_API_HASH"),
            session_name=_optional("TELEGRAM_SESSION", "signal_monitor"),
            phone=_optional("TELEGRAM_PHONE"),
            target_chat=parse_chat_ref(_require("TELEGRAM_TARGET_CHAT")),
            claude_cli_path=_optional("CLAUDE_CLI_PATH", "claude"),
            # Empty means "don't pass --model" — the CLI uses whatever the
            # user already has configured (via `/model` or their settings).
            model=_optional("CLAUDE_MODEL"),
            claude_max_turns=_int("CLAUDE_MAX_TURNS", 3, minimum=1),
            signal_parser_mode=signal_parser_mode,
            worker_count=_int("WORKER_COUNT", 2),
            queue_maxsize=_int("QUEUE_MAXSIZE", 200),
            max_message_chars=_int("MAX_MESSAGE_CHARS", 8000, minimum=100),
            analyse_edits=_bool("ANALYSE_EDITS", False),
            max_retries=_int("CLAUDE_MAX_RETRIES", 3, minimum=0),
            # CLI invocations spawn a fresh process and load hooks/CLAUDE.md/
            # MCP servers each time, so they run slower than a raw API call.
            request_timeout=float(_optional("CLAUDE_TIMEOUT_SECONDS", "180")),
            log_level=_optional("LOG_LEVEL", "INFO").upper(),
            log_file=Path(log_file) if log_file else None,
            jsonl_output=Path(jsonl_output) if jsonl_output else None,
            color=_bool("COLOR_OUTPUT", True) and not os.getenv("NO_COLOR"),
            show_json=_bool("SHOW_JSON", True),
            storage_db_path=Path(_optional("STORAGE_DB_PATH", "data/signals.db")),
            twelve_data_api_key=_optional("TWELVE_DATA_API_KEY"),
            market_data_cache_ttl=float(_optional("MARKET_DATA_CACHE_TTL", "30")),
            market_data_timeframe=Timeframe[market_data_timeframe_raw],
        )

    @property
    def session_path(self) -> str:
        """Telethon session file (without the ``.session`` suffix it appends)."""
        return self.session_name
