"""Logging configuration.

Application logs go to stderr so they stay separate from the analysis output on
stdout — ``python main.py > signals.txt`` keeps a clean transcript while errors
remain visible in the terminal — and to a rotating log file, so a bot that
stops unattended leaves a record of why (``config.Settings.log_file``:
``<database folder>/logs/bot.log`` unless ``LOG_FILE`` says otherwise).

Every handler redacts secrets before a line is written: the values of
credential-bearing environment variables, Telegram bot tokens, ``apikey=``/
``token=`` query parameters, bearer tokens and Anthropic keys — in the
message, its arguments and any traceback.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"

# Log-file rotation bounds. A plain FileHandler grows forever, which for a
# monitor meant to run unattended for weeks is a slow disk-fill: measured at
# ~174 bytes/line, a moderately busy INFO-level deployment writes on the
# order of hundreds of MB per month. These caps put a hard ceiling on it
# (max_bytes * (backup_count + 1)) — 50 MB total by default — while keeping
# enough history to investigate an incident from the previous days.
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 4

# Environment variables whose VALUES must never reach a log line.
_SECRET_ENV = re.compile(r"(TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|API_HASH|API_ID|PHONE|_B64|"
                         r"CREDENTIAL|ACCOUNT_ID|MT5_LOGIN)", re.IGNORECASE)
_REDACTED = "***"
_PATTERNS = (
    # Telegram bot token — also inside ".../bot<token>/sendMessage", where no
    # word boundary precedes the digits.
    (re.compile(r"(?<![0-9])\d{6,12}:[A-Za-z0-9_-]{30,}"), _REDACTED),
    (re.compile(r"(?i)\b((?:api[_-]?key|apikey|access_token|token|secret|password)=)[^&\s\"']+"),
     r"\1" + _REDACTED),                                                                # URL query secrets
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1" + _REDACTED),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{8,}"), _REDACTED),
)


def secret_values_from_env(environ: Optional[dict] = None) -> list:
    env = os.environ if environ is None else environ
    return [v.strip() for k, v in env.items() if _SECRET_ENV.search(k) and len(v.strip()) >= 6]


class RedactingFilter(logging.Filter):
    """Replaces secrets in the fully formatted message and traceback."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, _REDACTED)
        for rx, repl in _PATTERNS:
            text = rx.sub(repl, text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not drop the line
            message = f"{record.msg} {record.args}"
        record.msg, record.args = self.redact(message), None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self.redact(record.stack_info)
        return True


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    max_bytes: int = _MAX_BYTES,
    backup_count: int = _BACKUP_COUNT,
    secrets: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """Configure the root logger. Returns the log file actually in use, or
    ``None`` — including when ``log_file`` could not be opened: a missing or
    read-only log location is reported on the console and the process carries
    on with console logging. Only that OSError is handled here; anything else
    still raises."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    for handler in list(root.handlers):
        if getattr(handler, "_owned_by_logging_setup", False):
            handler.close()                  # release the log file (Windows locks it)
    root.handlers.clear()
    redactor = RedactingFilter(secret_values_from_env() if secrets is None else secrets)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    console.addFilter(redactor)
    console._owned_by_logging_setup = True
    root.addHandler(console)

    active: Optional[Path] = None
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            # Rotating rather than plain: bounded total disk usage (see above).
            # errors="replace" because Telegram text reaches the logs and can
            # contain lone surrogates, which would otherwise raise
            # UnicodeEncodeError from inside the logging call itself.
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            # A log file is a convenience, never a reason not to run.
            logging.getLogger(__name__).warning(
                "log_file_unavailable path=%s error=%s: %s — continuing with console logging only",
                log_file, type(exc).__name__, exc)
        else:
            file_handler.setFormatter(
                logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
            )
            file_handler.addFilter(redactor)
            file_handler._owned_by_logging_setup = True
            root.addHandler(file_handler)
            active = log_file

    # These libraries are chatty at INFO; keep them at WARNING unless debugging.
    quiet = logging.DEBUG if root.level <= logging.DEBUG else logging.WARNING
    for noisy in ("telethon", "anthropic"):
        logging.getLogger(noisy).setLevel(quiet)

    # httpx/httpcore are pinned to WARNING *unconditionally* — deliberately
    # NOT following the debug level above. httpx logs every request's full
    # URL at INFO ("HTTP Request: GET https://api.twelvedata.com/price?
    # symbol=...&apikey=<key> ..."), and market_data.py passes the Twelve
    # Data API key as a query parameter, so letting these follow LOG_LEVEL
    # would write the key into the console and any LOG_FILE the moment
    # someone debugs an unrelated problem. No application log line depends
    # on these two loggers. (The redacting filter is a second line of defence.)
    for secret_bearing in ("httpx", "httpcore"):
        logging.getLogger(secret_bearing).setLevel(logging.WARNING)
    return active


def install_exception_hooks() -> None:
    """Log anything that escapes every handler — main thread and others —
    before the process dies, so a crash is never silent."""
    crash = logging.getLogger("crash")

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        crash.critical("uncaught_exception type=%s", exc_type.__name__, exc_info=(exc_type, exc, tb))

    def _thread_hook(args):
        crash.critical("uncaught_thread_exception thread=%s type=%s",
                       getattr(args.thread, "name", None), args.exc_type.__name__,
                       exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _hook
    threading.excepthook = _thread_hook


def asyncio_exception_handler(loop, context: dict) -> None:
    """For ``loop.set_exception_handler``: an exception nobody awaited."""
    exc = context.get("exception")
    logging.getLogger("crash").error("asyncio_unhandled message=%s", context.get("message"),
                                     exc_info=(type(exc), exc, exc.__traceback__) if exc else None)
