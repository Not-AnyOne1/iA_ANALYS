"""Logging configuration.

Application logs go to stderr so they stay separate from the analysis output on
stdout — ``python main.py > signals.txt`` keeps a clean transcript while errors
remain visible in the terminal.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

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


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    max_bytes: int = _MAX_BYTES,
    backup_count: int = _BACKUP_COUNT,
) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(console)

    if log_file is not None:
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
        file_handler.setFormatter(
            logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        )
        root.addHandler(file_handler)

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
    # on these two loggers.
    for secret_bearing in ("httpx", "httpcore"):
        logging.getLogger(secret_bearing).setLevel(logging.WARNING)
