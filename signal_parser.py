"""Deterministic, regex-first signal parsing (RFC-001).

Runs before the Claude Code CLI for every message, when enabled via
``SIGNAL_PARSER_MODE`` (see ``config.py`` and ``pipeline.py``). Only messages
this parser cannot confidently classify are handed to
:class:`~claude_client.ClaudeAnalyzer` as a fallback.

Design principle: false negatives (deferring to Claude when this parser
could plausibly have handled it) are cheap — one extra CLI call. False
positives (confidently answering wrong) are not, since nothing double-checks
a "confident" verdict once running in "active" mode. Every pattern below is
deliberately conservative: it only claims confidence for two safe buckets —

  1. A fully-formed, unambiguous signal (symbol + direction + entry + stop
     loss + at least one take-profit, all present, with no wording that
     suggests this is actually a result report or a promotional wrapper).
  2. A message with no signal-like content at all (no digits, no direction/
     entry/stop-loss/take-profit/leverage keywords).

Everything else — partial matches, trade-management updates, result
reports, promotional posts, replies (which are more likely commentary on an
earlier post than a fresh signal), or a symbol/format this parser doesn't
recognise — is treated as ambiguous and deferred to Claude.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from models import SignalAnalysis, TradeSetup

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from telegram_client import IncomingMessage


@dataclass(frozen=True)
class ParseResult:
    """The regex parser's verdict for one message.

    ``analysis`` is ``None`` whenever the message is ambiguous and must fall
    back to Claude. ``reason`` is a short, stable tag for logs and
    shadow-mode comparisons — not shown to end users.
    """

    analysis: Optional[SignalAnalysis]
    reason: str


# --- symbol normalisation ----------------------------------------------------
# Mirrors claude_client.SYSTEM_PROMPT's own normalisation rules exactly, so a
# symbol produced by either path looks identical downstream.
_SYMBOL_ALIASES = {
    "GOLD": "XAUUSD",
    "XAU": "XAUUSD",
    "SILVER": "XAGUSD",
    "OIL": "USOIL",
}

_QUOTES = r"USDT|USDC|USD|EUR|GBP|JPY|BTC|ETH"

# An ALL-CAPS run directly against a quote currency is a deliberate ticker
# (BTCUSDT, EURUSD) — safe without a base whitelist because ordinary prose
# isn't typed in all caps. No separator allowed, so "BUY USD" can't match.
_SYMBOL_CAPS_RE = re.compile(rf"\b([A-Z]{{2,5}})({_QUOTES})\b")

# A "/" or "-" between two tokens is a deliberate pair separator, rare in
# ordinary prose — safe case-insensitively.
_SYMBOL_SEPARATED_RE = re.compile(rf"\b([A-Za-z]{{2,5}})[/\-]({_QUOTES})\b", re.IGNORECASE)

# Lowercase, no separator (e.g. "long btcusdt") only matches a known base —
# unrestricted, this would also catch ordinary words like "the usd".
_KNOWN_BASES = (
    "BTC", "ETH", "XRP", "SOL", "BNB", "ADA", "DOGE", "DOT", "LTC", "TRX",
    "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "XAU", "XAG",
)
_SYMBOL_KNOWN_BASE_RE = re.compile(
    r"\b(" + "|".join(_KNOWN_BASES) + rf")\s*({_QUOTES})\b", re.IGNORECASE
)

# --- direction ----------------------------------------------------------------
_LONG_RE = re.compile(r"\b(long|buy)\b", re.IGNORECASE)
_SHORT_RE = re.compile(r"\b(short|sell)\b", re.IGNORECASE)

# --- core fields ----------------------------------------------------------------
_ENTRY_RE = re.compile(
    r"entry\w*\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*(?:-|/|~|\bto\b)\s*([0-9]+(?:\.[0-9]+)?))?",
    re.IGNORECASE,
)
_STOP_LOSS_RE = re.compile(
    r"(?:stop\s*-?\s*loss|s\.?\s*l\.?)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_TAKE_PROFIT_RE = re.compile(
    r"(?:take\s*-?\s*profit|t\.?\s*p\.?)\s*\d*\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_LEVERAGE_RE = re.compile(
    r"(?:leverage|lev)\s*[:\-]?\s*((?:cross\s*|isolated\s*)?[0-9]+\s*x)",
    re.IGNORECASE,
)

# --- order type: narrow, specific phrases only. Bare "stop" collides with
# "stop loss", which is present in nearly every complete signal. -------------
_ORDER_MARKET_RE = re.compile(r"\b(market\s*order|at\s*market)\b", re.IGNORECASE)
_ORDER_LIMIT_RE = re.compile(r"\blimit\s*order\b", re.IGNORECASE)
_ORDER_STOP_RE = re.compile(r"\b(buy\s*stop|sell\s*stop|stop\s*order|breakout)\b", re.IGNORECASE)

# --- guards: force "ambiguous" even if the core fields otherwise look clean --
# A result report ("TP1 hit, +45%") or a promotional wrapper can easily
# contain the same numbers/keywords as a fresh signal; never let those
# through the fast path.
_RESULT_MARKER_RE = re.compile(
    r"\b(hit|closed|result|pnl|profit\s*:|loss\s*:)\b|\d+\s*%", re.IGNORECASE
)
_PROMO_MARKER_RE = re.compile(
    r"\b(vip|join|subscribe|discount|promo|link\s*in\s*bio)\b|https?://", re.IGNORECASE
)

# A message with none of these has essentially no chance of being signal or
# trade-management related. This is deliberately *not* a dedicated
# trade-update parser — it doesn't classify what kind of update a message
# is, it only ensures one can never be mistaken for "no signal-related
# content at all" and confidently mislabelled as `other`. An unrecognised
# trade-management message still falls through to the ambiguous/defer path
# below, exactly like any other partial match.
_ANY_SIGNAL_MARKER_RE = re.compile(
    r"\b(long|short|buy|sell|entry|stop\s*-?\s*loss|s\.?\s*l\.?|"
    r"take\s*-?\s*profit|t\.?\s*p\.?|leverage|"
    r"close|closed|exit|cancel|trail|breakeven)\b|[0-9]",
    re.IGNORECASE,
)

_CORE_FIELDS = ("symbol", "direction", "entries", "stop_loss", "take_profits")


def _is_missing(value: object) -> bool:
    """True for None or an empty list — but not for a legitimate 0.0 price."""
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0
    return False


def _extract_symbol(text: str) -> Optional[str]:
    for alias, resolved in _SYMBOL_ALIASES.items():
        if re.search(rf"\b{alias}\b", text, re.IGNORECASE):
            return resolved
    for pattern in (_SYMBOL_CAPS_RE, _SYMBOL_SEPARATED_RE, _SYMBOL_KNOWN_BASE_RE):
        match = pattern.search(text)
        if match:
            return f"{match.group(1).upper()}{match.group(2).upper()}"
    return None


def _extract_direction(text: str) -> Optional[str]:
    has_long = _LONG_RE.search(text) is not None
    has_short = _SHORT_RE.search(text) is not None
    if has_long and not has_short:
        return "long"
    if has_short and not has_long:
        return "short"
    return None  # both or neither present -> ambiguous, not a guess


def _extract_entries(text: str) -> List[float]:
    match = _ENTRY_RE.search(text)
    if not match:
        return []
    values = [float(match.group(1))]
    if match.group(2):
        values.append(float(match.group(2)))
    return values


def _extract_stop_loss(text: str) -> Optional[float]:
    match = _STOP_LOSS_RE.search(text)
    return float(match.group(1)) if match else None


def _extract_take_profits(text: str) -> List[float]:
    return [float(m.group(1)) for m in _TAKE_PROFIT_RE.finditer(text)]


def _extract_leverage(text: str) -> Optional[str]:
    match = _LEVERAGE_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_order_type(text: str) -> str:
    if _ORDER_STOP_RE.search(text):
        return "stop"
    if _ORDER_MARKET_RE.search(text):
        return "market"
    if _ORDER_LIMIT_RE.search(text):
        return "limit"
    return "unknown"


def parse_signal(message: "IncomingMessage") -> ParseResult:
    """Classify one message without calling Claude, when safely possible."""
    text = message.text

    if message.reply_to_id is not None:
        # Mirrors claude_client.SYSTEM_PROMPT's own treatment of replies: a
        # reply is much more likely to be commentary on an earlier post than
        # a fresh, standalone signal. Never fast-path these.
        return ParseResult(None, "is_reply")

    if _RESULT_MARKER_RE.search(text):
        return ParseResult(None, "contains_result_marker")
    if _PROMO_MARKER_RE.search(text):
        return ParseResult(None, "contains_promo_marker")

    symbol = _extract_symbol(text)
    direction = _extract_direction(text)
    entries = _extract_entries(text)
    stop_loss = _extract_stop_loss(text)
    take_profits = _extract_take_profits(text)

    fields = {
        "symbol": symbol,
        "direction": direction,
        "entries": entries,
        "stop_loss": stop_loss,
        "take_profits": take_profits,
    }
    missing = [name for name in _CORE_FIELDS if _is_missing(fields[name])]

    if not missing:
        setup = TradeSetup(
            symbol=symbol,
            direction=direction,
            order_type=_extract_order_type(text),
            entries=entries,
            stop_loss=stop_loss,
            take_profits=take_profits,
            leverage=_extract_leverage(text),
            timeframe=None,
        )
        analysis = SignalAnalysis(
            is_signal=True,
            category="signal",
            setup=setup,
            summary=(
                f"{direction.title()} setup on {symbol} (regex match — "
                f"entry {entries[0]:g}, SL {stop_loss:g}, "
                f"{len(take_profits)} target(s))."
            ),
            confidence=0.9,
            missing_fields=[],
            notes=None,
            source="regex",
        )
        return ParseResult(analysis, "clean_signal_match")

    if not _ANY_SIGNAL_MARKER_RE.search(text):
        analysis = SignalAnalysis(
            is_signal=False,
            category="other",
            setup=TradeSetup(),
            summary="No trading-signal language detected (regex match).",
            confidence=0.85,
            missing_fields=[],
            notes=None,
            source="regex",
        )
        return ParseResult(analysis, "no_signal_markers")

    return ParseResult(None, f"partial_match_missing_{'_'.join(missing)}")
