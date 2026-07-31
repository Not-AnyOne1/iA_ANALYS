"""Typed schemas for the structured analysis returned by Claude.

These Pydantic models are handed straight to the Claude API as a JSON schema
(via structured outputs), so the response is guaranteed to be parseable and to
match the shape below. Keep them free of constraints the API does not support
(``minimum``/``maximum``/``minLength`` etc.) — validate those in Python instead.

One exception: ``SignalAnalysis.source`` is local metadata (which code path
produced this result) and is deliberately excluded from what's sent to the
CLI — see ``claude_client._build_schema``. It's never requested from or
produced by the model itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Direction = Literal["long", "short"]
OrderType = Literal["market", "limit", "stop", "unknown"]
MessageCategory = Literal[
    "signal",       # A new, actionable trade setup.
    "update",       # Management of an existing trade (move SL, partial close...).
    "result",       # Reporting the outcome of a past trade.
    "commentary",   # Market talk, analysis, charts without an explicit setup.
    "promotion",    # Marketing, VIP upsells, referral links.
    "other",        # Greetings, admin notices, anything unrelated.
]


class TradeSetup(BaseModel):
    """The trade parameters extracted from a message, when present."""

    symbol: Optional[str] = Field(
        default=None,
        description=(
            "Trading pair or instrument, normalised to uppercase with no "
            "separators (e.g. BTCUSDT, XAUUSD, EURUSD). Null if not stated."
        ),
    )
    direction: Optional[Direction] = Field(
        default=None,
        description=(
            "Trade side. Map BUY/LONG to 'long' and SELL/SHORT to 'short'. "
            "Null if the message does not state a side."
        ),
    )
    order_type: OrderType = Field(
        default="unknown",
        description=(
            "'market' for immediate entry, 'limit' for a resting order below/above "
            "price, 'stop' for a breakout order, 'unknown' if unclear."
        ),
    )
    entries: List[float] = Field(
        default_factory=list,
        description=(
            "Entry price(s) in order given. Use both endpoints for a zone "
            "(e.g. 'entry 100-102' -> [100, 102]). Empty if no entry is stated."
        ),
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Stop-loss price. Null if the message gives no stop.",
    )
    take_profits: List[float] = Field(
        default_factory=list,
        description="Take-profit targets in the order listed (TP1, TP2, ...).",
    )
    leverage: Optional[str] = Field(
        default=None,
        description="Leverage as written, e.g. '10x', 'cross 20x'. Null if absent.",
    )
    timeframe: Optional[str] = Field(
        default=None,
        description="Chart timeframe if mentioned, e.g. '15m', '4H', 'daily'.",
    )

    @property
    def has_entry(self) -> bool:
        return bool(self.entries)

    @property
    def risk_reward(self) -> Optional[float]:
        """Reward-to-risk ratio against the first target, when computable."""
        if not self.entries or self.stop_loss is None or not self.take_profits:
            return None
        entry = sum(self.entries) / len(self.entries)
        risk = abs(entry - self.stop_loss)
        if risk == 0:
            return None
        return abs(self.take_profits[0] - entry) / risk


class SignalAnalysis(BaseModel):
    """Claude's verdict on a single Telegram message."""

    is_signal: bool = Field(
        description=(
            "True only when the message contains a concrete, actionable trade "
            "setup (at minimum an instrument and a direction). Trade management "
            "updates, results and market commentary are not signals."
        ),
    )
    category: MessageCategory = Field(
        description="Best-fitting category for the message.",
    )
    setup: TradeSetup = Field(
        description=(
            "Extracted trade parameters. Leave every field null/empty when the "
            "message is not a signal — never invent values that are not stated."
        ),
    )
    summary: str = Field(
        description=(
            "One or two plain-English sentences describing what the message says. "
            "For non-signals, say briefly what it is instead."
        ),
    )
    confidence: float = Field(
        description=(
            "Confidence between 0.0 and 1.0 that the classification and the "
            "extracted fields are correct. Lower it when the message is "
            "ambiguous, partially formatted, or relies on context you cannot see."
        ),
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description=(
            "Names of setup fields a trader would need but the message omits "
            "(e.g. ['stop_loss', 'entries']). Empty for non-signals."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description=(
            "Optional caveats: ambiguity, conflicting numbers, references to an "
            "earlier message, or anything that warrants human review."
        ),
    )
    source: Literal["regex", "claude"] = Field(
        default="claude",
        description=(
            "Which code path produced this analysis. Stamped locally by "
            "signal_parser.py or claude_client.py — never requested from or "
            "produced by the model itself (excluded from the CLI's schema)."
        ),
    )

    @property
    def confidence_pct(self) -> int:
        """Confidence clamped to 0-100 — the model is not constrained by schema."""
        return int(round(max(0.0, min(1.0, self.confidence)) * 100))

    def to_record(self, *, message: "object" = None) -> dict:
        """Flatten into a JSON-serialisable record for logging or persistence."""
        record: dict = {
            "analysed_at": datetime.now(timezone.utc).isoformat(),
            "analysis": self.model_dump(mode="json"),
        }
        rr = self.setup.risk_reward
        if rr is not None:
            record["analysis"]["setup"]["risk_reward"] = round(rr, 2)
        if message is not None and hasattr(message, "to_dict"):
            record["message"] = message.to_dict()
        return record
