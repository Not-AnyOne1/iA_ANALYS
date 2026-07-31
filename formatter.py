"""Terminal rendering for analysed messages.

Produces a compact human-readable panel followed by the structured JSON record,
with optional ANSI colour (disabled automatically when stdout is not a TTY or
when ``NO_COLOR`` is set).
"""

from __future__ import annotations

import json
import shutil
import sys
from typing import Iterable

from models import SignalAnalysis
from telegram_client import IncomingMessage

_RESET = "\033[0m"
_STYLES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
}

_CATEGORY_COLOR = {
    "signal": "green",
    "update": "cyan",
    "result": "blue",
    "commentary": "yellow",
    "promotion": "magenta",
    "other": "grey",
}


class Formatter:
    """Renders analysis results for the console."""

    def __init__(self, color: bool = True, show_json: bool = True) -> None:
        self.color = color and sys.stdout.isatty()
        self.show_json = show_json
        self.width = min(shutil.get_terminal_size((100, 24)).columns, 100)

    # ------------------------------------------------------------- utilities

    def _c(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        prefix = "".join(_STYLES.get(s, "") for s in styles)
        return f"{prefix}{text}{_RESET}"

    def _rule(self, char: str = "─") -> str:
        return self._c(char * self.width, "grey")

    @staticmethod
    def _fmt_numbers(values: Iterable[float]) -> str:
        return ", ".join(f"{v:g}" for v in values)

    def _confidence_bar(self, pct: int) -> str:
        filled = round(pct / 10)
        color = "green" if pct >= 75 else "yellow" if pct >= 45 else "red"
        bar = "█" * filled + "░" * (10 - filled)
        return f"{self._c(bar, color)} {pct}%"

    # --------------------------------------------------------------- renderer

    def render(self, message: IncomingMessage, analysis: SignalAnalysis) -> str:
        setup = analysis.setup
        category_color = _CATEGORY_COLOR.get(analysis.category, "grey")

        lines: list[str] = ["", self._rule("━")]

        badge = (
            self._c(" SIGNAL ", "bold", "green")
            if analysis.is_signal
            else self._c(f" {analysis.category.upper()} ", "bold", category_color)
        )
        header = f"{badge} {self._c(message.chat_title, 'bold')}"
        lines.append(header)
        lines.append(
            self._c(
                f"#{message.id} · {message.sender} · "
                f"{message.timestamp.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
                + (" · edited" if message.is_edit else "")
                + f" · {analysis.source}",
                "grey",
            )
        )
        if message.link:
            lines.append(self._c(message.link, "grey"))

        lines.append(self._rule())
        lines.append(self._c("message", "dim"))
        for raw_line in message.text.splitlines()[:12]:
            lines.append(f"  {self._c(raw_line, 'dim')}")
        if len(message.text.splitlines()) > 12:
            lines.append(self._c("  [...]", "dim"))

        lines.append(self._rule())

        if analysis.is_signal:
            direction = (setup.direction or "?").upper()
            direction_color = "green" if setup.direction == "long" else "red"
            lines.append(
                f"  {self._c('Symbol', 'bold'):<12} {setup.symbol or '—'}"
                f"   {self._c(direction, 'bold', direction_color)}"
                f"  {self._c(f'({setup.order_type})', 'grey')}"
            )
            lines.append(
                f"  {self._c('Entry', 'bold'):<12} "
                f"{self._fmt_numbers(setup.entries) or '—'}"
            )
            lines.append(
                f"  {self._c('Stop loss', 'bold'):<12} "
                + (self._c(f"{setup.stop_loss:g}", "red") if setup.stop_loss is not None else "—")
            )
            lines.append(
                f"  {self._c('Take profit', 'bold'):<12} "
                + (self._c(self._fmt_numbers(setup.take_profits), "green")
                   if setup.take_profits else "—")
            )
            extras = []
            if setup.leverage:
                extras.append(f"leverage {setup.leverage}")
            if setup.timeframe:
                extras.append(f"timeframe {setup.timeframe}")
            rr = setup.risk_reward
            if rr is not None:
                extras.append(f"R:R ≈ {rr:.2f}")
            if extras:
                lines.append(f"  {self._c('Extras', 'bold'):<12} {' · '.join(extras)}")

        lines.append(f"  {self._c('Summary', 'bold'):<12} {analysis.summary}")
        lines.append(f"  {self._c('Confidence', 'bold'):<12} {self._confidence_bar(analysis.confidence_pct)}")

        if analysis.missing_fields:
            lines.append(
                f"  {self._c('Missing', 'bold'):<12} "
                + self._c(", ".join(analysis.missing_fields), "yellow")
            )
        if analysis.notes:
            lines.append(f"  {self._c('Notes', 'bold'):<12} {self._c(analysis.notes, 'yellow')}")

        if self.show_json:
            lines.append(self._rule())
            payload = json.dumps(
                analysis.to_record(message=message), indent=2, ensure_ascii=False
            )
            lines.extend(self._c(line, "grey") for line in payload.splitlines())

        lines.append(self._rule("━"))
        return "\n".join(lines)

    def render_error(self, message: IncomingMessage, error: Exception) -> str:
        return "\n".join(
            [
                "",
                self._rule("━"),
                self._c(" ANALYSIS FAILED ", "bold", "red")
                + f" {message.chat_title} #{message.id}",
                self._c(f"  {type(error).__name__}: {error}", "red"),
                self._c(f"  message: {message.text[:200]}", "dim"),
                self._rule("━"),
            ]
        )
