"""The independent analysis as a compact Telegram card (HTML parse mode).

Shows only values that were computed: a level that does not exist is left
out, never filled with a placeholder. The original signal's verdict and the
independent proposal are separate sections, and the card always ends with
the shadow-mode line.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, List, Optional

from models import TradeSetup

from analyst.engine import Analysis
from analyst.strategies import INVALID, READY, WAITING, Proposal
from analyst.verdict import Reason

SHADOW_LINE = "Mode: SHADOW — NOT A LIVE TRADE INSTRUCTION"
TEST_LINE = "⚠️ TEST DATA — NOT REAL MARKET DATA"
_SIDE = {"long": "BUY", "short": "SELL", None: "NONE"}
_LIMIT = 3800                                   # Telegram allows 4096 characters


def _p(value: Optional[float]) -> str:
    return f"{value:.2f}"


def _short(text: str, limit: int = 220) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _reasons(items: Iterable[Reason]) -> str:
    return "; ".join(escape(_short(r.text)) for r in items)


def _entry(setup: TradeSetup) -> str:
    if setup.entries:
        lo, hi = min(setup.entries), max(setup.entries)
        levels = _p(lo) if lo == hi else f"{_p(lo)}–{_p(hi)}"
        return f"{levels} ({escape(setup.order_type)})"
    return "market" if setup.order_type in ("market", "unknown") else escape(setup.order_type)


def _proposal(p: Optional[Proposal]) -> List[str]:
    if p is None:
        return ["• Direction: NONE", "• Status: NO SETUP — no strategy's conditions are met"]
    lines = [f"• Direction: {_SIDE[p.direction]}"]
    if p.status in (READY, WAITING) and p.entry is not None:
        if p.order_type == "market" and p.entry_zone:
            lines.append(f"• Entry zone: {_p(p.entry_zone[0])}–{_p(p.entry_zone[1])} "
                         f"(market now {_p(p.entry)})")
        else:
            kind = {"stop": "stop entry", "limit": "limit entry"}.get(p.order_type or "", "entry")
            lines.append(f"• Entry: {_p(p.entry)} ({kind})")
        if p.stop_loss is not None:
            lines.append(f"• SL: {_p(p.stop_loss)}")
        if p.take_profit is not None:
            lines.append(f"• TP: {_p(p.take_profit)} ({escape(p.target)})")
        if p.rr_net is not None:
            lines.append(f"• RR: {p.rr_net:.2f} net of costs ({p.rr_gross:.2f} gross)")
    lines.append(f"• Setup type: {p.strategy} — {p.name}")
    if p.status == READY:
        lines.append("• Status: READY")
    elif p.status == WAITING:
        lines.append(f"• Status: WAITING — needs {escape(_short(p.confirmation))}")
    else:
        lines.append(f"• Status: INVALID — {escape(_short('; '.join(p.notes) or ', '.join(p.reasons)))}")
    if p.status != INVALID and p.invalidation:
        lines.append(f"• Invalidation: {escape(_short(p.invalidation))}")
    return lines


def render_card(a: Analysis, setup: TradeSetup, *, live_verdict: Optional[str] = None,
                test_data: bool = False) -> str:
    v = a.verdict
    lines = [f"<b>{escape(a.symbol)} — INDEPENDENT ANALYSIS</b>"]
    if test_data:
        lines.append(TEST_LINE)
    lines += ["", "<b>Original signal:</b>",
              f"• Direction: {_SIDE.get(setup.direction, 'NONE')}",
              f"• Entry: {_entry(setup)}",
              f"• SL: {_p(setup.stop_loss) if setup.stop_loss is not None else 'not given'}",
              f"• TP: {' / '.join(_p(t) for t in setup.take_profits) or 'not given'}",
              "", "<b>Market context:</b>",
              f"• H4 bias: {escape(_short(a.context.get('H4', 'unavailable')))}",
              f"• H1 bias: {escape(_short(a.context.get('H1', 'unavailable')))}",
              f"• M15 setup: {escape(a.m15_setup)}",
              f"• M5 confirmation: {escape(a.m5_confirmation)}",
              "", f"<b>Original signal decision: {v.verdict}</b>",
              f"• Main reasons: {_reasons(v.reasons)}"]
    if v.weaknesses:
        lines.append(f"• Weaknesses: {_reasons(v.weaknesses)}")
    if v.checks:
        lines.append(f"• Evidence: {v.checks_met}/{len(v.checks)} checks met "
                     f"(uncalibrated score — not a win probability)")
    if live_verdict:
        lines.append(f"• Live bot verdict (unchanged): {escape(live_verdict)}")
    lines += ["", "<b>Independent trade proposal:</b>", *_proposal(a.headline)]
    others = [p for side in a.proposals.values() for p in side.values()
              if p is not None and p is not a.headline]
    if others:
        lines.append("• Also: " + "; ".join(f"{p.strategy} {_SIDE[p.direction]} {p.status}" for p in others))
    lines += ["",
              f"Analysis timestamp: {a.as_of:%Y-%m-%d %H:%M:%S} UTC",
              f"Market data status: {escape(a.data_status)}"
              + (f" ({escape(_short('; '.join(a.view.problems), 160))})" if a.view.problems else ""),
              f"News: {escape(_short(a.news, 160))}",
              SHADOW_LINE]
    text = "\n".join(lines)
    if len(text) > _LIMIT:
        # Cut on a line boundary — every tag opens and closes on one line —
        # and keep the shadow line whatever happens.
        cut = text.rfind("\n", 0, _LIMIT - len(SHADOW_LINE) - 3)
        text = text[:cut] + "\n…\n" + SHADOW_LINE
    return text
