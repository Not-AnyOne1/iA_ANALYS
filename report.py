"""Human-readable reports for a market-validated signal.

Pure formatting: a :class:`market_context.MarketContext` plus an optional
:class:`trade_decision.TradeDecision` in, text out. No I/O, no computation —
every number shown was produced by a deterministic engine upstream.

Two audiences, two renderings — deliberately different, not two views of the
same text:

- :func:`render_text` — the full desk report, for the terminal and the logs.
  Every section, every number, monospace tables.
- :func:`render_telegram_report` — a short card for the phone: instrument,
  side, verdict, confidence, levels, and a few lines of why. At most 20
  lines. It is not a trimmed copy of the desk report; a wall of tables is
  unreadable on mobile and burying the verdict in it helps nobody.

Missing values print as ``—`` rather than being omitted or defaulted, so a
gap in the data is visible in the report instead of looking like it was
never relevant.
"""

from __future__ import annotations

import html
from typing import Optional

from market_context import MarketContext
from trade_decision import (
    ENTER_MIN_SCORE as report_enter_min,
    WAIT_MIN_SCORE as report_wait_min,
    Action,
    TradeDecision,
)

_DASH = "—"
_TELEGRAM_LIMIT = 3800   # headroom under Telegram's 4096 cap



def _num(value: Optional[float], digits: int = 2) -> str:
    return _DASH if value is None else f"{value:,.{digits}f}"


def _price(value: Optional[float]) -> str:
    """A price the way a trader writes it: 4100, not 4,100.00.

    Separate from :func:`_num` because the terminal report's columns want
    fixed width and grouping; a phone card wants the shortest exact form.
    Precision follows magnitude, so 4100 stays 4100 and 1.23456 keeps its
    pip. Trailing zeros are dropped, never significant digits.
    """
    if value is None:
        return _DASH
    magnitude = abs(value)
    digits = 2 if magnitude >= 100 else 5 if magnitude >= 1 else 8
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _pct(value: Optional[float], digits: int = 1) -> str:
    return _DASH if value is None else f"{value:.{digits}f}%"


def _yesno(value: Optional[bool]) -> str:
    return _DASH if value is None else ("yes" if value else "no")


def _verdict_marker(verdict: Action) -> str:
    return {Action.ENTER: "[ENTER]", Action.WAIT: "[WAIT]", Action.SKIP: "[SKIP]"}[verdict]


def render_text(context: MarketContext, decision: Optional[TradeDecision] = None) -> str:
    """The full report."""
    ctx = context
    primary = ctx.primary
    lines: list[str] = []

    rule = "=" * 66
    lines.append(rule)
    header = f"MARKET REPORT  {ctx.symbol}"
    if decision is not None:
        header += f"  {_verdict_marker(decision.verdict)}"
    lines.append(header)
    lines.append(f"generated {ctx.generated_at:%Y-%m-%d %H:%M:%S} UTC  ·  "
                 f"primary timeframe {ctx.primary_timeframe.value}")
    lines.append(rule)

    # --- price -------------------------------------------------------------
    lines.append("PRICE")
    lines.append(f"  Current price      {_num(ctx.current_price, 5)}")
    lines.append(f"  Spread             {_num(ctx.spread, 5)}")
    if ctx.quote is not None:
        lines.append(f"  Bid / Ask          {_num(ctx.quote.bid, 5)} / {_num(ctx.quote.ask, 5)}")
        lines.append(f"  Source             {ctx.quote.provider}")

    if ctx.quality is not None and ctx.quality.entry_price is not None:
        entry = ctx.quality.entry_price
        lines.append(f"  Stated entry       {_num(entry, 5)}")
        if ctx.current_price is not None:
            gap = abs(ctx.current_price - entry)
            atr_v = primary.atr if primary else None
            in_atr = f"  ({gap / atr_v:.2f} ATR)" if atr_v else ""
            lines.append(f"  Entry distance     {_num(gap, 5)}{in_atr}")

    # --- structure ---------------------------------------------------------
    lines.append("")
    lines.append("STRUCTURE")
    if ctx.structure is not None:
        lines.append(f"  Trend              {ctx.structure.trend.value}")
        last = ctx.structure.last_event
        lines.append(f"  Last event         "
                     f"{f'{last.type.value.upper()} {last.direction.value}' if last else _DASH}")
        bos = [e for e in ctx.structure.events if e.type.value == "bos"]
        choch = [e for e in ctx.structure.events if e.type.value == "choch"]
        lines.append(f"  BOS / CHoCH        {len(bos)} / {len(choch)}")
        lines.append(f"  Swing points       {len(ctx.structure.swing_points)}")
    else:
        lines.append(f"  {_DASH} (not computed)")

    # --- SMC ---------------------------------------------------------------
    lines.append("")
    lines.append("SMART MONEY")
    if ctx.smc is not None:
        smc = ctx.smc
        sweep = (max(smc.liquidity_sweeps, key=lambda s: s.candle_index)
                 if smc.liquidity_sweeps else None)
        lines.append(f"  Liquidity sweeps   {len(smc.liquidity_sweeps)}"
                     f"{f' (last: {sweep.direction.value})' if sweep else ''}")
        lines.append(f"  Liquidity pools    {len(smc.liquidity_pools)}")
        lines.append(f"  Equal highs/lows   {len(smc.equal_highs)} / {len(smc.equal_lows)}")
        lines.append(f"  FVG / inverse      {len(smc.fair_value_gaps)} / {len(smc.inverse_fvgs)}")
        lines.append(f"  Order blocks       {len(smc.order_blocks)}")
        lines.append(f"  Supply / demand    {len(smc.supply_zones)} / {len(smc.demand_zones)}")
        zone = ("premium" if smc.premium_zone else "") + ("/discount" if smc.discount_zone else "")
        lines.append(f"  Premium/discount   {zone or _DASH}")
        lines.append(f"  OTE zone           {'present' if smc.ote_zone else _DASH}")
    else:
        lines.append(f"  {_DASH} (not computed)")

    # --- indicators per timeframe -----------------------------------------
    lines.append("")
    lines.append("INDICATORS")
    if ctx.timeframes:
        lines.append(f"  {'TF':<5} {'trend':<9} {'EMA':<9} {'RSI':>6} {'ATR':>10} "
                     f"{'ADX':>6} {'vol':<7}")
        for name in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
            tf = ctx.timeframes.get(name)
            if tf is None:
                continue
            lines.append(
                f"  {name:<5} {tf.trend:<9} {tf.ema_alignment:<9} "
                f"{_num(tf.rsi, 1):>6} {_num(tf.atr, 5):>10} "
                f"{_num(tf.adx.adx, 1):>6} {tf.volatility:<7}"
            )
    else:
        lines.append(f"  {_DASH} (no timeframe analysed)")

    # --- levels ------------------------------------------------------------
    lines.append("")
    lines.append("LEVELS")
    if ctx.levels is not None:
        lv = ctx.levels
        if lv.day is not None:
            pos = lv.day.position_of(ctx.current_price) if ctx.current_price is not None else None
            lines.append(f"  Day high / low     {_num(lv.day.high, 5)} / {_num(lv.day.low, 5)}")
            lines.append(f"  Position in range  "
                         f"{_pct(pos * 100 if pos is not None else None)}")
        for label, dist in (("day high", lv.distance_to_day_high),
                            ("day low", lv.distance_to_day_low),
                            ("resistance", lv.nearest_resistance),
                            ("support", lv.nearest_support)):
            if dist is None:
                continue
            atr_txt = f"{dist.atr_multiple:.2f} ATR" if dist.atr_multiple is not None else _DASH
            lines.append(f"  To {label:<15} {_num(dist.level, 5)}  "
                         f"({_num(dist.absolute, 5)}, {atr_txt})")
    else:
        lines.append(f"  {_DASH} (not computed)")

    # --- context -----------------------------------------------------------
    lines.append("")
    lines.append("CONTEXT")
    lines.append(f"  Session            "
                 f"{ctx.session.label if ctx.session else _DASH}"
                 f"{'  (overlap)' if ctx.session and ctx.session.is_overlap else ''}")
    lines.append(f"  News               {ctx.news.summary if ctx.news else _DASH}")
    if ctx.news is not None and ctx.news.blocked:
        lines.append("                     ** trading blocked by imminent event **")

    # --- setup quality -----------------------------------------------------
    lines.append("")
    lines.append("SETUP QUALITY")
    if ctx.quality is not None:
        q = ctx.quality
        rr = f"1:{q.risk_reward:.2f}" if q.risk_reward is not None else _DASH
        lines.append(f"  Risk / reward      {rr}")
        lines.append(f"  Stop quality       {q.stop.quality.value}"
                     f"  ({_num(q.stop.atr_multiple)} ATR)")
        for reason in q.stop.reasons:
            lines.append(f"                     - {reason}")
        lines.append(f"  Target quality     {q.target.quality.value}"
                     f"  ({_num(q.target.atr_multiple)} ATR)")
        for reason in q.target.reasons:
            lines.append(f"                     - {reason}")
    else:
        lines.append(f"  {_DASH} (no setup supplied)")

    # --- scoring + risk ----------------------------------------------------
    lines.append("")
    lines.append("SCORING & RISK")
    if ctx.scoring is not None:
        lines.append(f"  Score direction    {ctx.scoring.direction.value}")
        lines.append(f"  Confidence         {ctx.scoring.confidence}%"
                     f"  ({ctx.scoring.net_score:+.0f} of {ctx.scoring.total_possible:.0f})")
    if ctx.risk is not None:
        lines.append(f"  Risk gate          "
                     f"{'APPROVED' if ctx.risk.approved else 'REJECTED'}")
        for check in ctx.risk.checks:
            mark = {"pass": "ok  ", "reject": "FAIL",
                    "warn": "warn", "abstain": "n/a "}[check.verdict.value]
            lines.append(f"    [{mark}] {check.name:<16} {check.detail}")

    # --- validation --------------------------------------------------------
    lines.append("")
    lines.append("VALIDATION")
    if ctx.validation is not None:
        v = ctx.validation
        lines.append(f"  Review adjustment  +{v.bonus} / -{v.penalty} confidence"
                     f"   (net {v.adjustment:+d})"
                     + ("   ** FATAL PROBLEM **" if v.has_fatal else ""))
        if v.fatal_problems:
            lines.append("")
            lines.append("  Fatal problems")
            for check in v.fatal_problems:
                lines.append(f"    XX {check.name:<26} {check.detail}")
        if v.strengths:
            lines.append("")
            lines.append("  Strengths")
            for check in v.strengths:
                lines.append(f"    +  {check.name:<26} {check.detail}")
        if v.weaknesses:
            lines.append("")
            lines.append("  Weaknesses")
            for check in v.weaknesses:
                lines.append(f"    -  {check.name:<26} {check.detail}")
        if v.unknowns:
            lines.append("")
            lines.append("  Unverified (counted against the score)")
            for check in v.unknowns:
                lines.append(f"    ?  {check.name:<26} {check.detail}")
    else:
        lines.append(f"  {_DASH} (not computed)")

    # --- decision ----------------------------------------------------------
    if decision is not None:
        lines.append("")
        lines.append(rule)
        lines.append(f"DECISION  {decision.verdict.value.upper()}"
                     f"   confidence {decision.confidence}%"
                     f"   (source: {decision.source.value})")
        # --- how the number was reached ------------------------------------
        # The complete arithmetic. There is one confidence in this system and
        # this is every term that produced it — nothing hidden, no min().
        lines.append("")
        lines.append("  Confidence")
        if decision.deterministic_score is None:
            lines.append(f"      deterministic score      {_DASH} not computed")
        else:
            lines.append(f"      deterministic score      {decision.deterministic_score}"
                         f"     (market conviction, scoring engine)")
        lines.append(f"    + validation bonus         {decision.validation_bonus}"
                     f"     (review credit, max {decision.max_adjustment})")
        lines.append(f"    - validation penalty       {decision.validation_penalty}"
                     f"     (review charge, max {decision.max_adjustment})")
        if decision.adjusted_score is None:
            lines.append(f"    = adjusted score           {_DASH} not computed")
        else:
            lines.append(f"    = adjusted score           {decision.adjusted_score}"
                         f"     (clamped once to 0-100)")

        lines.append("")
        lines.append(f"  Bands   SKIP 0-{report_wait_min - 1}"
                     f"   WAIT {report_wait_min}-{report_enter_min - 1}"
                     f"   ENTER {report_enter_min}-100")
        lines.append(f"  Verdict {decision.verdict.value.upper()}"
                     f"   — {decision.decision_reason}")

        if decision.risk_approved is not None:
            state = "approved" if decision.risk_approved else "REJECTED (objective blocker)"
            lines.append(f"  Risk gate                {state}")
            for item in decision.risk_rejections:
                lines.append(f"    XX {item}")
            for item in decision.risk_warnings:
                lines.append(f"    !  {item}   (warning — does not block)")

        lines.append(rule)
        lines.append(f"  {decision.reasoning}")
        if decision.fatal_problems:
            lines.append("")
            lines.append("  Fatal problems")
            for problem in decision.fatal_problems:
                lines.append(f"    XX {problem}")
        if decision.strengths:
            lines.append("")
            lines.append("  Strengths")
            for s in decision.strengths:
                lines.append(f"    + {s}")
        if decision.risks:
            lines.append("")
            lines.append("  Weaknesses / risks")
            for r in decision.risks:
                lines.append(f"    - {r}")
        if decision.strongest_reason_against:
            lines.append("")
            lines.append("  Strongest reason NOT to take this trade")
            lines.append(f"    ! {decision.strongest_reason_against}")
        for label, value in (("Alternative scenario", decision.alternative_scenario),
                             ("Worst case", decision.worst_case),
                             ("Best case", decision.best_case)):
            if value:
                lines.append("")
                lines.append(f"  {label}")
                lines.append(f"    {value}")
        lines.append("")
        lines.append(f"  Plan: {decision.execution_plan}")

    # --- data gaps ---------------------------------------------------------
    if ctx.warnings:
        lines.append("")
        lines.append("DATA GAPS")
        for w in ctx.warnings:
            lines.append(f"  ! {w}")

    lines.append(rule)
    return "\n".join(lines)


def render_telegram(context: MarketContext, decision: Optional[TradeDecision] = None) -> str:
    """Compact HTML for the Telegram bot.

    Every dynamic value is HTML-escaped: the symbol and the model's prose
    both ultimately trace back to untrusted third-party content, and an
    unescaped ``<`` makes Telegram reject the whole message with a 400.
    """
    def esc(v: object) -> str:
        return html.escape(str(v))

    ctx = context
    primary = ctx.primary
    out: list[str] = []

    if decision is not None:
        out.append(f"<b>{esc(decision.verdict.value.upper())}</b> — "
                   f"{esc(ctx.symbol)}  ({decision.confidence}%)")
    else:
        out.append(f"<b>{esc(ctx.symbol)}</b>")
    out.append("")

    out.append(f"<b>Price:</b> {_num(ctx.current_price, 5)}"
               f"   <b>Spread:</b> {_num(ctx.spread, 5)}")
    if ctx.structure is not None:
        out.append(f"<b>Trend:</b> {esc(ctx.structure.trend.value)}")
        last = ctx.structure.last_event
        if last is not None:
            out.append(f"<b>Last event:</b> {esc(last.type.value.upper())} "
                       f"{esc(last.direction.value)}")
    if primary is not None:
        out.append(f"<b>ATR:</b> {_num(primary.atr, 5)}  "
                   f"<b>RSI:</b> {_num(primary.rsi, 1)}  "
                   f"<b>ADX:</b> {_num(primary.adx.adx, 1)}")
        out.append(f"<b>EMA:</b> {esc(primary.ema_alignment)}  "
                   f"<b>Vol:</b> {esc(primary.volatility)}")
    if ctx.session is not None:
        out.append(f"<b>Session:</b> {esc(ctx.session.label)}")
    if ctx.news is not None:
        out.append(f"<b>News:</b> {esc(ctx.news.summary)}")
    if ctx.quality is not None and ctx.quality.risk_reward is not None:
        out.append(f"<b>R:R:</b> 1:{ctx.quality.risk_reward:.2f}")
    if ctx.risk is not None:
        out.append(f"<b>Risk gate:</b> "
                   f"{'approved' if ctx.risk.approved else 'REJECTED'}")
        for check in ctx.risk.rejections:
            out.append(f"  ✗ {esc(check.name)}: {esc(check.detail)}")
        for check in ctx.risk.warnings:
            out.append(f"  ! {esc(check.name)}: {esc(check.detail)}")
    if ctx.validation is not None:
        out.append(f"<b>Review:</b> +{ctx.validation.bonus} / "
                   f"-{ctx.validation.penalty} confidence")
        for check in ctx.validation.fatal_problems:
            out.append(f"  ✗ FATAL {esc(check.name)}: {esc(check.detail)}")

    if decision is not None:
        out.append("")
        out.append(f"<b>Reason:</b> {esc(decision.reasoning)}")
        if decision.risks:
            out.append("<b>Risks:</b>")
            for r in decision.risks[:3]:
                out.append(f"  - {esc(r)}")
        if decision.strongest_reason_against:
            out.append(f"<b>Strongest case against:</b> "
                       f"{esc(decision.strongest_reason_against)}")
        if decision.worst_case:
            out.append(f"<b>Worst case:</b> {esc(decision.worst_case)}")
        out.append(f"<i>{esc(decision.execution_plan)}</i>")

    text = "\n".join(out)
    if len(text) > _TELEGRAM_LIMIT:
        text = text[:_TELEGRAM_LIMIT] + "\n[... truncated]"
    return text


# ------------------------------------------------------- full report for Telegram

def _card_line(text: str, limit: int = 110) -> str:
    """One summary bullet: single line, escaped, hard length cap."""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[:limit - 1].rstrip(" ,;:.") + "…"
    return html.escape(flat)


def _summary_bullets(context: MarketContext,
                     decision: Optional[TradeDecision]) -> list[str]:
    """2-4 short bullets saying WHY, not what every indicator did.

    Drawn from what the pipeline already produced — the strongest confluence
    first, then the decision's own reason, then the single biggest risk. No
    new analysis and nothing asked of Claude that it was not already asked.
    """
    bullets: list[str] = []
    if decision is not None:
        strengths = [s for s in decision.strengths if s.strip()]

        # On a refusal the reason IS the message — a trader needs to know
        # what stopped it, not what was good about it. Otherwise lead with
        # the confluence that carried the trade.
        if decision.verdict is Action.SKIP and decision.decision_reason:
            bullets.append(_card_line(decision.decision_reason))
            bullets.extend(_card_line(s) for s in strengths[:1])
        else:
            bullets.extend(_card_line(s) for s in strengths[:2])
            if len(bullets) < 2 and decision.decision_reason:
                bullets.append(_card_line(decision.decision_reason))

        # The biggest risk, stated once. The self-critique is the sharpest
        # version of it; fall back to the first listed risk.
        risk = decision.strongest_reason_against or (
            decision.risks[0] if decision.risks else "")
        if risk.strip():
            bullets.append("⚠ " + _card_line(risk))

    if not bullets:
        v = context.validation
        if v is not None and v.strengths:
            bullets.append(_card_line(v.strengths[0].detail))
        if v is not None and v.weaknesses:
            bullets.append("⚠ " + _card_line(v.weaknesses[0].detail))
    if not bullets:
        bullets.append(_card_line("No qualitative review was available for this signal."))
    return bullets[:4]


def render_telegram_report(
    context: MarketContext, decision: Optional[TradeDecision] = None
) -> str:
    """The final Telegram message: a short card, readable on a phone.

    Deliberately NOT the terminal report. That runs to thousands of
    characters of tables and is unreadable on mobile; it stays in the
    terminal and the logs. This is only what a trader needs before acting —
    instrument, side, verdict, confidence, levels, and a few lines of why.

    At most 20 lines by construction: 8 fixed, up to 5 take-profits, up to 4
    summary bullets.
    """
    setup = context.setup
    esc = html.escape

    direction = getattr(setup, "direction", None)
    side = {"long": "BUY", "short": "SELL"}.get(direction, "—")
    symbol = esc(context.symbol or "?")

    lines = ["━" * 22, f"📊 <b>{symbol} — {side}</b>", ""]

    if decision is not None:
        mark = {Action.ENTER: "✅", Action.WAIT: "⏸",
                Action.SKIP: "⛔"}[decision.verdict]
        lines.append(f"{mark} <b>Decision:</b> {decision.verdict.value.upper()}")
        lines.append(f"🎯 <b>Confidence:</b> {decision.confidence}%")
    else:
        lines.append("⛔ <b>Decision:</b> —")
    lines.append("")

    entry = getattr(context.quality, "entry_price", None)
    if entry is None and setup is not None and setup.entries:
        entry = setup.entries[0]
    if entry is not None:
        lines.append(f"💰 <b>Entry:</b> {_price(entry)}")
    stop = getattr(setup, "stop_loss", None)
    if stop is not None:
        lines.append(f"🛑 <b>SL:</b> {_price(stop)}")
    for i, tp in enumerate(list(getattr(setup, "take_profits", None) or [])[:5], 1):
        lines.append(f"🎯 <b>TP{i}:</b> {_price(tp)}")

    lines.append("")
    lines.append("📌 <b>Summary:</b>")
    lines.extend(f"• {b}" for b in _summary_bullets(context, decision))

    message = "\n".join(lines)
    if len(message) > _TELEGRAM_LIMIT:      # unreachable in practice; never send nothing
        message = message[:_TELEGRAM_LIMIT - 1] + "…"
    return message
