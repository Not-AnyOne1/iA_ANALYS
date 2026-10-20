"""The strategy comparison, computed from the shadow tables. Read-only.

Unit of observation: the SIGNAL. Every strategy is judged on the same
signals and the same outcome per signal (one hypothetical trade, the
original Entry/SL/TP), so a strategy's decisions are never counted as
separate market observations. Signals re-forwarded within four hours
(``duplicate_of``) are excluded unless asked for.

* **Resolved** means a final tp / sl / timeout outcome. ``not_filled`` is
  neither a win nor a loss; an open or untracked trade is unresolved — none
  of them enter R statistics, all are counted.
* **False positive**: an ENTER whose trade lost (net R <= 0).
  **False negative**: a SKIP whose trade won (net R > 0).
* **Paired comparison**: a candidate and the baseline differ only on the
  signals where their verdicts differ, so those discordant signals are
  shown with their R, alongside the difference in R per ENTER.
* **Uncertainty**: 95% bootstrap intervals that resample whole UTC days,
  because signals cluster in market episodes.
* **Sample size**: fewer than 30 resolved ENTERs is "insufficient — do not
  draw conclusions"; fewer than 100 is "preliminary".
* **Cohorts**: each signal's cohort was fixed when it was recorded
  (``STRATEGY_SHADOW_CONFIRMATION_START``). Only the exploration cohort's
  results are shown unless ``include_confirmation`` is set, so a
  confirmation set stays untouched until it is deliberately opened once.

These are prospective (forward-collected) results. The historical replay
(``research/reports/PS_results.md``) used standardized signals and is a
different sample; it is quoted for context only.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from strategy_shadow.versions import STRATEGIES

RESOLVED = ("tp", "sl", "timeout")
MIN_SAMPLE, ADEQUATE = 30, 100
HISTORICAL = "historical replay 2024-26, standardized signals (different sample): net R per ENTER — " \
             "baseline -0.041, htf_regime +0.011, valid_events -0.035, range_filter -0.032"


def warning_for(n: int) -> str:
    if n < MIN_SAMPLE:
        return f"INSUFFICIENT SAMPLE ({n} resolved ENTERs < {MIN_SAMPLE}): descriptive only, no conclusion"
    if n < ADEQUATE:
        return f"PRELIMINARY ({n} resolved ENTERs < {ADEQUATE})"
    return f"{n} resolved ENTERs"


def summary(rs: Sequence[float]) -> Dict[str, Any]:
    n = len(rs)
    if not n:
        return {"n": 0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    out = {"n": n, "win_rate": round(len(wins) / n, 4),
           "avg_win_r": round(sum(wins) / len(wins), 3) if wins else None,
           "avg_loss_r": round(sum(losses) / len(losses), 3) if losses else None,
           "expectancy_r": round(sum(rs) / n, 4),
           "profit_factor": round(sum(wins) / -sum(losses), 3) if losses and sum(losses) < 0 else None,
           "total_r": round(sum(rs), 3)}
    if n >= 20:                                  # a drawdown of a handful of trades means nothing
        eq = peak = dd = 0.0
        for r in rs:
            eq += r
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        out["max_drawdown_r"] = round(dd, 3)
    return out


class DayBootstrap:
    def __init__(self, days: Sequence[str], n: int = 2000, seed: int = 5) -> None:
        self.days = sorted(set(days))
        self.index = {d: i for i, d in enumerate(self.days)}
        rng = random.Random(seed)
        self.draws = [[rng.randrange(len(self.days)) for _ in self.days] for _ in range(n)] if self.days else []

    def _sums(self, items, value):
        s, c = [0.0] * len(self.days), [0] * len(self.days)
        for day, item in items:
            v = value(item)
            if v is not None:
                s[self.index[day]] += v
                c[self.index[day]] += 1
        return s, c

    def mean(self, items, value) -> Optional[Tuple[float, float]]:
        s, c = self._sums(items, value)
        means = sorted(sum(s[i] for i in d) / n for d in self.draws if (n := sum(c[i] for i in d)))
        if len(means) < 100:
            return None
        return round(means[int(0.025 * len(means))], 4), round(means[int(0.975 * len(means)) - 1], 4)

    def diff(self, items, a, b) -> Optional[Tuple[float, float]]:
        sa, ca = self._sums(items, a)
        sb, cb = self._sums(items, b)
        out = []
        for d in self.draws:
            na, nb = sum(ca[i] for i in d), sum(cb[i] for i in d)
            if na and nb:
                out.append(sum(sa[i] for i in d) / na - sum(sb[i] for i in d) / nb)
        if len(out) < 100:
            return None
        out.sort()
        return round(out[int(0.025 * len(out))], 4), round(out[int(0.975 * len(out)) - 1], 4)


def _join(data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    decisions: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for d in data["decisions"]:
        decisions.setdefault(d["signal_id"], {})[d["strategy"]] = d
    outcomes: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for o in data["outcomes"]:
        outcomes.setdefault(o["signal_id"], {})[o["basis"]] = o
    rows = []
    for s in data["signals"]:
        rows.append({**s, "regime": json.loads(s["regime_json"]),
                     "decisions": decisions.get(s["id"], {}), "outcomes": outcomes.get(s["id"], {})})
    return rows


def _r(row, basis: str = "net") -> Optional[float]:
    o = row["outcomes"].get(basis)
    return o["r"] if o and o["status"] in RESOLVED else None


def _verdict(row, strategy: str) -> Optional[str]:
    d = row["decisions"].get(strategy)
    return d["verdict"] if d else None


def _relative(label: Optional[str], direction: Optional[str]) -> str:
    if label in ("bullish", "long"):
        return "with" if direction == "long" else "against"
    if label in ("bearish", "short"):
        return "with" if direction == "short" else "against"
    return label or "unknown"


def build(data: Dict[str, List[Dict[str, Any]]], *, include_confirmation: bool = False,
          include_duplicates: bool = False) -> Dict[str, Any]:
    rows = _join(data)
    cohorts = {c: sum(1 for r in rows if r["cohort"] == c) for c in ("exploration", "confirmation")}
    shown = [r for r in rows if include_confirmation or r["cohort"] == "exploration"]
    duplicates = sum(1 for r in shown if r["duplicate_of"] is not None)
    if not include_duplicates:
        shown = [r for r in shown if r["duplicate_of"] is None]
    tracked = [r for r in shown if r["trade_json"]]
    statuses: Dict[str, int] = {}
    for r in tracked:
        st = r["outcomes"].get("net", {}).get("status", "unresolved")
        statuses[st] = statuses.get(st, 0) + 1
    out: Dict[str, Any] = {
        "kind": "prospective shadow results (forward-collected Telegram signals)",
        "historical_context": HISTORICAL,
        "cohorts_recorded": cohorts,
        "cohorts_shown": ["exploration", "confirmation"] if include_confirmation else ["exploration"],
        "signals": len(shown), "duplicates_excluded": 0 if include_duplicates else duplicates,
        "tracked": len(tracked), "untracked": len(shown) - len(tracked),
        "outcome_status": statuses,
        "resolved": sum(1 for r in tracked if _r(r) is not None),
        "unresolved_rate": round(statuses.get("unresolved", 0) / len(tracked), 4) if tracked else None,
        "data_not_fresh_rate": round(sum(1 for r in shown if r["data_status"] != "fresh") / len(shown), 4) if shown else None,
        "data_status": {k: sum(1 for r in shown if r["data_status"] == k)
                        for k in sorted({r["data_status"] for r in shown})},
        "baseline_live_mismatches": sum(1 for r in shown if r["baseline_matches_live"] == 0),
        "strategies": {}, "paired_vs_baseline": {}, "agreement": {},
    }
    boot = DayBootstrap([r["decided_at"][:10] for r in shown])
    items = [(r["decided_at"][:10], r) for r in shown]
    for strategy in STRATEGIES:
        verdicts = {k: sum(1 for r in shown if _verdict(r, strategy) == k) for k in ("ENTER", "WAIT", "SKIP")}
        verdicts["error"] = sum(1 for r in shown if r["decisions"].get(strategy, {}).get("error"))
        enter = [r for r in tracked if _verdict(r, strategy) == "ENTER"]
        resolved = [r for r in enter if _r(r) is not None]
        skips = [r for r in tracked if _verdict(r, strategy) == "SKIP" and _r(r) is not None]
        waits = [r for r in tracked if _verdict(r, strategy) == "WAIT" and _r(r) is not None]
        entry = {
            "verdicts": verdicts,
            "enter_trades": {"tracked": len(enter), "resolved": len(resolved),
                             "not_filled": sum(1 for r in enter if r["outcomes"].get("net", {}).get("status") == "not_filled"),
                             "unresolved": sum(1 for r in enter if not r["outcomes"].get("net"))},
            "enter_net": summary([_r(r) for r in resolved]),
            "enter_gross": summary([_r(r, "gross") for r in resolved if _r(r, "gross") is not None]),
            "enter_net_ci95": boot.mean(items, lambda r, s=strategy: _r(r) if _verdict(r, s) == "ENTER" else None),
            "false_positives": sum(1 for r in resolved if _r(r) <= 0),
            "false_negatives": sum(1 for r in skips if _r(r) > 0),
            "skip_net": summary([_r(r) for r in skips]),
            "wait_net": summary([_r(r) for r in waits]),
            "by_direction_net": {d: summary([_r(r) for r in resolved if r["direction"] == d]) for d in ("long", "short")},
            "warning": warning_for(len(resolved)),
        }
        for key, label in (("h4", "h4_trend"), ("htf_regime", "htf_regime"), ("session", "session")):
            buckets: Dict[str, List[float]] = {}
            for r in resolved:
                v = r["regime"].get(key)
                bucket = v if key == "session" else _relative(v, r["direction"])
                buckets.setdefault(bucket, []).append(_r(r))
            entry[f"by_{label}_net"] = {k: summary(v) for k, v in sorted(buckets.items())}
        out["strategies"][strategy] = entry
    for strategy in STRATEGIES[1:]:
        both = [r for r in tracked if _r(r) is not None]
        groups = {"both_enter": [], "candidate_only": [], "baseline_only": [], "neither": []}
        for r in both:
            c, b = _verdict(r, strategy) == "ENTER", _verdict(r, "baseline") == "ENTER"
            groups["both_enter" if c and b else "candidate_only" if c else "baseline_only" if b else "neither"].append(_r(r))
        cand = [x for k in ("both_enter", "candidate_only") for x in groups[k]]
        base = [x for k in ("both_enter", "baseline_only") for x in groups[k]]
        out["paired_vs_baseline"][strategy] = {
            "resolved_signals": len(both),
            "discordant": {k: summary(groups[k]) for k in ("candidate_only", "baseline_only")},
            "concordant_enter": len(groups["both_enter"]), "concordant_not_enter": len(groups["neither"]),
            "delta_r_per_enter": (round(sum(cand) / len(cand) - sum(base) / len(base), 4)
                                  if cand and base else None),
            "delta_total_r": round(sum(cand) - sum(base), 3),
            "delta_ci95": boot.diff(items,
                                    lambda r, s=strategy: _r(r) if _verdict(r, s) == "ENTER" else None,
                                    lambda r: _r(r) if _verdict(r, "baseline") == "ENTER" else None),
            "warning": warning_for(min(len(cand), len(base))),
        }
    for i, a in enumerate(STRATEGIES):
        for b in STRATEGIES[i + 1:]:
            pairs = [(_verdict(r, a), _verdict(r, b)) for r in shown]
            pairs = [p for p in pairs if None not in p]
            out["agreement"][f"{a}~{b}"] = round(sum(1 for x, y in pairs if x == y) / len(pairs), 4) if pairs else None
    return out


def render(report: Dict[str, Any], *, compact: bool = False) -> str:
    """Plain text; ``compact`` for the Telegram command."""
    lines = ["STRATEGY SHADOW — EXPERIMENTAL, NOT A TRADE INSTRUCTION",
             f"{report['kind']}; cohort shown: {', '.join(report['cohorts_shown'])} "
             f"(recorded: {report['cohorts_recorded']})",
             f"signals {report['signals']} (duplicates excluded {report['duplicates_excluded']}) · "
             f"tracked {report['tracked']} · resolved {report['resolved']} · "
             f"outcomes {report['outcome_status']}",
             f"data not fresh: {report['data_not_fresh_rate']} {report['data_status']} · "
             f"baseline≠live: {report['baseline_live_mismatches']}"]
    for strategy, e in report["strategies"].items():
        n = e["enter_net"]
        lines.append(
            f"{strategy}: ENTER {e['verdicts']['ENTER']} WAIT {e['verdicts']['WAIT']} SKIP {e['verdicts']['SKIP']}"
            f" · resolved ENTER {n['n']}"
            + (f" · net {n['expectancy_r']:+.3f}R/ENTER (95% {e['enter_net_ci95']}) win {n['win_rate']:.0%}"
               f" PF {n.get('profit_factor')} · FP {e['false_positives']} FN {e['false_negatives']}"
               if n["n"] else "")
            + f" · {e['warning']}")
    for strategy, p in report["paired_vs_baseline"].items():
        d = p["discordant"]
        lines.append(f"{strategy} vs baseline: Δ {p['delta_r_per_enter']}R/ENTER (95% {p['delta_ci95']}) · "
                     f"only-candidate {d['candidate_only'].get('n', 0)} @ {d['candidate_only'].get('expectancy_r')} · "
                     f"only-baseline {d['baseline_only'].get('n', 0)} @ {d['baseline_only'].get('expectancy_r')}")
    if not compact:
        lines.append(f"agreement: {report['agreement']}")
        for strategy, e in report["strategies"].items():
            lines.append(f"{strategy} by direction: { {k: (v.get('n'), v.get('expectancy_r')) for k, v in e['by_direction_net'].items()} }")
            lines.append(f"{strategy} by H4 trend: { {k: (v.get('n'), v.get('expectancy_r')) for k, v in e['by_h4_trend_net'].items()} }")
            lines.append(f"{strategy} by session: { {k: (v.get('n'), v.get('expectancy_r')) for k, v in e['by_session_net'].items()} }")
    lines.append(f"context only — {report['historical_context']}")
    return "\n".join(lines)


def render_latest(data: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """The most recent signal's comparison, HTML for the bot. None if empty."""
    from html import escape
    rows = _join(data)
    if not rows:
        return None
    r = rows[-1]
    side = {"long": "BUY", "short": "SELL"}.get(r["direction"], "—")
    entries = json.loads(r["entries_json"])
    tps = json.loads(r["take_profits_json"])
    quality = json.loads(r["data_quality_json"])
    age = quality.get("frames", {}).get("M15", {}).get("last_closed_age_min")
    lines = ["<b>STRATEGY SHADOW — LATEST</b>",
             "<i>Experimental comparison, not a trade instruction.</i>",
             f"#{r['message_id']} · {side} {escape(r['symbol'])} · signal {r['signal_at'][:16]} UTC"
             + (" · duplicate" if r["duplicate_of"] else ""),
             f"Entry {', '.join(f'{x:g}' for x in entries) or 'market'} ({escape(r['order_type'] or '?')})"
             f" · SL {r['stop_loss'] if r['stop_loss'] is not None else '—'}"
             f" · TP {' / '.join(f'{x:g}' for x in tps) or '—'}",
             f"Data: {escape(r['data_status'])}" + (f" · M15 last close {age:g} min old" if age is not None else ""),
             f"Live verdict: {escape(r['live_verdict'] or '—')}"
             + (f" · setup score {r['live_setup_score']}/100 (heuristic, not a win probability)"
                if r["live_setup_score"] is not None else " · not scored")]
    for strategy in STRATEGIES:
        d = r["decisions"].get(strategy)
        if d is None:
            continue
        note = " (replayed — matches live)" if strategy == "baseline" and r["baseline_matches_live"] == 1 else \
            " (replayed — DIFFERS from live)" if strategy == "baseline" and r["baseline_matches_live"] == 0 else ""
        reason = (d["reason"] or d["error"] or "")[:110]
        lines.append(f"{escape(strategy)}: <b>{escape(d['verdict'] or 'error')}</b>{note} — {escape(reason)}")
    net = r["outcomes"].get("net")
    if not r["trade_json"]:
        lines.append(f"Outcome: not tracked — {escape(r['untracked_reason'] or '')}")
    elif net is None:
        lines.append("Outcome: pending")
    else:
        lines.append(f"Outcome (net): {escape(net['status'])}" + (f" {net['r']:+.2f}R" if net["r"] is not None else ""))
    return "\n".join(lines)
