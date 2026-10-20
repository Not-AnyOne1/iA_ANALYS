# Independent XAUUSD analyst — summary of the work

Branch `independent-analyst`, based on production `main` `197b2a0`. Local
commits only: nothing pushed, merged or deployed. Shadow mode is off by
default.

## 1. Strategy audit — `IA_audit_and_confidence.md`

Production verdict rules, from the code at `197b2a0`:

1. A risk rejection gives SKIP. Only two rules can reject: R:R below 1.10,
   and news (which abstains, because there is no calendar).
2. Otherwise `adjusted = clamp(directional_confidence + validation bonus −
   penalty)`.
3. Below 40 is SKIP, 40–59 is WAIT, 60 or more is ENTER.
4. Claude writes the narrative only.

Weaknesses found:
- the scoring curve is saturated: engine confidence was 60–99 on all 109
  recorded signals;
- WAIT is unreachable in practice;
- forming bars and weekend stubs reach the engines;
- the candle labels are about 11 h off UTC (newly measured; see 3);
- nothing is measured: production records no outcomes.

On the recorded signals, production's own code replayed (Baseline A) gave:

| Verdict | Signals | Mean net R | False rate |
| --- | --- | --- | --- |
| ENTER | 54 | −0.50 | 81% of ENTERs lost (false positives) |
| SKIP | 46 | +0.37 | 54% of SKIPs won (false negatives) |

The verdicts were inverted in this sample, but it covers only two market
episodes.

## 2. Why 0% / 100% — same file

The binary look is deterministic, not a bug in formatting, parsing or AI
interpretation:

| Shown | Signals | Cause |
| --- | --- | --- |
| 0% | 40 | `directional_confidence` forces 0 when the engine favours the other side |
| 0% | 8 | blocked signals have no score, and the card prints 0 for "none" |
| 100% | 5 | the bonus clamps at 100 |
| 60–99% | the rest | the saturated curve |

The core defect: a heuristic score is presented as a percentage, which reads
as a probability. There is no calibrated probability anywhere.

**Fix in the analyst:**
- no percentage;
- an evidence count instead ("4/6 checks met — uncalibrated score, not a win
  probability");
- reason codes for every verdict;
- "not scored" shown as a reason, never as 0.

On the recorded signals, the count did **not** rank outcomes, so it stays
labelled uncalibrated. Production's card is unchanged, as instructed.

## 3. Engine and rules — `analyst/`, README "Independent analyst (shadow)"

**Timeframe roles.** Closed bars only, no future bar:
- **H4 context and H1 bias:** the direction of the last BOS/CHoCH, provided
  the close is on that side of EMA-50; otherwise neutral.
- **M15 setup:** a 38.2–100% pullback in the last impulse leg (strategy B),
  or the first sweep of resting liquidity that closes back (strategy C).
- **M5 entry:** a break in the trade's direction after the setup's extreme,
  within the last hour.

**Stops and targets:**
- the stop sits 0.2 M15 ATR beyond the invalidation extreme, inside a band
  of 0.5–4 ATR;
- the target is the nearest resting liquidity; B targets beyond its impulse
  swing, or the 1.272 extension;
- R:R is net of costs and must be at least 1.5;
- a target is never moved to improve the ratio.

**Verdict on the original signal:** ENTER, WAIT, SKIP or NO TRADE, from an
ordered rule set with reason codes. It checks:
- the signal itself: incomplete, stale, duplicate;
- the data: missing, stale, misaligned or market closed gives NO TRADE;
- the levels at the live price;
- timing, plus news when a real calendar exists (none is fabricated);
- structure.

**Data safety:**
- weekend stubs are dropped and M5 is padded so 300 real bars survive;
- the grid is checked;
- freshness is measured in trading time;
- closes must agree across timeframes;
- the live price must be within 0.5% of the last M5 close.

**The label problem.** The live provider labels candles about 11 h ahead of
UTC: measured, same bars, labels only. Production's verdict does not depend
on the labels. The only label-dependent value is the "day range" in Claude's
narrative facts, which is grouped by the account-zone day. The analyst
measures the shift by exact close-matching and undoes it; an unprovable
shift gives NO TRADE.

**Shadow mode** (`INDEPENDENT_ANALYSIS_MODE=shadow`):
- runs after the live report, as a separate task;
- reuses the live context's candles and adds one M5 request;
- writes write-once SQLite rows: triggers refuse UPDATE and DELETE;
- tracks the signal's outcome whatever the verdict, and every READY
  proposal's outcome, with the same accounting as P1;
- sends nothing; `/shadow` answers only the configured chat.

## 4. Telegram example (TEST DATA) — synthetic scenario `b_long`

```
XAUUSD — INDEPENDENT ANALYSIS
⚠️ TEST DATA — NOT REAL MARKET DATA

Original signal:
• Direction: BUY
• Entry: market
• SL: 2353.00
• TP: 2372.00 / 2380.00

Market context:
• H4 bias: bullish (last BOS up through 2345.05 at 03-19 04:00, close above EMA50; swings ranging)
• H1 bias: bullish (last BOS up through 2360.05 at 03-19 14:00, close above EMA50; swings bullish)
• M15 setup: pullback 48% of 2344.95 -> 2365.05
• M5 confirmation: CHOCH up through 2359.55 at 16:35

Original signal decision: ENTER
• Main reasons: B (trend continuation): H4 bias bullish; H1 bias bullish; M15 pullback 48% of the leg 2344.95 -> 2365.05, extreme 2355.45; M5 CHOCH through 2359.55 at 16:35
• Weaknesses: stop 2353.00 sits above the M15 swing low 2344.95 — a sweep of it would stop the trade first; target 2372.00 lies beyond the resting M15 swing high 2365.05
• Evidence: 5/6 checks met (uncalibrated score — not a win probability)
• Live bot verdict (unchanged): SKIP

Independent trade proposal:
• Direction: BUY
• Entry zone: 2359.55–2360.00 (market now 2360.00)
• SL: 2355.07
• TP: 2370.52 (M15 leg 1.272 extension 2370.52)
• RR: 1.95 net of costs (2.13 gross)
• Setup type: B — trend continuation
• Status: READY
• Invalidation: an M15 close below 2355.45 (pullback extreme) or H1 bias turning

Analysis timestamp: 2026-03-19 16:40:00 UTC
Market data status: fresh
News: not checked — no economic calendar configured
Mode: SHADOW — NOT A LIVE TRADE INSTRUCTION
```

## 5. Backtest comparison

Full detail: `IA_backtest_signals.md`, `IA_backtest_systematic.md`; protocol
in `IA_backtest_protocol.md`.

| Strategy | Sample | Trades | Net expectancy (95% interval) | Verdict |
| --- | --- | --- | --- | --- |
| A — production baseline | 109 recorded signals, 2.5 weeks | 54 ENTER | −0.50R (−0.77, −0.21) on its ENTERs | INSUFFICIENT EVIDENCE: one clustered sample |
| Analyst as a signal filter | same 109 signals | 0 ENTER | its SKIP +0.04R, WAIT 0.00R, NO TRADE −0.45R | INSUFFICIENT EVIDENCE |
| B — trend continuation | 2024–2026 out of sample | 705 | −0.28R (−0.40, −0.15); gross −0.16R | NOT PROFITABLE |
| C — liquidity-sweep reversal | 2024–2026 out of sample | 861 | −0.19R (−0.30, −0.08); gross −0.10R | NOT PROFITABLE |

- B was negative every year from 2021 to 2026.
- C was negative in 2021–2025, and close to breakeven in the 2026 holdout
  (−0.01R).
- Doubling costs makes every figure worse.
- No threshold was tuned. Development, validation, test and holdout were
  each run once.

## 6. Test report

The full suite: **948 passed, 0 failed.** Of these:
- 844 existing tests, all passing (no regression);
- 19 copied `trade_accounting` tests;
- 85 new analyst tests, covering:
  - multi-timeframe alignment and the label offset;
  - missing, stale, off-grid and weekend data;
  - structure without lookahead;
  - sweeps and pullbacks;
  - B and C in both directions;
  - stops, targets and net R:R;
  - every verdict branch, including WAIT and NO TRADE;
  - card format, with no percentage and no placeholders;
  - write-once storage, restart persistence beside the live tables, and
    outcome resolution;
  - shadow isolation: the live verdict, storage and the single Telegram
    message are identical with shadow on or off, and a failing analyst
    cannot break the pipeline;
  - config defaults;
  - `/shadow` access.

Checks run: pyflakes clean on all new and changed files. Secret scan of all
34 files changed against `197b2a0`: 0 hits, no forbidden files.

Twelve Data use during this work: about 100 requests, throttled. That is the
M5 history at 1 request per 30 s, plus a few probes.

## 7. Deployment plan and rollback — `IA_deployment_plan.md`

Nothing is deployed. If approved:
1. PR into `main`.
2. Set `INDEPENDENT_ANALYSIS_MODE=shadow` on Railway.
3. Redeploy.

Rollback: set the variable to `off`, or revert the merge. The shadow tables
stay inert and live tables are untouched.

## 8. Profitability verdict

- **Strategies B and C: NOT PROFITABLE**, by the pre-declared rule. Each has
  more than 700 out-of-sample trades, and the 95% interval of net
  expectancy is entirely below zero.
- **Baseline A and the analyst's filtering of real signals: INSUFFICIENT
  EVIDENCE.** The recorded-signal sample is too short and clustered. On it,
  the baseline's ENTERs lost, and the analyst avoided them only by entering
  nothing.

**Recommendation:**
- Do not use B or C proposals for any live recommendation.
- Shadow mode is still useful as a measurement instrument. It records the
  baseline's live verdicts and the analyst's verdicts on real signals, with
  outcomes, so the filter question can be answered with real data over the
  coming months.
- Enable it only if that measurement is wanted.
