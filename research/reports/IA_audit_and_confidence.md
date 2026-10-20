# Audit of the deployed strategy and the 0% / 100% confidence

Production: GitHub `main` = `197b2a0` (`bc92a5a` plus the Railway session
fix), deployed on Railway on 2026-10-19. Compared with the integration line
(`integration-p1-e1-fixes` `1e0bc3b`, `fix-railway-session` `b75bcb4`).
Every rule below was read from the source at those commits. The numbers come
from replaying production's own code (`research/audit_production.py`) over
the 109 XAUUSD signals recorded locally between 2026-09-28 and 2026-10-15,
with closed bars served as of each signal's time.

## 1. The deployed pipeline

| Step | Production `197b2a0` |
| --- | --- |
| Parsing | Claude CLI turns the Telegram text into a `TradeSetup`. The regex parser runs in `SIGNAL_PARSER_MODE=shadow` (compared and logged, never used). Parser confidence (0.85/0.9) is console-only. |
| Market data | Twelve Data: one quote plus 300 bars each of M15, H1, H4, D1 (primary H1). Validated for symbol/timeframe match, non-empty, consistent OHLC and strictly increasing timestamps; 30 s cache; retries with backoff on timeouts, 5xx and 429. **The still-forming bar is included and weekend stub quotes are not removed.** Timestamps are assumed UTC; measured on 2026-10-20, they are the account zone, about 11 h ahead (same bars, labels only). |
| Indicators | EMA 20/50/200, RSI-14, ATR-14 (Wilder), ADX-14, a volatility state. |
| Structure / SMC | 5-bar fractal swings, BOS/CHoCH on closes (no future bar). SMC objects: equal highs/lows, liquidity pools, sweeps, FVG, inverse FVG, order blocks, breaker/mitigation blocks, supply/demand, premium/discount, OTE. |
| Scoring | A directional net score over the primary timeframe; confidence = `min(100, 100 · margin_curve · participation_curve)`. |
| Risk gate | Only two rules block: R:R below 1.10 and a news blackout. The news filter has no calendar, so it abstains. Trend, volatility, confidence, stop, target and spread only warn. |
| Validation | A weighted skeptical checklist (18 checks: HTF trend 12, R:R 10, MAE 9, liquidity 8, BOS 8, news 8, ...) giving a bonus (up to +19) or a penalty. |
| Verdict | `decide_deterministically`. A risk rejection, or a missing score, review or risk result, gives SKIP. Otherwise `adjusted = clamp(directional_confidence + bonus − penalty, 0, 100)`: below 40 SKIP, 40–59 WAIT, 60 or more ENTER. |
| Claude in the decision | Writes the narrative only; it cannot change the verdict. |
| Telegram card | `🎯 Confidence: {adjusted}%`, with 0 when there is no score. |
| Outcomes | **None recorded.** The `analyses` table keeps the verdict and the confidence only. |

On the integration line only (not deployed):
- closed-candle analysis;
- weekend-stub removal and padding to full 300-bar windows;
- the D1 repair;
- the trade-geometry gate;
- confluence = `adjusted × location × extension`, with ENTER at 50 or more and no WAIT;
- outcome tracking with realistic fills and costs (P1);
- the E1 breakeven shadow experiment;
- redacted rotating logs.

## 2. Weaknesses found

1. **The confidence curve is saturated.** Engine confidence was 60–99 for
   all 109 signals, so it does not tell signals apart.
2. **The direction gate shows as 0%.** `directional_confidence` returns 0
   whenever the net score favours the other side. That happened for 40 of
   the 109 signals.
3. **A blocked signal shows as 0%.** The 8 signals rejected on R:R had no
   score, but the card printed 0% rather than "not scored".
4. **The bonus pushes signals to 100%.** The validation bonus clamped 5
   signals to exactly 100%.
5. **WAIT is unreachable.** No signal reached WAIT: it needs an adjusted
   score of 40–59, which the saturated curve rarely produces.
6. **The direction call was inverted in this sample.** On 10-05/06 the
   engine favoured long while gold fell. Net R per signal, P1 accounting:

   | Signal | Engine verdict | n | Win rate | Mean net R |
   | --- | --- | --- | --- | --- |
   | Long | ENTER | 47 | 17% | −0.55 |
   | Short | SKIP | 31 | 52% | +0.40 |

   The sample covers about two market episodes, so this shows a failure
   mode, not a measured edge.
7. **Data problems.** The forming bar and the weekend stubs reach the
   engines.
8. **Nothing is measured.** Production records no outcomes, so false
   positives and false negatives cannot be counted there.

## 3. Root cause of the 0% / 100% confidence

The binary look is **not** a boolean-to-percentage bug, a parsing default, a
formatting bug or an AI-interpretation problem. Claude never sets the
number. It comes from three deterministic mechanisms:

| Displayed | Signals | Mechanism |
| --- | --- | --- |
| 0% | 40 | `directional_confidence` forces 0 when the engine favours the opposite side |
| 0% | 8 | blocked on R:R, so no score; the card prints 0 for "none" |
| 1–39% | 4 | penalties larger than the base |
| 60–99% | 52 | the saturated curve (engine confidence 60–99 on every signal) |
| 100% | 5 | the bonus clamped at 100 |

The recorded production-era live data agree: of 64 live verdicts, 25 showed
0% and 1 showed 100%. The real defect is that the number is a heuristic
**score** displayed as a percentage, which reads as a probability. It was
never calibrated, and no outcome data existed to calibrate it against.

These are four different things:

- **Heuristic score** — what production shows: a rule-based number.
- **Model-estimated probability** — the parser's 0.85/0.9 is one, of the
  *parse*, not of the trade.
- **Calibrated probability** — does not exist anywhere in this project.
- **Decision** — ENTER / WAIT / SKIP.

**Fix, implemented in the shadow analyst:**
- it shows no percentage;
- it shows the count of conditions met ("4/6 checks met"), labelled
  *uncalibrated score — not a win probability*;
- it gives the verdict with named reason codes;
- "not scored" is a reason (`DATA_*`, `SIG_*`), never a 0.

Calibration can be evaluated once enough tracked outcomes exist (see the
backtest report). Production's card is unchanged, as instructed. Changing it
is a separate decision.
