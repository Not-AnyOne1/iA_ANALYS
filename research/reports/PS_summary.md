# Production strategy fix — summary

Branch `production-strategy-fix`, on top of `independent-analyst`, which is
kept intact with all its tests. Local commits only: nothing pushed, merged or
deployed, and the live ENTER/SKIP rule is unchanged.

## 1. Root cause of the BUY/SELL bias — `PS_root_cause.md`

- **The verdict is the H1 vote.** Production's verdict is in effect one
  question: does the H1 scoring vote agree with the signal? On the 109
  recorded signals, every vote-agreeing signal but 5 was ENTER, and every
  disagreeing one was SKIP.
- **The vote reads stale evidence.** It sums ten "latest object" factors
  (latest BOS, latest CHoCH, latest order block, the supply/demand zone
  count, and others), none checked for age or validity.
- **What happened on 10-05/06.** The selloff made its new lows by wick. No
  H1 close broke a confirmed swing low (the last bearish break was 82 bars
  old), while bounces through lower highs fired bullish CHoCH and BOS
  events. 61–62 of the 62 signals had bullish BOS, CHoCH, order-block and
  zone-count votes (48 of 106 points), 23 of them with the bullish BOS
  already invalidated. D1 was bearish throughout, but the higher timeframes
  only reach one validation check and cannot override the vote.
- **Not a code asymmetry.** A price-mirror test gave identical decisions in
  218 of 218 cases. Stale or placeholder candles were not the cause either:
  a closed-bar replay agrees on 108 of 109 verdicts.
- **The vote has no predictive value out of sample.** On 2024–2026 test
  signals, signals that agree with it average −0.047R and signals against
  it −0.050R.

## 2. Changes implemented, and why

| Change | Status |
| --- | --- |
| `strategy_variants.py`: four one-change candidates (`clean_data`, `htf_regime`, `valid_events`, `range_filter`), each aimed at one suspected cause | Evaluation only; not wired into the bot |
| `MarketContextBuilder(clean_candles=True)` (closed bars, stubs removed, padded windows) and a replay clock `now=` | Off by default; live behaviour unchanged |
| Card: "Setup score: N/100 (heuristic, not a win probability)"; "not scored (reason)" instead of 0%; "engine reads BUY/SELL" when the score was zeroed for disagreement. Text report likewise; console "Parse conf." | **Display change on the live card, needs your approval to deploy.** No verdict changes |
| `research/production_replay.py`, `diagnose_bias.py`, `strategy_backtest.py` | Research tools |

## 3–4. Before and after, and whether it holds out of sample — `PS_results.md`

Out-of-sample net R per ENTER (2024 to 2026-10, about 17,000 standardized
signals):

| Strategy | Net R per ENTER (95% interval) | Δ vs baseline (95% interval) |
| --- | --- | --- |
| baseline | −0.041 (−0.097, +0.014) | — |
| `clean_data` | −0.043 | −0.002 |
| `htf_regime` | +0.011 (−0.072, +0.097) | +0.052 (−0.002, +0.106) |
| `valid_events` | −0.035 | +0.006 (−0.016, +0.026) |
| `range_filter` | −0.032 | +0.009 (−0.008, +0.026) |

`htf_regime` comes closest, but its gain is concentrated in 2025's uptrend
and it is worse than the baseline in the 2026 holdout (−0.040R). Nothing
holds out of sample by the pre-declared rule.

## 5. Confidence scoring

- The 0% and 100% come from rules, not from a calibration: a direction
  mismatch forces 0, a blocked signal prints 0, and a bonus clamps at 100.
- The card now shows a labelled heuristic score and never a percentage.
- No probability is claimed. None is calibrated, and on the recorded signals
  the analyst's evidence count did not rank outcomes either.

## 6. Tests and deployment risks

**Tests.** 959 passed, 0 failed:
- 11 new tests for the variants, the mirror symmetry, the clean-candle mode
  and the score line;
- two existing card tests updated for the new wording: they asserted the
  old "Confidence:" and "72%" labels.

**Checks.** pyflakes clean. Secret scan of all 50 files changed against
`197b2a0`: 0 hits.

**Deployment risks.**
- Only the card wording changes for users. Anything parsing "Confidence: X%"
  from the Telegram card would break.
- Telegram session handling, the Twelve Data integration, storage and the
  bot commands are untouched.

## 7. Verdict

**INSUFFICIENT EVIDENCE.**
- No candidate is IMPROVED.
- The baseline is not profitable and shows no measurable directional skill.

Recommendation:
- Keep the live rule.
- Deploy only the honest score display, if you approve.
- Record the three undecided candidates' verdicts in shadow mode next to the
  live verdict, so the decision can be made on real signals and outcomes.
