# Production strategy fix — test protocol (fixed before any result)

Committed before `research/strategy_backtest.py` was run for the first time.

## Baseline

Production's verdict (`trade_decision.decide_deterministically`, code as on
`main` `197b2a0`), replayed as the live bot computes it
(`research/production_replay.py`):

- 300-bar M15/H1/H4/D1 windows, with the forming bar rebuilt from M5 and the
  weekend stubs included;
- the quote at the last M5 close;
- the decision clock at the replayed time.

## Candidates

Each changes one thing and is tested separately against the baseline. They
were designed from the diagnosis of the 109 recorded signals
(`PS_root_cause.md`); no threshold was fitted to any outcome.

| Name | Change |
| --- | --- |
| `clean_data` | Closed bars only, weekend stubs removed, full 300-bar windows. Decision unchanged. |
| `htf_regime` | The direction the score must agree with comes from H4 and H1 bias (last BOS/CHoCH plus the EMA-50 side) instead of the H1 vote. No agreement means SKIP. |
| `valid_events` | The H1 vote without stale evidence: BOS, CHoCH and order-block votes count only while price is still beyond their level or zone; supply/demand counts zones from the last 100 bars only. |
| `range_filter` | Baseline, but H1 structure ranging/unknown with H1 ADX below 20 is SKIP. |

## Test signals

There are only 109 recorded signals. They cover about two episodes, and the
candidates were designed from them, so they are not the test.

- **When:** every 2 h from 2021-01-01 to 2026-10-17, while the market is
  open and more than 30 min before the weekly close.
- **What:** one BUY and one SELL each time, shaped like the recorded signals
  (medians): market entry, stop 1.0 H1 ATR, target 2.0R.
- **Scoring:** every signal is scored with `trade_accounting` (stop first on
  a same-bar touch, gap-through exits at the open, 5-day timeout), gross,
  net (spread 0.30 and slippage 0.10, assumed) and net ×2.
- **Excluded:** 2026-09-25 to 2026-10-17, the window the candidates were
  designed from.

## Periods

All chronological; nothing is refitted.

| Period | Dates |
| --- | --- |
| development | 2021–2023 |
| validation | 2024 |
| test | 2025 |
| holdout | 2026-01-01 to 2026-10-17, minus the excluded window |

Out of sample = validation + test + holdout.

## Measures, per candidate and period

- ENTER, WAIT and SKIP counts;
- for ENTER trades: count, win rate, average win and loss, expectancy
  (gross / net / net ×2), profit factor and maximum drawdown;
- a one-position-at-a-time version of the same;
- false-positive rate (losing ENTER) and false-negative rate (winning SKIP);
- ENTER minus SKIP separation;
- the share of ENTERs that are longs;
- out-of-sample results by regime: D1/H4 trend relative to the trade, H1
  ADX, session.

Intervals are 95% bootstraps that resample whole ISO weeks, because the
signals overlap in time.

## Verdict per candidate

The comparison is net ENTER expectancy against the baseline, out of sample.

- **IMPROVED** — all of:
  - the 95% week-bootstrap interval of (candidate − baseline) is above 0;
  - the difference is positive in the holdout and in at least two of the
    three out-of-sample years;
  - at least 100 out-of-sample ENTERs;
  - the difference is still positive at ×2 costs.
- **NOT IMPROVED** — the point difference is 0 or below.
- **INSUFFICIENT EVIDENCE** — anything else.

## Overall verdict

- **IMPROVED** if at least one candidate is IMPROVED.
- **NOT IMPROVED** if every candidate is NOT IMPROVED.
- **INSUFFICIENT EVIDENCE** otherwise.

Four candidates are tested, so a single pass is reported alongside that
multiplicity. Profitability is a separate question: a candidate can improve
on the baseline and still lose money. No candidate is called profitable
unless its own out-of-sample net expectancy interval is above 0.

The 109 recorded signals are reported descriptively only.
