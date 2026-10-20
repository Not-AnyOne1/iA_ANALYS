# Backtest protocol for the independent analyst (fixed before any result)

Committed before the systematic backtest was run for the first time. The
analyst's rules are those of `analyst/rules.py` at `7aba14c`. Since then only
data-plumbing fixes have been made, none touching a threshold:

- `3af7e94` — live candle labels;
- `37bc2ae` — weekend stub padding.

## Data

- **Bars:** Twelve Data XAU/USD research bars in UTC (`data/research/`):
  - M5 from 2020-12-01;
  - M15 from 2020-12-01;
  - H1 from 2020-10-01;
  - H4 from 2020-06-01.
- **What each decision sees:** at decision time `t`, each timeframe gets the
  bars the live bot would receive — the last 300 that had opened (M5:
  padded for weekend stubs). The analyst keeps only closed, non-stub bars.

## Trades

- **Entry:** a READY proposal is a market entry at the last closed M5 close
  at `t`.
- **Outcome:** scored by `trade_accounting.account`:
  - the stop counts first when a bar touches both levels;
  - a bar that opens through the stop exits at its open;
  - a trade still open after five days exits at the last close.
- **Costs**, three bases:
  - gross;
  - net — spread 0.30 and slippage 0.10, an assumption;
  - net ×2 — spread 0.60 and slippage 0.20.
- **Positions:** one at a time per strategy. A setup is taken only after the
  previous trade of the same strategy has exited.

## Periods

All chronological. No threshold is fitted on any of them.

| Period | From | To |
| --- | --- | --- |
| development | 2021-01-01 | 2023-12-31 |
| validation | 2024-01-01 | 2024-12-31 |
| test | 2025-01-01 | 2025-12-31 |
| holdout | 2026-01-01 | 2026-10-17 |

Walk-forward is reported as the per-year results of these fixed rules: there
is nothing to refit.

## Verdict rule, per strategy

Out of sample means validation + test + holdout, net of ×1 costs.

- **PROFITABLE** — all of:
  - at least 100 trades out of sample;
  - the 95% bootstrap interval of expectancy is above 0;
  - expectancy is positive in the holdout and in at least two of the three
    out-of-sample years;
  - expectancy is still above 0 at ×2 costs.
- **NOT PROFITABLE** — at least 100 trades out of sample, and the 95%
  interval of expectancy is below 0.
- **INSUFFICIENT EVIDENCE** — anything else.

The recorded Telegram signals (109, 2026-09-28 to 2026-10-15, about two
market episodes) are reported descriptively. They cannot support a verdict
on their own.
