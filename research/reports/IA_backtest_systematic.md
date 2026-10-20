# Strategies B and C, systematic backtest 2021–2026

Protocol: `IA_backtest_protocol.md`, committed (`a683d69`) before the first
run. Rules: `analyst/rules.py` as of `7aba14c`; no threshold changed.

**Method:**
- every M15 close from 2021-01-01 to 2026-10-17 is a decision point;
- 300-bar windows per timeframe, as the live bot sees them (forming bar and
  weekend stubs included, both dropped by the analyst);
- a READY proposal becomes a market entry at the last M5 close;
- one position at a time per strategy;
- outcomes from `trade_accounting`: stop wins a same-bar tie, gap-through
  exits at the open, 5-day timeout;
- costs: gross; net (spread 0.30, slippage 0.10, assumed); net ×2.

Mean R per trade, 95% bootstrap interval in brackets.

## Strategy B — trend continuation

| Period | Trades | Win rate | Avg win | Avg loss | Gross | Net | Net ×2 | Profit factor (net) | Max drawdown (net) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development 2021–23 | 520 | 19% | 3.41R | −1.19R | −0.09 | **−0.33** [−0.49, −0.16] | −0.55 | 0.66 | 177R |
| validation 2024 | 239 | 20% | 2.79R | −1.16R | −0.15 | **−0.38** [−0.59, −0.17] | −0.54 | 0.59 | 94R |
| test 2025 | 263 | 22% | 2.92R | −1.08R | −0.13 | **−0.20** [−0.42, +0.04] | −0.30 | 0.77 | 100R |
| holdout 2026 | 203 | 20% | 2.81R | −1.04R | −0.22 | **−0.26** [−0.49, −0.02] | −0.33 | 0.68 | 62R |
| **out of sample** | **705** | 21% | 2.85R | −1.10R | −0.16 [−0.29, −0.02] | **−0.28** [−0.40, −0.15] | −0.39 | 0.68 | 199R |

Net expectancy by year: −0.36, −0.32, −0.31, −0.38, −0.20, −0.26 (2021–2026).
Negative every year.

By direction (all years): long −0.29, short −0.32.

## Strategy C — liquidity-sweep reversal

| Period | Trades | Win rate | Avg win | Avg loss | Gross | Net | Net ×2 | Profit factor (net) | Max drawdown (net) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development 2021–23 | 601 | 23% | 2.80R | −1.16R | −0.04 | **−0.26** [−0.39, −0.11] | −0.47 | 0.71 | 169R |
| validation 2024 | 284 | 24% | 2.38R | −1.13R | −0.14 | **−0.28** [−0.46, −0.09] | −0.43 | 0.68 | 80R |
| test 2025 | 296 | 22% | 2.66R | −1.07R | −0.18 | **−0.27** [−0.45, −0.08] | −0.35 | 0.68 | 85R |
| holdout 2026 | 281 | 29% | 2.52R | −1.04R | +0.02 | **−0.01** [−0.21, +0.20] | −0.06 | 0.98 | 26R |
| **out of sample** | **861** | 25% | 2.52R | −1.08R | −0.10 [−0.21, +0.01] | **−0.19** [−0.30, −0.08] | −0.28 | 0.77 | 178R |

Net expectancy by year: −0.21, −0.06, −0.40, −0.28, −0.27, −0.01 (2021–2026).

By direction (all years): long −0.16, short −0.27.

**Out of sample by regime, net, descriptive only.** These were not used to
select anything. Picking the best cell after the fact would be the
in-sample selection the protocol rules out.

| Regime | Bucket: trades, mean R |
| --- | --- |
| H1 bias relative to trade | with: 217, −0.27 · neutral: 166, −0.01 · against: 478, −0.21 |
| H4 bias relative to trade | with: 298, −0.15 · neutral: 176, −0.15 · against: 387, −0.24 |
| Session (UTC) | asia: 259, −0.15 · london: 171, −0.32 · overlap: 175, −0.32 · new_york: 177, −0.12 · late: 79, +0.11 |

## Mechanics, checked

**Stops are small.** Median stop $2.18 (B) and $2.88 (C). Median target
reward-to-risk 3.4 (B) and 2.8 (C). Costs therefore take about 0.12–0.16R
per trade.

**Many trades end in the first bar.** About a quarter of all trades stop
out within the first 15 minutes. That is a property of a stop placed 0.2
ATR beyond the setup's extreme, not an accounting fault. Two trades were
checked bar by bar against the raw data and booked correctly.

**Gross is negative too.** Before costs, B is below zero out of sample (the
interval excludes 0), and C is at or below zero.

## Verdict (pre-declared rule)

| Strategy | Out-of-sample trades | Net expectancy, 95% interval | Verdict |
| --- | --- | --- | --- |
| B — trend continuation | 705 | −0.28R [−0.40, −0.15] | **NOT PROFITABLE** |
| C — liquidity-sweep reversal | 861 | −0.19R [−0.30, −0.08] | **NOT PROFITABLE** |

C's holdout year is close to breakeven (−0.01R). One year does not overturn
the out-of-sample result, and the rule needs positive expectancy in two of
three years plus a positive interval.

## Limits

- **Costs are an assumption.** Real spreads around the rollover and news
  are wider, which would make results worse, not better.
- **Fills use M15 bars.** A stop and a target inside one bar count as the
  stop, which is conservative.
- **Research prices are Twelve Data mid prices.** A broker's feed differs
  slightly.
- **Only these two specifications were tested.** Variants (wider stops,
  other targets, filters) were not, and testing many of them on this data
  would itself be a selection exercise. A new variant needs fresh
  out-of-sample data or shadow results.
