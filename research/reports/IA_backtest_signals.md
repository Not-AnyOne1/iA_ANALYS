# Recorded Telegram signals: baseline versus the independent analyst

**Sample:**
- 109 XAUUSD signals recorded locally, 2026-09-28 to 2026-10-15;
- 95 distinct after removing re-forwards (same direction, stop and TP1
  within 4 h);
- 100 have a resolved outcome.

**Outcome scoring.** Every signal is scored the same way whatever its
verdict (P1 accounting):
- stated order type;
- market orders fill at the live price;
- the stop wins a same-bar tie;
- a bar that gaps through the stop exits at its open;
- 5-day timeout;
- net of 0.30 spread and 0.10 slippage (assumed).

All figures are net R per signal; 95% intervals are bootstrap.

**Sources compared:**
- **Live (recorded)** — the verdict the bot gave at the time. That is the
  engine of the day: 103 verdicts predate outcome tracking, mostly earlier
  than `197b2a0`, and 6 come from the integration engine.
- **Baseline A (production replay)** — `197b2a0`'s own code replayed over the
  same bars (`research/audit_production.py`).
- **Analyst** — the shadow analyst (`analyst/`, rules as of `7aba14c`).

## All 109 signals

Taking every signal: 100 trades, win rate 35%, expectancy **−0.10R**
(95% interval −0.35 to +0.16), profit factor 0.84.

| Source | ENTER | WAIT | SKIP | NO TRADE | False-positive rate (losing ENTER) | False-negative rate (winning SKIP) |
| --- | --- | --- | --- | --- | --- | --- |
| Live (recorded) | 32 | 13 | 64 | — | 76% | 33% |
| Baseline A | 57 | 0 | 52 | — | 81% | 54% |
| Analyst | 0 | 25 | 57 | 27 | n/a (no ENTER) | 44% |

Mean net R of the signals in each bucket:

| Source | ENTER | WAIT | SKIP | NO TRADE |
| --- | --- | --- | --- | --- |
| Live (recorded) | −0.45 (n 29; interval −0.79 to −0.08) | +0.92 (n 13; +0.13 to +1.64) | −0.16 (n 58) | — |
| Baseline A | −0.50 (n 54; interval −0.77 to −0.21) | — | +0.37 (n 46; −0.03 to +0.75) | — |
| Analyst | — | 0.00 (n 25) | +0.04 (n 48) | −0.45 (n 27; −0.84 to −0.02) |

Distinct signals only (95): Baseline A ENTER −0.44R (n 44), SKIP +0.42R
(n 42); the analyst's SKIP +0.31R (n 34), WAIT 0.00R (n 25), NO TRADE −0.45R
(n 27).

## What this shows

- **Baseline A's verdicts were inverted on this sample.** Its ENTERs lost
  about half a risk unit each, while its SKIPs would have won. That matches
  the audit: the engine favoured long through the 10-05/06 decline.
- **The analyst entered nothing.** Its reasons, in order of frequency:

  | Reason | Signals |
  | --- | --- |
  | entry not reached (limit entries away from price) | 35 |
  | no structure either way | 27 |
  | waiting for confirmation | 22 |
  | against H1 structure | 16 |
  | duplicate | 14 |
  | R:R below minimum | 10 |
  | stale | 6 |
  | stop too wide | 5 |
  | already stopped out | 3 |

  It avoided Baseline A's losing ENTERs only by entering none. That is not
  evidence of skill.
- **The analyst's NO TRADE bucket lost** (−0.45R). Those were signals with
  no supporting structure.
- **The analyst's SKIP bucket broke even** (+0.04R), so its SKIPs were not
  filtering out losers either.
- **Its own proposals did no better.** Only 3 READY proposals arose at signal
  times, and all 3 lost.
- **The evidence count does not rank outcomes.** By checks met:

  | Checks met | Signals | Mean net R |
  | --- | --- | --- |
  | 0 | 34 | −0.25 |
  | 1 | 6 | +1.36 |
  | 2 | 18 | +0.24 |
  | 3 | 21 | −0.34 |
  | 4 | 18 | −0.21 |
  | 5 | 3 | −1.02 |

  There is no monotonic relation. The count stays labelled uncalibrated and
  must not be read as a probability.

**Limits.** 2.5 weeks and roughly two market episodes. One clustered move
explains most of the baseline's losses. Neither the baseline nor the analyst
can be judged on this sample: **INSUFFICIENT EVIDENCE** from the signals
alone.
