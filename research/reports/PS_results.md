# Production strategy fix — results

Protocol: `PS_protocol.md`, committed (`bc84996`) before the run. Root cause:
`PS_root_cause.md`. Candidate code: `strategy_variants.py`; none is wired
into the live bot.

**Test set:**
- 35,680 standardized signals, one BUY and one SELL every 2 h, 2021-01 to
  2026-10;
- shaped like the recorded signals: market entry, stop 1.0 H1 ATR, target
  2.0R;
- 2026-09-25 to 10-17 excluded: the window the candidates were designed
  from;
- every signal scored whatever its verdict, net of assumed costs (spread
  0.30, slippage 0.10).

Intervals are 95% bootstraps over whole ISO weeks.

## 1. The baseline has no measurable directional skill

Out of sample (2024 to 2026-10):

| | Signals | Net R per signal |
| --- | --- | --- |
| H1 vote agrees with the signal | 8,342 | −0.047 |
| H1 vote disagrees | 8,342 | −0.050 |
| Production ENTER | 8,093 | −0.041 (−0.097, +0.014) |
| Production SKIP | 8,621 | −0.053 |
| Every signal | 16,804 | −0.049 |

- **The vote does not separate winners from losers.** ENTER minus SKIP is
  +0.012R.
- **Before costs, ENTERs are flat:** gross +0.019R out of sample, −0.011R
  in development.
- **In 2021–2023 ENTERs lost after costs:** −0.125R per trade (interval
  −0.177 to −0.072).

Production's verdict is close to a coin flip on these signals. On the
recorded October signals it was worse than that (§4).

## 2. Each candidate against the unchanged baseline

Net R per ENTER; Δ = candidate minus baseline, with its 95% interval.

| Candidate | ENTERs out of sample | Out-of-sample net R | Δ out of sample | Δ 2024 | Δ 2025 | Δ 2026 holdout | Δ at ×2 costs | Development Δ (2021–23) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 8,093 | −0.041 (−0.097, +0.014) | — | — | — | — | — | — | — |
| `clean_data` | 8,124 | −0.043 | **−0.002** (−0.018, +0.013) | 0.000 | +0.009 | −0.019 | −0.001 | −0.001 | **NOT IMPROVED** |
| `htf_regime` | 4,388 | +0.011 (−0.072, +0.097) | **+0.052** (−0.002, +0.106) | +0.027 | +0.142 | **−0.040** | +0.054 | +0.029 (−0.021, +0.078) | **INSUFFICIENT EVIDENCE** |
| `valid_events` | 7,831 | −0.035 | **+0.006** (−0.016, +0.026) | −0.009 | +0.023 | +0.001 | +0.005 | −0.003 | **INSUFFICIENT EVIDENCE** |
| `range_filter` | 7,012 | −0.032 | **+0.009** (−0.008, +0.026) | +0.003 | +0.019 | +0.001 | +0.010 | −0.017 | **INSUFFICIENT EVIDENCE** |

Other measures, out of sample:

| | False-positive rate (losing ENTER) | False-negative rate (winning SKIP) | ENTER − SKIP | Share of ENTERs long | One position at a time: trades, net R, max drawdown |
| --- | --- | --- | --- | --- | --- |
| baseline | 66.5% | 33.3% | +0.012 | 60% | 2,931 · −0.068 · 201R |
| `clean_data` | 66.5% | 33.5% | +0.004 | 60% | 2,952 · −0.069 · 204R |
| `htf_regime` | 64.9% | 32.8% | +0.079 | 65% | 1,680 · +0.023 · 46R |
| `valid_events` | 66.3% | 33.1% | +0.024 | 59% | 2,858 · −0.026 · 105R |
| `range_filter` | 66.3% | 33.2% | +0.026 | 60% | 2,716 · −0.053 · 149R |

**`htf_regime`** is the only candidate with a material effect, and it fails
two required conditions:
- its interval reaches below 0;
- it is worse than the baseline in the 2026 holdout (−0.069R against
  −0.029R).

Almost all of its gain comes from 2025, a strong gold uptrend in which 73%
of its ENTERs were longs. Out of sample, longs averaged about +0.04 to
+0.09R and shorts about −0.15R across all verdicts. A rule that follows the
H4 trend benefits from that drift, which is not the same as having skill.

**`valid_events`** and **`range_filter`** fix the stale-vote mechanism in
the intended direction, but the effect is a few hundredths of an R: too
small to tell from noise. Both were slightly negative in development.

**`clean_data`** changes almost nothing (8,877 vs 8,891 development ENTERs;
Δ ≈ 0). Stale and placeholder candles are not what limits this strategy.

## 3. By regime, out of sample (net R per ENTER, trades)

| Regime | baseline | `htf_regime` |
| --- | --- | --- |
| H4 trend with the trade | −0.002 (2,807) | +0.044 (2,263) |
| H4 trend ranging | −0.047 (3,230) | +0.020 (1,643) |
| H4 trend against | −0.087 (1,938) | −0.207 (418) |
| D1 trend with | −0.047 (2,801) | −0.067 (1,456) |
| D1 trend against | −0.058 (2,078) | +0.041 (1,063) |
| H1 ADX < 20 | −0.051 (2,201) | +0.013 (898) |
| H1 ADX ≥ 25 | −0.046 (4,330) | −0.005 (2,817) |
| Session: New York | 0.000 (2,003) | +0.089 (1,109) |
| Session: late (21–24 UTC) | −0.147 (613) | −0.022 (340) |

No regime gives the baseline a positive edge after costs.

## 4. The 109 recorded signals (descriptive only — they shaped the candidates)

| | ENTER | Net R per ENTER | False-positive rate | Longs among ENTERs |
| --- | --- | --- | --- | --- |
| baseline | 56 | −0.49 (n 53) | 81% | 48 |
| `clean_data` | 58 | −0.39 (n 55) | 76% | 48 |
| `htf_regime` | 11 | −0.57 (n 9) | 78% | 5 |
| `valid_events` | 42 | −0.56 (n 39) | 85% | 33 |
| `range_filter` | 48 | −0.53 (n 45) | 82% | 41 |

None of the candidates turns the October ENTERs into winners. `htf_regime`
mostly avoids them by entering only 11.

## 5. Verdict (pre-declared rule)

**INSUFFICIENT EVIDENCE.**
- No candidate meets IMPROVED.
- One is NOT IMPROVED (`clean_data`).
- Three show positive point estimates with intervals that include 0.

The baseline itself is not profitable: out of sample its interval spans 0,
and in development it was significantly negative. No candidate is
profitable either: `htf_regime`'s +0.011R has an interval of −0.072 to
+0.097. Four candidates were tested, so a borderline result like
`htf_regime`'s should be read with that multiplicity in mind.

**Recommendation:**
- Do not change the live ENTER/SKIP rule on this evidence.
- The most useful next step is to measure on real signals rather than to
  tune further. Record `htf_regime`, `valid_events` and `range_filter`
  verdicts next to the live verdict in the shadow analyst's table.
- After that, decide on real outcomes.
