# Why production favoured BUY in the October selloff — root cause

Source: production code at `197b2a0`, replayed as the live bot computed it.
`research/production_replay.py` serves live-like windows: the forming bar is
rebuilt from M5, the weekend stubs are included, and the decision clock is
the signal's own time. `research/diagnose_bias.py` records every vote and
check behind each of the 109 recorded signals.

## 1. The verdict is one question: does the H1 vote agree with the signal?

`decide_deterministically` computes `base = directional_confidence(...)`:
- the scoring confidence if the H1 net vote favours the trade's direction;
- 0 otherwise.

It then adds the validation bonus and penalty (±25 at most) and applies the
60/40 bands. Because the confidence curve sits at 60–99, the base decides
almost everything:

| H1 vote vs signal | ENTER | SKIP |
| --- | --- | --- |
| agrees (61) | 56 | 5 |
| disagrees (48) | 0 | 48 |

Validation flipped 5 of 109 verdicts. In effect the verdict is the H1 vote.

## 2. The H1 vote reads stale evidence

The vote sums ten factors on the primary timeframe (H1) only. Each takes
the **most recent** object of its kind in the 300-bar window. None asks
whether that object still holds at the current price, or how old it is.

On 10-05/06 (62 signals) gold was in a selloff (D1 structure bearish on all
62). The votes:

| Factor (weight) | Bullish | Bearish | No vote |
| --- | --- | --- | --- |
| latest BOS (15) | 61 | 1 | 0 |
| latest CHoCH (15) | 62 | 0 | 0 |
| latest order block (10) | 61 | 1 | 0 |
| supply/demand zone count (8) | 62 | 0 | 0 |
| liquidity sweep (10) | 10 | 52 | 0 |
| FVG (8) | 44 | 18 | 0 |
| inverse FVG (8) | 18 | 44 | 0 |
| trend (20) | 26 | 17 | 19 |
| OTE (6) / premium-discount (6) | split | split | 0 |

Four factors, 48 of 106 points, voted long almost every time. Why:

- **Breaks only count on closes, and the selloff made its new lows by
  wick.** At the sample signal (10-06 01:15, price 4149) the H1 lows were
  4130.4 → 4126.1 → 4125.4. No H1 close went below a confirmed swing low,
  so no bearish break fired. The last bearish event was 82 H1 bars old.
  Meanwhile each bounce through a lower high (4144.0) fired a bullish
  CHoCH, then a BOS. Median age of the latest BOS: 11 H1 bars; of the
  latest CHoCH: 22.
- **No invalidation.** In 23 of the 62 signals the latest bullish BOS was
  already invalidated: price had fallen back below the level it broke. The
  BOS kept voting bullish anyway.
- **The zone count is an inventory, not a state.** Supply/demand counts
  every unbroken zone in the window (14 demand against 7 supply, left from
  the long uptrend), so it voted bullish constantly.
- **The higher timeframes never reach the direction.** D1 (bearish) and H4
  enter only one validation check (12 of about 130 points), which cannot
  outweigh a 60–99 base.

## 3. Not a code asymmetry

Every price was mirrored around the decision price (p → 2·price − p, highs
and lows swapped). A BUY on the real data and a SELL on its mirror then got
identical decisions in **218 of 218** cases.

A first run with a fixed mirror point showed 12 small penalty differences.
They came from the liquidity-pool tolerance, which is relative to price, so
moving prices changed the grouping — a test artifact. There is no
hard-coded preference for longs. `tests/test_strategy_variants.py` keeps the
mirror test as a regression check.

## 4. Data handling, checked

- **Labels.** The live provider's candle labels are about 11 h off UTC
  (account zone; bars and OHLC identical). No verdict rule reads a label.
  The only label-dependent value is the day range shown to Claude.
- **Forming bar.** It is included: it moves until it closes, which mainly
  affects the last bar's trend and EMA.
- **Weekend stubs.** They are included: 49 H1, 196 M15 and about 12 H4
  placeholder bars per weekend.
- **Effect on these 109 signals.** A replay on closed bars only, made
  before this work, agreed with the live-like replay on 108 of 109
  verdicts. Stale candles are not what caused the October failure.
- Data cleaning is still tested as its own candidate (`clean_data`), since
  its effect elsewhere is unknown.

## Root cause, in one line

The verdict is decided by a single H1 vote of "latest object" factors that
are never checked for age or validity. In a grinding decline that makes new
lows by wick, stale bullish breaks, order blocks and zone counts outvote the
real direction, and the higher timeframes that showed the decline cannot
override that vote.
