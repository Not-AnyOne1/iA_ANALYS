# Independent analyst — deployment plan and rollback (proposed, not done)

Nothing here has been pushed, merged or deployed. Railway still runs `main` =
`197b2a0`.

## What the branch contains

`independent-analyst` sits on top of production `main` (`197b2a0`), **not** on
the integration line. As a result:

- deploying it changes no live ENTER / WAIT / SKIP rule;
- deploying it changes nothing on the live card;
- the integration branch's unreleased decision changes stay a separate
  decision.

With `INDEPENDENT_ANALYSIS_MODE` unset or `off`, the only differences from
`197b2a0` at runtime are:

- `MarketContext` keeps a reference to the candle series it already fetched.
  Claude does not see it, and a test checks that.
- The `/shadow` command handler exists but is not registered.

No new table, request or task.

## Enabling shadow mode (after approval)

1. Push `independent-analyst` and open a PR into `main`. Review it, then merge.
   Railway deploys `main`.
2. In Railway → service → Variables, add `INDEPENDENT_ANALYSIS_MODE=shadow`.
   Nothing else is required. Optional variables:
   - `ANALYST_MIN_RR` (default 1.5);
   - `ANALYST_SIGNAL_MIN_RR` (default 1.0);
   - `COST_SPREAD_XAUUSD` (default 0.30);
   - `COST_SLIPPAGE_XAUUSD` (default 0.10).
3. Redeploy. Expected startup log line:
   `independent_analysis_enabled mode=shadow (records only; live verdicts and reports unchanged)`.
4. Per signal, expect one `independent_analysis message_id=... verdict=...`
   log line. The Telegram output is unchanged: still exactly one message per
   signal. From the configured bot chat, `/shadow` shows the latest card.

The new tables live in the existing `STORAGE_DB_PATH` on the `/data` volume.
They are created with `CREATE TABLE IF NOT EXISTS`; no existing table is
altered. Telegram session handling, the entrypoint and the Twelve Data key are
untouched.

## Twelve Data budget

| Source | Requests |
| --- | --- |
| Live context, now | 5 per signal (quote + 4 timeframes) |
| Shadow analyst | +1 per signal (M5), plus at most 1 M15 request per 30 min while outcomes are pending (≤ 48/day) |
| Free tier | 8/min, 800/day |

The M5 request runs after Claude's decision, outside the burst of 5.

## Rollback

| Situation | Action |
| --- | --- |
| Turn shadow off | Set `INDEPENDENT_ANALYSIS_MODE=off` (or delete the variable) and redeploy. The tables stay, inert. |
| Remove the code | Revert the merge commit on `main` (`git revert -m 1 <merge>`), push, and let Railway redeploy `197b2a0`'s behaviour. The tag `pre-p1-2026-10-10` and local `main` `7d18b09` are untouched. |
| Drop the shadow data (only if you decide to) | The tables are write-once by trigger. Back up `/data/signals.db` first, then drop the triggers and the two `independent_*` tables. No live table depends on them. |

## Not done, deliberately

- No live mode. The analyst cannot change a verdict, and there is no flag that
  would let it.
- No automatic execution.
- No push, merge or deploy.
- No change to production's confidence display. The audit recommends
  replacing "Confidence: X%" with an uncalibrated evidence count; that is a
  live-card change and needs its own approval.

## Recommendation after the backtest

Strategies B and C are **NOT PROFITABLE** out of sample
(`IA_backtest_systematic.md`). Shadow mode records their proposals but
must not be used for recommendations.

What shadow mode measures that nothing else can: on real signals, with
outcomes, whether the analyst's ENTER / WAIT / SKIP / NO TRADE filter beats
the live verdict. Enable it only if that measurement is wanted.

## Found during this work, not changed

The live Twelve Data provider receives candles labelled about 11 h ahead of
UTC (account zone; same bars and OHLC). Production's verdict does not depend
on the labels: no rule compares a candle with the clock. The one
label-dependent value is `levels.day_range`, which groups by the
account-zone day instead of the UTC day, and it reaches only Claude's
narrative facts. The integration line already requests `timezone=UTC`, and
the shadow analyst corrects for it on its own.
