# Strategy shadow evaluation — implementation and readiness

Branch `strategy-shadow-evaluation`. It contains production `197b2a0` with the
Railway session fix, the independent analyst (unchanged; strategies B/C stay
off), the four variants and the score-display fix. Local commits only:
nothing pushed, merged or deployed. Remote `main` is still `197b2a0`.

## A. Implementation

**New package `strategy_shadow/`:**
- `versions.py` — strategy versions, a hash of the rule files, and the
  deployed commit (`RAILWAY_GIT_COMMIT_SHA`).
- `snapshot.py` — the snapshot fingerprint, the data-quality record, and the
  cached UTC label calibration.
- `evaluate.py` — the four decisions on one snapshot, each isolated; the
  labelled setup score.
- `store.py` — write-once tables and the schema-version table; read-only
  loader.
- `outcomes.py` — outcome resolution of the original signal (P1 accounting).
- `report.py` — the comparison report and the latest-comparison card.
- `service.py` — the pipeline hook.

**New research tools:**
- `research/strategy_shadow_report.py` — CLI, read-only.
- `research/strategy_shadow_integration.py` — the local integration check.

**Changed:**

| File | Change |
| --- | --- |
| `config.py` | `STRATEGY_SHADOW_MODE`, `STRATEGY_SHADOW_REPORTING`, `STRATEGY_SHADOW_CONFIRMATION_START` |
| `pipeline.py` | one bounded spawner for both shadow systems: at most 16 tasks in flight, dropped (logged) beyond that, 60 s limit each, 10 s drain at shutdown |
| `main.py` | wiring when on; one shared UTC-labelled provider for all shadow work; the strategy shadow refuses to start if `STORAGE_DB_PATH` is not on the Railway volume |
| `telegram_bot.py` | `/strategy_shadow` and `/strategy_shadow_latest`, configured chat only, registered only when reporting is on |
| `report.py` | card: "Setup score: N/100 (heuristic, not a win probability)"; "Setup score: Not scored — reason" |
| `statistics.py`, `dashboard.py`, notification and console labels | "setup score" for the rule score, "parse confidence" for the parser's |
| `analyst/data.py` | `label_offset` accepts any UTC reference timeframe (default unchanged) |

**New configuration:** all default off. `INDEPENDENT_ANALYSIS_MODE` is
unchanged.

**Database migration:** additive only. Three new tables
(`strategy_shadow_signals`, `strategy_shadow_decisions`,
`strategy_shadow_outcomes`) and `strategy_shadow_meta` (schema version 1) are
created with `CREATE TABLE IF NOT EXISTS` in the existing `STORAGE_DB_PATH`.
No existing table is altered. A newer schema makes older code refuse to
write.

## B. Comparison design

**Strategies.** Exactly as tested in `PS_results.md`, unchanged (rules hash
recorded per signal):

| Strategy | Rule |
| --- | --- |
| baseline | production's `decide_deterministically`, replayed on the live context; `baseline_matches_live` records whether it equals the actual live verdict |
| htf_regime | the favoured direction is H4 bias = H1 bias (last BOS/CHoCH, close on that side of EMA-50); no agreement means score 0, so SKIP |
| valid_events | the H1 vote with BOS/CHoCH/order-block votes counted only while price still holds their level or zone, and supply/demand counting zones from the last 100 bars |
| range_filter | baseline, but SKIP when H1 structure is ranging/unknown and ADX < 20 |

**Snapshot.** The decisions read the live `MarketContext` object itself:
- the same candles, forming bar and stubs included (production's data
  policy);
- the same quote and the same moment;
- no extra request.

Its fingerprint is recorded. Data quality is recorded alongside, and does not
change the decisions:
- the label shift measured against UTC;
- forming bar and stub counts;
- weekday gaps;
- age of the last closed bar in trading time;
- quote against the last bar;
- signal levels within 10% of price (a units check);
- status: fresh / stale / incomplete / unverified / market_closed.

**Pairing.** One hypothetical trade per signal, fixed at decision time:
- the original Entry/SL/TP and order type, never modified;
- the P1 definition (zone midpoint; market orders filled at the live price;
  pending orders must fill within 24 h);
- the stop first on a same-bar touch; gap-through exits at the open; 5-day
  timeout;
- gross and net R.

Every strategy is scored on the same outcome, so a signal is one observation
however many strategies judged it:
- `not_filled` is neither a win nor a loss;
- open and untracked trades are counted but excluded from R;
- re-forwarded duplicates (same direction, SL and TP1 within 4 h) are
  flagged and excluded by default.

**Uncertainty.**
- 95% bootstrap intervals that resample whole UTC days;
- paired comparison through the discordant signals (candidate-only and
  baseline-only ENTERs);
- warnings below 30 resolved ENTERs ("insufficient") and below 100
  ("preliminary").

**Cohorts.** Each signal's cohort is fixed when recorded, from
`STRATEGY_SHADOW_CONFIRMATION_START`. Reports show only the exploration
cohort unless `--include-confirmation` is given, so the confirmation set stays
untouched until it is opened once.

## C. Test results

- **Full suite: 982 passed, 0 failed.** That includes 23 new
  strategy-shadow tests covering:
  - flags and defaults;
  - isolation, same snapshot, BUY/SELL symmetry;
  - stale, missing and closed data, units;
  - calibration and its caching, provider failures;
  - migration and idempotent init; restart and forwarded duplicates;
  - immutability;
  - TP/SL with costs and correction revisions;
  - unfilled and unresolved trades;
  - outage and idle resolver; report metrics;
  - admin-only commands; volume check;
  - on/off pipeline identity, a failing shadow, the task cap.
- **Integration** (`research/strategy_shadow_integration.py`, the 109
  recorded signals through the real pipeline): **12/12 checks passed.**

  | Check | Result |
  | --- | --- |
  | Live verdicts and stored rows | identical ON vs OFF |
  | Telegram | one message per signal, identical ON vs OFF |
  | Card vs production `197b2a0` | only the score line differs |
  | Shadow records | one per signal, four decisions each |
  | Replayed baseline vs live | equal in 109/109 |
  | Snapshot quality | all 109 "fresh", offset measured |
  | Outcomes | 108 resolved, equal to an independent P1 computation (0 mismatches) with one M15 request |
  | Restart on the same database | no new rows, nothing lost |
  | Calibration source down | all processed, recorded "unverified" |
  | Outcome source down | nothing written, trades still pending |
  | Live provider requests | unchanged (545 with shadow on and off) |
  | Calibration requests | 20 over 17 days of signals |
  | Shadow time per signal | 0.60 s mean, 1.19 s max, in a background task (decisions in a worker thread) |
  | Peak memory | 12.9 MB traced for the whole run |
  | Tasks dropped | 0 |

- **Found and fixed by the integration check.** Right after the weekend
  reopen, the calibration reference was mostly flat placeholder bars, so the
  match was ambiguous and 6 signals came out "unverified". The reference now
  drops placeholder bars (window 300, one credit) and an ambiguous match keeps
  the last measured offset.
- **Lint:** pyflakes clean on all new and changed files.
- **Secret scan:** 63 files changed against `197b2a0`, 0 hits, no forbidden
  files.

The recorded signals inside the integration report also give a first
descriptive comparison. It is exploratory, the same signals the variants
were designed from, and not evidence. Over 97 distinct signals and 88
resolved:

| Strategy | Resolved ENTERs | Net R per ENTER | Sample warning |
| --- | --- | --- | --- |
| baseline | 44 | −0.44 | preliminary |
| htf_regime | 7 | −0.44 | insufficient |
| valid_events | 32 | −0.46 | preliminary |
| range_filter | 37 | −0.48 | preliminary |

## D. Railway deployment plan (after approval)

1. Confirm in Railway → Variables that `STORAGE_DB_PATH` points at the volume
   (`/data/signals.db`). The code default is `data/signals.db` under `/app`,
   which is ephemeral, and the entrypoint does not set it.
   - The startup log now prints `storage_not_on_volume` if it is wrong, and
     strategy shadow refuses to start.
   - If it is wrong today, the live history itself is not persisting. That
     needs correcting either way; do not move files without a backup.
2. Back up `/data/signals.db` (Railway volume snapshot, or copy the file).
3. Push the branch and open a PR into `main`. Review, then merge. Railway
   redeploys `main`.
4. Deploy first with the flags still off. Confirm normal operation and that
   the card shows "Setup score".
5. Set `STRATEGY_SHADOW_MODE=on` and redeploy. Expect
   `strategy_shadow mode=on reporting=off` at startup and one
   `strategy_shadow_recorded ... baseline_matches=1` per signal.
   `baseline_matches=0` is logged as a warning and must be investigated.
6. Optionally set `STRATEGY_SHADOW_REPORTING=on` for the admin commands.
7. When the hypotheses to confirm are fixed, set
   `STRATEGY_SHADOW_CONFIRMATION_START=<date>`. Later signals are kept apart.

**Twelve Data budget:**
- live use is unchanged (5 per signal);
- shadow adds about 4 calibration requests a day and at most 48 outcome
  requests a day while trades are pending;
- the independent analyst, if also on, adds 1 per signal.

**Rollback:**
- set `STRATEGY_SHADOW_MODE=off` (and `REPORTING=off`) and redeploy: baseline
  only, with the stored rows intact and inert;
- or revert the merge commit on `main` (`git revert -m 1 <merge>`);
- the tag `pre-p1-2026-10-10` and local `main` `7d18b09` are untouched;
- the shadow tables are never dropped automatically.

## E. Risks and limitations

- **Data quality.**
  - Decisions use production's live-like candles (forming bar, stubs)
    because that is what the baseline uses; the quality fields make that
    visible.
  - Freshness depends on the UTC calibration. Before the first successful
    calibration, signals are "unverified".
- **Card wording.** Anything parsing "Confidence: X%" from the Telegram card
  would break. Nothing in this repository does.
- **API usage.** Small, as above. Two signals 30–60 s apart could already
  exceed the 8-per-minute limit on the live path alone: the 30 s cache only
  covers closer ones. Shadow adds nothing to such a burst, because its
  decisions need no request.
- **Performance.** About 0.6 s of background work per signal, mostly in a
  worker thread. The task cap prevents pile-ups.
- **Sample size.** At recent signal rates it takes months to reach 100
  resolved ENTERs for the baseline. `htf_regime` enters rarely: about 9 of
  97 signals here, so it will take longest.
- **Unresolved trades.** Pending orders that never fill, and trades still
  open, are counted separately and never as zero-R results.
- **Single process.** SQLite stays safe with one replica, which is what
  `railway.json` sets. Do not scale replicas.

## F. Recommendation

**READY FOR SHADOW DEPLOYMENT**, on one condition: verify step D.1 first
(`STORAGE_DB_PATH` on the volume). That is a Railway variable I cannot see.
If it is wrong, the shadow mode will refuse to start and say why.

This verdict is about safe deployment of the measurement, not about strategy
profitability. No strategy has shown reliable profitability.
