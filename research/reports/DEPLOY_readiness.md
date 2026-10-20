# Deployment readiness — `strategy-shadow-evaluation` → `main` → Railway

Prepared 2026-10-20. Nothing has been pushed, merged or deployed, and no
Railway variable or production data was touched. Railway, Docker and the
GitHub CLI are not installed on this machine. Railway facts therefore come
from the repository's own configuration, and every check that needs the live
service is written as a step for you.

## Checklist

| # | Item | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Database path is `/data/signals.db` and the volume is mounted at `/data` | **NEEDS YOUR CHECK (blocking)** | Repo side confirmed; the live side cannot be seen from here. See §1 |
| 2 | Backup and migration if the database is off the volume | Procedure ready (§2) | Not executed |
| 3 | Telegram session resolves to `/data/signal_monitor`, never deleted or replaced while valid | Code confirmed unchanged; one log check for you | §3 |
| 4 | Live ENTER/SKIP decisions unchanged | **PASS** | §4: 2,259/2,259 identical vs production `197b2a0`; decision files byte-identical |
| 5 | Shadow flags default off; no experimental strategy can reach users | **PASS** | §5 |
| 6 | Branch history and diff vs remote `main` reviewed | **PASS** | §6: 10 commits, base `197b2a0`, no conflicts |
| 7 | PR, deploy, verification and rollback steps | Ready | §7, §8 |
| 8 | Final tests and secret scans | **PASS** | §9: 983 passed, 0 failed; lint clean; 0 secret hits |

**Blocking issue:** only item 1. Confirm it before merging, because the merge
itself triggers a Railway redeploy.

**Non-blocking findings:**
- The Dockerfile's `ENV DATA_DIR=/data` means the entrypoint always uses
  `/data` and ignores `RAILWAY_VOLUME_MOUNT_PATH`, despite its comment. This
  is pre-existing and unchanged; it is why the mount must be exactly `/data`.
- Local `main` (`7d18b09`) has diverged from `origin/main` (`197b2a0`). Never
  push local `main`; the PR goes from the feature branch.
- Users will see the card wording change: "Setup score: N/100 (heuristic,
  not a win probability)" instead of "Confidence: X%". That is the approved
  label correction.

## 1. Persistence path (blocking)

**Confirmed in the repository:**
- The README's Railway variables set `STORAGE_DB_PATH=/data/signals.db` and
  `TELEGRAM_SESSION=/data/signal_monitor`.
- The code default for `STORAGE_DB_PATH` is `data/signals.db`, which resolves
  to `/app/data/signals.db` — container storage, wiped on every redeploy. The
  entrypoint does not set it.
- `DATA_DIR` is always `/data` (Dockerfile `ENV`).

**Check in Railway (read-only):**
1. Service → Settings → **Volumes**: a volume is attached with mount path
   exactly **`/data`**.
2. Service → **Variables**: `STORAGE_DB_PATH` = `/data/signals.db`, and
   `TELEGRAM_SESSION` = `/data/signal_monitor` or unset. Both values are paths,
   not secrets; there is no need to reveal any other value.
3. Service → **Settings → Source**: the deploy branch is `main`.
4. Deployments → the current deployment (19 Oct) → **Logs**: look for
   `[entrypoint] Persistent data directory: /data` and
   `[entrypoint] Using existing Telegram session at /data/signal_monitor.session`.
   - "Using existing" on a deployment after the first one means `/data`
     survived a redeploy, so it is persistent.
   - "Seeded Telegram session" on every deployment would mean it is not.

If 1–3 hold, item 1 passes. After the merge the new
`persistence_check ... storage_on_volume=True data_dir_is_mount=True` line
proves it from the logs (§8).

## 2. Backup, and migration if the database is off the volume

**Before any merge, either way:**
- Railway → the volume → **Backups** → create a backup (snapshot of `/data`).
- Optionally keep a local copy; see step 2 below.

**If `STORAGE_DB_PATH` is unset or points outside `/data`, the live database
is in the running container and will be lost at the next redeploy.** Move it
first, while the old deployment is still running. Do not merge, restart or
change any variable before step 5.

```bash
# 1. Open a shell in the running service (Railway CLI, logged in as you)
railway ssh

# 2. Inside the container: consistent online copy to the volume, never overwriting
python - <<'EOF'
import os, sqlite3, time
src = os.environ.get("STORAGE_DB_PATH") or "data/signals.db"
dst = f"/data/signals.migrated-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.db"
assert os.path.exists(src), f"no database at {src}"
assert not os.path.exists(dst)
s, d = sqlite3.connect(src), sqlite3.connect(dst)
s.backup(d)                                     # safe while the bot is writing (WAL)
print("integrity:", d.execute("PRAGMA integrity_check").fetchone()[0])
for t in ("analyses", "subscribers"):
    print(t, s.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], "->",
          d.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
d.close(); s.close(); print("copied to", dst)
EOF

# 3. Only if /data/signals.db does NOT exist yet: put the copy in place
#    (use the exact path printed by step 2)
test -e /data/signals.db && echo "STOP: /data/signals.db exists — compare before choosing" \
  || cp /data/signals.migrated-<STAMP>.db /data/signals.db
```

`railway ssh` needs the Railway CLI logged in as you. The bot keeps writing
to the old file until step 5 redeploys it, so do steps 2–5 back to back, in a
quiet moment: rows written in between are not in the copy.

4. If `/data/signals.db` already existed, stop. Compare row counts and
   decide which to keep; nothing is overwritten automatically.
5. Then set `STORAGE_DB_PATH=/data/signals.db` in Railway. That redeploys the
   current code, which now opens the volume copy.
6. Confirm `/stats` shows the same totals as before, and only then continue
   with §8.

## 3. Telegram session

**Unchanged by this branch:** `docker-entrypoint.sh`, `telegram_session.py`,
`telegram_client.py`, `config.py`'s session handling and `railway_secrets.py`
are byte-identical to `origin/main`.

**What a redeploy does with the session:**
1. The session prefix is `$TELEGRAM_SESSION`, else `/data/signal_monitor`.
   It is exported, so the app opens exactly that file.
2. If `/data/signal_monitor.session` exists, it is used as-is. It is never
   re-seeded from `TELEGRAM_SESSION_B64`.
3. `telegram_session.py ensure` asks Telegram whether it is authorised:
   - **authorised:** kept unchanged;
   - **inconclusive or API error:** kept unchanged;
   - **rejected:** replaced only by a `TELEGRAM_SESSION_B64` session that is
     verified as authorised and different, with the rejected file kept as
     `.rejected-<UTC>`.

**Expected logs on deploy:**
- `[entrypoint] Using existing Telegram session at /data/signal_monitor.session`
- `[session] kept: /data/signal_monitor.session is authorised — kept unchanged`
- `persistence_check ... session_file=/data/signal_monitor.session session_exists=True session_on_volume=True`

## 4. Live decisions unchanged

- **Production rule files are byte-identical to `origin/main`:**
  `trade_decision`, `scoring_engine`, `validation_engine`, `risk_engine`,
  `structure_engine`, `smc_engine`, `indicators`, `trade_quality`, `levels`,
  `sessions`, `news_filter`, `signal_parser`, `claude_client`,
  `decision_engine`, `storage`, `models`, `market_providers`.
- **Changed files on the live path only gain opt-in behaviour.** The defaults
  match production:
  - `MarketContextBuilder(clean_candles=False, now=None)`;
  - `TwelveDataProvider(label_timezone=None)`;
  - the context keeps a reference to the series it already fetched;
  - `pipeline` starts shadow tasks only if a shadow object was passed (None
    when the flags are off).
- **Same verdicts on identical inputs.**
  `research/compare_live_decisions.py` ran production `197b2a0` (fresh
  `git archive`) and this branch's head (`06f2bd5`) on 2,259 identical inputs:
  109 recorded signals and 2,150 standardized ones, 2021–2026, with live-like
  bars and a frozen clock. Verdict, base score, bonus, penalty, adjusted
  score, blocked flag and reason were identical for 2,259 of 2,259, and the
  two output files are byte-identical.
- **Shadow on vs off.** The integration check
  (`research/strategy_shadow_integration.py`, 12/12) gave the same stored
  verdicts and the same Telegram messages for 109 of 109 signals. Against
  production's own renderer, only the score line differs.

## 5. Flags and isolation

- **Defaults:**
  - `INDEPENDENT_ANALYSIS_MODE=off`
  - `STRATEGY_SHADOW_MODE=off`
  - `STRATEGY_SHADOW_REPORTING=off`
  - `STRATEGY_SHADOW_CONFIRMATION_START` unset

  These hold in `config.py` and `.env.example`, are tested, and anything other
  than off/on (or a valid date) is refused at startup.
- **Off means off.** No experimental module is imported, no table is created
  and no request is made: the imports in `main.py` sit inside the flag
  checks, `pipeline.py` uses them for type hints only, and `telegram_bot.py`
  only inside command handlers that are registered only when attached.
- **The only Telegram send of a signal** is `pipeline.py:319`
  `send_report(message, render_telegram_report(context, decision))`, the
  live decision. Shadow systems write to their own tables only.
- **`/shadow` and `/strategy_shadow*`** are pull commands, answered only for
  `TELEGRAM_BOT_CHAT_ID`, never subscribers, and labelled experimental / not
  a trade instruction.
- **Strategies B and C** (`analyst/`) have no code path to any message: they
  are recorded only, and only with `INDEPENDENT_ANALYSIS_MODE=shadow`.

## 6. History and diff vs `origin/main`

- **Merge base** = `origin/main` = `197b2a0`: the PR merges with no
  conflicts.
- **Commits:** 10, all authored by `MAA`. That email is already public on
  `origin/main`.

  | Commit | Branch | Subject |
  | --- | --- | --- |
  | `7aba14c` | independent-analyst | Independent XAUUSD analyst, shadow mode only (off by default) |
  | `3af7e94` | independent-analyst | Shadow analyst: undo the live provider's account-zone candle labels |
  | `37bc2ae` | independent-analyst | Analyst: pad M5 for weekend stubs; backtest harness, protocol fixed |
  | `a683d69` | independent-analyst | Backtest protocol and verdict rule, fixed before the first systematic run |
  | `694ae40` | independent-analyst | Independent analyst reports |
  | `bc84996` | production-strategy-fix | Production strategy: root cause, four candidate fixes, protocol |
  | `c6fb958` | production-strategy-fix | Card: a labelled heuristic score instead of "Confidence: X%" |
  | `f54c86c` | production-strategy-fix | Production strategy fix: results and summary |
  | `f5c72f2` | strategy-shadow-evaluation | Strategy shadow evaluation: baseline and three candidates |
  | `06f2bd5` | strategy-shadow-evaluation | Startup persistence_check log line; live-decision comparison tool |

- **Size:** 65 files, +9,701 / −21.
  - **New:** `analyst/` (11), `strategy_shadow/` (8), `research/` (20 scripts
    and reports, no data), `tests/` (9); plus `strategy_variants.py`,
    `costs.py`, `market_hours.py` and `trade_accounting.py`, the last three
    unchanged from the integration line.
  - **Modified (11):** `main.py`, `pipeline.py`, `config.py`,
    `market_context.py`, `market_data.py`, `report.py`, `telegram_bot.py`,
    `statistics.py`, `formatter.py`, `dashboard.py`, `README.md`,
    `.env.example`.
- **No new dependency;** `requirements.txt` is unchanged. The image's
  `COPY . .` includes the new packages, and `.dockerignore` still excludes
  `.env`, sessions, `data/` and `tests/`.

## 7. GitHub PR

There is no `gh` CLI here. After your approval:

```bash
git push -u origin strategy-shadow-evaluation
```

Then open https://github.com/Not-AnyOne1/iA_ANALYS/compare/main...strategy-shadow-evaluation:
- Base: `main`; compare: `strategy-shadow-evaluation`.
- Title: **Strategy shadow evaluation (off by default) + honest setup-score label**.
- Body: `research/reports/PR_strategy_shadow.md`.
- Review the "Files changed" tab. The decision-path files in §4 should not
  appear.
- Merge with **"Create a merge commit"**, so one commit can be reverted.
  Don't squash and don't rebase.
- Do not push local `main`.

## 8. Railway rollout and verification

Do these one at a time, and wait for each deployment to finish.

**Phase 0 — before merging.**
1. Complete §1 and the backup in §2.
2. Note the current deployment ID (Deployments tab); it is the rollback
   target.

**Phase 1 — merge with every new flag off.** Railway deploys `main`.

Expected in the deploy logs:

```
[entrypoint] Persistent data directory: /data
[entrypoint] Using existing Telegram session at /data/signal_monitor.session
[session] kept: /data/signal_monitor.session is authorised — kept unchanged (auth key fp now=…)
persistence_check storage_db=/data/signals.db storage_db_exists=True storage_on_volume=True session_file=/data/signal_monitor.session session_exists=True session_on_volume=True volume_mount=/data data_dir_is_mount=True
market_validation_enabled primary_timeframe=H1 provider=twelve_data
Listening for new messages — press Ctrl-C to stop
```

- There must be no `strategy_shadow` or `independent_analysis` lines.
- On the next signal: one Telegram card, with "Setup score: …/100
  (heuristic, not a win probability)" or "Setup score: Not scored — …".
- `/stats` totals continue from before.

**Stop and roll back if** you see any of:
- `storage_not_on_volume`;
- `storage_db_exists=False` when a database existed before;
- `session_exists=False`;
- `data_dir_is_mount=False`;
- a `FATAL` entrypoint line;
- missing cards.

**Phase 2 — set `STRATEGY_SHADOW_MODE=on`** (Railway redeploys).

Expected:

```
strategy_shadow mode=on reporting=off confirmation_start=None (records only; live verdicts and reports unchanged)
```

Per signal:

```
strategy_shadow_recorded signal_key=… id=… data=fresh live=ENTER baseline=ENTER htf_regime=… valid_events=… range_filter=… baseline_matches=1
```

Later: `strategy_shadow_outcomes_recorded rows=…`. Still exactly one Telegram
message per signal.

**Investigate or roll back on:**
- `baseline_matches=0` (logged as a warning);
- `strategy_shadow_disabled`;
- `Failed to initialize strategy shadow`;
- repeated `shadow_task_dropped`;
- `strategy_shadow_calibration_failed` on every signal.

**Phase 3 (optional) — set `STRATEGY_SHADOW_REPORTING=on`.**
- `/strategy_shadow` and `/strategy_shadow_latest` answer only the
  configured bot chat.
- Later, when the hypotheses are fixed, set
  `STRATEGY_SHADOW_CONFIRMATION_START=YYYY-MM-DD`.

**Variables to add:** only those in phases 2–3. Do not change any existing
variable, except `STORAGE_DB_PATH` if §2 applied.

## Rollback

| Situation | Action | Data |
| --- | --- | --- |
| Shadow misbehaves | Delete `STRATEGY_SHADOW_MODE` (or set `off`); Railway redeploys baseline-only | Shadow rows stay, inert |
| Anything else after the merge | Railway → Deployments → the Phase 0 deployment → **Redeploy** (fastest); then revert the merge on GitHub (PR → Revert, or `git revert -m 1 <merge-sha>`) so `main` matches | Untouched: no migration alters existing tables |
| Data damage (not expected) | Restore the Phase 0 volume backup | — |

Tag `pre-p1-2026-10-10` and local `main` `7d18b09` are untouched.

## 9. Final tests and scans (`06f2bd5`)

- **Tests:** full suite 983 passed, 0 failed (`.venv`, Python 3.11.5).
- **Lint:** pyflakes clean on every new and changed file.
- **Secret scan, tree:** the 65 files changed vs `origin/main` — 0 hits, no
  forbidden files.
- **Secret scan, history:** every added line of every commit
  `origin/main..HEAD` (9,760 lines) — 0 hits, no `.env`, session,
  `.railway-secrets` or `data/` file ever added.
- **Tracked files:** no `.env`, `*.session`, `*.session-journal` or `data/`.
