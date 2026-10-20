## Strategy shadow evaluation (off by default) + honest setup-score label

**What changes for users right now:** the card's "Confidence: X%" becomes
"Setup score: N/100 (heuristic, not a win probability)", or "Setup score:
Not scored — reason". Nothing else. No ENTER/WAIT/SKIP rule, threshold or
stored value changes.

**What is added, all off by default:**
- `STRATEGY_SHADOW_MODE=on`: each XAUUSD signal's live snapshot is also
  judged by the baseline replay and three candidates (`htf_regime`,
  `valid_events`, `range_filter`). The results are recorded write-once in
  new tables beside the existing ones, and the original signal's outcome is
  tracked, so strategies can be compared on forward-collected signals.
- `STRATEGY_SHADOW_REPORTING=on`: admin-only `/strategy_shadow` and
  `/strategy_shadow_latest` for `TELEGRAM_BOT_CHAT_ID`.
- `INDEPENDENT_ANALYSIS_MODE=shadow`: the independent analyst, recording
  only. Its strategies B and C tested NOT PROFITABLE and never reach a user.
- A startup `persistence_check` log line: where the database and session
  resolve to, and whether they are on the volume.

**Evidence:**
- Production `197b2a0` vs this branch on 2,259 identical inputs: verdicts
  and every score identical (`research/compare_live_decisions.py`).
- Integration on the 109 recorded signals through the real pipeline: 12/12
  checks, including identical verdicts and Telegram messages with shadow on
  vs off (`research/strategy_shadow_integration.py`).
- 983 tests pass; lint clean; secret scans of the tree and of every added
  line: 0 hits.

**Unchanged files:** decision code (`trade_decision`, `scoring_engine`,
`validation_engine`, `risk_engine`, `structure_engine`, `smc_engine`, …),
`docker-entrypoint.sh`, `telegram_session.py`, `Dockerfile`, `railway.json`,
`requirements.txt`.

**Research behind it:**
- `research/reports/IA_summary.md`
- `research/reports/PS_summary.md`
- `research/reports/SS_shadow_evaluation.md`

**Deployment:** `research/reports/DEPLOY_readiness.md` — merge with a merge
commit and deploy with flags off first.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
