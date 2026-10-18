# E1 — its evaluation window spans two engine versions

Status note only. The E1 registration (`data/experiment_E1_breakeven_2026-10-10.md`), its start
time and its recorded rows are **unchanged**; resetting or annotating E1 awaits the owner's
approval.

## What spans what

| | before the data/accounting fixes | after (branch `fix-data-accounting`) |
|---|---|---|
| engine version stamp | `52ae890+pa73ea9a4ab` | `<commit>+p732199a42b` |
| intraday metals candles | weekend stub quotes included | stubs removed; windows padded so 300 real bars remain |
| stop gapped through (both arms) | booked at the stop price | the bar's open bid/ask plus slippage |

* E1 started at **2026-10-11 22:26:20 UTC** (first signal recorded by E1 code).
* Signals in the window so far: **6**, all stamped `52ae890+pa73ea9a4ab` (database as of
  2026-10-15; 24 E1 accounting rows).
* Any signal the bot records once it runs the fixed code carries the new stamp, so the
  window will hold signals from **both** engine versions.

## What it does and does not affect

* Both arms are accounted with the same rules on the same signals, so E1's paired comparison
  stays like-for-like within each signal.
* The data fix changes which signals the live strategy ENTERs (6 of 109 replayed development
  signals flipped SKIP → ENTER), so the **ENTER-only** subset, one of E1's registered
  criteria, will mix decisions from two engines.
* The six signals already in the window were re-accounted under the new rules: E1 results
  identical for all 6 (`research/recalc_baseline.py`).
* `experiments.py` pools every signal since the start; it does not split by engine version.

## Options for the owner (none applied)

1. Keep the window as is and report results split by engine version alongside the pooled ones.
2. Restart E1's window at the first signal recorded by the fixed engine.
3. Annotate the registration with the change and continue.
