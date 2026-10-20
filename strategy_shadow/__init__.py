"""Prospective shadow evaluation of production ENTER/SKIP strategies.

For every valid XAUUSD signal the live bot has already decided, the four
strategies below are evaluated on the SAME market snapshot the live verdict
used (the live ``MarketContext`` object itself — no extra candle request),
recorded once, immutably, and the original signal's outcome is tracked
whatever any strategy said. Nothing here can change the live verdict, the
live report or what the bot sends.

    baseline       production's decide_deterministically, replayed on the
                   snapshot (and checked against the live verdict)
    htf_regime     strategy_variants: direction from H4+H1 bias agreement
    valid_events   strategy_variants: stale/invalidated H1 votes removed
    range_filter   strategy_variants: SKIP in a ranging, ADX<20 H1 market

Modules:

    versions.py    strategy versions, rules hash, code version
    snapshot.py    snapshot fingerprint, data-quality record, UTC calibration
    evaluate.py    the four decisions, each isolated, with labelled scores
    store.py       write-once SQLite tables beside the existing ones
    outcomes.py    outcome resolution of the original signal (P1 accounting)
    report.py      the comparison report (CLI and bot command)
    service.py     the pipeline hook, enabled by STRATEGY_SHADOW_MODE=on

Independent-analyst proposals (strategies B and C) are a separate system
(``analyst/``) and play no part in this comparison.
"""
