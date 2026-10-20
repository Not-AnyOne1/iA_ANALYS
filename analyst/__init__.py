"""Independent XAUUSD analyst — shadow mode only.

Reads the market itself on four timeframes and judges each Telegram signal
against what it finds, without trusting the signal's direction:

    H4  context      bias from the last break of structure and the EMA-50 side
    H1  bias         the same rule, one timeframe down
    M15 setup        pullback inside an impulse leg (B) or liquidity sweep (C)
    M5  entry        a break of M5 structure after the setup's extreme

Modules, in the order data flows through them:

    rules.py       every threshold, fixed before any backtest was run
    data.py        closed candles only, weekend stubs dropped, freshness and
                   cross-timeframe alignment checks
    structure.py   per-timeframe facts: swings, BOS/CHoCH, bias, liquidity,
                   sweeps, M5 breaks — nothing that needs a future bar
    strategies.py  B (trend continuation) and C (liquidity-sweep reversal),
                   isolated from each other
    verdict.py     the original signal's checks and its ENTER / WAIT / SKIP /
                   NO TRADE verdict, with reason codes
    engine.py      one pure call that runs all of the above
    report.py      the Telegram card
    store.py       write-once SQLite tables beside the existing ones
    tracker.py     outcome resolution for recorded proposals
    service.py     the pipeline hook, enabled by INDEPENDENT_ANALYSIS_MODE=shadow

Nothing here changes the live verdict, the live report or what the bot
sends. The live strategy stays the baseline (strategy A).
"""
