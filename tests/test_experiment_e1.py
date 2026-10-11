"""Experiment E1 — breakeven stop after +0.5R — as a challenger accounting arm.

The challenger moves the stop to the fill once a CLOSED M15 bar has carried
the bid (long) / ask (short) 0.5 x planned risk beyond it, from the next bar.
Nothing here may change the live strategy's accounting, decision or card.
"""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import experiments as ex
from costs import NO_COSTS, CostModel, Costs
from market_data import Candle, CandleSeries
from outcome_tracker import OutcomeTracker
from storage import Storage
from trade_accounting import BAR, account, account_both

T0 = datetime(2026, 11, 2, 10, 7, tzinfo=timezone.utc)      # decision time (a Monday)
NEXT = datetime(2026, 11, 2, 10, 15, tzinfo=timezone.utc)   # first bar opening after it
COSTS = Costs(spread=0.4, slippage=0.1, source="assumed")    # half spread 0.2
LATE = NEXT + timedelta(days=10)
BE = 0.5


def _bar(i, o, h, l, c):
    return Candle(timestamp=NEXT + i * BAR, open=o, high=h, low=l, close=c, volume=1.0)


def _flat(i, price):
    return _bar(i, price, price + 0.1, price - 0.1, price)


# The bar the decision fell inside: accounting needs history reaching back to
# the decision (otherwise it is "no_data"), but never reads this bar.
PRE = _bar(-1, 100, 100.1, 99.9, 100)


def _acc(row, candles, now, **kw):
    return account(row, [PRE] + candles, now, **kw)


def _long(**over):
    row = {"recorded_at": T0.isoformat(), "direction": "long", "order_type": "market",
           "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0,
           "take_profits_json": json.dumps([110.0]), "price_at_decision": 100.0}
    row.update(over)
    return row


def _short(**over):
    return _long(direction="short", stop_loss=105.0, take_profit=90.0,
                 take_profits_json=json.dumps([90.0]), **over)


# Long market at 100, stop 95 (risk 5). Gross: fill 100, arms when the high
# reaches 102.5, breakeven when the low comes back to 100. Net: fill at the ask
# + slippage = 100.3, arms when the BID reaches 102.8 (mid 103.0), breakeven
# when the bid falls to 100.3 (mid 100.5), exit 100.3 - 0.1 = 100.2.


# ------------------------------------------------------------- the trigger

def test_a_long_arms_on_the_bid_not_the_mid():
    candles = [_bar(0, 100, 102.9, 100, 102), _bar(1, 102, 102, 99.9, 100)] + \
        [_flat(i, 101) for i in range(2, 8)]
    gross = _acc(_long(), candles, NEXT + 8 * BAR, breakeven_at=BE)
    net = _acc(_long(), candles, NEXT + 8 * BAR, costs=COSTS, breakeven_at=BE)
    assert gross.status == "breakeven" and gross.r == pytest.approx(0.0)
    assert net.status == "open"                     # the bid peaked at 102.7 < 102.8


def test_a_short_arms_on_the_ask_not_the_mid():
    # Short: gross fill 100 arms at a low of 97.5; net fill 99.7 needs the ASK at
    # 97.2 (mid 97.0) to arm, and the ask back at 99.7 (mid 99.5) to stop out.
    candles = [_bar(0, 100, 100, 97.1, 98), _bar(1, 98, 100.1, 98, 100)] + \
        [_flat(i, 99) for i in range(2, 8)]
    gross = _acc(_short(), candles, NEXT + 8 * BAR, breakeven_at=BE)
    net = _acc(_short(), candles, NEXT + 8 * BAR, costs=COSTS, breakeven_at=BE)
    assert gross.status == "breakeven" and gross.r == pytest.approx(0.0)
    assert net.status == "open"                     # the ask bottomed at 97.3 > 97.2

    armed = [_bar(0, 100, 100, 96.9, 98), _bar(1, 98, 99.6, 98, 99.5)]
    net = _acc(_short(), armed, NEXT + 2 * BAR, costs=COSTS, breakeven_at=BE)
    assert net.status == "breakeven"
    assert net.exit_price == pytest.approx(99.8)    # buys back at the stop + slippage
    assert net.r == pytest.approx(-0.1 / 5)


def test_the_moved_stop_applies_from_the_next_bar_only():
    """The trigger bar also trades back through the entry: on M15 bars we
    cannot know which came first, so that bar still uses the original stop."""
    candles = [_bar(0, 100, 103.5, 99.0, 101), _flat(1, 101), _bar(2, 101, 101, 100.0, 100.4)]
    net = _acc(_long(), candles, NEXT + 3 * BAR, costs=COSTS, breakeven_at=BE)
    assert net.status == "breakeven" and net.exit_at == NEXT + 2 * BAR
    gross = _acc(_long(), candles[:1], NEXT + BAR, breakeven_at=BE)
    assert gross.status == "open"                   # not stopped on the trigger bar


def test_only_closed_bars_can_arm_or_stop():
    candles = [_bar(0, 100, 103.5, 100.6, 103), _bar(1, 103, 103, 100.0, 100.4)]
    still_forming = _acc(_long(), candles, NEXT + BAR + timedelta(minutes=10),
                            costs=COSTS, breakeven_at=BE)
    closed = _acc(_long(), candles, NEXT + 2 * BAR, costs=COSTS, breakeven_at=BE)
    assert still_forming.status == "open"
    assert closed.status == "breakeven"
    not_yet_closed_trigger = _acc(_long(), candles[:1], NEXT + timedelta(minutes=10),
                                     costs=COSTS, breakeven_at=BE)
    assert not_yet_closed_trigger.status == "open"


def test_bars_before_the_decision_cannot_arm():
    """The bar the decision fell inside rallied — but partly before the decision."""
    before = Candle(timestamp=NEXT - BAR, open=100, high=104, low=100, close=100, volume=1.0)
    candles = [before, _bar(0, 100, 101, 100.0, 100.4)]
    net = account(_long(), candles, NEXT + BAR, costs=COSTS, breakeven_at=BE)
    assert net.status == "open"


def test_a_resting_orders_fill_bar_cannot_arm():
    """Its high may have printed before the fill — same rule as for the target."""
    row = _long(order_type="limit", entry=99.0, take_profits_json="[110.0]")
    fill_bar = _bar(0, 100, 103, 98.5, 100)        # ask reaches 99; high would arm
    back = _bar(1, 100, 100, 98.9, 99.5)           # bid back at 99 -> would be breakeven
    net = _acc(row, [fill_bar, back], NEXT + 2 * BAR, costs=COSTS, breakeven_at=BE)
    assert net.status == "open"                     # filled, but not armed by the fill bar
    arms_later = [fill_bar, _bar(1, 100, 103, 100, 102), _bar(2, 100, 100, 98.9, 99.5)]
    net = _acc(row, arms_later, NEXT + 3 * BAR, costs=COSTS, breakeven_at=BE)
    # Filled at the limit (99, no slippage); the breakeven exit is a stop order.
    assert net.status == "breakeven" and net.r == pytest.approx(-0.1 / 4)


# ------------------------------------------------------ collisions, costs

def test_breakeven_and_target_in_the_same_bar_count_as_the_stop():
    candles = [_bar(0, 100, 103.5, 100.6, 103), _bar(1, 103, 111, 99, 105)]
    net = _acc(_long(), candles, NEXT + 2 * BAR, costs=COSTS, breakeven_at=BE)
    assert net.status == "breakeven" and "same bar" in net.note
    assert _acc(_long(), candles, NEXT + 2 * BAR, costs=COSTS).status == "tp"


def test_a_target_on_the_trigger_bar_still_wins():
    candles = [_bar(0, 100, 111, 100.6, 110)]
    assert _acc(_long(), candles, NEXT + BAR, costs=COSTS, breakeven_at=BE).status == "tp"


def test_the_breakeven_stop_needs_the_bid_to_reach_the_fill():
    armed = _bar(0, 100, 103.5, 100.6, 103)
    above = _acc(_long(), [armed, _bar(1, 103, 103, 100.51, 101)], NEXT + 2 * BAR,
                    costs=COSTS, breakeven_at=BE)
    touched = _acc(_long(), [armed, _bar(1, 103, 103, 100.49, 101)], NEXT + 2 * BAR,
                      costs=COSTS, breakeven_at=BE)
    assert above.status == "open"                   # bid low 100.31 > fill 100.3
    assert touched.status == "breakeven"


def test_a_breakeven_exit_pays_slippage_and_is_exactly_zero_gross():
    candles = [_bar(0, 100, 103.5, 100.6, 103), _bar(1, 103, 103, 99.5, 100)]
    net = _acc(_long(), candles, NEXT + 2 * BAR, costs=COSTS, breakeven_at=BE)
    gross = _acc(_long(), candles, NEXT + 2 * BAR, breakeven_at=BE)
    assert net.entry_fill == pytest.approx(100.3) and net.exit_price == pytest.approx(100.2)
    assert net.r == pytest.approx(-0.1 / 5)
    assert gross.r == 0.0


def test_an_armed_trade_that_never_returns_times_out_like_the_baseline():
    candles = [_bar(0, 100, 103.5, 100.6, 103)] + [_flat(i, 104) for i in range(1, 30)]
    kw = dict(costs=COSTS, horizon=timedelta(hours=2))
    challenger = _acc(_long(), candles, NEXT + 30 * BAR, breakeven_at=BE, **kw)
    assert challenger.status == "timeout"
    assert challenger == _acc(_long(), candles, NEXT + 30 * BAR, **kw)


# --------------------------------------------------- isolation, invariance

def _walk(seed, n=200):
    rnd = random.Random(seed)
    price, bars = 100.0, [PRE]
    for i in range(n):
        o = price
        price += rnd.gauss(0, 0.9)
        hi = max(o, price) + abs(rnd.gauss(0, 0.5))
        lo = min(o, price) - abs(rnd.gauss(0, 0.5))
        bars.append(_bar(i, o, hi, lo, price))
    return bars


@pytest.mark.parametrize("seed", range(60))
def test_the_challenger_equals_the_baseline_unless_it_exits_at_breakeven(seed):
    """Before arming the arms are identical; after it, any path that never
    touches the moved stop never touches the original (further) one either."""
    candles = _walk(seed)
    for row in (_long(), _short()):
        for costs in (NO_COSTS, COSTS):
            base = account(row, candles, LATE, costs=costs)
            chall = account(row, candles, LATE, costs=costs, breakeven_at=BE)
            if chall.status != "breakeven":
                assert (chall.status, chall.r, chall.exit_at) == (base.status, base.r, base.exit_at)
            else:                                # never later than the baseline's exit
                assert base.exit_at is None or chall.exit_at <= base.exit_at


@pytest.mark.parametrize("seed", range(20))
def test_the_baseline_arm_is_exactly_the_live_accounting(seed):
    candles = _walk(seed)
    arms = ex.account_arms(ex.E1, _long(), candles, LATE, costs=COSTS)
    live = account_both(_long(), candles, LATE, costs=COSTS)
    assert arms["baseline/gross"] == live["gross"]
    assert arms["baseline/net"] == live["net"]


def test_the_sensitivity_basis_doubles_both_costs_and_says_so():
    arms = ex.account_arms(ex.E1, _long(), _walk(1), LATE, costs=COSTS)
    assert set(arms) == {f"{a}/{b}" for a in ex.ARMS for b in ex.BASES}
    doubled = arms["baseline/net_2x"].costs
    assert (doubled.spread, doubled.slippage) == (0.8, 0.2) and doubled.source == "assumed x2"
    only_gross = ex.account_arms(ex.E1, _long(), _walk(1), LATE, costs=None)
    assert set(only_gross) == {"baseline/gross", "challenger/gross"}


def test_the_registered_rules_are_the_ones_in_force():
    """Changing any of these after seeing results breaks the registration."""
    assert ex.E1.breakeven_at_r == 0.5
    assert (ex.E1.min_signals, ex.E1.min_enter) == (150, 30)
    assert ex.E1.max_duration == timedelta(weeks=8)
    assert ex.E1.cost_multiplier == 2.0
    assert (ex.E1.direction_min_signals, ex.E1.direction_floor_r) == (20, -0.10)
    doc = Path(__file__).resolve().parent.parent / ex.E1.registration
    if not doc.exists():
        pytest.skip("registration document is local (data/ is not in the repository)")
    text = doc.read_text(encoding="utf-8")
    for fragment in ("+0.5R", ">= 150", ">= 30", "8 weeks", "-0.10R", "doubled"):
        assert fragment in text


# ---------------------------------------------------------------- report

def _arm(status, r, exit_at, source="assumed"):
    return {"status": status, "r": r, "exit_at": exit_at.isoformat(), "cost_source": source}


def _signal(i, base, chall, *, decision="enter", direction="long", at=None, entry=None,
            base2=None, chall2=None):
    at = at or (T0 + timedelta(hours=7 * i))        # > 6 h apart: never duplicates
    exit_at = at + timedelta(hours=1)
    arms = {}
    for basis, (b, c) in {"gross": (base, chall), "net": (base, chall),
                          "net_2x": (base2 or base, chall2 or chall)}.items():
        arms[f"baseline/{basis}"] = _arm(b[0], b[1], exit_at)
        arms[f"challenger/{basis}"] = _arm(c[0], c[1], exit_at)
    return {"id": i, "recorded_at": at.isoformat(), "symbol": "XAUUSD", "direction": direction,
            "decision": decision, "entry": entry if entry is not None else 100.0 + i,
            "stop_loss": 95.0, "take_profit": 110.0, "status": "tp", "duplicate_of": None,
            "arms": arms}


SAVED = (("sl", -1.02), ("breakeven", -0.02))
CUT = (("tp", 1.98), ("breakeven", -0.02))
SAME_WIN = (("tp", 1.98), ("tp", 1.98))
SAME_LOSS = (("sl", -1.02), ("sl", -1.02))


def test_pairs_are_compared_signal_by_signal():
    rows = [_signal(1, *SAVED), _signal(2, *CUT), _signal(3, *SAME_WIN)]
    rep = ex.build_experiment_report(rows, started_at=None, period="development", now=LATE)
    c = rep.net["every signal"]
    assert c.resolved == 3
    assert c.difference == pytest.approx((1.0 - 2.0 + 0.0) / 3, abs=1e-4)
    assert (c.losers_saved, c.winners_cut) == (1, 1)
    assert c.baseline_r == pytest.approx((-1.02 + 1.98 + 1.98) / 3, abs=1e-4)


def test_drawdown_is_measured_per_arm_in_exit_order():
    rows = [_signal(1, *SAME_WIN), _signal(2, *SAVED), _signal(3, *SAVED)]
    c = ex.build_experiment_report(rows, started_at=None, period="development", now=LATE).net["every signal"]
    assert c.baseline_drawdown == pytest.approx(2.04)
    assert c.challenger_drawdown == pytest.approx(0.04)


def test_duplicates_count_once():
    first = _signal(1, *SAVED, entry=100.0)
    repeat = _signal(2, *SAVED, entry=100.05, at=T0 + timedelta(hours=7, minutes=30))
    repeat["recorded_at"] = (datetime.fromisoformat(first["recorded_at"])
                             + timedelta(minutes=30)).isoformat()
    rep = ex.build_experiment_report([first, repeat], started_at=None, period="development", now=LATE)
    assert rep.duplicates_removed == 1 and rep.net["every signal"].signals == 1


def test_development_data_never_gets_a_verdict():
    rows = [_signal(i, *SAVED) for i in range(1, 200)]
    rep = ex.build_experiment_report(rows, started_at=None, period="development", now=LATE)
    assert rep.verdict.startswith("NO VERDICT")
    md = ex.render_markdown(rep)
    assert "NOT validation" in md and "IN-SAMPLE" in md


def test_only_signals_inside_the_window_count():
    start = T0 + timedelta(days=1)
    rows = [_signal(1, *SAVED, at=T0),                                   # before the start
            _signal(2, *SAVED, at=start),                                # the starting signal
            _signal(3, *SAVED, at=start + timedelta(weeks=8, minutes=1))]  # after the hard stop
    rep = ex.build_experiment_report(rows, started_at=start.isoformat(), now=start + timedelta(days=1))
    assert rep.signals_seen == 1 and rep.net["every signal"].signals == 1
    assert rep.verdict.startswith("COLLECTING")
    assert ex.build_experiment_report([], started_at=None, now=LATE).verdict.startswith("NOT STARTED")
    gone = ex.build_experiment_report(rows, started_at=start.isoformat(),
                                      now=start + timedelta(weeks=9))
    assert gone.verdict.startswith("INSUFFICIENT DATA")


def _enough(kind_enter, kind_skip, n_enter=30, n_skip=120, start=T0):
    rows, i = [], 0
    for kind, n, decision in ((kind_enter, n_enter, "enter"), (kind_skip, n_skip, "skip")):
        for _ in range(n):
            i += 1
            rows.append(_signal(i, *kind[i % len(kind)], decision=decision,
                                direction="long" if i % 2 else "short",
                                at=start + timedelta(hours=i)))
            rows[-1]["entry"] = 100.0 + i
    return rows


def test_a_success_needs_every_registered_criterion():
    rows = _enough([SAVED, SAME_WIN, SAME_WIN], [SAVED, SAME_LOSS, SAME_WIN])
    rep = ex.build_experiment_report(rows, started_at=T0.isoformat(), now=T0 + timedelta(weeks=3))
    assert rep.evaluation_point["reached"]
    assert [c["passed"] for c in rep.criteria] == [True] * 6
    assert rep.verdict.startswith("SUCCESS") and "approval" in rep.verdict


def test_cutting_more_winners_than_it_saves_is_not_a_success():
    rows = _enough([CUT, SAME_WIN, SAVED, SAVED], [CUT, CUT, SAVED, SAVED, SAVED])
    rep = ex.build_experiment_report(rows, started_at=T0.isoformat(), now=T0 + timedelta(weeks=3))
    assert not rep.verdict.startswith("SUCCESS")


def test_a_challenger_that_does_not_beat_the_baseline_fails():
    rows = _enough([CUT, SAME_WIN], [CUT, SAME_LOSS])
    rep = ex.build_experiment_report(rows, started_at=T0.isoformat(), now=T0 + timedelta(weeks=3))
    assert rep.verdict.startswith("FAILED")


def test_later_signals_cannot_change_a_reached_verdict():
    rows = _enough([SAVED, SAME_WIN, SAME_WIN], [SAVED, SAME_LOSS, SAME_WIN])
    tail = [_signal(1000 + k, *CUT, at=T0 + timedelta(hours=200 + k)) for k in range(80)]
    for k, r in enumerate(tail):
        r["entry"] = 500.0 + k
    now = T0 + timedelta(weeks=3)
    before = ex.build_experiment_report(rows, started_at=T0.isoformat(), now=now)
    after = ex.build_experiment_report(rows + tail, started_at=T0.isoformat(), now=now)
    assert after.verdict == before.verdict
    assert after.net["every signal"].difference == before.net["every signal"].difference


def test_an_open_signal_inside_the_evaluation_set_holds_the_verdict():
    rows = _enough([SAVED, SAME_WIN, SAME_WIN], [SAVED, SAME_LOSS, SAME_WIN])
    rows += [_signal(500 + k, *SAME_WIN, at=T0 + timedelta(hours=151 + k)) for k in range(5)]
    for k in range(5):
        rows[-1 - k]["entry"] = 700.0 + k
    for basis in ex.BASES:                           # one early signal still open
        rows[3]["arms"].pop(f"baseline/{basis}")
        rows[3]["arms"].pop(f"challenger/{basis}")
    rep = ex.build_experiment_report(rows, started_at=T0.isoformat(), now=T0 + timedelta(weeks=3))
    assert rep.verdict.startswith("COLLECTING") and "still open" in rep.verdict


def test_a_direction_is_judged_only_with_enough_signals():
    rows = _enough([SAVED, SAME_WIN, SAME_WIN], [SAVED, SAME_LOSS, SAME_WIN])
    for r in rows[:-10]:
        r["direction"] = "long"                      # shorts: 10 signals -> not judged
    rep = ex.build_experiment_report(rows, started_at=T0.isoformat(), now=T0 + timedelta(weeks=3))
    six = rep.criteria[5]
    assert set(six["value"]) == {"long"} and six["passed"]


# ------------------------------------------------- storage and the tracker

def _storage(tmp_path) -> Storage:
    s = Storage(tmp_path / "e1.db")
    asyncio.run(s.initialize())
    return s


def _outcome(at, **over):
    row = {"recorded_at": at.isoformat(), "symbol": "XAUUSD", "direction": "long",
           "order_type": "market", "entry": 100.0, "stop_loss": 95.0, "take_profit": 110.0,
           "take_profits_json": "[110.0]", "price_at_decision": 100.0, "decision": "enter",
           "confluence_score": 60, "status": "open"}
    row.update(over)
    return row


class _Market:
    def __init__(self, candles):
        self.candles = candles

    async def get_candles(self, symbol, timeframe, limit):
        return CandleSeries(symbol=symbol, timeframe=timeframe, provider="stub",
                            candles=tuple(self.candles))


# Armed on bar 0, back to the entry on bar 1, target on bar 3: the baseline
# wins, the challenger exits at breakeven.
_PATH = [PRE, _bar(0, 100, 103.5, 100.6, 103), _bar(1, 103, 103, 99.5, 100), _flat(2, 101),
         _bar(3, 101, 111, 101, 110)]


def test_an_existing_database_gains_the_experiment_tables_untouched(tmp_path):
    path = tmp_path / "old.db"
    from storage import _SCHEMA
    old_schema = _SCHEMA.split("-- Pre-registered accounting experiments")[0]
    conn = sqlite3.connect(path)
    conn.executescript(old_schema)
    conn.execute("INSERT INTO signal_outcomes (recorded_at, symbol, status) VALUES (?, 'XAUUSD', 'tp')",
                 (T0.isoformat(),))
    conn.commit()
    before = conn.execute("SELECT * FROM signal_outcomes").fetchall()
    conn.close()
    asyncio.run(Storage(path).initialize())
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"experiments", "experiment_accounting"} <= tables
    assert conn.execute("SELECT * FROM signal_outcomes").fetchall()[0][:3] == before[0][:3]
    conn.close()


def test_the_start_is_fixed_once_by_the_first_recorded_signal(tmp_path):
    s = _storage(tmp_path)
    first = asyncio.run(s.register_experiment("E1", T0.isoformat(), "doc", {"x": 1}))
    again = asyncio.run(s.register_experiment("E1", (T0 + timedelta(days=3)).isoformat(), "doc", {}))
    assert first == again == T0.isoformat()
    assert asyncio.run(s.experiment_start("E1")) == T0.isoformat()


def test_recording_a_signal_starts_the_experiment(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_confluence_redesign import _build
    from models import TradeSetup
    from trade_decision import decide_deterministically
    s = _storage(tmp_path)
    ctx = _build(TradeSetup(symbol="XAUUSD", direction="long", order_type="market", entries=[],
                            stop_loss=90.0, take_profits=[130.0]))
    tracker = OutcomeTracker(s, _Market([]), cost_model=CostModel())
    oid = asyncio.run(tracker.record(ctx, decide_deterministically(ctx)))
    assert oid is not None
    start, rows = asyncio.run(s.fetch_experiment_rows("E1"))
    assert start is not None and [r["id"] for r in rows] == [oid]


def _tracked(tmp_path, experiments=ex.REGISTERED, start=True):
    s = Storage(tmp_path / "tracked.db")
    asyncio.run(s.initialize())
    if start:
        asyncio.run(s.register_experiment("E1", T0.isoformat(), ex.E1.registration, {}))
    early = asyncio.run(s.record_signal_outcome(_outcome(T0 - timedelta(hours=1))))
    oid = asyncio.run(s.record_signal_outcome(_outcome(T0, cost_spread=0.4, cost_slippage=0.1,
                                                       cost_source="assumed")))
    tracker = OutcomeTracker(s, _Market(_PATH), cost_model=CostModel(), experiments=experiments,
                             clock=lambda: NEXT + 4 * BAR)
    asyncio.run(tracker.resolve_pending())
    return s, tracker, early, oid


def _dump(path, table):
    conn = sqlite3.connect(path)
    try:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})") if r[1] != "computed_at"]
        return conn.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY 1, 2").fetchall()
    finally:
        conn.close()


def test_the_tracker_writes_both_arms_once_for_signals_since_the_start(tmp_path):
    s, tracker, early, oid = _tracked(tmp_path)
    start, rows = asyncio.run(s.fetch_experiment_rows("E1"))
    by_id = {r["id"]: r for r in rows}
    assert early not in by_id                        # recorded before the start
    arms = by_id[oid]["arms"]
    assert set(arms) == {f"{a}/{b}" for a in ex.ARMS for b in ex.BASES}
    assert arms["baseline/net"]["status"] == "tp"
    assert arms["challenger/net"]["status"] == "breakeven"
    assert arms["challenger/gross"]["r"] == 0.0
    stamp = arms["challenger/net"]["computed_at"]
    asyncio.run(tracker.resolve_pending())
    assert asyncio.run(s.fetch_experiment_rows("E1"))[1][-1]["arms"]["challenger/net"]["computed_at"] == stamp
    with pytest.raises(ValueError):
        asyncio.run(s.record_experiment_accounting(oid, "E1", "challenger", "net",
                                                   {"status": "open"}, 7200))


def test_the_experiment_never_changes_the_live_records(tmp_path):
    """Same signals, same candles: with and without E1, signal_outcomes and the
    live strategy's trade_accounting are identical row for row."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with_e1, *_ = _tracked(tmp_path / "a")
    without, *_ = _tracked(tmp_path / "b", experiments=(), start=False)
    assert _dump(with_e1._db_path, "trade_accounting") == _dump(without._db_path, "trade_accounting")
    assert _dump(with_e1._db_path, "signal_outcomes") == _dump(without._db_path, "signal_outcomes")
    assert _dump(without._db_path, "experiment_accounting") == []
