"""Performance metrics: correct arithmetic and honest labelling."""

from __future__ import annotations

from datetime import timedelta

import pytest

import performance as pf

OOS = pf.DEV_CUTOFF + timedelta(days=1)
DEV = pf.DEV_CUTOFF - timedelta(days=1)


def _row(i, decision, r_gross, *, r_net="same", at=OOS, direction="long", conf=55,
         status=None, entry=None, stop=None, tp=None, duplicate_of=None, symbol="XAUUSD",
         variants=None, version="v1", cost_spread=0.3):
    at_i = at + timedelta(minutes=10 * i)
    def acct(basis, r):
        if r is None:
            return None
        st = status or ("tp" if r > 0 else "sl")
        return {"basis": basis, "status": st, "r": r, "holding_minutes": 30.0 + i,
                "exit_at": (at_i + timedelta(minutes=30 + i)).isoformat(),
                "cost_source": "assumed" if basis == "net" else "none"}
    net_r = (r_gross - 0.05 if r_gross is not None else None) if r_net == "same" else r_net
    accounting = {k: v for k, v in (("gross", acct("gross", r_gross)), ("net", acct("net", net_r))) if v}
    return {"id": i, "recorded_at": at_i.isoformat(), "symbol": symbol, "direction": direction,
            "decision": decision, "confluence_score": conf, "engine_version": version,
            "duplicate_of": duplicate_of, "entry": entry if entry is not None else 100.0 + i,
            "stop_loss": stop if stop is not None else 95.0, "take_profit": tp if tp is not None else 110.0,
            "cost_spread": cost_spread, "accounting": accounting, "variants": variants or {}}


def test_the_core_metrics_are_computed_exactly():
    rows = [_row(1, "enter", 2.0), _row(2, "enter", -1.0), _row(3, "enter", -1.0),
            _row(4, "enter", 1.5), _row(5, "enter", -1.0)]
    s = pf.build_report(rows, basis="gross").enter
    assert s.resolved == 5 and s.wins == 2 and s.losses == 3
    assert s.win_rate == pytest.approx(0.4)
    assert s.avg_win == pytest.approx(1.75) and s.avg_loss == pytest.approx(-1.0)
    assert s.expectancy == pytest.approx(0.5 / 5)
    assert s.profit_factor == pytest.approx(3.5 / 3.0, abs=1e-3)
    assert s.total_r == pytest.approx(0.5)
    # Equity in exit order: 2, 1, 0, 1.5, 0.5 -> peak 2, trough 0.
    assert s.max_drawdown == pytest.approx(2.0)
    assert s.longest_losing_streak == 2


def test_drawdown_follows_exit_order_not_signal_order():
    """Signal 1 is recorded first but exits last."""
    rows = [_row(1, "enter", -1.0), _row(2, "enter", 2.0)]
    rows[0]["accounting"]["gross"]["exit_at"] = (OOS + timedelta(days=2)).isoformat()
    s = pf.build_report(rows, basis="gross").enter
    assert s.max_drawdown == pytest.approx(1.0)       # +2 first, then -1 from the peak


def test_no_losing_trade_means_no_profit_factor_not_infinity():
    s = pf.build_report([_row(1, "enter", 2.0)], basis="gross").enter
    assert s.profit_factor is None


def test_missed_winners_and_avoided_losers_come_from_skips():
    rows = [_row(1, "skip", 2.0), _row(2, "skip", -1.0), _row(3, "skip", 1.5), _row(4, "enter", 2.0)]
    rep = pf.build_report(rows, basis="gross")
    assert rep.missed_winners == {"count": 2, "total_r": pytest.approx(3.5)}
    assert rep.avoided_losers == {"count": 1, "total_r": pytest.approx(-1.0)}
    assert rep.all_signals.resolved == 4


def test_gross_and_net_are_separate_and_labelled():
    rows = [_row(1, "enter", 2.0), _row(2, "enter", -1.0)]
    gross = pf.build_report(rows, basis="gross")
    net = pf.build_report(rows, basis="net")
    assert gross.basis == "gross" and net.basis == "net"
    assert net.enter.expectancy < gross.enter.expectancy
    assert "NOT a net result" in pf.render_markdown(gross)
    assert "after costs" in pf.render_markdown(net)
    with pytest.raises(ValueError):
        pf.trades_from_rows(rows, basis="blended")


def test_a_trade_without_a_cost_figure_is_excluded_from_net_and_counted():
    rows = [_row(1, "enter", 2.0), _row(2, "enter", -1.0, r_net=None, cost_spread=None,
                                         symbol="BTCUSDT")]
    net = pf.build_report(rows, basis="net")
    assert net.excluded_no_costs == 1
    assert net.enter.signals == 1
    assert pf.build_report(rows, basis="gross").enter.signals == 2


def test_assumed_costs_are_flagged_in_the_report():
    rep = pf.build_report([_row(1, "enter", 2.0)], basis="net")
    assert any("ASSUMED" in n for n in rep.notes)


def test_development_data_is_separate_from_out_of_sample():
    rows = [_row(1, "enter", 2.0, at=DEV), _row(2, "enter", -1.0, at=OOS)]
    oos = pf.build_report(rows, basis="gross", period="oos")
    dev = pf.build_report(rows, basis="gross", period="dev")
    assert oos.enter.resolved == 1 and oos.enter.total_r == pytest.approx(-1.0)
    assert dev.enter.resolved == 1 and dev.enter.total_r == pytest.approx(2.0)
    assert any("debugging only" in n for n in dev.notes)
    assert "DEVELOPMENT DATA" in pf.render_markdown(dev)


def test_duplicates_count_once_by_default():
    rows = [_row(1, "enter", 2.0, entry=100.0), _row(2, "enter", 2.0, entry=100.05),
            _row(3, "enter", -1.0, entry=105.0)]
    assert pf.duplicate_ids(rows) == {2}
    assert pf.build_report(rows, basis="gross").enter.resolved == 2
    assert pf.build_report(rows, basis="gross", deduplicate=False).enter.resolved == 3


def test_a_stored_duplicate_tag_always_wins():
    rows = [_row(1, "enter", 2.0), _row(2, "enter", 2.0, entry=150.0, duplicate_of=1)]
    assert pf.duplicate_ids(rows) == {2}


def test_repeats_more_than_six_hours_apart_are_new_signals():
    rows = [_row(1, "enter", 2.0, entry=100.0)]
    late = _row(2, "enter", 2.0, entry=100.0)
    late["recorded_at"] = (OOS + timedelta(hours=7)).isoformat()
    assert pf.duplicate_ids(rows + [late]) == set()


def test_the_interval_is_seeded_and_brackets_the_mean():
    rows = [_row(i, "enter", 2.0 if i % 3 == 0 else -1.0) for i in range(1, 31)]
    a = pf.build_report(rows, basis="gross").enter
    b = pf.build_report(rows, basis="gross").enter
    assert a.expectancy_interval == b.expectancy_interval
    low, high = a.expectancy_interval
    assert low <= a.expectancy <= high


def test_small_samples_are_flagged():
    rep = pf.build_report([_row(1, "enter", 2.0)], basis="gross")
    assert rep.enter.small_sample
    assert any("far too few" in n for n in rep.notes)


def test_breakdowns_by_direction_band_session_and_version():
    rows = [_row(1, "enter", 2.0, direction="long", conf=72, version="v1"),
            _row(2, "enter", -1.0, direction="short", conf=51, version="v2"),
            _row(3, "skip", 1.0, direction="long", conf=30, version="v2")]
    rep = pf.build_report(rows, basis="gross")
    assert set(rep.by_direction) == {"long", "short"}
    assert set(rep.by_confluence) == {"70-100", "50-59", "0-49"}
    assert set(rep.by_version) == {"v1", "v2"}
    assert rep.by_session


def test_every_signal_is_also_split_by_engine_version():
    """A SKIP-only engine version still gets its own row, and mixing is flagged."""
    rows = [_row(1, "enter", 2.0, version="v1"), _row(2, "skip", -1.0, version="v2"),
            _row(3, "skip", 1.5, version="v2")]
    rep = pf.build_report(rows, basis="gross")
    assert set(rep.by_version) == {"v1"}                        # ENTER only
    assert set(rep.by_version_all) == {"v1", "v2"}
    assert rep.by_version_all["v2"].resolved == 2 and rep.by_version_all["v2"].total_r == pytest.approx(0.5)
    assert any("2 engine versions" in n for n in rep.notes)
    assert "Every signal by engine version" in pf.render_markdown(rep)


def test_a_variant_is_judged_on_its_own_verdicts():
    rows = [_row(1, "skip", 2.0, variants={"timing": {"verdict": "enter", "confluence_score": 60}}),
            _row(2, "enter", -1.0, variants={"timing": {"verdict": "skip", "confluence_score": 40}}),
            _row(3, "enter", 1.0)]                         # timing did not judge this one
    champion = pf.build_report(rows, basis="gross")
    timing = pf.build_report(rows, basis="gross", variant="timing")
    assert champion.enter.total_r == pytest.approx(0.0)
    assert timing.enter.total_r == pytest.approx(2.0)
    assert timing.all_signals.signals == 2


def test_pending_and_unfilled_trades_are_counted_but_have_no_r():
    rows = [_row(1, "enter", None), _row(2, "enter", None)]
    rows[1]["accounting"] = {"gross": {"basis": "gross", "status": "not_filled", "r": None,
                                       "holding_minutes": None, "exit_at": None,
                                       "cost_source": "none"}}
    s = pf.build_report(rows, basis="gross").enter
    assert s.resolved == 0 and s.pending == 1 and s.not_filled == 1
    assert s.expectancy is None and s.win_rate is None


def test_empty_data_reports_nothing_without_crashing():
    rep = pf.build_report([], basis="net")
    assert rep.enter.resolved == 0 and rep.enter.expectancy is None
    assert pf.render_markdown(rep)
    assert pf.report_json(rep)["basis"] == "net"
