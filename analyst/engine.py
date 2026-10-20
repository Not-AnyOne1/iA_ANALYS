"""One pure call: candles + signal in, independent analysis out.

``analyse`` is what the live shadow hook and the backtests both run, so a
backtest result is a result of exactly this code. It does no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence

from costs import Costs
from market_data import Candle, Timeframe
from models import TradeSetup
from news_filter import NewsStatus

from analyst.data import TIMEFRAMES, MarketView, build_view
from analyst.rules import Rules
from analyst.strategies import Proposal, evaluate, headline
from analyst.structure import LONG, MarketRead, find_pullback, find_sweep, m5_break, read_market
from analyst.verdict import RecentSignal, Verdict, judge

ENGINE_VERSION = "analyst-1"


@dataclass(frozen=True)
class Analysis:
    symbol: str
    as_of: datetime
    price: Optional[float]
    view: MarketView
    context: Dict[str, str]                 # timeframe -> one-line reading
    biases: Dict[str, str]                  # timeframe -> bullish / bearish / neutral / unavailable
    m15_setup: str
    m5_confirmation: str
    proposals: Dict[str, Dict[str, Optional[Proposal]]]
    headline: Optional[Proposal]
    verdict: Verdict
    news: str
    engine_version: str = ENGINE_VERSION

    @property
    def data_status(self) -> str:
        return self.view.status


def _m15_setup(read: MarketRead, direction: str) -> str:
    parts = []
    pb = find_pullback(read, direction)
    if pb is not None:
        parts.append(f"pullback {pb.depth:.0%} of {pb.origin.price:.2f} -> {pb.impulse.price:.2f}")
    sw = find_sweep(read, direction)
    if sw is not None:
        parts.append(f"sweep of {sw.level.label()} at {sw.bar.timestamp:%H:%M}")
    return "; ".join(parts) or f"no {direction} setup"


def _m5(read: MarketRead, direction: str) -> str:
    e = m5_break(read, direction, read.view.as_of - timedelta(days=1))
    if e is None:
        return f"no {direction} M5 break in the last hour"
    return f"{e.type.value.upper()} {'up' if direction == LONG else 'down'} through " \
           f"{e.broken_swing.price:.2f} at {e.timestamp:%H:%M}"


def analyse(symbol: str, setup: TradeSetup, message_at: datetime, as_of: datetime,
            price: Optional[float], candles_by_tf: Dict[Timeframe, Optional[Sequence[Candle]]],
            rules: Rules, costs: Costs, *, recent: Sequence[RecentSignal] = (),
            news: Optional[NewsStatus] = None, reader=None) -> Analysis:
    view = build_view(symbol, as_of, price, candles_by_tf, rules)
    read = read_market(view, rules, reader)
    if read is None:
        proposals: Dict[str, Dict[str, Optional[Proposal]]] = {"B": {}, "C": {}}
        empty = {"B": {"long": None, "short": None}, "C": {"long": None, "short": None}}
        verdict = judge(setup, message_at, view, None, empty, rules, costs, recent=recent, news=news)
        context = {tf.value: f"unavailable ({view.status})" for tf in TIMEFRAMES}
        return Analysis(view.symbol, as_of, price, view, context,
                        {tf.value: "unavailable" for tf in TIMEFRAMES}, "not analysed",
                        "not analysed", proposals, None, verdict, _news(news))
    proposals = evaluate(read, costs)
    verdict = judge(setup, message_at, view, read, proposals, rules, costs, recent=recent, news=news)
    head = headline(proposals)
    direction = setup.direction or (head.direction if head else LONG)
    context = {tf.value: read.frame(tf).describe() for tf in (Timeframe.H4, Timeframe.H1)}
    return Analysis(view.symbol, as_of, price, view, context,
                    {tf.value: read.frame(tf).bias for tf in (Timeframe.H4, Timeframe.H1)},
                    _m15_setup(read, direction), _m5(read, direction), proposals, head, verdict,
                    _news(news))


def _news(news: Optional[NewsStatus]) -> str:
    if news is None or not news.available:
        return "not checked — no economic calendar configured"
    return news.reason
