"""The Python Analysis Engine: assembles every computed fact for one signal.

This is the layer that sits between live market data and Claude. It fetches
real candles, runs every deterministic engine over them, and produces a
single :class:`MarketContext` of *facts only*. Claude receives that object
(as JSON) and reasons about it — it never computes any of it.

    Telegram signal
          ↓
    signal extraction        (signal_parser.py / claude_client.py)
          ↓
    live market data         (market_data.py + market_providers.py)
          ↓
    THIS MODULE              structure, SMC, indicators, levels, session,
          ↓                  trade quality, news, risk gate
    Claude reasoning         (decision_engine.py)
          ↓
    decision + report        (report.py)

Design rules carried over from the rest of the project:

- **Nothing is fabricated.** Every field is ``None``/``unknown`` when the
  data needed for it was unavailable, and :attr:`MarketContext.warnings`
  records why. A missing indicator is reported as missing, never defaulted.
- **Partial results are still useful.** A failure fetching the H4 series
  does not abort the whole context; the primary timeframe's analysis still
  comes back, with the gap recorded.
- **Deterministic once data is fixed.** Given the same candles, this module
  always produces the same context. Only the fetch is non-deterministic.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import entry_quality
import indicators
import levels as levels_mod
import trade_quality
from market_data import (
    Candle, CandleSeries, MarketDataError, MarketDataService, Quote, Timeframe, closed_candles,
)
from models import TradeSetup
from news_filter import NewsFilter, NewsStatus
from risk_engine import RiskAssessment, RiskSettings, evaluate as evaluate_risk
from scoring_engine import ScoringEngine, ScoringResult
from sessions import SessionInfo, session_info
from smc_engine import SMCAnalysis, SMCEngine
from structure_engine import StructureAnalysis, StructureEngine
from validation_engine import ValidationEngine, ValidationResult

log = logging.getLogger(__name__)

# Timeframes fetched for every signal. The primary one drives structure/SMC
# and the trade-quality maths; the others provide higher-timeframe context.
DEFAULT_TIMEFRAMES: Tuple[Timeframe, ...] = (
    Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1,
)
DEFAULT_PRIMARY = Timeframe.H1

# Enough bars for EMA-200 plus headroom for ADX/volatility lookbacks.
DEFAULT_CANDLE_LIMIT = 300


@dataclass(frozen=True)
class TimeframeAnalysis:
    """Everything computed for one timeframe."""

    timeframe: Timeframe
    candles_analysed: int
    trend: str                       # structure_engine's TrendDirection value
    ema: indicators.EMASet
    rsi: Optional[float]
    atr: Optional[float]
    atr_percent: Optional[float]
    volatility: str
    adx: indicators.ADXResult
    last_close: Optional[float]

    @property
    def ema_alignment(self) -> str:
        return self.ema.alignment


@dataclass
class MarketContext:
    """The complete factual picture handed to Claude."""

    symbol: str
    generated_at: datetime
    primary_timeframe: Timeframe

    # Live market state
    quote: Optional[Quote] = None
    current_price: Optional[float] = None
    spread: Optional[float] = None

    # Per-timeframe deterministic analysis
    timeframes: Dict[str, TimeframeAnalysis] = field(default_factory=dict)

    # Primary-timeframe structure / SMC (the full objects the engines produce)
    structure: Optional[StructureAnalysis] = None
    smc: Optional[SMCAnalysis] = None
    scoring: Optional[ScoringResult] = None

    # Context
    levels: Optional[levels_mod.LevelContext] = None
    session: Optional[SessionInfo] = None
    news: Optional[NewsStatus] = None

    # The stated setup, judged. ``setup`` is the parsed signal itself, kept
    # so renderers can show the trader-facing levels (direction, stop, every
    # take-profit) — ``quality`` judges them but only summarises one target.
    setup: Optional[TradeSetup] = None
    quality: Optional[trade_quality.TradeQuality] = None
    risk: Optional[RiskAssessment] = None

    # Is this a good trade from THIS price? Location in the dealing range and
    # extension of the move so far (entry_quality.py). ``None`` means the
    # assessment did not run, which the decision treats as unmeasured.
    entry: Optional[entry_quality.EntryQuality] = None

    # The skeptical challenge (validation_engine.py). Populated last,
    # because it reasons over everything above.
    validation: Optional[ValidationResult] = None

    # Primary-timeframe candles, retained only so validation_engine can
    # measure historical adverse excursion. Deliberately NOT part of what
    # Claude sees: ``trade_decision.build_facts`` is an explicit allowlist
    # and never serialises this, which is what keeps "Claude cannot compute
    # an indicator" true — there is no series in its input to compute from.
    primary_candles: Tuple[Candle, ...] = ()

    # Open time of the last CLOSED bar each timeframe was analysed through.
    # Everything structural is computed from closed bars only; recording
    # where each series stopped makes every decision replayable exactly.
    closed_through: Dict[str, datetime] = field(default_factory=dict)

    # Major levels price must clear on the way to a target — multi-touch
    # liquidity, unswept H4 swings, previous-day high/low — as (price, label).
    major_levels: List[Tuple[float, str]] = field(default_factory=list)

    # Anything that could not be computed, and why.
    warnings: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """True when at least the primary timeframe was analysed.

        Below this, there is nothing factual to reason about and the caller
        should skip Claude entirely rather than ask it to judge a blank.
        """
        return self.primary_timeframe.value in self.timeframes

    @property
    def primary(self) -> Optional[TimeframeAnalysis]:
        return self.timeframes.get(self.primary_timeframe.value)


class MarketContextBuilder:
    """Builds a :class:`MarketContext` for one signal.

    Stateless per call and safe to share across pipeline workers: the
    engines it holds are themselves stateless, and nothing is cached here
    (``MarketDataService`` owns caching).
    """

    def __init__(
        self,
        market_data: MarketDataService,
        *,
        structure_engine: Optional[StructureEngine] = None,
        smc_engine: Optional[SMCEngine] = None,
        scoring_engine: Optional[ScoringEngine] = None,
        news_filter: Optional[NewsFilter] = None,
        risk_settings: Optional[RiskSettings] = None,
        validation_engine: Optional[ValidationEngine] = None,
        timeframes: Sequence[Timeframe] = DEFAULT_TIMEFRAMES,
        primary_timeframe: Timeframe = DEFAULT_PRIMARY,
        candle_limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> None:
        self._market_data = market_data
        self._structure = structure_engine or StructureEngine()
        self._smc = smc_engine or SMCEngine()
        self._scoring = scoring_engine or ScoringEngine()
        self._news = news_filter or NewsFilter()
        self._risk_settings = risk_settings or RiskSettings()
        self._validation = validation_engine or ValidationEngine()
        self._timeframes = tuple(timeframes)
        self._primary = primary_timeframe
        self._limit = candle_limit

        if self._primary not in self._timeframes:
            self._timeframes = (self._primary, *self._timeframes)

    async def build(self, symbol: str, setup: Optional[TradeSetup] = None, *,
                    now: Optional[datetime] = None) -> MarketContext:
        """Assemble every fact available for ``symbol``. Never raises.

        ``now`` is the decision time; it defaults to the clock. It decides
        which bars count as closed, so passing a past time replays a
        decision exactly as it would have been made then.
        """
        symbol = symbol.strip().upper()
        ctx = MarketContext(
            symbol=symbol,
            generated_at=now or datetime.now(timezone.utc),
            primary_timeframe=self._primary,
        )

        await self._add_quote(ctx)
        series_by_tf = await self._fetch_series(ctx)
        self._add_timeframe_analysis(ctx, series_by_tf)
        self._add_structure_and_smc(ctx, series_by_tf)
        self._add_levels(ctx, series_by_tf)
        self._add_major_levels(ctx, series_by_tf)
        ctx.session = session_info(ctx.generated_at)
        ctx.news = self._news.check(symbol, ctx.generated_at)
        primary_series = series_by_tf.get(self._primary)
        if primary_series is not None:
            ctx.primary_candles = primary_series.candles
        ctx.setup = setup
        if setup is not None:
            self._add_quality_and_risk(ctx, setup)
            self._add_entry_quality(ctx, setup)
        self._add_validation(ctx, setup)
        return ctx

    def _add_entry_quality(self, ctx: MarketContext, setup: TradeSetup) -> None:
        """Where the entry sits in the dealing range, and how far price has run.

        The dealing range is the SMC one — latest confirmed swing high and
        low, whose midpoint smc_engine already uses as equilibrium — so this
        introduces no second definition of a range. Measured from the
        effective entry (the stated entry, or the live price for a market
        order), because the question is whether *that* price is a good one.
        """
        smc = ctx.smc
        low = smc.discount_zone.price_low if smc and smc.discount_zone else None
        high = smc.premium_zone.price_high if smc and smc.premium_zone else None
        entry = trade_quality.effective_entry(setup, ctx.current_price)
        try:
            ctx.entry = entry_quality.assess(
                setup.direction, entry, low, high, ctx.primary_candles,
                current_price=ctx.current_price, targets=setup.take_profits,
                levels=ctx.major_levels, atr=ctx.primary.atr if ctx.primary else None,
            )
        except Exception as exc:  # noqa: BLE001 - must not cost the rest of the analysis
            ctx.warnings.append(f"entry quality failed: {type(exc).__name__}: {exc}")
            log.exception("entry_quality_failed symbol=%s", ctx.symbol)

    def _add_major_levels(
        self, ctx: MarketContext, series_by_tf: Dict[Timeframe, CandleSeries]
    ) -> None:
        """Levels a trade has to clear on its way to a target — the major ones.

        Three kinds, all from closed bars, each a level traders watch for a
        reason rather than any swing that happens to sit in the way:

        * **multi-touch liquidity** — H1 equal highs and equal lows (two or
          more swings at one price) that price has not traded through since;
        * **unswept H4 swings** — higher-timeframe highs and lows not yet
          exceeded;
        * **previous day high/low** — from the last closed daily bar.

        A failure here leaves the list empty: room is then graded "clear",
        and a warning says why.
        """
        levels: List[Tuple[float, str]] = []
        try:
            primary = series_by_tf.get(self._primary)
            if ctx.smc is not None and primary is not None:
                highs = [c.high for c in primary.candles]
                lows = [c.low for c in primary.candles]
                for eq in ctx.smc.equal_highs:
                    top = max(s.price for s in eq.swing_points)
                    last = max(s.index for s in eq.swing_points)
                    if all(h <= top for h in highs[last + 1:]):
                        levels.append((top, "equal highs"))
                for eq in ctx.smc.equal_lows:
                    bottom = min(s.price for s in eq.swing_points)
                    last = max(s.index for s in eq.swing_points)
                    if all(lo >= bottom for lo in lows[last + 1:]):
                        levels.append((bottom, "equal lows"))

            h4 = series_by_tf.get(Timeframe.H4)
            if h4 is not None:
                highs = [c.high for c in h4.candles]
                lows = [c.low for c in h4.candles]
                for swing in self._structure.analyze(h4).swing_points:
                    later = slice(swing.index + 1, None)
                    if swing.type.value == "high" and all(h <= swing.price for h in highs[later]):
                        levels.append((swing.price, "H4 swing high"))
                    elif swing.type.value == "low" and all(lo >= swing.price for lo in lows[later]):
                        levels.append((swing.price, "H4 swing low"))

            d1 = series_by_tf.get(Timeframe.D1)
            if d1 is not None and d1.candles:
                previous_day = d1.candles[-1]
                levels.append((previous_day.high, "previous day high"))
                levels.append((previous_day.low, "previous day low"))
        except Exception as exc:  # noqa: BLE001 - must not cost the rest of the analysis
            ctx.warnings.append(f"major levels failed: {type(exc).__name__}: {exc}")
            log.exception("major_levels_failed symbol=%s", ctx.symbol)
            levels = []
        ctx.major_levels = levels

    def _add_validation(self, ctx: MarketContext, setup: Optional[TradeSetup]) -> None:
        """Run the skeptical challenge over everything computed above.

        Last, deliberately: it reasons about the other results, so it needs
        them all in place. Isolated like every other step — a validation
        bug must not cost the analysis that already succeeded.
        """
        try:
            ctx.validation = self._validation.validate(
                ctx, direction=setup.direction if setup is not None else None
            )
        except Exception as exc:  # noqa: BLE001
            ctx.warnings.append(f"validation failed: {type(exc).__name__}: {exc}")
            log.exception("validation_failed symbol=%s", ctx.symbol)

    # ------------------------------------------------------------- fetching

    async def _add_quote(self, ctx: MarketContext) -> None:
        try:
            quote = await self._market_data.get_quote(ctx.symbol)
        except MarketDataError as exc:
            ctx.warnings.append(f"quote unavailable: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - context building must never raise
            ctx.warnings.append(f"quote failed unexpectedly: {type(exc).__name__}: {exc}")
            return
        ctx.quote = quote
        ctx.current_price = quote.price
        ctx.spread = quote.spread
        if quote.spread is None:
            ctx.warnings.append(
                f"provider '{quote.provider}' exposes no bid/ask — spread unknown"
            )

    async def _fetch_series(self, ctx: MarketContext) -> Dict[Timeframe, CandleSeries]:
        """Fetch every timeframe concurrently; record each failure."""
        async def one(tf: Timeframe):
            try:
                return tf, await self._market_data.get_candles(ctx.symbol, tf, self._limit)
            except MarketDataError as exc:
                return tf, exc
            except Exception as exc:  # noqa: BLE001
                return tf, MarketDataError(
                    f"{type(exc).__name__}: {exc}", retryable=False)

        results = await asyncio.gather(*(one(tf) for tf in self._timeframes))
        series: Dict[Timeframe, CandleSeries] = {}
        for tf, outcome in results:
            if isinstance(outcome, MarketDataError):
                ctx.warnings.append(f"{tf.value} candles unavailable: {outcome}")
                continue
            # Closed bars only. The provider's last bar is usually still
            # forming; a BOS, a sweep or an indicator read off it is
            # provisional and was measured to flip live verdicts. The live
            # price still reaches the decision — through the entry, the
            # location, R:R and the latest move — never through structure.
            closed = closed_candles(outcome.candles, tf, ctx.generated_at)
            if not closed:
                ctx.warnings.append(f"{tf.value} candles unavailable: no closed bar yet")
                continue
            series[tf] = replace(outcome, candles=closed)
            ctx.closed_through[tf.value] = closed[-1].timestamp
        return series

    # ------------------------------------------------------------ computing

    def _add_timeframe_analysis(
        self, ctx: MarketContext, series_by_tf: Dict[Timeframe, CandleSeries]
    ) -> None:
        for tf, series in series_by_tf.items():
            candles = series.candles
            try:
                structure = self._structure.analyze(series)
                ctx.timeframes[tf.value] = TimeframeAnalysis(
                    timeframe=tf,
                    candles_analysed=len(candles),
                    trend=structure.trend.value,
                    ema=indicators.ema_set(candles),
                    rsi=indicators.rsi(candles),
                    atr=indicators.atr(candles),
                    atr_percent=indicators.atr_percent(candles),
                    volatility=indicators.volatility_state(candles),
                    adx=indicators.adx(candles),
                    last_close=candles[-1].close if candles else None,
                )
            except Exception as exc:  # noqa: BLE001 - one bad timeframe must not sink the rest
                ctx.warnings.append(f"{tf.value} analysis failed: {type(exc).__name__}: {exc}")
                log.exception("timeframe_analysis_failed symbol=%s tf=%s", ctx.symbol, tf.value)

    def _add_structure_and_smc(
        self, ctx: MarketContext, series_by_tf: Dict[Timeframe, CandleSeries]
    ) -> None:
        series = series_by_tf.get(self._primary)
        if series is None:
            ctx.warnings.append(
                f"primary timeframe {self._primary.value} unavailable — "
                "no structure/SMC/scoring computed"
            )
            return
        try:
            ctx.structure = self._structure.analyze(series)
            ctx.smc = self._smc.analyze(series, ctx.structure)
            # ``last_index`` lets the scoring engine weigh each structure
            # event by how recently it happened.
            ctx.scoring = self._scoring.score(ctx.structure, ctx.smc,
                                              last_index=len(series.candles) - 1)
        except Exception as exc:  # noqa: BLE001
            ctx.warnings.append(f"structure/SMC/scoring failed: {type(exc).__name__}: {exc}")
            log.exception("structure_smc_failed symbol=%s", ctx.symbol)

    def _add_levels(
        self, ctx: MarketContext, series_by_tf: Dict[Timeframe, CandleSeries]
    ) -> None:
        series = series_by_tf.get(self._primary)
        price = ctx.current_price or (series.candles[-1].close if series and series.candles else None)
        if series is None or price is None:
            ctx.warnings.append("levels not computed (no primary candles or price)")
            return
        primary = ctx.primary
        try:
            ctx.levels = levels_mod.level_context(
                series.candles,
                ctx.structure.swing_points if ctx.structure else (),
                price,
                primary.atr if primary else None,
            )
        except Exception as exc:  # noqa: BLE001
            ctx.warnings.append(f"levels failed: {type(exc).__name__}: {exc}")

    def _add_quality_and_risk(self, ctx: MarketContext, setup: TradeSetup) -> None:
        primary = ctx.primary
        atr_value = primary.atr if primary else None
        try:
            ctx.quality = trade_quality.assess(
                setup,
                current_price=ctx.current_price,
                atr_value=atr_value,
                swings=ctx.structure.swing_points if ctx.structure else (),
                levels=ctx.levels,
            )
            ctx.risk = evaluate_risk(
                ctx.quality,
                direction=setup.direction,
                trend=ctx.structure.trend if ctx.structure else None,
                atr_value=atr_value,
                spread=ctx.spread,
                volatility=primary.volatility if primary else None,
                confidence=ctx.scoring.confidence if ctx.scoring else None,
                news=ctx.news,
                settings=self._risk_settings,
            )
        except Exception as exc:  # noqa: BLE001
            ctx.warnings.append(f"quality/risk failed: {type(exc).__name__}: {exc}")
            log.exception("quality_risk_failed symbol=%s", ctx.symbol)
