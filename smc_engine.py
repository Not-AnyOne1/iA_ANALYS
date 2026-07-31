"""Smart Money Concepts (SMC) Engine (RFC-006): deterministic SMC objects.

Consumes an already-computed ``market_data.CandleSeries`` and
``structure_engine.StructureAnalysis`` (RFC-004/RFC-005 outputs, unchanged)
and derives the SMC objects listed below. Pure and deterministic — no I/O,
no async, no AI or probabilistic logic, no randomness. The exact same
input always produces the exact same output.

Because this engine consumes an already-computed StructureAnalysis rather
than confirming swings itself, every rule below that scans forward from a
swing point starts at the swing's own candle index (not the confirmation
lag `structure_engine.StructureEngine` used internally) — this module
trusts every SwingPoint/StructureEvent it's given as already-confirmed
input data; it never re-derives or second-guesses structural confirmation.

Out of scope entirely, by design: scoring, trade decisions, and any wiring
into Claude, the Telegram bot, or the analysis pipeline.

Detection rules, stated precisely:

1. Equal Highs / Equal Lows — confirmed swing highs (or lows) are sorted by
   price and greedily chain-grouped: a swing joins the current group if it
   is within ``equal_tolerance`` (relative, default 0.05%) of the group's
   most recently added price. Any group of 2+ swings becomes one
   EqualLevel, priced at the group's average and timestamped/indexed at
   its most recent (highest-index) member.

2. Liquidity Pools — one per Equal Level (buy-side above clustered highs,
   sell-side below clustered lows) PLUS one singleton pool per confirmed
   swing that isn't already part of an Equal Level — every confirmed swing
   represents resting liquidity, clusters just represent more of it.
   Strength = number of swings composing the pool.

3. Liquidity Sweeps — for each confirmed swing, the first subsequent
   candle whose wick trades beyond the swing's price but whose CLOSE
   rejects back on the original side: high > swing high AND close < swing
   high (sweeping buy-side liquidity, bearish reaction), or low < swing
   low AND close > swing low (sweeping sell-side liquidity, bullish
   reaction). This is the one rule in this module that legitimately uses
   wicks — a sweep is definitionally a wick event, unlike BOS/CHoCH. At
   most one sweep per swing (the first qualifying candle); a swing that
   gets cleanly closed through instead of rejected produces no sweep.

4. Fair Value Gap (FVG) — the classic 3-candle imbalance: at candle i,
   bullish if candle[i-1].high < candle[i+1].low (gap = that range);
   bearish if candle[i-1].low > candle[i+1].high (gap = that range).

5. Inverse FVG (IFVG) — an FVG flips once price closes all the way through
   it in the opposite direction: the first candle (starting two candles
   after the FVG's own index, i.e. once the 3-candle pattern is complete)
   that closes beyond the gap's far boundary. A bullish FVG flips bearish
   on a close below its low; a bearish FVG flips bullish on a close above
   its high. The zone and strength are carried over unchanged from the
   source FVG; only direction and the triggering candle are new.

6. Order Blocks — anchored to a confirmed StructureEvent (BOS or CHoCH,
   both count — each represents an impulsive structural break). Scanning
   backward from the breakout candle to (but not including) the broken
   swing's own candle, the nearest opposite-colored candle (bearish body
   for a bullish break, bullish body for a bearish break) becomes the
   order block; its full high/low range is the zone. If no such candle
   exists in that window, no order block is produced for that event.
   Strength = distance from the order block's close to the breakout close.

7. Breaker Blocks / Mitigation Blocks — both derived from the FIRST candle
   after an order block that trades into its zone (low <= zone high AND
   high >= zone low). If that candle's CLOSE goes all the way through the
   zone (below the low of a bullish OB, or above the high of a bearish
   OB), the order block is invalidated and flips into a Breaker Block
   (opposite direction, zone and strength carried over). If the candle
   only touches the zone without closing through, the order block is
   Mitigated (Mitigation Block, same direction, zone and strength carried
   over) rather than invalidated. Only the first interaction with a given
   order block's zone is used.

8. Supply Zones / Demand Zones — every order block that has NOT produced a
   Breaker Block (a Mitigation does not retire a zone, only a full
   close-through breaker does) is re-surfaced as a zone object: bearish
   order blocks become Supply Zones, bullish ones become Demand Zones.
   Same fields as the source order block, unchanged.

9. Premium / Discount Zones — using the single most recent confirmed swing
   high and most recent confirmed swing low (by index, regardless of which
   is more recent), the dealing range is [min(high,low), max(high,low)]
   with an equilibrium at its midpoint. Premium = upper half (equilibrium
   to range high); Discount = lower half (range low to equilibrium).
   Strength = range width. Requires at least one confirmed swing of each
   type; produces neither zone otherwise (or if the range has zero width).

10. OTE (Optimal Trade Entry) — using the same most recent swing high/low
    pair, whichever swing has the higher index is the end of the most
    recent impulse leg. A low-to-high leg (high is more recent) yields a
    BULLISH OTE: the classic 61.8%-79% retracement band measured down from
    the high. A high-to-low leg (low is more recent) yields a BEARISH OTE:
    the same band measured up from the low. Strength = leg size. Produces
    no OTE if the two swings tie on index (ambiguous ordering) or the leg
    has zero size.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from market_data import CandleSeries, Timeframe
from structure_engine import StructureAnalysis, StructureEvent, SwingPoint, SwingType, TrendDirection


class LiquiditySide(str, Enum):
    BUY_SIDE = "buy_side"    # resting above swing highs
    SELL_SIDE = "sell_side"  # resting below swing lows


@dataclass(frozen=True)
class SMCObjectBase:
    """Fields every detected SMC object carries, per RFC-006's requirement."""

    timeframe: Timeframe
    candle_index: int
    timestamp: datetime
    price_high: float
    price_low: float
    strength: float


@dataclass(frozen=True)
class EqualLevel(SMCObjectBase):
    side: LiquiditySide
    swing_points: Tuple[SwingPoint, ...]


@dataclass(frozen=True)
class LiquidityPool(SMCObjectBase):
    side: LiquiditySide
    swing_points: Tuple[SwingPoint, ...]


@dataclass(frozen=True)
class LiquiditySweep(SMCObjectBase):
    side: LiquiditySide           # which side's liquidity was swept
    direction: TrendDirection     # expected reaction direction after the sweep
    swept_level: SwingPoint


@dataclass(frozen=True)
class FairValueGap(SMCObjectBase):
    direction: TrendDirection


@dataclass(frozen=True)
class InverseFVG(SMCObjectBase):
    direction: TrendDirection     # the new (flipped) direction
    source_fvg: FairValueGap


@dataclass(frozen=True)
class OrderBlock(SMCObjectBase):
    direction: TrendDirection
    source_event: StructureEvent


@dataclass(frozen=True)
class BreakerBlock(SMCObjectBase):
    direction: TrendDirection     # the new (flipped) direction
    source_order_block: OrderBlock


@dataclass(frozen=True)
class MitigationBlock(SMCObjectBase):
    direction: TrendDirection     # unchanged from the source order block
    source_order_block: OrderBlock


@dataclass(frozen=True)
class SupplyZone(SMCObjectBase):
    source_order_block: OrderBlock


@dataclass(frozen=True)
class DemandZone(SMCObjectBase):
    source_order_block: OrderBlock


@dataclass(frozen=True)
class PremiumZone(SMCObjectBase):
    pass


@dataclass(frozen=True)
class DiscountZone(SMCObjectBase):
    pass


@dataclass(frozen=True)
class OTEZone(SMCObjectBase):
    direction: TrendDirection


@dataclass(frozen=True)
class SMCAnalysis:
    """The full SMC picture derived from one CandleSeries + StructureAnalysis."""

    symbol: str
    timeframe: Timeframe
    liquidity_pools: Tuple[LiquidityPool, ...]
    liquidity_sweeps: Tuple[LiquiditySweep, ...]
    equal_highs: Tuple[EqualLevel, ...]
    equal_lows: Tuple[EqualLevel, ...]
    fair_value_gaps: Tuple[FairValueGap, ...]
    inverse_fvgs: Tuple[InverseFVG, ...]
    order_blocks: Tuple[OrderBlock, ...]
    breaker_blocks: Tuple[BreakerBlock, ...]
    mitigation_blocks: Tuple[MitigationBlock, ...]
    supply_zones: Tuple[SupplyZone, ...]
    demand_zones: Tuple[DemandZone, ...]
    premium_zone: Optional[PremiumZone]
    discount_zone: Optional[DiscountZone]
    ote_zone: Optional[OTEZone]


class SMCEngine:
    """Deterministic SMC detector. Stateless across calls."""

    _OTE_NEAR = 0.618
    _OTE_FAR = 0.79

    def __init__(self, equal_tolerance: float = 0.0005) -> None:
        if equal_tolerance < 0:
            raise ValueError(f"equal_tolerance must be >= 0, got {equal_tolerance}")
        self._equal_tolerance = equal_tolerance

    def analyze(self, series: CandleSeries, structure: StructureAnalysis) -> SMCAnalysis:
        if series.symbol != structure.symbol or series.timeframe != structure.timeframe:
            raise ValueError(
                f"series ({series.symbol}/{series.timeframe}) and structure "
                f"({structure.symbol}/{structure.timeframe}) do not describe the same data"
            )

        candles = series.candles
        timeframe = series.timeframe
        swings = structure.swing_points

        equal_highs = self._detect_equal_levels(swings, SwingType.HIGH, timeframe)
        equal_lows = self._detect_equal_levels(swings, SwingType.LOW, timeframe)
        liquidity_pools = self._detect_liquidity_pools(swings, equal_highs, equal_lows, timeframe)
        liquidity_sweeps = self._detect_liquidity_sweeps(candles, swings, timeframe)

        fvgs = self._detect_fvgs(candles, timeframe)
        inverse_fvgs = self._detect_inverse_fvgs(candles, fvgs, timeframe)

        order_blocks = self._detect_order_blocks(candles, structure.events, timeframe)
        breaker_blocks, mitigation_blocks = self._detect_breakers_and_mitigations(
            candles, order_blocks, timeframe
        )

        broken_ob_indices = {b.source_order_block.candle_index for b in breaker_blocks}
        supply_zones = tuple(
            SupplyZone(**self._base_fields(ob), source_order_block=ob)
            for ob in order_blocks
            if ob.direction == TrendDirection.BEARISH and ob.candle_index not in broken_ob_indices
        )
        demand_zones = tuple(
            DemandZone(**self._base_fields(ob), source_order_block=ob)
            for ob in order_blocks
            if ob.direction == TrendDirection.BULLISH and ob.candle_index not in broken_ob_indices
        )

        premium_zone, discount_zone = self._detect_premium_discount(swings, timeframe)
        ote_zone = self._detect_ote(swings, timeframe)

        return SMCAnalysis(
            symbol=series.symbol,
            timeframe=timeframe,
            liquidity_pools=liquidity_pools,
            liquidity_sweeps=liquidity_sweeps,
            equal_highs=equal_highs,
            equal_lows=equal_lows,
            fair_value_gaps=fvgs,
            inverse_fvgs=inverse_fvgs,
            order_blocks=order_blocks,
            breaker_blocks=breaker_blocks,
            mitigation_blocks=mitigation_blocks,
            supply_zones=supply_zones,
            demand_zones=demand_zones,
            premium_zone=premium_zone,
            discount_zone=discount_zone,
            ote_zone=ote_zone,
        )

    @staticmethod
    def _base_fields(obj: SMCObjectBase) -> Dict[str, object]:
        return dict(
            timeframe=obj.timeframe, candle_index=obj.candle_index, timestamp=obj.timestamp,
            price_high=obj.price_high, price_low=obj.price_low, strength=obj.strength,
        )

    # --------------------------------------------------------- equal levels

    def _detect_equal_levels(
        self, swings: Tuple[SwingPoint, ...], swing_type: SwingType, timeframe: Timeframe
    ) -> Tuple[EqualLevel, ...]:
        relevant = sorted((s for s in swings if s.type == swing_type), key=lambda s: s.price)
        side = LiquiditySide.BUY_SIDE if swing_type == SwingType.HIGH else LiquiditySide.SELL_SIDE

        groups: List[List[SwingPoint]] = []
        for swing in relevant:
            if groups:
                last = groups[-1][-1]
                if abs(swing.price - last.price) <= self._equal_tolerance * max(swing.price, last.price):
                    groups[-1].append(swing)
                    continue
            groups.append([swing])

        results: List[EqualLevel] = []
        for group in groups:
            if len(group) < 2:
                continue
            ordered = tuple(sorted(group, key=lambda s: s.index))
            latest = ordered[-1]
            avg_price = sum(s.price for s in group) / len(group)
            results.append(EqualLevel(
                timeframe=timeframe, candle_index=latest.index, timestamp=latest.timestamp,
                price_high=avg_price, price_low=avg_price, strength=float(len(group)),
                side=side, swing_points=ordered,
            ))
        results.sort(key=lambda e: e.candle_index)
        return tuple(results)

    # ------------------------------------------------------- liquidity pools

    def _detect_liquidity_pools(
        self,
        swings: Tuple[SwingPoint, ...],
        equal_highs: Tuple[EqualLevel, ...],
        equal_lows: Tuple[EqualLevel, ...],
        timeframe: Timeframe,
    ) -> Tuple[LiquidityPool, ...]:
        pools: List[LiquidityPool] = []
        pooled_indices = set()

        for level in (*equal_highs, *equal_lows):
            pools.append(LiquidityPool(
                timeframe=timeframe, candle_index=level.candle_index, timestamp=level.timestamp,
                price_high=level.price_high, price_low=level.price_low, strength=level.strength,
                side=level.side, swing_points=level.swing_points,
            ))
            pooled_indices.update(s.index for s in level.swing_points)

        for swing in swings:
            if swing.index in pooled_indices:
                continue
            side = LiquiditySide.BUY_SIDE if swing.type == SwingType.HIGH else LiquiditySide.SELL_SIDE
            pools.append(LiquidityPool(
                timeframe=timeframe, candle_index=swing.index, timestamp=swing.timestamp,
                price_high=swing.price, price_low=swing.price, strength=1.0,
                side=side, swing_points=(swing,),
            ))

        pools.sort(key=lambda p: p.candle_index)
        return tuple(pools)

    # ------------------------------------------------------ liquidity sweeps

    def _detect_liquidity_sweeps(
        self, candles: Tuple, swings: Tuple[SwingPoint, ...], timeframe: Timeframe
    ) -> Tuple[LiquiditySweep, ...]:
        sweeps: List[LiquiditySweep] = []
        for swing in swings:
            for j in range(swing.index + 1, len(candles)):
                candle = candles[j]
                if swing.type == SwingType.HIGH:
                    if candle.high > swing.price and candle.close < swing.price:
                        sweeps.append(LiquiditySweep(
                            timeframe=timeframe, candle_index=j, timestamp=candle.timestamp,
                            price_high=candle.high, price_low=swing.price,
                            strength=candle.high - swing.price,
                            side=LiquiditySide.BUY_SIDE, direction=TrendDirection.BEARISH,
                            swept_level=swing,
                        ))
                        break
                else:
                    if candle.low < swing.price and candle.close > swing.price:
                        sweeps.append(LiquiditySweep(
                            timeframe=timeframe, candle_index=j, timestamp=candle.timestamp,
                            price_high=swing.price, price_low=candle.low,
                            strength=swing.price - candle.low,
                            side=LiquiditySide.SELL_SIDE, direction=TrendDirection.BULLISH,
                            swept_level=swing,
                        ))
                        break
        sweeps.sort(key=lambda s: s.candle_index)
        return tuple(sweeps)

    # ------------------------------------------------------------------ FVG

    def _detect_fvgs(self, candles: Tuple, timeframe: Timeframe) -> Tuple[FairValueGap, ...]:
        fvgs: List[FairValueGap] = []
        for i in range(1, len(candles) - 1):
            prev_candle, next_candle = candles[i - 1], candles[i + 1]
            if prev_candle.high < next_candle.low:
                fvgs.append(FairValueGap(
                    timeframe=timeframe, candle_index=i, timestamp=candles[i].timestamp,
                    price_high=next_candle.low, price_low=prev_candle.high,
                    strength=next_candle.low - prev_candle.high, direction=TrendDirection.BULLISH,
                ))
            elif prev_candle.low > next_candle.high:
                fvgs.append(FairValueGap(
                    timeframe=timeframe, candle_index=i, timestamp=candles[i].timestamp,
                    price_high=prev_candle.low, price_low=next_candle.high,
                    strength=prev_candle.low - next_candle.high, direction=TrendDirection.BEARISH,
                ))
        return tuple(fvgs)

    def _detect_inverse_fvgs(
        self, candles: Tuple, fvgs: Tuple[FairValueGap, ...], timeframe: Timeframe
    ) -> Tuple[InverseFVG, ...]:
        inverses: List[InverseFVG] = []
        for fvg in fvgs:
            for j in range(fvg.candle_index + 2, len(candles)):
                candle = candles[j]
                if fvg.direction == TrendDirection.BULLISH and candle.close < fvg.price_low:
                    inverses.append(InverseFVG(
                        timeframe=timeframe, candle_index=j, timestamp=candle.timestamp,
                        price_high=fvg.price_high, price_low=fvg.price_low, strength=fvg.strength,
                        direction=TrendDirection.BEARISH, source_fvg=fvg,
                    ))
                    break
                if fvg.direction == TrendDirection.BEARISH and candle.close > fvg.price_high:
                    inverses.append(InverseFVG(
                        timeframe=timeframe, candle_index=j, timestamp=candle.timestamp,
                        price_high=fvg.price_high, price_low=fvg.price_low, strength=fvg.strength,
                        direction=TrendDirection.BULLISH, source_fvg=fvg,
                    ))
                    break
        inverses.sort(key=lambda x: x.candle_index)
        return tuple(inverses)

    # ------------------------------------------------------------ order blocks

    def _detect_order_blocks(
        self, candles: Tuple, events: Tuple[StructureEvent, ...], timeframe: Timeframe
    ) -> Tuple[OrderBlock, ...]:
        order_blocks: List[OrderBlock] = []
        for event in events:
            start = event.candle_index - 1
            end = event.broken_swing.index  # exclusive lower bound
            found_index: Optional[int] = None
            for k in range(start, end, -1):
                candle = candles[k]
                if event.direction == TrendDirection.BULLISH and candle.close < candle.open:
                    found_index = k
                    break
                if event.direction == TrendDirection.BEARISH and candle.close > candle.open:
                    found_index = k
                    break
            if found_index is None:
                continue
            candle = candles[found_index]
            order_blocks.append(OrderBlock(
                timeframe=timeframe, candle_index=found_index, timestamp=candle.timestamp,
                price_high=candle.high, price_low=candle.low,
                strength=abs(event.break_price - candle.close),
                direction=event.direction, source_event=event,
            ))
        order_blocks.sort(key=lambda ob: ob.candle_index)
        return tuple(order_blocks)

    # ----------------------------------------------- breaker / mitigation blocks

    def _detect_breakers_and_mitigations(
        self, candles: Tuple, order_blocks: Tuple[OrderBlock, ...], timeframe: Timeframe
    ) -> Tuple[Tuple[BreakerBlock, ...], Tuple[MitigationBlock, ...]]:
        breakers: List[BreakerBlock] = []
        mitigations: List[MitigationBlock] = []

        for ob in order_blocks:
            for j in range(ob.candle_index + 1, len(candles)):
                candle = candles[j]
                touches = candle.low <= ob.price_high and candle.high >= ob.price_low
                if not touches:
                    continue

                if ob.direction == TrendDirection.BULLISH:
                    invalidated = candle.close < ob.price_low
                else:
                    invalidated = candle.close > ob.price_high

                if invalidated:
                    new_direction = (
                        TrendDirection.BEARISH if ob.direction == TrendDirection.BULLISH
                        else TrendDirection.BULLISH
                    )
                    breakers.append(BreakerBlock(
                        timeframe=timeframe, candle_index=j, timestamp=candle.timestamp,
                        price_high=ob.price_high, price_low=ob.price_low, strength=ob.strength,
                        direction=new_direction, source_order_block=ob,
                    ))
                else:
                    mitigations.append(MitigationBlock(
                        timeframe=timeframe, candle_index=j, timestamp=candle.timestamp,
                        price_high=ob.price_high, price_low=ob.price_low, strength=ob.strength,
                        direction=ob.direction, source_order_block=ob,
                    ))
                break  # only the first interaction with the zone matters

        breakers.sort(key=lambda b: b.candle_index)
        mitigations.sort(key=lambda m: m.candle_index)
        return tuple(breakers), tuple(mitigations)

    # ------------------------------------------------------- premium / discount

    def _detect_premium_discount(
        self, swings: Tuple[SwingPoint, ...], timeframe: Timeframe
    ) -> Tuple[Optional[PremiumZone], Optional[DiscountZone]]:
        highs = [s for s in swings if s.type == SwingType.HIGH]
        lows = [s for s in swings if s.type == SwingType.LOW]
        if not highs or not lows:
            return None, None

        latest_high = max(highs, key=lambda s: s.index)
        latest_low = max(lows, key=lambda s: s.index)
        range_high = max(latest_high.price, latest_low.price)
        range_low = min(latest_high.price, latest_low.price)
        if range_high == range_low:
            return None, None

        equilibrium = (range_high + range_low) / 2
        anchor = latest_high if latest_high.index >= latest_low.index else latest_low
        width = range_high - range_low

        premium = PremiumZone(
            timeframe=timeframe, candle_index=anchor.index, timestamp=anchor.timestamp,
            price_high=range_high, price_low=equilibrium, strength=width,
        )
        discount = DiscountZone(
            timeframe=timeframe, candle_index=anchor.index, timestamp=anchor.timestamp,
            price_high=equilibrium, price_low=range_low, strength=width,
        )
        return premium, discount

    # ------------------------------------------------------------------- OTE

    def _detect_ote(
        self, swings: Tuple[SwingPoint, ...], timeframe: Timeframe
    ) -> Optional[OTEZone]:
        highs = [s for s in swings if s.type == SwingType.HIGH]
        lows = [s for s in swings if s.type == SwingType.LOW]
        if not highs or not lows:
            return None

        latest_high = max(highs, key=lambda s: s.index)
        latest_low = max(lows, key=lambda s: s.index)
        if latest_high.index == latest_low.index:
            return None

        leg = abs(latest_high.price - latest_low.price)
        if leg == 0:
            return None

        if latest_high.index > latest_low.index:
            ote_high = latest_high.price - self._OTE_NEAR * leg
            ote_low = latest_high.price - self._OTE_FAR * leg
            direction = TrendDirection.BULLISH
            anchor = latest_high
        else:
            ote_low = latest_low.price + self._OTE_NEAR * leg
            ote_high = latest_low.price + self._OTE_FAR * leg
            direction = TrendDirection.BEARISH
            anchor = latest_low

        return OTEZone(
            timeframe=timeframe, candle_index=anchor.index, timestamp=anchor.timestamp,
            price_high=ote_high, price_low=ote_low, strength=leg, direction=direction,
        )
