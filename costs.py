"""Trading costs — spread and slippage — real when available, assumed otherwise.

Every decision records the costs a trade on it would have paid, and where
they came from:

* ``bid/ask:<provider>`` — the spread measured from a real bid/ask quote at
  decision time (the decision's own quote, or a dedicated OANDA/MT5 quote
  when one is configured). Slippage is still an assumption: no feed reports
  the fill a trader would actually have got.
* ``assumed`` — no bid/ask available; the configured assumption for the
  symbol is used, and the label says so everywhere it is reported.

A symbol with no assumption and no bid/ask gets **no** cost figure. Its
trades are reported gross only and never appear in a net figure: gross is
never passed off as net.

Default assumption for XAUUSD (USD per ounce), overridable with
``COST_SPREAD_XAUUSD`` / ``COST_SLIPPAGE_XAUUSD``:

* spread 0.30 — the middle of the 0.20-0.40 typical retail spread in liquid
  hours. It is wider around the daily rollover and in thin Asian hours, so
  this flatters those trades rather than penalising them;
* slippage 0.10 — applied to every fill that executes at market: market
  entries, stop-entry triggers, stop-loss exits and timeout exits. Limit
  entries and take-profits fill at their price.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from market_data import Quote

log = logging.getLogger(__name__)

DEFAULT_ASSUMPTIONS: Dict[str, Tuple[float, float]] = {
    "XAUUSD": (0.30, 0.10),
}


@dataclass(frozen=True)
class Costs:
    spread: float        # full spread, price units
    slippage: float      # per market-executed fill, price units
    source: str          # "bid/ask:<provider>" / "assumed" / "none"

    @property
    def describe(self) -> str:
        return f"spread {self.spread:g}, slippage {self.slippage:g} ({self.source})"


NO_COSTS = Costs(0.0, 0.0, "none")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    value = float(raw)                    # a malformed value is a config error
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


class CostModel:
    """Decides the costs recorded for one decision."""

    def __init__(self, assumptions: Optional[Dict[str, Tuple[float, float]]] = None,
                 quote_source=None) -> None:
        self._assumptions = dict(DEFAULT_ASSUMPTIONS if assumptions is None else assumptions)
        self._quote_source = quote_source    # optional provider with real bid/ask

    @classmethod
    def from_env(cls) -> "CostModel":
        """Assumptions from the environment; a real bid/ask source if one is
        configured (OANDA credentials, or ``MT5_ENABLED=1`` with a terminal)."""
        assumptions = {
            symbol: (_env_float(f"COST_SPREAD_{symbol}", spread),
                     _env_float(f"COST_SLIPPAGE_{symbol}", slip))
            for symbol, (spread, slip) in DEFAULT_ASSUMPTIONS.items()
        }
        source = None
        if os.getenv("OANDA_API_KEY") and os.getenv("OANDA_ACCOUNT_ID"):
            from market_providers import OandaProvider
            source = OandaProvider(os.environ["OANDA_API_KEY"], os.environ["OANDA_ACCOUNT_ID"])
        elif os.getenv("MT5_ENABLED", "").strip() == "1":
            from market_providers import MT5Provider
            mt5 = MT5Provider()
            if mt5.available:
                source = mt5
            else:
                log.warning("cost_source_unavailable mt5 reason=%s", mt5.unavailable_reason)
        return cls(assumptions, source)

    @property
    def has_real_source(self) -> bool:
        return self._quote_source is not None

    def assumption(self, symbol: str) -> Optional[Costs]:
        values = self._assumptions.get(symbol)
        return None if values is None else Costs(values[0], values[1], "assumed")

    async def costs_for(self, symbol: str, quote: Optional[Quote] = None) -> Optional[Costs]:
        """Real spread if any bid/ask is available, else the assumption, else None."""
        assumed = self.assumption(symbol)
        slippage = assumed.slippage if assumed else 0.0
        for candidate in (quote, await self._dedicated_quote(symbol)):
            if candidate is not None and candidate.bid and candidate.ask and candidate.ask >= candidate.bid:
                return Costs(round(candidate.ask - candidate.bid, 6), slippage,
                             f"bid/ask:{candidate.provider}")
        return assumed

    async def _dedicated_quote(self, symbol: str) -> Optional[Quote]:
        if self._quote_source is None:
            return None
        try:
            return await self._quote_source.get_quote(symbol)
        except Exception as exc:  # noqa: BLE001 - costs fall back to the assumption
            log.warning("cost_quote_failed symbol=%s error=%s", symbol, exc)
            return None

    async def aclose(self) -> None:
        closer = getattr(self._quote_source, "aclose", None)
        if closer is not None:
            await closer()
