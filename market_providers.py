"""Additional market data adapters and the provider fallback chain.

Kept separate from ``market_data.py`` so that module — which is covered by
an established test suite and is the stable public interface every engine
imports — stays untouched. Everything here implements the existing
``market_data.MarketDataProvider`` Protocol, so any of these can be handed
to ``MarketDataService`` with no change to a single caller.

Preferred order, per the trading spec:

    1. MT5          local terminal, real bid/ask/spread
    2. OANDA        REST v20, real bid/ask
    3. Polygon      REST v2
    4. Twelve Data  REST (implemented in market_data.py)
    5. Alpha Vantage REST

:class:`FallbackProvider` chains them: the first provider that answers wins,
and one that is unavailable or failing is skipped rather than taking the
whole layer down.

**Verification status, stated plainly.** The HTTP adapters below are written
against each vendor's documented REST response shape and are unit-tested
with mocked transports, so their parsing, symbol mapping and error handling
are verified. They have **not** been run against the live services — no
account or API key for OANDA, Polygon or Alpha Vantage was available. Treat
the first live call to any of them as the real integration test.
:class:`MT5Provider` is different again: see its docstring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import httpx

from market_data import (
    Candle,
    CandleSeries,
    MarketDataError,
    MarketDataProvider,
    Quote,
    Timeframe,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- MT5

class MT5Provider:
    """MetaTrader 5 adapter.

    MT5 is the spec's preferred source because it is the only one here that
    gives a genuine broker bid/ask/spread rather than a last-traded price.

    It is also the only provider that cannot run on this project's current
    deployment, and that is not a bug in this adapter:

    - ``MetaTrader5`` is a Windows-only package that talks to a *locally
      running MT5 terminal* over IPC. There is no Linux build and no remote
      protocol.
    - This project deploys to a Linux container (see the Railway section of
      the README), where neither the package nor a terminal exists.

    So this adapter is written to detect its own unavailability and say so,
    rather than to pretend. On a Windows machine with MetaTrader5 installed
    and a terminal logged in, it works; anywhere else :attr:`available` is
    ``False`` and :class:`FallbackProvider` skips straight past it. The
    import is deliberately lazy so merely importing this module never fails.
    """

    name = "mt5"

    # MT5 exposes timeframes as module constants (mt5.TIMEFRAME_M1 etc.);
    # mapped by attribute name so no import is needed to define this.
    _TIMEFRAME_ATTR: Dict[Timeframe, str] = {
        Timeframe.M1: "TIMEFRAME_M1",
        Timeframe.M5: "TIMEFRAME_M5",
        Timeframe.M15: "TIMEFRAME_M15",
        Timeframe.M30: "TIMEFRAME_M30",
        Timeframe.H1: "TIMEFRAME_H1",
        Timeframe.H4: "TIMEFRAME_H4",
        Timeframe.D1: "TIMEFRAME_D1",
    }

    def __init__(self) -> None:
        self._mt5 = None
        self._unavailable_reason: Optional[str] = None
        try:
            import MetaTrader5 as mt5  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001 - any import problem means unavailable
            self._unavailable_reason = (
                f"MetaTrader5 package not importable ({type(exc).__name__}). "
                "It is Windows-only and requires a local MT5 terminal."
            )
            return
        if not mt5.initialize():
            self._unavailable_reason = (
                "MetaTrader5 terminal not running or not logged in "
                f"(last_error={mt5.last_error()})"
            )
            return
        self._mt5 = mt5

    @property
    def available(self) -> bool:
        return self._mt5 is not None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    def _require(self):
        if self._mt5 is None:
            raise MarketDataError(
                f"MT5 unavailable: {self._unavailable_reason}", retryable=False
            )
        return self._mt5

    async def get_quote(self, symbol: str) -> Quote:
        mt5 = self._require()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MarketDataError(f"MT5 has no tick for {symbol}", retryable=False)
        # The one provider that gives a true bid/ask, so pass both through.
        return Quote(
            symbol=symbol,
            price=(tick.bid + tick.ask) / 2 if tick.bid and tick.ask else tick.last,
            timestamp=datetime.now(timezone.utc),
            provider=self.name,
            bid=tick.bid or None,
            ask=tick.ask or None,
        )

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        mt5 = self._require()
        tf = getattr(mt5, self._TIMEFRAME_ATTR[timeframe])
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)
        if rates is None or len(rates) == 0:
            raise MarketDataError(f"MT5 returned no candles for {symbol}", retryable=True)
        candles = tuple(
            Candle(
                timestamp=datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
                open=float(r["open"]), high=float(r["high"]),
                low=float(r["low"]), close=float(r["close"]),
                volume=float(r["tick_volume"]),
            )
            for r in rates
        )
        # MT5 returns oldest-first already; the service re-validates ordering.
        return CandleSeries(symbol=symbol, timeframe=timeframe,
                            provider=self.name, candles=candles)


# ------------------------------------------------------------- HTTP adapters

class _HTTPProvider:
    """Shared plumbing for the REST-based adapters."""

    name = "http"
    _BASE_URL = ""

    def __init__(self, api_key: str, *, timeout: float = 10.0,
                 client: Optional[httpx.AsyncClient] = None) -> None:
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(base_url=self._BASE_URL, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict, headers: Optional[dict] = None) -> dict:
        try:
            response = await self._client.get(path, params=params, headers=headers or {})
        except httpx.TimeoutException as exc:
            raise MarketDataError(f"{self.name} request timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise MarketDataError(f"{self.name} request failed: {exc}", retryable=True) from exc

        if response.status_code != 200:
            raise MarketDataError(
                f"{self.name} returned HTTP {response.status_code}",
                retryable=response.status_code >= 500 or response.status_code == 429,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise MarketDataError(f"{self.name} returned non-JSON", retryable=True) from exc


class OandaProvider(_HTTPProvider):
    """OANDA v20 REST. Gives real bid/ask, unlike the last-price feeds."""

    name = "oanda"
    _BASE_URL = "https://api-fxtrade.oanda.com"

    _GRANULARITY: Dict[Timeframe, str] = {
        Timeframe.M1: "M1", Timeframe.M5: "M5", Timeframe.M15: "M15",
        Timeframe.M30: "M30", Timeframe.H1: "H1", Timeframe.H4: "H4",
        Timeframe.D1: "D",
    }

    def __init__(self, api_key: str, account_id: str, **kwargs) -> None:
        super().__init__(api_key, **kwargs)
        self._account_id = account_id

    @staticmethod
    def _instrument(symbol: str) -> str:
        """OANDA uses BASE_QUOTE (EUR_USD, XAU_USD)."""
        s = symbol.upper().replace("/", "").replace("_", "")
        return f"{s[:3]}_{s[3:]}" if len(s) == 6 else symbol.upper()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def get_quote(self, symbol: str) -> Quote:
        data = await self._get(
            f"/v3/accounts/{self._account_id}/pricing",
            {"instruments": self._instrument(symbol)},
            self._headers(),
        )
        prices = data.get("prices") or []
        if not prices:
            raise MarketDataError(f"OANDA returned no price for {symbol}", retryable=False)
        p = prices[0]
        try:
            bid = float(p["bids"][0]["price"])
            ask = float(p["asks"][0]["price"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise MarketDataError(
                f"OANDA price payload malformed for {symbol}", retryable=False) from exc
        return Quote(symbol=symbol, price=(bid + ask) / 2,
                     timestamp=datetime.now(timezone.utc), provider=self.name,
                     bid=bid, ask=ask)

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        data = await self._get(
            f"/v3/instruments/{self._instrument(symbol)}/candles",
            {"granularity": self._GRANULARITY[timeframe], "count": limit, "price": "M"},
            self._headers(),
        )
        rows = data.get("candles") or []
        if not rows:
            raise MarketDataError(f"OANDA returned no candles for {symbol}", retryable=False)
        candles: List[Candle] = []
        for row in rows:
            if not row.get("complete", True):
                continue  # skip the still-forming candle
            try:
                mid = row["mid"]
                candles.append(Candle(
                    timestamp=datetime.fromisoformat(row["time"].replace("Z", "+00:00")),
                    open=float(mid["o"]), high=float(mid["h"]),
                    low=float(mid["l"]), close=float(mid["c"]),
                    volume=float(row.get("volume", 0)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(
                    f"OANDA candle malformed for {symbol}", retryable=False) from exc
        if not candles:
            raise MarketDataError(f"OANDA returned no complete candles for {symbol}",
                                  retryable=True)
        return CandleSeries(symbol=symbol, timeframe=timeframe,
                            provider=self.name, candles=tuple(candles))


class PolygonProvider(_HTTPProvider):
    """Polygon.io v2 aggregates."""

    name = "polygon"
    _BASE_URL = "https://api.polygon.io"

    # (multiplier, timespan)
    _SPAN: Dict[Timeframe, Tuple[int, str]] = {
        Timeframe.M1: (1, "minute"), Timeframe.M5: (5, "minute"),
        Timeframe.M15: (15, "minute"), Timeframe.M30: (30, "minute"),
        Timeframe.H1: (1, "hour"), Timeframe.H4: (4, "hour"),
        Timeframe.D1: (1, "day"),
    }

    @staticmethod
    def _ticker(symbol: str) -> str:
        """Polygon prefixes forex/metal pairs with C: (C:XAUUSD)."""
        s = symbol.upper().replace("/", "")
        return f"C:{s}" if len(s) == 6 else s

    async def get_quote(self, symbol: str) -> Quote:
        data = await self._get(f"/v2/last/nbbo/{self._ticker(symbol)}",
                               {"apiKey": self._api_key})
        results = data.get("results") or {}
        bid, ask = results.get("p"), results.get("P")
        if bid is None and ask is None:
            raise MarketDataError(f"Polygon returned no quote for {symbol}", retryable=False)
        bid_f = float(bid) if bid is not None else None
        ask_f = float(ask) if ask is not None else None
        price = ((bid_f + ask_f) / 2) if (bid_f and ask_f) else (bid_f or ask_f)
        return Quote(symbol=symbol, price=float(price),
                     timestamp=datetime.now(timezone.utc), provider=self.name,
                     bid=bid_f, ask=ask_f)

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        mult, span = self._SPAN[timeframe]
        data = await self._get(
            f"/v2/aggs/ticker/{self._ticker(symbol)}/range/{mult}/{span}/"
            f"1970-01-01/2100-01-01",
            {"apiKey": self._api_key, "limit": limit, "sort": "desc"},
        )
        rows = data.get("results") or []
        if not rows:
            raise MarketDataError(f"Polygon returned no candles for {symbol}", retryable=False)
        candles: List[Candle] = []
        for row in rows:
            try:
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc),
                    open=float(row["o"]), high=float(row["h"]),
                    low=float(row["l"]), close=float(row["c"]),
                    volume=float(row.get("v", 0)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(
                    f"Polygon candle malformed for {symbol}", retryable=False) from exc
        candles.reverse()   # sort=desc -> oldest-first contract
        return CandleSeries(symbol=symbol, timeframe=timeframe,
                            provider=self.name, candles=tuple(candles))


class AlphaVantageProvider(_HTTPProvider):
    """Alpha Vantage FX endpoints. Last resort: heavily rate-limited."""

    name = "alpha_vantage"
    _BASE_URL = "https://www.alphavantage.co"

    _INTERVAL: Dict[Timeframe, str] = {
        Timeframe.M1: "1min", Timeframe.M5: "5min", Timeframe.M15: "15min",
        Timeframe.M30: "30min", Timeframe.H1: "60min",
    }

    @staticmethod
    def _pair(symbol: str) -> Tuple[str, str]:
        s = symbol.upper().replace("/", "")
        if len(s) != 6:
            raise MarketDataError(
                f"Alpha Vantage needs a 6-character FX pair, got {symbol!r}",
                retryable=False)
        return s[:3], s[3:]

    def _check_notes(self, data: dict) -> None:
        # Alpha Vantage signals rate limiting with a 200 + "Note"/"Information".
        if "Note" in data or "Information" in data:
            raise MarketDataError(
                f"Alpha Vantage rate limited: {data.get('Note') or data.get('Information')}",
                retryable=True)
        if "Error Message" in data:
            raise MarketDataError(f"Alpha Vantage error: {data['Error Message']}",
                                  retryable=False)

    async def get_quote(self, symbol: str) -> Quote:
        base, quote = self._pair(symbol)
        data = await self._get("/query", {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": base, "to_currency": quote, "apikey": self._api_key,
        })
        self._check_notes(data)
        rate = data.get("Realtime Currency Exchange Rate") or {}
        try:
            price = float(rate["5. Exchange Rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError(f"Alpha Vantage payload malformed for {symbol}",
                                  retryable=False) from exc
        bid = rate.get("8. Bid Price")
        ask = rate.get("9. Ask Price")
        return Quote(symbol=symbol, price=price, timestamp=datetime.now(timezone.utc),
                     provider=self.name,
                     bid=float(bid) if bid else None,
                     ask=float(ask) if ask else None)

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        base, quote = self._pair(symbol)
        if timeframe not in self._INTERVAL:
            # H4/D1 use a different function with a different payload shape.
            if timeframe is Timeframe.D1:
                data = await self._get("/query", {
                    "function": "FX_DAILY", "from_symbol": base,
                    "to_symbol": quote, "apikey": self._api_key,
                })
                series_key = "Time Series FX (Daily)"
            else:
                raise MarketDataError(
                    f"Alpha Vantage does not offer {timeframe.value} for FX",
                    retryable=False)
        else:
            data = await self._get("/query", {
                "function": "FX_INTRADAY", "from_symbol": base, "to_symbol": quote,
                "interval": self._INTERVAL[timeframe], "apikey": self._api_key,
            })
            series_key = f"Time Series FX ({self._INTERVAL[timeframe]})"

        self._check_notes(data)
        series = data.get(series_key) or {}
        if not series:
            raise MarketDataError(f"Alpha Vantage returned no candles for {symbol}",
                                  retryable=False)
        candles: List[Candle] = []
        for stamp, row in sorted(series.items())[-limit:]:
            try:
                candles.append(Candle(
                    timestamp=datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc),
                    open=float(row["1. open"]), high=float(row["2. high"]),
                    low=float(row["3. low"]), close=float(row["4. close"]),
                    volume=None,   # Alpha Vantage FX carries no volume
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(f"Alpha Vantage candle malformed for {symbol}",
                                      retryable=False) from exc
        return CandleSeries(symbol=symbol, timeframe=timeframe,
                            provider=self.name, candles=tuple(candles))


# ------------------------------------------------------------ fallback chain

class FallbackProvider:
    """Tries each provider in order; the first to answer wins.

    Implements the same Protocol as its members, so ``MarketDataService``
    cannot tell it apart from a single provider — the chain is invisible to
    every caller.

    A provider that raises is logged and skipped. Only when *all* of them
    fail does this raise, and the raised error carries the last failure so
    the reason is never lost. A non-retryable failure from one provider
    (e.g. "symbol not found") is still worth trying the next for: a symbol
    absent from one venue may exist on another.
    """

    name = "fallback"

    def __init__(self, providers: Sequence[MarketDataProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self._providers = tuple(providers)

    @property
    def provider_names(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self._providers)

    async def _try(self, method: str, *args):
        last: Optional[MarketDataError] = None
        for provider in self._providers:
            # Skip a provider that already knows it can't work (e.g. MT5 on
            # Linux) without paying for a call that is certain to fail.
            if getattr(provider, "available", True) is False:
                log.debug("market_provider_skipped provider=%s reason=%s",
                          provider.name, getattr(provider, "unavailable_reason", "unavailable"))
                continue
            try:
                return await getattr(provider, method)(*args)
            except MarketDataError as exc:
                last = exc
                log.warning("market_provider_failed provider=%s method=%s error=%s",
                            provider.name, method, exc)
            except Exception as exc:  # noqa: BLE001 - a broken adapter must not end the chain
                last = MarketDataError(
                    f"{provider.name} raised {type(exc).__name__}: {exc}", retryable=True)
                log.exception("market_provider_crashed provider=%s method=%s",
                              provider.name, method)
        raise MarketDataError(
            f"all providers failed ({', '.join(self.provider_names)}); "
            f"last error: {last}",
            retryable=bool(last and last.retryable),
        )

    async def get_quote(self, symbol: str) -> Quote:
        return await self._try("get_quote", symbol)

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        return await self._try("get_candles", symbol, timeframe, limit)
