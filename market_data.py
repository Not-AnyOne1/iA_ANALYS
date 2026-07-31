"""Pluggable market data layer (RFC-004).

Provides live quotes and OHLC candle series through a provider-agnostic
service. Twelve Data is the initial (and only) implementation, but every
detail specific to it — REST endpoints, its own interval strings, its
symbol format ("XAU/USD" vs this project's "XAUUSD"), its two distinct
failure response shapes — is confined to :class:`TwelveDataProvider`.
:class:`MarketDataService` and everything it returns (:class:`Quote`,
:class:`Candle`, :class:`CandleSeries`, :class:`Timeframe`,
:class:`MarketDataError`) know nothing about Twelve Data. A future provider
implements :class:`MarketDataProvider` and swaps in behind the same
service with no change required in any caller.

Not wired into the analysis pipeline yet (RFC-004 scope) — this is the
foundation the Structure/SMC/Scoring/Claude Decision engines will call
directly in later RFCs, via exactly two methods:
:meth:`MarketDataService.get_quote` and
:meth:`MarketDataService.get_candles`.

Contracts every caller (current and future) can rely on:

- :attr:`CandleSeries.candles` is always ordered oldest to newest,
  regardless of what order the underlying provider's API answers in.
- Both service methods either return complete, valid data or raise
  :class:`MarketDataError` — never a partial/empty/fabricated result.
- Retries for transient failures happen once, centrally, in
  :class:`MarketDataService` — so every provider gets that resilience for
  free instead of re-implementing it.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

import httpx

log = logging.getLogger(__name__)

T = TypeVar("T")


class Timeframe(str, Enum):
    """Provider-neutral candle timeframe labels.

    Each provider maps these to its own interval strings internally (see
    ``TwelveDataProvider._INTERVAL_MAP``) — callers never see a
    provider-specific value, which is what keeps swapping providers a no-op
    for every caller.
    """

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


@dataclass(frozen=True)
class Quote:
    """A single current-price snapshot.

    ``bid``/``ask`` are optional because not every provider exposes them:
    Twelve Data's /price endpoint returns a last price only. They default to
    ``None`` rather than being derived from ``price``, since a fabricated
    spread would feed straight into the risk engine's spread rule.
    """

    symbol: str
    price: float
    timestamp: datetime
    provider: str
    bid: Optional[float] = None
    ask: Optional[float] = None

    @property
    def spread(self) -> Optional[float]:
        """Ask minus bid, or ``None`` when the provider gives neither."""
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def spread_percent(self) -> Optional[float]:
        spread = self.spread
        if spread is None or not self.price:
            return None
        return spread / self.price * 100.0


@dataclass(frozen=True)
class Candle:
    """One OHLC bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


@dataclass(frozen=True)
class CandleSeries:
    """A run of candles for one symbol/timeframe.

    ``candles`` is guaranteed oldest-to-newest by the time it reaches a
    caller — :class:`MarketDataService` enforces this itself (see
    :meth:`MarketDataService._validate_candle_series`) rather than trusting
    each provider to get it right, so a future provider that returns the
    wrong order fails loudly instead of silently breaking every consumer.
    """

    symbol: str
    timeframe: Timeframe
    provider: str
    candles: Tuple[Candle, ...]


class MarketDataError(RuntimeError):
    """Raised whenever valid, complete data cannot be returned.

    Mirrors ``claude_client.AnalysisError``'s shape deliberately: a message
    plus a ``retryable`` flag, so callers already familiar with that
    convention don't need to learn a new one for this layer. Never
    provider-specific in the sense that matters — callers only need
    ``retryable`` to decide how to react, never the message text.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class MarketDataProvider(Protocol):
    """What a market data provider must implement.

    Every future provider only needs to satisfy this shape —
    :class:`MarketDataService` and everything above it are written entirely
    against this Protocol, never against a concrete provider class.
    """

    name: str

    async def get_quote(self, symbol: str) -> Quote: ...

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries: ...


# Hard ceiling on cached quote/candle entries, per cache. Well above the
# handful of symbols a single monitored group realistically trades, but low
# enough that an unbounded stream of distinct symbols can't grow memory
# without limit over weeks of uptime.
_MAX_CACHE_ENTRIES = 512


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: object, expires_at: float) -> None:
        self.value = value
        self.expires_at = expires_at


class MarketDataService:
    """Provider-agnostic facade every future engine should call directly.

    Construct with ``provider=None`` to represent "not configured" (e.g. no
    API key set) — both methods then raise :class:`MarketDataError`
    immediately, no network attempted, the same pattern
    ``TelegramBot``/``BotSettings.enabled`` already uses.
    """

    def __init__(
        self,
        provider: Optional[MarketDataProvider],
        *,
        cache_ttl: float = 30.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        max_cache_entries: int = _MAX_CACHE_ENTRIES,
    ) -> None:
        self._provider = provider
        self._cache_ttl = cache_ttl
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._max_cache_entries = max_cache_entries
        self._quote_cache: "OrderedDict[str, _CacheEntry]" = OrderedDict()
        self._candle_cache: "OrderedDict[Tuple[str, Timeframe, int], _CacheEntry]" = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def provider_name(self) -> str:
        """Which provider is currently answering — for status/logging only;
        callers must never branch on this value."""
        return self._provider.name if self._provider is not None else "none"

    async def get_quote(self, symbol: str) -> Quote:
        if self._provider is None:
            raise MarketDataError("Market data is not configured", retryable=False)

        symbol = symbol.strip().upper()
        cached = await self._get_cached(self._quote_cache, symbol)
        if cached is not None:
            return cached

        provider = self._provider
        quote = await self._call_with_retry(lambda: provider.get_quote(symbol))
        self._validate_quote(quote, expected_symbol=symbol)

        await self._set_cached(self._quote_cache, symbol, quote)
        return quote

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int = 100) -> CandleSeries:
        if self._provider is None:
            raise MarketDataError("Market data is not configured", retryable=False)
        if limit < 1:
            raise MarketDataError(f"limit must be >= 1, got {limit}", retryable=False)

        symbol = symbol.strip().upper()
        key = (symbol, timeframe, limit)
        cached = await self._get_cached(self._candle_cache, key)
        if cached is not None:
            return cached

        provider = self._provider
        series = await self._call_with_retry(lambda: provider.get_candles(symbol, timeframe, limit))
        self._validate_candle_series(series, expected_symbol=symbol, expected_timeframe=timeframe)

        await self._set_cached(self._candle_cache, key, series)
        return series

    # ------------------------------------------------------------- retry

    async def _call_with_retry(self, call: Callable[[], Awaitable[T]]) -> T:
        """Retry a provider call on transient (``retryable``) failures only.

        Centralised here rather than in each provider so every current and
        future provider gets the same resilience for free.
        """
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return await call()
            except MarketDataError as exc:
                if not exc.retryable or attempt == attempts:
                    raise
                delay = self._retry_base_delay * (2 ** (attempt - 1))
                log.warning(
                    "market_data_retry attempt=%d/%d delay=%.1fs error=%s",
                    attempt, attempts, delay, exc,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # loop always returns or raises

    # ------------------------------------------------------------- caching

    async def _get_cached(self, cache: Dict, key: object) -> Optional[object]:
        async with self._lock:
            entry = cache.get(key)
            if entry is not None and entry.expires_at > time.monotonic():
                return entry.value
            return None

    async def _set_cached(self, cache: Dict, key: object, value: object) -> None:
        async with self._lock:
            cache[key] = _CacheEntry(value, time.monotonic() + self._cache_ttl)
            cache.move_to_end(key)
            self._evict(cache)

    def _evict(self, cache: "OrderedDict") -> None:
        """Drop expired entries, then oldest-first until under the cap.

        Without this the caches grew forever: nothing ever removed an entry,
        not even an expired one, so a long-running process accumulated one
        permanent entry per distinct symbol/timeframe/limit ever requested
        (measured: 20,000 symbols -> 20,000 retained entries, none evicted).
        Caller already holds ``self._lock``.
        """
        now = time.monotonic()
        for key in [k for k, entry in cache.items() if entry.expires_at <= now]:
            del cache[key]
        while len(cache) > self._max_cache_entries:
            cache.popitem(last=False)  # oldest insertion first

    # ---------------------------------------------------------- validation

    @staticmethod
    def _validate_quote(quote: Quote, *, expected_symbol: str) -> None:
        if quote.symbol != expected_symbol:
            raise MarketDataError(
                f"Provider returned a quote for {quote.symbol!r} when {expected_symbol!r} was requested",
                retryable=False,
            )
        if quote.price <= 0:
            raise MarketDataError(
                f"Provider returned an invalid price for {expected_symbol}", retryable=False
            )

    @staticmethod
    def _validate_candle_series(
        series: CandleSeries, *, expected_symbol: str, expected_timeframe: Timeframe
    ) -> None:
        if series.symbol != expected_symbol or series.timeframe != expected_timeframe:
            raise MarketDataError(
                f"Provider returned candles for {series.symbol!r}/{series.timeframe!r} when "
                f"{expected_symbol!r}/{expected_timeframe!r} was requested",
                retryable=False,
            )
        if not series.candles:
            raise MarketDataError(f"Provider returned no candles for {expected_symbol}", retryable=False)

        previous_ts: Optional[datetime] = None
        for candle in series.candles:
            if candle.low > candle.high:
                raise MarketDataError(
                    "invalid candle: low greater than high",
                    retryable=False,
                )
            if not (candle.low <= candle.open <= candle.high and candle.low <= candle.close <= candle.high):
                raise MarketDataError(
                    f"Provider returned an inconsistent OHLC bar for {expected_symbol} at {candle.timestamp}",
                    retryable=False,
                )
            if previous_ts is not None and candle.timestamp <= previous_ts:
                raise MarketDataError(
                    f"Provider returned candles out of order for {expected_symbol}", retryable=False,
                )
            previous_ts = candle.timestamp


class TwelveDataProvider:
    """Twelve Data REST API implementation of :class:`MarketDataProvider`.

    Every Twelve-Data-specific detail is confined to this class: its base
    URL, its own interval strings, its symbol format, and its two distinct
    failure shapes (non-200 HTTP, and 200 OK with an error body). None of
    this ever surfaces past :class:`MarketDataService`'s callers. This
    class does its own bounded-nothing single attempt per call — retry
    policy lives in :class:`MarketDataService`, not here.
    """

    name = "twelve_data"

    _BASE_URL = "https://api.twelvedata.com"

    _INTERVAL_MAP: Dict[Timeframe, str] = {
        Timeframe.M1: "1min",
        Timeframe.M5: "5min",
        Timeframe.M15: "15min",
        Timeframe.M30: "30min",
        Timeframe.H1: "1h",
        Timeframe.H4: "4h",
        Timeframe.D1: "1day",
    }

    # Bare 6-letter currency/metal-style symbols (XAUUSD, EURUSD, GBPJPY, ...)
    # need a "/" inserted for Twelve Data's forex/metals format. Anything
    # else (BTCUSDT, AAPL, already-slashed symbols) is passed through
    # unchanged; an unresolvable symbol simply surfaces as a normal
    # "symbol not found" MarketDataError rather than a special case.
    _CURRENCY_STYLE_RE = re.compile(r"^[A-Z]{6}$")

    def __init__(self, api_key: str, *, timeout: float = 10.0, client: Optional[httpx.AsyncClient] = None) -> None:
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(base_url=self._BASE_URL, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    @classmethod
    def _normalize_symbol(cls, symbol: str) -> str:
        if "/" in symbol:
            return symbol
        if cls._CURRENCY_STYLE_RE.match(symbol):
            return f"{symbol[:3]}/{symbol[3:]}"
        return symbol

    async def get_quote(self, symbol: str) -> Quote:
        data = await self._request("/price", {"symbol": self._normalize_symbol(symbol)})

        price_raw = data.get("price")
        if price_raw is None:
            raise MarketDataError(f"Twelve Data returned no price for {symbol}", retryable=False)
        try:
            price = float(price_raw)
        except (TypeError, ValueError) as exc:
            raise MarketDataError(
                f"Twelve Data returned a non-numeric price for {symbol}", retryable=False
            ) from exc

        return Quote(symbol=symbol, price=price, timestamp=datetime.now(timezone.utc), provider=self.name)

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        interval = self._INTERVAL_MAP[timeframe]
        data = await self._request(
            "/time_series",
            {"symbol": self._normalize_symbol(symbol), "interval": interval, "outputsize": limit},
        )

        values = data.get("values")
        if not isinstance(values, list) or not values:
            raise MarketDataError(f"Twelve Data returned no candles for {symbol}", retryable=False)

        candles: List[Candle] = []
        for row in values:
            try:
                # Twelve Data timestamps carry no explicit offset; treated as
                # UTC for internal consistency (documented assumption, not a
                # guarantee from the API).
                ts = datetime.fromisoformat(row["datetime"]).replace(tzinfo=timezone.utc)
                candles.append(
                    Candle(
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]) if row.get("volume") is not None else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(
                    f"Twelve Data returned a malformed candle for {symbol}", retryable=False
                ) from exc

        # Twelve Data's time_series returns newest-first; this module's
        # public contract (CandleSeries.candles) is always oldest-first.
        candles.reverse()

        return CandleSeries(symbol=symbol, timeframe=timeframe, provider=self.name, candles=tuple(candles))

    async def _request(self, path: str, params: dict) -> dict:
        query = {**params, "apikey": self._api_key}
        try:
            response = await self._client.get(path, params=query)
        except httpx.TimeoutException as exc:
            raise MarketDataError(f"Twelve Data request timed out: {exc}", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise MarketDataError(f"Twelve Data request failed: {exc}", retryable=True) from exc

        if response.status_code != 200:
            raise MarketDataError(
                f"Twelve Data returned HTTP {response.status_code}",
                retryable=response.status_code >= 500 or response.status_code == 429,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise MarketDataError("Twelve Data returned a non-JSON response", retryable=True) from exc

        # Twelve Data's second failure shape: 200 OK with an error body,
        # e.g. {"code": 400, "message": "...", "status": "error"}.
        if isinstance(data, dict) and data.get("status") == "error":
            code = data.get("code", 0)
            raise MarketDataError(
                f"Twelve Data error {code}: {data.get('message', 'unknown error')}",
                retryable=code >= 500,
            )

        return data
