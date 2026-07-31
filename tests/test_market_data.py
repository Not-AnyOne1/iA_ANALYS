"""Unit tests for market_data.py (RFC-004).

Two layers are tested separately:

- TwelveDataProvider: HTTP is stubbed via httpx.MockTransport (no real
  network/API key needed) — covers Twelve Data's own quirks (symbol
  normalization, interval mapping, its two failure response shapes, newest
  -> oldest reordering).
- MarketDataService: exercised against a hand-written stub provider (not
  TwelveDataProvider) — covers the provider-agnostic contract every future
  engine depends on: caching, centralised retry, and validation that
  rejects out-of-order/inconsistent/mismatched data regardless of which
  provider produced it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from market_data import (
    Candle,
    CandleSeries,
    MarketDataError,
    MarketDataService,
    Quote,
    Timeframe,
    TwelveDataProvider,
)


# --------------------------------------------------------------------------- helpers

def _ts(offset_minutes: int = 0) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes)


def _candle(offset_minutes: int, *, low=100.0, high=110.0, open_=105.0, close=106.0) -> Candle:
    return Candle(timestamp=_ts(offset_minutes), open=open_, high=high, low=low, close=close, volume=10.0)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """MarketDataService's retry backoff sleeps between attempts; keep tests instant."""
    monkeypatch.setattr(asyncio, "sleep", __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock())


def _client_with_handler(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=TwelveDataProvider._BASE_URL)


# --------------------------------------------------------------------------- TwelveDataProvider: symbol normalization

def test_normalizes_bare_six_letter_currency_symbol():
    assert TwelveDataProvider._normalize_symbol("XAUUSD") == "XAU/USD"
    assert TwelveDataProvider._normalize_symbol("EURUSD") == "EUR/USD"


def test_leaves_already_slashed_symbol_unchanged():
    assert TwelveDataProvider._normalize_symbol("XAU/USD") == "XAU/USD"


def test_leaves_non_currency_style_symbol_unchanged():
    assert TwelveDataProvider._normalize_symbol("BTCUSDT") == "BTCUSDT"
    assert TwelveDataProvider._normalize_symbol("AAPL") == "AAPL"


def test_interval_map_covers_every_timeframe():
    for tf in Timeframe:
        assert tf in TwelveDataProvider._INTERVAL_MAP


# --------------------------------------------------------------------------- TwelveDataProvider: get_quote

def test_get_quote_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbol"] == "XAU/USD"
        return httpx.Response(200, json={"price": "2400.50"})

    provider = TwelveDataProvider("key", client=_client_with_handler(handler))
    quote = asyncio.run(provider.get_quote("XAUUSD"))

    assert quote.symbol == "XAUUSD"
    assert quote.price == 2400.50
    assert quote.provider == "twelve_data"


def test_get_quote_missing_price_field_raises():
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(200, json={})
    ))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("XAUUSD"))
    assert exc_info.value.retryable is False


def test_get_quote_http_500_is_retryable():
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(500, text="internal error")
    ))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("XAUUSD"))
    assert exc_info.value.retryable is True


def test_get_quote_http_400_is_not_retryable():
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(400, text="bad request")
    ))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("XAUUSD"))
    assert exc_info.value.retryable is False


def test_get_quote_200_with_error_body_unknown_symbol_is_not_retryable():
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(200, json={"code": 400, "message": "symbol not found", "status": "error"})
    ))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("NOTREAL"))
    assert exc_info.value.retryable is False
    assert "symbol not found" in str(exc_info.value)


def test_get_quote_200_with_error_body_server_error_is_retryable():
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(200, json={"code": 503, "message": "upstream down", "status": "error"})
    ))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("XAUUSD"))
    assert exc_info.value.retryable is True


def test_get_quote_timeout_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    provider = TwelveDataProvider("key", client=_client_with_handler(handler))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("XAUUSD"))
    assert exc_info.value.retryable is True


def test_get_quote_non_json_response_is_retryable():
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(200, text="not json")
    ))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("XAUUSD"))
    assert exc_info.value.retryable is True


# --------------------------------------------------------------------------- TwelveDataProvider: get_candles

def test_get_candles_reorders_newest_first_response_to_oldest_first():
    # Twelve Data returns values newest-first.
    values = [
        {"datetime": "2026-01-01 00:02:00", "open": "3", "high": "4", "low": "2", "close": "3.5", "volume": "10"},
        {"datetime": "2026-01-01 00:01:00", "open": "2", "high": "3", "low": "1", "close": "2.5", "volume": "10"},
        {"datetime": "2026-01-01 00:00:00", "open": "1", "high": "2", "low": "0", "close": "1.5", "volume": "10"},
    ]
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(200, json={"values": values})
    ))

    series = asyncio.run(provider.get_candles("XAUUSD", Timeframe.M1, limit=3))

    assert [c.timestamp for c in series.candles] == sorted(c.timestamp for c in series.candles)
    assert series.candles[0].open == 1.0
    assert series.candles[-1].open == 3.0
    assert series.provider == "twelve_data"
    assert series.timeframe == Timeframe.M1


def test_get_candles_empty_values_raises():
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(200, json={"values": []})
    ))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_candles("XAUUSD", Timeframe.M1, limit=3))
    assert exc_info.value.retryable is False


def test_get_candles_malformed_row_raises():
    values = [{"datetime": "2026-01-01 00:00:00", "open": "1", "high": "2", "low": "0"}]  # missing "close"
    provider = TwelveDataProvider("key", client=_client_with_handler(
        lambda r: httpx.Response(200, json={"values": values})
    ))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_candles("XAUUSD", Timeframe.M1, limit=1))
    assert exc_info.value.retryable is False


# --------------------------------------------------------------------------- MarketDataService: stub provider

class _StubProvider:
    name = "stub"

    def __init__(self, *, quote=None, quote_error=None, series=None, series_error=None):
        self._quote = quote
        self._quote_error = quote_error
        self._series = series
        self._series_error = series_error
        self.quote_calls = 0
        self.candle_calls = 0

    async def get_quote(self, symbol: str) -> Quote:
        self.quote_calls += 1
        if self._quote_error is not None:
            raise self._quote_error
        return self._quote

    async def get_candles(self, symbol: str, timeframe: Timeframe, limit: int) -> CandleSeries:
        self.candle_calls += 1
        if self._series_error is not None:
            raise self._series_error
        return self._series


def _ok_quote(symbol="XAUUSD", price=2400.0) -> Quote:
    return Quote(symbol=symbol, price=price, timestamp=_ts(), provider="stub")


def _ok_series(symbol="XAUUSD", timeframe=Timeframe.M1, candles=None) -> CandleSeries:
    candles = candles if candles is not None else (_candle(0), _candle(1), _candle(2))
    return CandleSeries(symbol=symbol, timeframe=timeframe, provider="stub", candles=tuple(candles))


def test_service_disabled_without_provider_raises_immediately():
    service = MarketDataService(None)
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(service.get_quote("XAUUSD"))
    assert exc_info.value.retryable is False

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_candles("XAUUSD", Timeframe.M1))


def test_service_provider_name_reflects_attached_provider():
    service = MarketDataService(_StubProvider(quote=_ok_quote()))
    assert service.provider_name == "stub"
    assert MarketDataService(None).provider_name == "none"


def test_service_get_quote_returns_provider_result():
    provider = _StubProvider(quote=_ok_quote(symbol="XAUUSD", price=2400.0))
    service = MarketDataService(provider)

    quote = asyncio.run(service.get_quote("xauusd"))  # lowercase in, normalized to match provider's symbol

    assert quote.price == 2400.0


def test_service_get_quote_caches_within_ttl():
    provider = _StubProvider(quote=_ok_quote())
    service = MarketDataService(provider, cache_ttl=60.0)

    asyncio.run(service.get_quote("XAUUSD"))
    asyncio.run(service.get_quote("XAUUSD"))

    assert provider.quote_calls == 1


def test_service_get_quote_refetches_after_ttl_expiry():
    provider = _StubProvider(quote=_ok_quote())
    service = MarketDataService(provider, cache_ttl=60.0)

    asyncio.run(service.get_quote("XAUUSD"))
    # Force expiry deterministically rather than sleeping in the test.
    for entry in service._quote_cache.values():
        entry.expires_at = 0.0
    asyncio.run(service.get_quote("XAUUSD"))

    assert provider.quote_calls == 2


def test_service_get_quote_rejects_zero_or_negative_price():
    provider = _StubProvider(quote=_ok_quote(price=0.0))
    service = MarketDataService(provider)

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_quote("XAUUSD"))


def test_service_get_quote_rejects_symbol_mismatch():
    provider = _StubProvider(quote=_ok_quote(symbol="WRONG"))
    service = MarketDataService(provider)

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_quote("XAUUSD"))


def test_service_get_candles_returns_oldest_first_series():
    provider = _StubProvider(series=_ok_series(candles=[_candle(0), _candle(1)]))
    service = MarketDataService(provider)

    series = asyncio.run(service.get_candles("XAUUSD", Timeframe.M1))

    assert [c.timestamp for c in series.candles] == [_ts(0), _ts(1)]


def test_service_get_candles_caches_within_ttl():
    provider = _StubProvider(series=_ok_series())
    service = MarketDataService(provider, cache_ttl=60.0)

    asyncio.run(service.get_candles("XAUUSD", Timeframe.M1, limit=3))
    asyncio.run(service.get_candles("XAUUSD", Timeframe.M1, limit=3))

    assert provider.candle_calls == 1


def test_service_get_candles_different_timeframe_is_a_separate_cache_entry():
    provider = _StubProvider(series=_ok_series())
    service = MarketDataService(provider, cache_ttl=60.0)

    asyncio.run(service.get_candles("XAUUSD", Timeframe.M1, limit=3))
    provider._series = _ok_series(timeframe=Timeframe.H1)
    asyncio.run(service.get_candles("XAUUSD", Timeframe.H1, limit=3))

    assert provider.candle_calls == 2


def test_service_rejects_out_of_order_candles_from_a_buggy_provider():
    # Timestamps deliberately descending — a provider bug the service must
    # catch regardless of which provider produced it.
    buggy_series = _ok_series(candles=[_candle(2), _candle(1), _candle(0)])
    provider = _StubProvider(series=buggy_series)
    service = MarketDataService(provider)

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_candles("XAUUSD", Timeframe.M1))


def test_service_rejects_inconsistent_ohlc_bar():
    bad_candle = _candle(0, low=200.0, high=100.0)  # low > high
    provider = _StubProvider(series=_ok_series(candles=[bad_candle]))
    service = MarketDataService(provider)

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_candles("XAUUSD", Timeframe.M1))


def test_service_rejects_candle_where_low_exceeds_high():
    bad_candle = _candle(0, low=100.0, high=90.0, open_=95.0, close=95.0)
    provider = _StubProvider(series=_ok_series(candles=[bad_candle]))
    service = MarketDataService(provider)

    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(service.get_candles("XAUUSD", Timeframe.M1))

    assert exc_info.value.retryable is False


def test_service_rejects_empty_candle_series():
    provider = _StubProvider(series=_ok_series(candles=[]))
    service = MarketDataService(provider)

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_candles("XAUUSD", Timeframe.M1))


def test_service_rejects_series_for_wrong_symbol_or_timeframe():
    provider = _StubProvider(series=_ok_series(symbol="WRONG"))
    service = MarketDataService(provider)

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_candles("XAUUSD", Timeframe.M1))


def test_service_rejects_non_positive_limit():
    service = MarketDataService(_StubProvider())
    with pytest.raises(MarketDataError):
        asyncio.run(service.get_candles("XAUUSD", Timeframe.M1, limit=0))


# --------------------------------------------------------------------------- MarketDataService: retry

def test_service_retries_retryable_error_then_succeeds():
    provider = _StubProvider()
    attempts = {"n": 0}

    async def flaky_get_quote(symbol):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise MarketDataError("transient", retryable=True)
        return _ok_quote()

    provider.get_quote = flaky_get_quote
    service = MarketDataService(provider, max_retries=3, retry_base_delay=0.01)

    quote = asyncio.run(service.get_quote("XAUUSD"))

    assert quote.price == 2400.0
    assert attempts["n"] == 3


def test_service_does_not_retry_non_retryable_error():
    provider = _StubProvider(quote_error=MarketDataError("permanent", retryable=False))
    service = MarketDataService(provider, max_retries=3, retry_base_delay=0.01)

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_quote("XAUUSD"))

    assert provider.quote_calls == 1


def test_service_retries_exhausted_raises_final_error():
    provider = _StubProvider(quote_error=MarketDataError("always fails", retryable=True))
    service = MarketDataService(provider, max_retries=2, retry_base_delay=0.01)

    with pytest.raises(MarketDataError):
        asyncio.run(service.get_quote("XAUUSD"))

    assert provider.quote_calls == 3  # 1 + max_retries


# --------------------------------------------------------------- cache bounds (regression)

def test_cache_evicts_expired_entries():
    """Expired entries used to linger forever — nothing ever removed them."""
    provider = _StubProvider()
    service = MarketDataService(provider, cache_ttl=60.0)

    provider._quote = Quote(symbol="XAUUSD", price=1.0, timestamp=_ts(), provider="stub")
    asyncio.run(service.get_quote("XAUUSD"))
    for entry in service._quote_cache.values():
        entry.expires_at = 0.0                      # force expiry

    provider._quote = Quote(symbol="EURUSD", price=1.0, timestamp=_ts(), provider="stub")
    asyncio.run(service.get_quote("EURUSD"))        # any write triggers a sweep

    assert "XAUUSD" not in service._quote_cache
    assert "EURUSD" in service._quote_cache


def test_quote_cache_is_bounded():
    """One permanent entry per distinct symbol ever seen was an unbounded
    leak for a process meant to run for weeks."""
    provider = _StubProvider()
    service = MarketDataService(provider, cache_ttl=3600.0, max_cache_entries=10)

    async def run():
        for i in range(200):
            provider._quote = Quote(symbol=f"SYM{i}", price=1.0,
                                    timestamp=_ts(), provider="stub")
            await service.get_quote(f"SYM{i}")

    asyncio.run(run())

    assert len(service._quote_cache) <= 10


def test_candle_cache_is_bounded():
    provider = _StubProvider()
    service = MarketDataService(provider, cache_ttl=3600.0, max_cache_entries=10)

    async def run():
        for i in range(200):
            provider._series = _ok_series(symbol=f"SYM{i}")
            await service.get_candles(f"SYM{i}", Timeframe.M1, limit=3)

    asyncio.run(run())

    assert len(service._candle_cache) <= 10


def test_bounded_cache_evicts_oldest_first():
    provider = _StubProvider()
    service = MarketDataService(provider, cache_ttl=3600.0, max_cache_entries=3)

    async def run():
        for i in range(5):
            provider._quote = Quote(symbol=f"SYM{i}", price=1.0,
                                    timestamp=_ts(), provider="stub")
            await service.get_quote(f"SYM{i}")

    asyncio.run(run())

    assert "SYM0" not in service._quote_cache   # oldest gone
    assert "SYM4" in service._quote_cache       # newest kept


def test_caching_still_works_within_the_bound():
    provider = _StubProvider(quote=_ok_quote())
    service = MarketDataService(provider, cache_ttl=3600.0, max_cache_entries=10)

    asyncio.run(service.get_quote("XAUUSD"))
    asyncio.run(service.get_quote("XAUUSD"))

    assert provider.quote_calls == 1   # eviction must not defeat caching
