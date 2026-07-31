"""Unit tests for market_providers.py.

HTTP is stubbed with httpx.MockTransport — same pattern as the Twelve Data
tests — so response parsing, symbol mapping and error classification are all
verified without touching a live vendor endpoint.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from market_data import CandleSeries, MarketDataError, Quote, Timeframe
from market_providers import (
    AlphaVantageProvider,
    FallbackProvider,
    MT5Provider,
    OandaProvider,
    PolygonProvider,
)


def _client(handler, base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)


# --------------------------------------------------------------------------- MT5

def test_mt5_reports_itself_unavailable_without_the_package():
    """MetaTrader5 is Windows-only and needs a local terminal; on this
    machine (and on the Linux deployment) it must say so, not pretend."""
    provider = MT5Provider()
    assert provider.available is False
    assert provider.unavailable_reason
    assert "Windows" in provider.unavailable_reason or "not importable" in provider.unavailable_reason


def test_mt5_calls_raise_a_clear_error_when_unavailable():
    provider = MT5Provider()
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("XAUUSD"))
    assert "MT5 unavailable" in str(exc_info.value)
    assert exc_info.value.retryable is False


def test_mt5_covers_every_timeframe():
    for tf in Timeframe:
        assert tf in MT5Provider._TIMEFRAME_ATTR


# --------------------------------------------------------------------------- OANDA

def test_oanda_maps_symbols_to_instruments():
    assert OandaProvider._instrument("XAUUSD") == "XAU_USD"
    assert OandaProvider._instrument("EUR/USD") == "EUR_USD"


def test_oanda_quote_uses_real_bid_and_ask():
    def handler(request):
        return httpx.Response(200, json={"prices": [
            {"bids": [{"price": "2400.10"}], "asks": [{"price": "2400.50"}]}
        ]})

    provider = OandaProvider("key", "acct",
                             client=_client(handler, OandaProvider._BASE_URL))
    quote = asyncio.run(provider.get_quote("XAUUSD"))

    assert quote.bid == pytest.approx(2400.10)
    assert quote.ask == pytest.approx(2400.50)
    assert quote.price == pytest.approx(2400.30)
    assert quote.spread == pytest.approx(0.40)


def test_oanda_candles_are_oldest_first_and_skip_incomplete():
    rows = [
        {"time": "2026-01-01T00:00:00Z", "complete": True,
         "mid": {"o": "1", "h": "2", "l": "0", "c": "1.5"}, "volume": 10},
        {"time": "2026-01-01T01:00:00Z", "complete": True,
         "mid": {"o": "2", "h": "3", "l": "1", "c": "2.5"}, "volume": 12},
        {"time": "2026-01-01T02:00:00Z", "complete": False,
         "mid": {"o": "3", "h": "4", "l": "2", "c": "3.5"}, "volume": 1},
    ]
    provider = OandaProvider("key", "acct", client=_client(
        lambda r: httpx.Response(200, json={"candles": rows}), OandaProvider._BASE_URL))

    series = asyncio.run(provider.get_candles("XAUUSD", Timeframe.H1, 10))

    assert len(series.candles) == 2                      # forming candle dropped
    assert series.candles[0].timestamp < series.candles[1].timestamp


def test_oanda_empty_response_raises():
    provider = OandaProvider("key", "acct", client=_client(
        lambda r: httpx.Response(200, json={"prices": []}), OandaProvider._BASE_URL))
    with pytest.raises(MarketDataError):
        asyncio.run(provider.get_quote("XAUUSD"))


# --------------------------------------------------------------------------- Polygon

def test_polygon_prefixes_forex_tickers():
    assert PolygonProvider._ticker("XAUUSD") == "C:XAUUSD"
    assert PolygonProvider._ticker("AAPL") == "AAPL"


def test_polygon_quote_parses_nbbo():
    provider = PolygonProvider("key", client=_client(
        lambda r: httpx.Response(200, json={"results": {"p": 2400.1, "P": 2400.5}}),
        PolygonProvider._BASE_URL))
    quote = asyncio.run(provider.get_quote("XAUUSD"))
    assert quote.bid == pytest.approx(2400.1)
    assert quote.ask == pytest.approx(2400.5)


def test_polygon_reverses_descending_aggregates():
    rows = [
        {"t": 3000, "o": 3, "h": 4, "l": 2, "c": 3.5, "v": 1},
        {"t": 2000, "o": 2, "h": 3, "l": 1, "c": 2.5, "v": 1},
        {"t": 1000, "o": 1, "h": 2, "l": 0, "c": 1.5, "v": 1},
    ]
    provider = PolygonProvider("key", client=_client(
        lambda r: httpx.Response(200, json={"results": rows}), PolygonProvider._BASE_URL))

    series = asyncio.run(provider.get_candles("XAUUSD", Timeframe.H1, 10))

    stamps = [c.timestamp for c in series.candles]
    assert stamps == sorted(stamps)     # oldest-first contract upheld


def test_polygon_covers_every_timeframe():
    for tf in Timeframe:
        assert tf in PolygonProvider._SPAN


# --------------------------------------------------------------------- AlphaVantage

def test_alpha_vantage_rate_limit_note_is_retryable():
    provider = AlphaVantageProvider("key", client=_client(
        lambda r: httpx.Response(200, json={"Note": "call frequency exceeded"}),
        AlphaVantageProvider._BASE_URL))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("EURUSD"))
    assert exc_info.value.retryable is True


def test_alpha_vantage_error_message_is_not_retryable():
    provider = AlphaVantageProvider("key", client=_client(
        lambda r: httpx.Response(200, json={"Error Message": "invalid symbol"}),
        AlphaVantageProvider._BASE_URL))
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(provider.get_quote("EURUSD"))
    assert exc_info.value.retryable is False


def test_alpha_vantage_parses_a_quote():
    payload = {"Realtime Currency Exchange Rate": {
        "5. Exchange Rate": "1.0850", "8. Bid Price": "1.0849", "9. Ask Price": "1.0851"}}
    provider = AlphaVantageProvider("key", client=_client(
        lambda r: httpx.Response(200, json=payload), AlphaVantageProvider._BASE_URL))
    quote = asyncio.run(provider.get_quote("EURUSD"))
    assert quote.price == pytest.approx(1.0850)
    assert quote.spread == pytest.approx(0.0002)


def test_alpha_vantage_rejects_a_non_fx_symbol():
    provider = AlphaVantageProvider("key", client=_client(
        lambda r: httpx.Response(200, json={}), AlphaVantageProvider._BASE_URL))
    with pytest.raises(MarketDataError):
        asyncio.run(provider.get_quote("BTCUSDT"))


# ------------------------------------------------------------------ FallbackProvider

class _Stub:
    def __init__(self, name, *, fail=False, available=True):
        self.name = name
        self.fail = fail
        self.available = available
        self.quote_calls = 0

    async def get_quote(self, symbol):
        self.quote_calls += 1
        if self.fail:
            raise MarketDataError(f"{self.name} failed", retryable=True)
        return Quote(symbol=symbol, price=1.0, timestamp=datetime.now(timezone.utc),
                     provider=self.name)

    async def get_candles(self, symbol, timeframe, limit):
        if self.fail:
            raise MarketDataError(f"{self.name} failed", retryable=True)
        return CandleSeries(symbol=symbol, timeframe=timeframe, provider=self.name,
                            candles=())


def test_fallback_uses_the_first_working_provider():
    first, second = _Stub("first"), _Stub("second")
    quote = asyncio.run(FallbackProvider([first, second]).get_quote("XAUUSD"))
    assert quote.provider == "first"
    assert second.quote_calls == 0


def test_fallback_moves_on_when_a_provider_fails():
    broken, working = _Stub("broken", fail=True), _Stub("working")
    quote = asyncio.run(FallbackProvider([broken, working]).get_quote("XAUUSD"))
    assert quote.provider == "working"


def test_fallback_skips_a_provider_that_knows_it_is_unavailable():
    """MT5 on Linux must not cost a call attempt on every signal."""
    offline, working = _Stub("mt5", available=False), _Stub("twelve")
    quote = asyncio.run(FallbackProvider([offline, working]).get_quote("XAUUSD"))
    assert quote.provider == "twelve"
    assert offline.quote_calls == 0


def test_fallback_raises_only_when_everything_fails():
    a, b = _Stub("a", fail=True), _Stub("b", fail=True)
    with pytest.raises(MarketDataError) as exc_info:
        asyncio.run(FallbackProvider([a, b]).get_quote("XAUUSD"))
    detail = str(exc_info.value)
    assert "all providers failed" in detail
    assert "b failed" in detail          # last error preserved


def test_fallback_survives_a_provider_that_raises_a_non_market_error():
    class Exploding:
        name = "boom"
        async def get_quote(self, symbol):
            raise RuntimeError("adapter bug")
        async def get_candles(self, symbol, timeframe, limit):
            raise RuntimeError("adapter bug")

    quote = asyncio.run(FallbackProvider([Exploding(), _Stub("good")]).get_quote("X"))
    assert quote.provider == "good"


def test_fallback_reports_its_chain():
    chain = FallbackProvider([_Stub("mt5"), _Stub("oanda"), _Stub("twelve_data")])
    assert chain.provider_names == ("mt5", "oanda", "twelve_data")


def test_fallback_needs_at_least_one_provider():
    with pytest.raises(ValueError):
        FallbackProvider([])
