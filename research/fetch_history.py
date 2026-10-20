"""Download XAU/USD history from Twelve Data for offline research.

Pages backwards with ``end_date`` until ``--start`` is reached and writes one
JSON file per interval to ``data/research/`` (gitignored): bars oldest-first
as [utc_iso, open, high, low, close] plus the request metadata. An existing
file is never overwritten: the new download goes to a different name and the
script refuses to replace a file that is already there.

Throttled to one request every ``--pause`` seconds (default 30). Twelve Data's
free tier allows 8 requests a minute and 800 a day, and the deployed bot
shares that quota, so research fetches stay far below it. The API key is read
from the environment or ``.env`` and is never printed.

Usage::

    python research/fetch_history.py --interval 5min --start 2020-12-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market_data import MarketDataError, TwelveDataProvider  # noqa: E402

OUT = ROOT / "data" / "research"
INTERVALS = ("5min", "15min", "1h", "4h", "1day")


def api_key() -> str:
    """TWELVE_DATA_API_KEY from the environment, else from .env. Never echoed."""
    value = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    env = ROOT / ".env"
    if not value and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, raw = line.partition("=")
            if sep and name.strip() == "TWELVE_DATA_API_KEY":
                value = raw.strip().strip('"').strip("'")
    if not value:
        raise SystemExit("TWELVE_DATA_API_KEY is not set")
    return value


async def fetch(provider: TwelveDataProvider, interval: str, start: str, pause: float) -> list:
    bars: dict = {}
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    while True:
        for attempt in range(4):
            try:
                data = await provider._request("/time_series", {
                    "symbol": "XAU/USD", "interval": interval, "outputsize": 5000,
                    "timezone": "UTC", "end_date": end})
                break
            except MarketDataError as exc:
                # A burst from the deployed bot can use up the minute's quota;
                # wait out the minute rather than lose the pages already fetched.
                if not exc.retryable or attempt == 3:
                    raise
                print(f"  {interval}: retryable error ({type(exc).__name__}); waiting 65 s",
                      flush=True)
                time.sleep(65)
        values = data.get("values") or []
        before = len(bars)
        for v in values:
            bars[v["datetime"]] = [v["datetime"], float(v["open"]), float(v["high"]),
                                   float(v["low"]), float(v["close"])]
        oldest = min((v["datetime"] for v in values), default=None)
        print(f"  {interval}: page to {end} -> {len(values)} bars, oldest {oldest}, "
              f"total {len(bars)}", flush=True)
        if not values or len(bars) == before or oldest is None or oldest <= start:
            break
        end = oldest
        time.sleep(pause)
    rows = sorted(b for k, b in bars.items() if k >= start)
    return [[datetime.fromisoformat(r[0]).replace(tzinfo=timezone.utc).isoformat(), *r[1:]]
            for r in rows]


async def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--interval", choices=INTERVALS, required=True)
    p.add_argument("--start", required=True, help="first date to keep (YYYY-MM-DD)")
    p.add_argument("--pause", type=float, default=30.0)
    args = p.parse_args(argv)
    path = OUT / f"xauusd_{args.interval}.json"
    if path.exists():
        print(f"{path.name} exists — kept; move it away first to refetch")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    provider = TwelveDataProvider(api_key(), timeout=60)
    try:
        rows = await fetch(provider, args.interval, args.start, args.pause)
    except MarketDataError as exc:
        print(f"{args.interval}: FAILED — {type(exc).__name__}")
        return 1
    finally:
        await provider.aclose()
    if not rows:
        print(f"{args.interval}: no bars returned")
        return 1
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps({
        "symbol": "XAU/USD", "interval": args.interval, "provider": "twelve_data",
        "price": "provider bar prices (treated as mid)", "timezone": "UTC",
        "fetched_at": datetime.now(timezone.utc).isoformat(), "start": args.start,
        "bars": rows}))
    if path.exists():                       # appeared while we were fetching
        print(f"{path.name} appeared meanwhile — new data left in {partial.name}")
        return 1
    partial.rename(path)
    print(f"{path.name}: {len(rows)} bars {rows[0][0]} .. {rows[-1][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
