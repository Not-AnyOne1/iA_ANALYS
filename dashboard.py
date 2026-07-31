"""Read-only Dashboard (RFC-010): a lightweight local HTTP view over
Storage (RFC-003) and Statistics (RFC-009).

No business logic: every endpoint function below is a thin pass-through
that calls exactly one ``storage.Storage``/``statistics.Statistics``
method and reshapes its result into JSON-serialisable data via
``decision_engine._to_json`` (reused, not duplicated — it already handles
dataclasses/Pydantic models/Enums/datetimes generically). Nothing here
computes a new aggregate, talks to Claude, or performs a market
calculation. The only "logic" present is presentational filtering over
data Storage/Statistics already returned (e.g. "the signals among the
last N analyses"), documented at each function that does it.

Independent of the analysis pipeline: this module is never imported by
``main.py``/``pipeline.py`` and has its own standalone entry point
(``python dashboard.py``). It only ever *reads* — it never writes to
Storage, never calls Claude, and never touches market_data.py.

Endpoints:

  GET /                          minimal HTML dashboard (fetches the below)
  GET /api/summary               Statistics.summary()
  GET /api/today                 Statistics.today()
  GET /api/week                  Statistics.this_week()
  GET /api/month                 Statistics.this_month()
  GET /api/by-symbol?symbol=X    Statistics.by_symbol(X)
  GET /api/by-timeframe?timeframe=X   Statistics.by_timeframe(X)  (X in M1/M5/M15/M30/H1/H4/D1)
  GET /api/confidence-chart      Statistics.summary().confidence_distribution, chart-shaped
  GET /api/recent?limit=N        Storage.history(N) — most recent analyses (default 20)
  GET /api/signals?limit=N       the signals among Storage.history(N) — a "signal timeline"
  GET /api/decisions?limit=N     the most recent N rows with a recorded decision — "decision history"

``limit`` defaults to 20 and is capped at 200 to keep responses bounded on
this stdlib-only, single-threaded-per-request server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from decision_engine import _to_json
from logging_setup import configure_logging
from market_data import Timeframe
from statistics import Statistics
from storage import Storage

log = logging.getLogger(__name__)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 200


def _clamp_limit(raw: Optional[str]) -> int:
    if raw is None:
        return _DEFAULT_LIMIT
    value = int(raw)  # ValueError propagates to the caller, mapped to 400
    if value < 1:
        raise ValueError("limit must be >= 1")
    return min(value, _MAX_LIMIT)


# ------------------------------------------------------------------ endpoints

async def get_summary(statistics: Statistics) -> Dict[str, Any]:
    return _to_json(await statistics.summary())


async def get_today(statistics: Statistics) -> Dict[str, Any]:
    return _to_json(await statistics.today())


async def get_this_week(statistics: Statistics) -> Dict[str, Any]:
    return _to_json(await statistics.this_week())


async def get_this_month(statistics: Statistics) -> Dict[str, Any]:
    return _to_json(await statistics.this_month())


async def get_by_symbol(statistics: Statistics, symbol: str) -> Dict[str, Any]:
    return _to_json(await statistics.by_symbol(symbol))


async def get_by_timeframe(statistics: Statistics, timeframe: Timeframe) -> Dict[str, Any]:
    return _to_json(await statistics.by_timeframe(timeframe))


async def get_confidence_chart(statistics: Statistics) -> Dict[str, Any]:
    """Statistics.summary().confidence_distribution, reshaped into parallel
    label/count arrays in bucket order — a presentation shape for a chart,
    not a new computation; the counts are exactly what Statistics produced."""
    summary = await statistics.summary()
    buckets = sorted(
        summary.confidence_distribution.items(),
        key=lambda item: int(item[0].split("-")[0]),
    )
    return {"labels": [label for label, _ in buckets], "counts": [count for _, count in buckets]}


async def get_recent_analyses(storage: Storage, limit: int = _DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """The most recent analyses, most recent first — straight from
    Storage.history(), the same source /latest and /history already use."""
    records = await storage.history(limit=limit)
    return [_to_json(record) for record in records]


async def get_signal_timeline(storage: Storage, limit: int = _DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """The signals among the last ``limit`` analyses (Storage.history()
    doesn't support a separate is_signal filter, so this is a
    presentational filter over an already-limited window, not a new
    query — a sparse signal group therefore may need a larger ``limit``
    to surface, exactly like Storage.history() itself)."""
    records = await storage.history(limit=limit)
    return [_to_json(record) for record in records if record.analysis.is_signal]


async def get_decision_history(storage: Storage, limit: int = _DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """The most recent rows that have a recorded decision, most recent
    first. Storage.fetch_analyses() has no built-in ordering/limit (it's
    built for Statistics' full-set aggregation), so the filter/sort/
    truncate happens here in Python — presentational shaping, not a new
    aggregate."""
    records = await storage.fetch_analyses()
    decided = [r for r in records if r.decision_verdict is not None]
    decided.sort(key=lambda r: r.recorded_at, reverse=True)
    return [_to_json(r) for r in decided[:limit]]


# --------------------------------------------------------------------- server

_DASHBOARD_HTML = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Signal Monitor Dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.05rem; margin-top: 2rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  th, td { border: 1px solid #ddd; padding: 0.35rem 0.6rem; font-size: 0.85rem; text-align: left; }
  th { background: #f4f4f4; }
  .bar-row { display: flex; align-items: center; gap: 0.5rem; margin: 2px 0; }
  .bar-label { width: 70px; font-size: 0.8rem; }
  .bar { background: #4a7; height: 14px; }
  .bar-count { font-size: 0.8rem; }
  .stat { display: inline-block; margin-right: 2rem; }
  .stat b { font-size: 1.3rem; display: block; }
</style>
</head>
<body>
<h1>Signal Monitor Dashboard</h1>
<div id="summary"></div>
<h2>Confidence distribution</h2>
<div id="chart"></div>
<h2>Recent analyses</h2>
<table id="recent"><thead><tr><th>Time</th><th>Symbol</th><th>Category</th><th>Confidence</th></tr></thead><tbody></tbody></table>
<h2>Decision history</h2>
<table id="decisions"><thead><tr><th>Time</th><th>Symbol</th><th>Verdict</th><th>Confidence</th></tr></thead><tbody></tbody></table>
<h2>Signal timeline</h2>
<table id="signals"><thead><tr><th>Time</th><th>Symbol</th><th>Direction</th><th>Summary</th></tr></thead><tbody></tbody></table>

<script>
async function j(url) { const r = await fetch(url); return r.json(); }

async function load() {
  const s = await j('/api/summary');
  document.getElementById('summary').innerHTML =
    `<div class="stat"><b>${s.total_analyses}</b>analyses</div>` +
    `<div class="stat"><b>${s.total_signals}</b>signals</div>` +
    Object.entries(s.verdict_counts).map(([k,v]) =>
      `<div class="stat"><b>${v}</b>${k}</div>`).join('');

  const chart = await j('/api/confidence-chart');
  const max = Math.max(1, ...chart.counts);
  document.getElementById('chart').innerHTML = chart.labels.map((label, i) =>
    `<div class="bar-row"><span class="bar-label">${label}</span>` +
    `<div class="bar" style="width:${(chart.counts[i]/max*200)|0}px"></div>` +
    `<span class="bar-count">${chart.counts[i]}</span></div>`).join('');

  const recent = await j('/api/recent?limit=20');
  document.querySelector('#recent tbody').innerHTML = recent.map(r =>
    `<tr><td>${r.timestamp}</td><td>${r.analysis.setup.symbol||''}</td>` +
    `<td>${r.analysis.category}</td><td>${(r.analysis.confidence*100).toFixed(0)}%</td></tr>`).join('');

  const decisions = await j('/api/decisions?limit=20');
  document.querySelector('#decisions tbody').innerHTML = decisions.map(d =>
    `<tr><td>${d.recorded_at}</td><td>${d.symbol||''}</td>` +
    `<td>${d.decision_verdict||''}</td><td>${d.decision_confidence!=null?d.decision_confidence+'%':''}</td></tr>`).join('');

  const signals = await j('/api/signals?limit=20');
  document.querySelector('#signals tbody').innerHTML = signals.map(sig =>
    `<tr><td>${sig.timestamp}</td><td>${sig.analysis.setup.symbol||''}</td>` +
    `<td>${sig.analysis.setup.direction||''}</td><td>${sig.analysis.summary}</td></tr>`).join('');
}
load();
</script>
</body>
</html>
"""


def _make_handler(storage: Storage, statistics: Statistics) -> type:
    """Builds a BaseHTTPRequestHandler bound to this storage/statistics
    pair via closure — the handler class itself holds no state."""

    class DashboardRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
            log.info("dashboard_request %s", fmt % args)

        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            parsed = urlparse(self.path)
            path = parsed.path
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            if path == "/":
                self._respond_html(_DASHBOARD_HTML)
                return

            route = _ROUTES.get(path)
            if route is None:
                self._respond_json({"error": "not found"}, status=404)
                return

            try:
                data = asyncio.run(route(storage, statistics, params))
            except ValueError as exc:
                self._respond_json({"error": str(exc)}, status=400)
            except Exception:  # noqa: BLE001 - a bad request must not crash the server
                log.exception("dashboard_endpoint_failed path=%s", path)
                self._respond_json({"error": "internal error"}, status=500)
            else:
                self._respond_json(data)

        def _respond_json(self, data: Any, *, status: int = 200) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _respond_html(self, html: str, *, status: int = 200) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardRequestHandler


async def _route_summary(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    return await get_summary(statistics)


async def _route_today(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    return await get_today(statistics)


async def _route_week(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    return await get_this_week(statistics)


async def _route_month(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    return await get_this_month(statistics)


async def _route_by_symbol(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    symbol = params.get("symbol")
    if not symbol:
        raise ValueError("missing 'symbol' query parameter")
    return await get_by_symbol(statistics, symbol)


async def _route_by_timeframe(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    raw = params.get("timeframe")
    if not raw or raw not in Timeframe.__members__:
        raise ValueError(
            f"missing or invalid 'timeframe' query parameter (expected one of "
            f"{sorted(Timeframe.__members__)})"
        )
    return await get_by_timeframe(statistics, Timeframe[raw])


async def _route_confidence_chart(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    return await get_confidence_chart(statistics)


async def _route_recent(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    return await get_recent_analyses(storage, _clamp_limit(params.get("limit")))


async def _route_signals(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    return await get_signal_timeline(storage, _clamp_limit(params.get("limit")))


async def _route_decisions(storage: Storage, statistics: Statistics, params: Dict[str, str]) -> Any:
    return await get_decision_history(storage, _clamp_limit(params.get("limit")))


_ROUTES: Dict[str, Callable] = {
    "/api/summary": _route_summary,
    "/api/today": _route_today,
    "/api/week": _route_week,
    "/api/month": _route_month,
    "/api/by-symbol": _route_by_symbol,
    "/api/by-timeframe": _route_by_timeframe,
    "/api/confidence-chart": _route_confidence_chart,
    "/api/recent": _route_recent,
    "/api/signals": _route_signals,
    "/api/decisions": _route_decisions,
}


class DashboardServer:
    """Owns the stdlib HTTP server. Construct, call :meth:`serve_forever`
    (blocks) or run it in a thread, call :meth:`shutdown` to stop it."""

    def __init__(self, storage: Storage, statistics: Statistics, *, host: str = "127.0.0.1", port: int = 8787) -> None:
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(storage, statistics))

    @property
    def address(self) -> Tuple[str, int]:
        return self._httpd.server_address

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


# ------------------------------------------------------------------- __main__

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dashboard",
        description="Read-only local dashboard over the signal monitor's SQLite storage.",
    )
    parser.add_argument("--env-file", default=".env", help="path to the .env file (default: .env)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8787, help="bind port (default: 8787)")
    parser.add_argument("--log-level", default="INFO", help="override LOG_LEVEL (DEBUG, INFO, ...)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    from config import Settings  # local import: avoids requiring full Settings just to import this module

    args = _parse_args(argv)
    configure_logging(args.log_level)

    settings = Settings.load(args.env_file)
    storage = Storage(settings.storage_db_path)
    asyncio.run(storage.initialize())
    statistics = Statistics(storage)

    server = DashboardServer(storage, statistics, host=args.host, port=args.port)
    host, port = server.address
    log.info("Dashboard serving at http://%s:%d/ (read-only, independent of the monitor)", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Dashboard stopped")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
