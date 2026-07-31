# Telegram Signal Monitor

Monitors a **single** Telegram trading-signals group with your own Telegram
account (Telethon / MTProto — not the Bot API), sends every new message to
your locally installed **Claude Code CLI**, and prints a structured analysis
in the terminal.

Analysis runs entirely through `claude` as a subprocess, authenticated with
your existing Claude subscription (`claude auth login`). **No
`ANTHROPIC_API_KEY` and no Anthropic Console account are used or required.**

For each message Claude decides whether it is an actionable trading signal,
extracts the symbol, direction, entry, stop loss and take profits, writes a
short summary, and assigns a confidence score. Everything else — market
commentary, results, promotions, chatter — is classified and passed through
without inventing trade parameters.

```
┌──────────────┐   new message   ┌───────────┐   worker pool   ┌──────────────┐
│  Telethon    │ ──────────────▶ │  asyncio  │ ──────────────▶ │  claude  CLI │
│  (1 group)   │                 │   queue   │                 │ (subprocess, │
└──────────────┘                 └───────────┘                 │ --json-schema)│
                                                               └──────┬───────┘
                                                                      ▼
                                                        terminal panel + JSON
                                                        (optional .jsonl file)
```

The Telegram event loop never blocks on the CLI: messages are queued and
analysed by a small pool of workers, so a slow call can't cause you to miss a
signal.

---

## Project layout

| File | Responsibility |
| --- | --- |
| `main.py` | CLI entry point, wiring, graceful shutdown |
| `config.py` | Loads and validates `.env`, exposes an immutable `Settings` |
| `telegram_client.py` | Telethon client, single-group filtering, reconnection |
| `claude_client.py` | Shells out to the `claude` CLI with `--json-schema`, retries, timeouts |
| `signal_parser.py` | Regex-first classification/extraction (RFC-001); Claude is the fallback for anything ambiguous |
| `models.py` | Pydantic schema for the analysis (also the CLI's `--json-schema`) |
| `pipeline.py` | Routes each message to the regex parser and/or Claude, queue workers, stats, optional JSONL persistence |
| `formatter.py` | Terminal rendering (ANSI colour, JSON block) |
| `telegram_bot.py` | Optional Telegram Bot notifications + /start /help /status /stats /latest /history (RFC-002) |
| `logging_setup.py` | Logging to stderr and optionally to a file |
| `market_data.py` | Pluggable market data provider (Twelve Data), quotes + OHLC candles |
| `structure_engine.py` | Deterministic market structure: swings, trend, BOS/CHoCH |
| `smc_engine.py` | Deterministic Smart Money Concepts objects |
| `scoring_engine.py` | Deterministic confluence scoring (0-100) |
| `decision_engine.py` | Claude as the final reasoning layer over the deterministic engines |
| `storage.py` | SQLite persistence for completed analyses |
| `statistics.py` | Read-only aggregation over stored analyses |
| `indicators.py` | Deterministic ATR, ADX, EMA 20/50/200, RSI, volatility |
| `sessions.py` | Asia / London / New York session detection |
| `levels.py` | Day high/low, support/resistance, distances in price/%/ATR |
| `trade_quality.py` | Risk/reward, stop quality, target quality |
| `news_filter.py` | High-impact economic event filter (pluggable calendar) |
| `risk_engine.py` | Deterministic pre-trade rejection gate |
| `validation_engine.py` | Skeptical 19-check challenge — validation score, fatal problems |
| `market_context.py` | Python Analysis Engine — assembles every fact for Claude |
| `market_providers.py` | MT5/OANDA/Polygon/AlphaVantage adapters + fallback chain |
| `trade_decision.py` | Claude's ENTER/WAIT/SKIP verdict over computed facts |
| `report.py` | Full desk report (terminal + Telegram HTML) |
| `dashboard.py` | Standalone read-only local dashboard (not deployed by default) |
| `Dockerfile` | Deployment image: Python 3.11 + Node.js + the Claude Code CLI |
| `docker-entrypoint.sh` | Seeds the Telegram session and Claude credentials at boot |
| `railway.json` | Railway build/deploy config — Dockerfile builder, restart policy |
| `railway_secrets.py` | Deployment helper: encodes the session + Claude credentials for Railway |

---

## Requirements

* Python 3.10 or newer
* Node.js (to install the Claude Code CLI)
* A Telegram account that is **already a member** of the group you want to watch
* Telegram API credentials from <https://my.telegram.org/apps>
* A Claude subscription, signed in via the Claude Code CLI — **not** an
  Anthropic API key

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Get Telegram API credentials

1. Sign in at <https://my.telegram.org/apps> with your phone number.
2. Create an application (any name/description works).
3. Copy `api_id` and `api_hash`.

These identify the *application*, not your account — you still log in with your
phone number on first run.

### 3. Install and authenticate the Claude Code CLI

```bash
npm install -g @anthropic-ai/claude-code
claude auth login
```

Use `claude auth login` **without** `--console` — that flag switches to
Anthropic Console / API-key billing, which is exactly what this project avoids.
Confirm it worked with `claude auth status` (exits 0 and prints your login
state if you're signed in).

### 4. Configure

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Fill in at minimum:

```ini
TELEGRAM_API_ID=1234567
TELEGRAM_API_HASH=0123456789abcdef0123456789abcdef
TELEGRAM_TARGET_CHAT=@my_signals_group
TELEGRAM_PHONE=+15551234567
```

There is no API key to fill in. If `claude` isn't on your `PATH` under that
exact name, set `CLAUDE_CLI_PATH` to its full path.

`.env` is listed in `.gitignore` — keep it that way.

### 5. Verify the setup

```bash
python main.py --check
```

This logs in to Telegram (prompting for the code Telegram sends you the first
time), confirms the target group is reachable, confirms the Claude Code CLI is
signed in, runs one sample signal through it, and exits. Use it whenever
something looks wrong.

### 6. Run

```bash
python main.py
```

Press `Ctrl-C` to stop. A session summary is printed on exit.

---

## Finding the group id

A public group can be referenced by username (`@my_signals_group`) or link.
Private groups — including ones where you're only a member, not an admin —
need the numeric id. List every chat you have access to with:

```bash
python main.py --list-chats
```

This logs in with your existing Telethon session (the placeholder
`TELEGRAM_TARGET_CHAT=@my_signals_group` from `.env.example` is fine — this
command doesn't need it to resolve to anything) and prints every chat:

```
-------------------------------------------
Title: VIP GOLD SIGNALS
ID: -1002485632145
Type: Supergroup
Username: None
-------------------------------------------
```

`Type` is one of `Group` (a legacy basic group), `Supergroup`, `Channel`, or
`Private` (a one-on-one chat). Copy the `ID` of the group you want into
`TELEGRAM_TARGET_CHAT`. Private *invite* links (`t.me/joinchat/...`,
`t.me/+...`) cannot be used directly as `TELEGRAM_TARGET_CHAT` — join the
group first, then use its id from this list.

---

## Example output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SIGNAL  VIP Signals
#4821 · Head Analyst (@analyst) · 2026-07-20 11:42:07
https://t.me/c/1234567890/4821
───────────────────────────────────────────────────────────────────────────────
message
  🔥 BTC/USDT LONG 🔥
  Entry: 61200 - 61450
  SL: 60350
  TP1: 62400  TP2: 63500  TP3: 65000
  Leverage: cross 10x
───────────────────────────────────────────────────────────────────────────────
  Symbol       BTCUSDT   LONG  (limit)
  Entry        61200, 61450
  Stop loss    60350
  Take profit  62400, 63500, 65000
  Extras       leverage cross 10x · R:R ≈ 1.10
  Summary      New long setup on BTCUSDT with a defined entry zone, a stop
               below the range and three scaled targets.
  Confidence   █████████░ 93%
───────────────────────────────────────────────────────────────────────────────
{
  "analysed_at": "2026-07-20T11:42:09.114Z",
  "analysis": {
    "is_signal": true,
    "category": "signal",
    "setup": {
      "symbol": "BTCUSDT",
      "direction": "long",
      "order_type": "limit",
      "entries": [61200.0, 61450.0],
      "stop_loss": 60350.0,
      "take_profits": [62400.0, 63500.0, 65000.0],
      "leverage": "cross 10x",
      "timeframe": null,
      "risk_reward": 1.1
    },
    "summary": "New long setup on BTCUSDT ...",
    "confidence": 0.93,
    "missing_fields": [],
    "notes": null
  },
  "message": { "id": 4821, "chat_title": "VIP Signals", ... }
}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Non-signals are rendered with their category badge (`UPDATE`, `RESULT`,
`COMMENTARY`, `PROMOTION`, `OTHER`) and empty setup fields.

---

## Output schema

Every analysis conforms to this shape — enforced by the CLI's `--json-schema`,
so it never needs to be repaired or re-parsed:

| Field | Type | Meaning |
| --- | --- | --- |
| `is_signal` | bool | True only for a concrete, actionable new setup |
| `category` | enum | `signal` · `update` · `result` · `commentary` · `promotion` · `other` |
| `setup.symbol` | string \| null | Normalised instrument (`BTCUSDT`, `XAUUSD`) |
| `setup.direction` | `long` \| `short` \| null | Trade side |
| `setup.order_type` | `market` \| `limit` \| `stop` \| `unknown` | Entry style |
| `setup.entries` | float[] | Entry price(s); a zone yields two values |
| `setup.stop_loss` | float \| null | Stop-loss price |
| `setup.take_profits` | float[] | Targets in listed order (TP1 first) |
| `setup.leverage` | string \| null | As written, e.g. `cross 10x` |
| `setup.timeframe` | string \| null | e.g. `15m`, `4H` |
| `setup.risk_reward` | float | Computed locally against TP1 (JSON output only) |
| `summary` | string | One or two sentences in plain English |
| `confidence` | float | 0.0–1.0 for the classification and extraction |
| `missing_fields` | string[] | Setup fields a trader would need but that are absent |
| `notes` | string \| null | Ambiguities and anything worth a human look |
| `source` | `regex` \| `claude` | Which code path produced this analysis — see "Signal Parser" below |

Set `JSONL_OUTPUT=data/signals.jsonl` to append one JSON object per line for
later backtesting or import into a database.

---

## Market-validated analysis

Every signal is checked against **live market data** before any verdict. The
text of a Telegram message is treated as a claim to be validated, never as
the basis for a decision.

```
Telegram signal
      ↓
Signal extraction          signal_parser.py / claude_client.py
      ↓
Live market data           market_data.py + market_providers.py
      ↓
Python Analysis Engine     market_context.py
      ↓                    (structure, SMC, indicators, levels,
      ↓                     session, news, trade quality, risk gate)
Validation Engine          validation_engine.py
      ↓                    (19 checks → score, strengths, weaknesses,
      ↓                     fatal problems)
Claude reasoning           trade_decision.py   → ENTER / WAIT / SKIP
      ↓
Decision + report          report.py
```

The rule the whole design rests on: **Claude never calculates anything.** It
receives finished numbers and reasons about them. `build_facts()` in
`trade_decision.py` is the complete list of what the model can see — and
there is no candle series in it, so there is nothing to derive an indicator
from even in principle. A test asserts this (`test_facts_never_include_a_candle_series`).

### What Python computes

| Module | Produces |
| --- | --- |
| `structure_engine.py` | trend, swing highs/lows, BOS, CHoCH |
| `smc_engine.py` | FVG, inverse FVG, order blocks, breaker/mitigation blocks, liquidity pools & sweeps, equal highs/lows, supply/demand, premium/discount, OTE |
| `indicators.py` | ATR (+%), ADX with ±DI, EMA 20/50/200 and their alignment, RSI, volatility state |
| `levels.py` | day high/low, position in day range, nearest support/resistance, distances in price / % / ATR |
| `sessions.py` | Asia / London / New York, including overlaps |
| `trade_quality.py` | risk/reward, stop quality, target quality |
| `news_filter.py` | high-impact event detection |
| `scoring_engine.py` | weighted confluence score (0-100) |
| `risk_engine.py` | the pre-trade rejection gate |

Everything is deterministic: the same candles always produce the same
analysis. Indicators use Wilder smoothing for RSI/ATR/ADX and the standard
`2/(n+1)` EMA, and return `None` rather than a padded value when the series
is too short — a missing number is reported as missing, never defaulted.

### The risk gate

`risk_engine.py` runs **before** Claude and can reject a setup outright, so
the model is never asked to reason about a trade that already failed a hard
rule — and cannot argue its way past one.

Six rules **reject** (`approved = false`, verdict SKIP, Claude never asked):

| Rule | Rejects when |
| --- | --- |
| `risk_reward` | R:R strictly below 1:1 — risking more than the trade can win (configurable) |
| `stop_quality` | stop inside the noise band (<0.5 ATR), absurdly wide (>5 ATR), or resting beyond a swing price is likely to sweep |
| `target_quality` | target inside the noise band, or a level stands between entry and target |
| `spread` | spread wider than 0.25 ATR |
| `confidence` | scoring confidence below 40 |
| `news` | a high-impact event is inside the blackout window |

Two rules **warn** (`RiskVerdict.WARN`) — fully evaluated, reported, and sent
to Claude, but they do not gate:

| Rule | Warns when | Why it does not block |
| --- | --- | --- |
| `trend_alignment` | trading against the structural trend | Reversal and mean-reversion entries are a legitimate style |
| `volatility` | volatility is low | A market condition that can change within the life of the trade |

Both facts are still *scored*, at the full weight of the matching
`validation_engine` check (`higher_timeframe_trend`, `volatility`) — so they
push the validation score down and let the decision bands judge, rather than
vetoing. Only a `REJECT` sets `approved = false`.

A rule whose input is unavailable **abstains** rather than rejecting — the
gate exists to reject known-bad setups, not everything it cannot measure.
Abstentions are shown in the report so an absent check is never mistaken for
a passed one, and warnings are shown distinctly from both.

### Claude's verdict

Claude returns `ENTER`, `WAIT` or `SKIP` with reasoning, strengths, risks and
an execution plan. Two guarantees are enforced in Python, not delegated:

- **Confidence is clamped** to the deterministic scoring ceiling. Claude may
  go lower; it can never inflate the number the engines produced.
- **The fallback never enters.** If the CLI is unavailable, the verdict is
  WAIT (when the risk gate approved and data exists) or SKIP — entering is
  precisely the decision that must not be made without the reasoning step.

### The Validation Engine

`validation_engine.py` sits between the analysis and Claude and does the
opposite job to the scoring engine. Where `scoring_engine.py` asks *"what
supports this trade?"*, this asks **"what argues against it?"** — and makes
the setup earn its score.

```
Python Analysis  ->  Validation Engine  ->  Claude  ->  Decision
```

Nineteen checks, each contributing positively or negatively:

| Group | Checks |
| --- | --- |
| Trend | higher-timeframe trend, lower-timeframe confirmation, distance to EMA200 |
| Structure | BOS alignment, CHoCH alignment |
| Smart money | liquidity direction, order block quality, FVG quality, distance from liquidity |
| Conditions | ATR condition, volatility, spread, session quality |
| Trade | risk/reward, maximum adverse excursion, distance to support, distance to resistance |
| Other | news, probability score |

Output: a **validation score (0-100)**, plus named **strengths**,
**weaknesses**, **fatal problems** and **unverified** findings.

Three properties make the skepticism structural rather than a matter of
prompt wording:

**1. Unvalidated is not the same as fine.** The score is
`(positives − negatives) / total_possible`. A check whose data is missing
contributes nothing to the numerator while still counting in the
denominator — so a setup nobody could verify scores *low*, not neutral. An
empty context scores ≤10, and the report lists every unverified check under
"counted against the score".

**2. Fatal problems are absolute — and rare.** Any single fatal finding
forces the score to 0 regardless of how many positives exist. A trade with
excellent structure taken into an imminent FOMC release is not a good trade
with one flaw; it is a trade that must not be taken.

Fatal is reserved for **objective blockers** — conditions under which the
trade cannot sensibly be placed at all, rather than ones that merely make it
a poor idea. Exactly three qualify:

| Fatal | Why it is a blocker, not an opinion |
|---|---|
| `spread` wider than 0.25 ATR | The cost of entry eats the edge outright |
| `risk_reward` below 1:1 | Risking more than the reward — an invalid trade parameter |
| `news` blackout | A scheduled high-impact release makes price unmodellable |

Everything that is a matter of **degree or likelihood** is a weakness. It
still costs the full weight of its check — the score keeps reflecting the
risk — but it cannot veto on its own; the decision bands judge. Five checks
work this way, each a serious hazard and each a judgement rather than an
impossibility:

| Weakness (at full weight) | Why it is a judgement |
|---|---|
| Counter-trend against every higher timeframe | Reversal and mean-reversion trades are a legitimate style |
| Liquidity swept against the trade | The sweep may already be exhausted |
| Low volatility | A market condition that can change within the life of the trade |
| Probability model favours the other side | One model's read over a fixed lookback |
| Stop inside the typical adverse excursion | A statistical tendency, not an impossibility |

Measured on the test fixtures, each of these alone moves a clean 57/100
setup to: counter-trend **16** (SKIP by score), model-disagrees **45**,
low-volatility **49**, sweep-against **51** (all WAIT). Serious, scored, but
not vetoed.

**3. Maximum adverse excursion is measured, not assumed.** For every candle
in the lookback window, `estimate_mae_atr()` measures how far price moved
*against* the trade direction over the following 20 candles, then takes the
80th percentile. That answers "how much heat does a position of this
direction usually take on this instrument at this volatility?"

A stop inside that range is a **weakness, not a fatal problem**. It is a
statistical tendency rather than an impossibility — the trade can still be
valid at a shallower entry, in a different regime, or as a deliberate
tight-stop scalp. So it costs the check's full weight (the severe case) or
half of it (a stop that barely clears the excursion), and the score bands
decide from there. In practice a setup whose only serious flaw is a tight
stop lands around the 40s — the WAIT band — instead of being forced to SKIP.

The validation engine cannot approve anything. It only produces findings;
the decision stays with `trade_decision.py` and the hard gate stays with
`risk_engine.py`, which is unchanged and still runs. The overlap is
deliberate: `risk_engine` is the fast pass/reject gate, this is the detailed
review Claude is then asked to explain.

### Claude as skeptic

The prompt reframes the model's job entirely. It is a **risk reviewer**, not
an analyst looking for entries:

- The Telegram signal is "an unverified third-party claim [that] carries no
  authority" — Claude is explicitly expected to disagree with it.
- Its primary task is to **explain the Validation Engine's findings** in a
  trader's language, not to form an independent opinion.
- Default to `skip`. `enter` must be earned and is the rare answer.
- Unverified checks are risks, not neutrals.

**Mandatory self-critique.** `strongest_reason_against` is a required schema
field, so the model physically cannot return a verdict without arguing the
opposing case:

> "must contain the single most compelling argument for NOT taking this
> trade, stated as forcefully as an opposing analyst would put it. Never
> write 'none'. […] if it is as strong as or stronger than your positives,
> the verdict is 'skip'."

It must also supply `alternative_scenario`, `worst_case` and `best_case`.

### Confidence can only fall

```
final confidence = min(deterministic scoring, validation score, Claude's own)
```

Enforced in Python, not requested in the prompt. No source can raise the
number above another's, and the report shows the arithmetic
(`confidence = min(validation 42, scoring 70, model) = 42`) so a low number is
always traceable to whichever assessment was most pessimistic.

Two further guarantees are structural rather than prompt-dependent:

- **A fatal problem downgrades ENTER to SKIP** after the fact. Claude is
  still asked (so the finding gets explained), but its approval cannot
  survive a fatal contradiction — verified by
  `test_a_fatal_problem_downgrades_enter_to_skip`.
- **The fallback is blocked by fatal problems too**, so a CLI outage cannot
  turn a fatally flawed setup into a WAIT.

### Report

The report gained a `VALIDATION` section (score, fatal problems, strengths,
weaknesses, unverified) and the decision block gained the self-critique,
alternative scenario, worst case, best case, and the confidence arithmetic.

```
VALIDATION
  Validation score   0/100   ** FATAL PROBLEM **

  Fatal problems
    XX news                       FOMC Statement in 12 min — high impact on USD

  Strengths
    +  higher_timeframe_trend     aligned with D1=bullish, H4=bullish
    +  bos_alignment              latest BOS is bullish, with the trade
    +  max_adverse_excursion      stop (4.84 ATR) comfortably clears the typical 1.56 ATR adverse excursion
    …

  Weaknesses
    -  liquidity_direction        last liquidity sweep reacted bearish, against a long trade
    -  lower_timeframe_confirmation M15 contradicts the trade (bearish)
    -  distance_to_ema200         price is on the wrong side of EMA200 (2466.63) for a long

  Unverified (counted against the score)
    ?  order_block_quality        no order block detected
    ?  distance_from_liquidity    no liquidity pool detected
```

---

### Providers

`market_providers.py` adds an adapter chain. `FallbackProvider` tries each in
order and skips one that is unavailable or failing, so a single provider
outage does not take the layer down.

| Order | Provider | Status |
| --- | --- | --- |
| 1 | MT5 | **Cannot run on this deployment** — see below |
| 2 | OANDA | implemented, unit-tested against mocked HTTP; not verified live |
| 3 | Polygon | implemented, unit-tested against mocked HTTP; not verified live |
| 4 | Twelve Data | implemented and in use (`market_data.py`) |
| 5 | Alpha Vantage | implemented, unit-tested against mocked HTTP; not verified live |

> **MT5 is Windows-only.** The `MetaTrader5` package talks to a *locally
> running MT5 terminal* over IPC — there is no Linux build and no remote
> protocol. This project deploys to a Linux container, where neither exists,
> so `MT5Provider` detects that and reports `available = False` with the
> reason instead of pretending. On a Windows host with a logged-in terminal
> it works, and `FallbackProvider` skips it everywhere else without paying
> for a call. (Note this also conflicts with the original "no MT5, no
> Windows VPS" constraint the market data layer was built under.)
>
> The OANDA/Polygon/Alpha Vantage adapters are written against each vendor's
> documented REST shape and their parsing, symbol mapping and error handling
> are covered by tests — but no account was available to exercise them
> against the live services. Treat the first real call as the integration
> test.

Only MT5 and OANDA expose a genuine bid/ask; `Quote.spread` is `None` for
the others, and the report says so rather than deriving a fake spread. The
risk engine's spread rule abstains in that case.

### Economic calendar

`news_filter.py` ships one implementation, `StaticCalendar`, which returns
exactly the events you give it. **It will never invent a release schedule** —
guessing that NFP is "the first Friday" would silently block or allow real
trades. With no calendar configured the filter reports `available=False` and
the risk gate abstains, so "no data" is always distinguishable from "clear".

To wire a real feed, implement the `EconomicCalendar` protocol
(`events_between(start, end)`) and pass it to `NewsFilter`.

### Enabling it

Market validation activates automatically when a market data provider is
configured — set `TWELVE_DATA_API_KEY` (see the configuration reference).
Without a provider there is nothing to validate against, and the pipeline
falls back to the signal-only path unchanged. `MARKET_DATA_TIMEFRAME`
selects the primary timeframe (default `H1`); M15/H1/H4/D1 are fetched for
context on every signal.

### Example report

```
==================================================================
MARKET REPORT  XAUUSD
generated 2026-03-10 14:22:07 UTC  ·  primary timeframe H1
==================================================================
PRICE
  Current price      2,455.00000
  Spread             0.40000
  Bid / Ask          2,454.80000 / 2,455.20000
  Stated entry       2,455.00000
  Entry distance     0.00000  (0.00 ATR)

STRUCTURE
  Trend              bullish
  Last event         BOS bullish
  BOS / CHoCH        17 / 7

INDICATORS
  TF    trend     EMA          RSI        ATR    ADX vol
  M15   ranging   bullish     58.2    5.91358    9.7 normal
  H1    bullish   bullish     65.2    6.62066   17.1 normal
  H4    ranging   bullish     47.4    6.26311   16.7 normal
  D1    bullish   bullish     68.5    6.34136   30.7 normal

CONTEXT
  Session            new_york+london  (overlap)
  News               no economic calendar configured

SETUP QUALITY
  Risk / reward      1:2.60
  Stop quality       poor  (3.78 ATR)
                     - stop sits beyond a swing level price may sweep first

SCORING & RISK
  Score direction    buy
  Confidence         38%  (+40 of 106)
  Risk gate          REJECTED
    [ok  ] risk_reward      R:R 2.60
    [FAIL] stop_quality     stop sits beyond a swing level price may sweep first
    [ok  ] trend_alignment  long aligns with bullish trend
    [FAIL] confidence       confidence 38 below the 40 minimum
    [n/a ] news             no economic calendar configured — news risk unknown
==================================================================
```

The `[n/a]` line is the point: the news check did not pass, it did not run.

---

## Signal Parser (RFC-001)

Not every message needs a full Claude call — a cleanly formatted signal
(symbol, direction, entry, stop loss, at least one target, all present) can be
extracted deterministically with regex, and a message with no trading
language at all can be classified just as confidently. Everything in
between — trade-management updates, result reports, promotional posts, replies,
partial matches, or a symbol/format the parser doesn't recognise — is treated
as ambiguous and handed to Claude exactly as before. `signal_parser.py` never
guesses: when it isn't sure, it says so and defers.

Controlled by `SIGNAL_PARSER_MODE`:

| Mode | Behaviour |
| --- | --- |
| `shadow` (default) | Claude still analyses **every** message — identical output to `off`. The regex parser also runs on each message, but only to compare its verdict against Claude's and log agreement/disagreement. Zero production risk; safe to leave running indefinitely. |
| `active` | A confident regex verdict is used directly and Claude is skipped for that message. An ambiguous one still falls back to Claude, unchanged. |
| `off` | Pre-RFC-001 behaviour — the regex parser isn't consulted at all. |

**Recommended rollout:** start in `shadow`, let it run for a while, then check
the shadow-agreement rate in the session summary printed at shutdown
(`shadow agreement: N/M (X%)`) and in `[shadow]`-tagged log lines (`DISAGREE`
entries are logged at `WARNING`). Only switch to `active` once you're
comfortable with that rate. Switching back to `off` (or `shadow`) at any time
fully reverts to the all-Claude path.

---

## Telegram Bot (RFC-002)

An optional, separate Telegram **bot** (Bot API, created via
[@BotFather](https://t.me/BotFather)) that sends every completed analysis to
a chat automatically, and answers a handful of commands there. It is
independent of the Telethon user-account session used for monitoring — a
different credential, a different client, and it can never affect message
ingestion or analysis.

**Disabled by default.** Leave `TELEGRAM_BOT_TOKEN` / `TELEGRAM_BOT_CHAT_ID`
empty and nothing about the rest of the app changes. Set both to enable it.

**Guarantees:**

- Every completed analysis (regardless of source — `regex` or `claude`) is
  sent automatically, at most once. Duplicate delivery for the same message
  is prevented by an in-memory, bounded de-duplication cache
  (`TELEGRAM_BOT_DEDUP_CACHE_SIZE`) — process-lifetime only; there's no
  persistent storage yet, so this doesn't survive a restart.
- Transient failures (rate limits, network errors) are retried with
  exponential backoff (`TELEGRAM_BOT_MAX_RETRIES`, `TELEGRAM_BOT_RETRY_BASE_DELAY`,
  `TELEGRAM_BOT_RETRY_MAX_DELAY`); a `RetryAfter` response from Telegram is
  honoured using Telegram's own requested delay rather than the usual backoff.
- Permanent failures (bot blocked/kicked, chat not found) and exhausted
  retries are logged and the notification is simply dropped — **never**
  raised into the pipeline. A bot that fails even to *start* (e.g. an invalid
  token) is logged and skipped; the Telegram monitor and Claude analysis
  continue exactly as if the bot were never configured.

**Commands** (only answered in the configured `TELEGRAM_BOT_CHAT_ID`, to
avoid leaking pipeline details to anyone else who messages the bot):

| Command | Shows |
| --- | --- |
| `/start` | A short introduction |
| `/help` | This command list |
| `/status` | Bot uptime, send/failure/duplicate counters, and the live pipeline summary (same line as the shutdown summary and `/stats`) |
| `/stats` | The pipeline summary on its own |
| `/latest` | The most recently sent notification |
| `/history [n]` | The last `n` notifications (default `TELEGRAM_BOT_HISTORY_DISPLAY_DEFAULT`), most recent first |

---

## Configuration reference

All values live in `.env`; real environment variables take precedence.

| Variable | Default | Description |
| --- | --- | --- |
| `TELEGRAM_API_ID` | *required* | From my.telegram.org |
| `TELEGRAM_API_HASH` | *required* | From my.telegram.org |
| `TELEGRAM_TARGET_CHAT` | *required* | Group id, `@username`, or `t.me` link |
| `TELEGRAM_PHONE` | — | Skips the phone prompt on first login |
| `TELEGRAM_SESSION` | `signal_monitor` | Session file name |
| `CLAUDE_CLI_PATH` | `claude` | Path to the CLI executable, if not on `PATH` under that name |
| `CLAUDE_MODEL` | *(empty)* | `--model` value (`sonnet`, `opus`, `haiku`, `fable`, or a full id). Empty = use the CLI's own configured default |
| `CLAUDE_MAX_TURNS` | `3` | Safety bound on agentic turns per call (`--max-turns`) |
| `CLAUDE_MAX_RETRIES` | `3` | Retries on transient CLI failures (rate limits, overload, timeouts) |
| `CLAUDE_TIMEOUT_SECONDS` | `180` | Per-call timeout — higher than a raw API call since the CLI spawns a fresh process and loads context each time |
| `SIGNAL_PARSER_MODE` | `shadow` | `shadow` \| `active` \| `off` — see "Signal Parser" above |
| `TELEGRAM_BOT_TOKEN` | *(empty)* | From @BotFather. Empty = bot disabled entirely |
| `TELEGRAM_BOT_CHAT_ID` | *(empty)* | Numeric chat id or `@channelusername` to notify |
| `TELEGRAM_BOT_MAX_RETRIES` | `5` | Retries on transient send failures |
| `TELEGRAM_BOT_RETRY_BASE_DELAY` | `2.0` | Base backoff delay (seconds) |
| `TELEGRAM_BOT_RETRY_MAX_DELAY` | `60.0` | Backoff cap (seconds) |
| `TELEGRAM_BOT_HISTORY_SIZE` | `50` | In-memory `/history`/`/latest` buffer size |
| `TELEGRAM_BOT_HISTORY_DISPLAY_DEFAULT` | `5` | Default count for a bare `/history` |
| `TELEGRAM_BOT_DEDUP_CACHE_SIZE` | `2000` | Bound on the duplicate-notification guard |
| `WORKER_COUNT` | `2` | Concurrent analyses |
| `QUEUE_MAXSIZE` | `200` | Buffered messages before dropping (with a loud error) |
| `MAX_MESSAGE_CHARS` | `8000` | Longer messages are truncated before sending |
| `ANALYSE_EDITS` | `false` | Also analyse edited messages |
| `LOG_LEVEL` | `INFO` | `DEBUG` shows Telethon/HTTP internals |
| `LOG_FILE` | — | Mirror logs to a file |
| `JSONL_OUTPUT` | — | Append structured results, one JSON object per line |
| `COLOR_OUTPUT` | `true` | ANSI colour (also honours `NO_COLOR`) |
| `SHOW_JSON` | `true` | Print the JSON block under each panel |

Analysis output goes to **stdout**, logs go to **stderr**, so
`python main.py > signals.txt` keeps a clean transcript while errors stay
visible.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/test_signal_parser.py` covers the regex parser directly — clean
signals in several formats, updates/results/promotions/replies correctly
deferred, and the symbol-matching false-positive guards. `tests/
test_pipeline_routing.py` covers `SIGNAL_PARSER_MODE`'s three modes against a
stubbed analyzer. `tests/test_telegram_bot.py` covers the bot's retry/backoff
classification (which Telegram errors are retried vs. permanent), duplicate
suppression, HTML-escaping of dynamic content, and all six commands —
against a mocked `telegram.Bot`, no real bot token or network calls anywhere.

---

## Deploying to Railway

The monitor deploys as a **worker service**: it holds a long-lived Telegram
(MTProto) connection and never serves HTTP, so it exposes no port and Railway
performs no HTTP health check. Liveness is simply "the process is still
running", and the restart policy in [`railway.json`](railway.json) brings it
back if it exits.

Files involved:

| File | Purpose |
| --- | --- |
| [`railway.json`](railway.json) | Build/deploy config — Dockerfile builder, start command, restart policy |
| [`Dockerfile`](Dockerfile) | Python 3.11 + Node.js + the Claude Code CLI |
| [`docker-entrypoint.sh`](docker-entrypoint.sh) | Decodes credentials onto the volume before starting |
| [`railway_secrets.py`](railway_secrets.py) | Generates the two credential environment variables |

### Read this first: two logins cannot happen on the server

Both of this project's authentications are *interactive by design*, and neither
can be completed inside a Railway container:

| Credential | Why it can't be created on Railway |
| --- | --- |
| **Telethon session** (`*.session`) | Telegram's login sends a code to your phone and reads it from a terminal prompt. `telegram_client.py` deliberately refuses to prompt when stdin isn't a TTY — a Railway worker never has one. |
| **Claude Code CLI auth** (`~/.claude/.credentials.json`) | `claude auth login` is a browser-based OAuth flow. |

Both are therefore created **on your machine first** and moved to Railway as
environment variables. Railway has no secret-file upload, so
`railway_secrets.py` gzips and base64-encodes each file, and
`docker-entrypoint.sh` decodes them onto the volume at boot. (Compression is
not cosmetic: the raw session is ~38 KB as base64, which risks environment
variable size limits; gzipped it is ~3 KB.)

> **Claude authentication — choose one.**
> **(a) Subscription** (what this project was built for): set
> `CLAUDE_CREDENTIALS_B64`. The entrypoint points `~/.claude` at the volume so
> the CLI's own token refreshes persist. The OAuth *refresh* token still
> expires eventually, and when it does the worker fails until you regenerate
> the variable. Check Anthropic's terms for your plan before running
> subscription credentials on a server.
> **(b) API key**: set `ANTHROPIC_API_KEY` and omit `CLAUDE_CREDENTIALS_B64`.
> The CLI picks the key up automatically — **no code change needed** — but
> this switches you from subscription to metered API billing.
> The entrypoint supports both and prefers `ANTHROPIC_API_KEY` when set.

### Step 1 — Create both credentials locally

```bash
# Telethon session: log in once, interactively. Creates <TELEGRAM_SESSION>.session
python main.py --list-chats

# Claude CLI: subscription login (skip if you chose the API-key route)
claude auth login
claude auth status
```

Confirm the app runs locally before deploying anything:

```bash
python main.py
```

### Step 2 — Generate the credential variables

```bash
python railway_secrets.py
```

This prints `TELEGRAM_SESSION_B64` and `CLAUDE_CREDENTIALS_B64`. Use
`--out .railway-secrets.txt` to write them to a file instead (already
gitignored; delete it once the values are in Railway).

> **Treat this output as a password.** The session grants full access to your
> Telegram account and the credentials to your Claude account. Never commit it
> or paste it anywhere public.

### Step 3 — Push the repository

`.gitignore` excludes `.env`, `*.session`, and `.railway-secrets.txt`;
`.dockerignore` keeps all three out of the image. Verify before pushing:

```bash
git status --porcelain | grep -E '\.env$|\.session$|railway-secrets'   # must print nothing
```

### Step 4 — Create the service

1. <https://railway.app> → **New Project** → **Deploy from GitHub repo**.
2. Select the repository. Railway reads `railway.json` and builds with the
   `Dockerfile` (not Nixpacks — the image needs Node.js for the Claude CLI,
   which a Python Nixpacks build would not install).
3. **Do not** add a public domain. This is a worker; there is no port to expose.

### Step 5 — Attach a volume

Service → **Settings** → **Volumes** → *Add Volume*, mounted at **`/data`**.

This is required, not optional. Without it the SQLite database, the Telegram
session, and the Claude credentials are wiped on every restart and redeploy —
and the session cannot be recreated on the server.

Railway injects `RAILWAY_VOLUME_MOUNT_PATH` automatically, and the entrypoint
uses it, so the mount path only needs to match what you set in
`TELEGRAM_SESSION` and `STORAGE_DB_PATH` below.

### Step 6 — Set the variables

Service → **Variables** → **Raw Editor**, then paste:

```ini
# --- Telegram user account (from https://my.telegram.org/apps) ---
TELEGRAM_API_ID=1234567
TELEGRAM_API_HASH=0123456789abcdef0123456789abcdef
TELEGRAM_TARGET_CHAT=-1001234567890
TELEGRAM_SESSION=/data/signal_monitor

# --- Credentials generated in Step 2 (paste the real values) ---
TELEGRAM_SESSION_B64=<from railway_secrets.py>
CLAUDE_CREDENTIALS_B64=<from railway_secrets.py>
# ...or, instead of CLAUDE_CREDENTIALS_B64:
# ANTHROPIC_API_KEY=sk-ant-...

# --- Claude CLI ---
CLAUDE_CLI_PATH=claude
CLAUDE_MAX_TURNS=3
CLAUDE_MAX_RETRIES=3
CLAUDE_TIMEOUT_SECONDS=180

# --- Signal parser: shadow | active | off ---
SIGNAL_PARSER_MODE=shadow

# --- Storage (on the volume) ---
STORAGE_DB_PATH=/data/signals.db

# --- Pipeline ---
WORKER_COUNT=2
QUEUE_MAXSIZE=200
MAX_MESSAGE_CHARS=8000
ANALYSE_EDITS=false

# --- Output ---
LOG_LEVEL=INFO
COLOR_OUTPUT=false
SHOW_JSON=true

# --- Optional: Telegram Bot notifications (omit to disable) ---
TELEGRAM_BOT_TOKEN=
TELEGRAM_BOT_CHAT_ID=

# --- Optional: market data (omit to disable enrichment) ---
TWELVE_DATA_API_KEY=
MARKET_DATA_CACHE_TTL=30
MARKET_DATA_TIMEFRAME=H1
```

Only `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and `TELEGRAM_TARGET_CHAT` are
strictly required by `config.py`; everything else has a documented default.
`TELEGRAM_SESSION` and `STORAGE_DB_PATH` must point at the volume, or nothing
persists.

### Step 7 — Deploy and verify

Railway builds and starts the worker. In **Deployments → Logs** you should see:

```
[entrypoint] Persistent data directory: /data
[entrypoint] Seeded Telegram session -> /data/signal_monitor.session (28672 bytes)
[entrypoint] Seeded Claude CLI credentials -> /data/claude/.credentials.json
[entrypoint] Starting: python main.py
... telegram_client: Signed in as <your name> (id=...)
... telegram_client: Monitoring '<group name>' (id=...)
... monitor: Listening for new messages — press Ctrl-C to stop
... pipeline: Started 2 analysis worker(s)
```

On every **later** boot the first two lines change to `Using existing ...`,
confirming the volume is working:

```
[entrypoint] Using existing Telegram session at /data/signal_monitor.session
[entrypoint] Using existing Claude credentials on the volume (token refreshes preserved)
```

If `TELEGRAM_BOT_TOKEN` is set, `/status` in your Telegram chat is the quickest
end-to-end confirmation.

### Persistence

Everything under the volume survives restarts, redeploys, and crashes:

| Path | Purpose |
| --- | --- |
| `/data/signal_monitor.session` | live Telethon session |
| `/data/claude/.credentials.json` | Claude CLI credentials, including refreshed tokens |
| `/data/signals.db` | SQLite history behind `/stats`, `/latest`, `/history` |

The entrypoint seeds each file from its environment variable **only when it is
missing**. Once the volume has a copy, that copy always wins — so Telethon's
session updates and the CLI's token refreshes are never reverted to the
originally uploaded snapshot. The database is created on first boot and
migrated in place afterwards.

### Restarts and reconnection

* **Process restarts:** `railway.json` sets `restartPolicyType: ON_FAILURE`
  with `restartPolicyMaxRetries: 10`. `main.py` exits `2` on a runtime failure
  (restarted) and `1` on a configuration error — the retry cap stops a bad
  config from looping forever instead of surfacing in the dashboard.
* **Reconnecting after a restart:** automatic, and the reason the volume
  matters — the persisted session means the app re-authenticates with Telegram
  with no phone-code prompt.
* **Reconnecting without a restart:** Telethon reconnects on its own
  indefinitely with backoff (`connection_retries=None`, `retry_delay=5`), and
  `TelegramMonitor.run_forever()` supervises on top with its own exponential
  backoff. Brief network drops never reach the process level.
* **Never sleeps:** `sleepApplication: false`. Railway's app-sleep would
  suspend the worker and silently miss messages.
* **Single replica:** `numReplicas: 1`, and it must stay 1. Two replicas would
  share one Telegram session and double-analyse every message.
* **Failure isolation** (pre-existing): a failing Claude call, storage write,
  market-data fetch, or Telegram notification is contained per-message and
  never stops the monitoring loop.

### Logs

`logging_setup.py` writes to **stderr**, which Railway captures automatically.
The Dockerfile sets `PYTHONUNBUFFERED=1`; without it Python block-buffers
output when stdout isn't a TTY and log lines appear minutes late or only at
exit. `COLOR_OUTPUT=false` keeps ANSI escapes out of the log viewer.

Raise verbosity with `LOG_LEVEL=DEBUG`. `httpx`/`httpcore` stay pinned at
`WARNING` regardless — they log full request URLs, which would print your
`TWELVE_DATA_API_KEY` into the log stream.

### Rotating credentials

Update `TELEGRAM_SESSION_B64` / `CLAUDE_CREDENTIALS_B64` **and delete the stale
file from the volume** — otherwise the existing volume copy keeps winning and
the new value is ignored. Either attach a shell to the service and remove the
file, or detach and recreate the volume (which also clears `signals.db`).

### What the dashboard needs

`dashboard.py` is not deployed by this configuration. It reads the same SQLite
file, and a Railway volume can only attach to one service, so a separate
service could not see it. Run it locally against a copy of `signals.db`, or add
it to this same service later if you want it served.

## How it stays reliable

* **The regex parser only claims confidence when it's mechanically certain** —
  a full field match with no result/promo/reply markers, or no signal-like
  content at all. Anything partial, ambiguous, or unrecognised defers to
  Claude; it never guesses. `shadow` mode lets you validate this against
  Claude's own judgment before trusting it to skip Claude in `active` mode.
* **The Telegram bot can't take down the monitor** — a bot that fails to
  start (bad token) is logged and skipped; a `notify()` failure at send time
  is caught in two places (inside `telegram_bot.py` itself, and again where
  `pipeline.py` calls it) specifically so a future bug in one layer can't
  turn into a pipeline-stopping exception. Dynamic content (chat title,
  sender, summary) is HTML-escaped before being sent, the same "untrusted
  third-party content" treatment the Claude prompt already gets.
* **Single-group isolation** — Telethon filters updates server-side by chat, and
  the handler re-checks `chat_id` before enqueuing. Messages from any other chat
  are never sent to Claude.
* **Reconnection** — the client is configured with unlimited connection retries
  and auto-reconnect; a supervisor loop additionally re-establishes the session
  with exponential backoff (5s → 5min) if the client drops out entirely.
* **Retries** — transient CLI failures (rate limits, overload, timeouts,
  unexpected output) are retried with exponential backoff and jitter.
  Non-retryable failures — not signed in, an invalid `--json-schema` — are
  surfaced immediately instead.
* **No tool access on untrusted content** — every call passes
  `--disallowedTools "*"`, so the CLI has no filesystem, shell, or network
  tools available while it reads the group's message text. The message body
  and all metadata (chat title, sender) travel via stdin, never as command-line
  arguments — both to keep arbitrarily long or adversarial content out of argv,
  and because Windows' `claude.cmd` shim mangles embedded newlines in an
  argument (discovered and fixed while building this).
* **Full process-tree termination on timeout** — a background watchdog kills
  the entire process tree (`taskkill /T` on Windows, the process group on
  POSIX) when a call exceeds `CLAUDE_TIMEOUT_SECONDS`, rather than relying on
  cancelling the read, which does not reliably interrupt a hung Windows batch
  shim in time.
* **Failure isolation** — one unanalysable message logs an error panel and the
  worker moves on; it never takes down the monitor.
* **Backpressure** — if the queue fills, messages are dropped with an explicit
  error rather than stalling Telegram reception. Raise `WORKER_COUNT` or
  `QUEUE_MAXSIZE` if you see that message.
* **Prompt-injection resistance** — the group's text (and its metadata) is
  passed as delimited data and the system prompt instructs the model to treat
  it as untrusted content, never as instructions.
* **Graceful shutdown** — `Ctrl-C` stops intake, drains in-flight analyses (up
  to 30s), and prints a session summary.

---

## Troubleshooting

**`Could not resolve TELEGRAM_TARGET_CHAT`**
Your account must be a member of the group, and private invite links can't be
used directly. Join it, then use the numeric id (see above).

**`No authorised session found and stdin is not a terminal`**
The first login is interactive. Run `python main.py --check` once from a real
terminal; afterwards the `.session` file is reused and the app can run headless.

**`The session was revoked from another device`**
Delete `signal_monitor.session` and log in again.

**`database is locked`**
Two processes are sharing one session file. Run a single instance, or give each
one its own `TELEGRAM_SESSION`.

**`Could not find the Claude Code CLI on PATH`**
Install it (`npm install -g @anthropic-ai/claude-code`) and confirm `claude
--version` works in a fresh terminal. If it's installed somewhere not on
`PATH`, set `CLAUDE_CLI_PATH` to the full path (including `.cmd` on Windows).

**`Claude Code CLI is not signed in`**
Run `claude auth login` (without `--console`) and check `claude auth status`.
If you previously ran `claude auth login --console`, that's API-key billing —
log out and back in without the flag to use your subscription instead.

**`Analysis queue full`**
Messages arrive faster than they're analysed. Raise `WORKER_COUNT` (to 4–6).

**Nothing appears when messages arrive**
Run with `--log-level DEBUG`. If you see "Ignoring message from unrelated chat",
`TELEGRAM_TARGET_CHAT` points at the wrong group.

**Analysis is noticeably slower than you'd expect**
Each call spawns a fresh `claude` process, which loads hooks, CLAUDE.md, and
any configured MCP servers before it can respond — this is inherently slower
than a raw API call. Raise `WORKER_COUNT` to parallelise across messages rather
than trying to speed up a single call.

---

## Cost and rate limits

There's no per-token API billing here — every call runs through your Claude
subscription via the CLI, so usage is governed by whatever limits apply to your
plan, not a pay-per-token invoice. A busy group making many calls per minute
can still hit your plan's rate limits; if you see frequent retries in the logs,
spread load with a lower `WORKER_COUNT` or check your plan's usage limits at
<https://claude.com/pricing>.

---

## Security notes

* The `.session` file grants **full access to your Telegram account**. It is
  gitignored — treat it like a password and never share or commit it.
* Automating a user account is against Telegram's rules if used for spam or bulk
  actions. This app only *reads* one group you already belong to; it never
  posts, joins, or scrapes.
* Message text is sent to Claude (via the local CLI, under your account) for
  analysis. Don't point it at a group whose contents you aren't permitted to
  share.
* Nothing is stored unless you set `JSONL_OUTPUT` or `LOG_FILE`.
* Each analysis call runs with `--disallowedTools "*"` and
  `--setting-sources ""`, so it can't act on your filesystem, run hooks, or
  reach configured MCP servers — group content is analysed as inert text, not
  as instructions to an agent with tool access.

---

## Limitations

* Text only — images, charts and voice notes are skipped (their captions are
  analysed if present).
* Each message is analysed independently. A bare "TP2 hit ✅" reply is
  classified as a `result` with low confidence because the referenced setup is
  not visible; the reply id is included in the prompt as context.
* Nothing is traded. This is an analysis and logging tool, not an execution
  engine, and none of its output is financial advice.
