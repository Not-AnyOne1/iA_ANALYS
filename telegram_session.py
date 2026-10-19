"""Telegram session check and safe recovery, for non-interactive deployments.

A session FILE existing is not a working LOGIN. This module asks Telegram:

* ``probe`` — is this session authorised? ``authorized`` / ``unauthorized``
  (Telegram rejected it: logged out, revoked, duplicated, empty) /
  ``api_rejected`` (TELEGRAM_API_ID / TELEGRAM_API_HASH refused) /
  ``inconclusive`` (network, flood wait, anything else — no conclusion).
  An empty session (no authorization key) is ``unauthorized`` without
  connecting, so no new key is minted into the file.
* ``ensure`` — the policy the container entrypoint runs before the app:
  a working session is NEVER replaced; an inconclusive one is kept; only a
  session Telegram REJECTS is replaced from ``TELEGRAM_SESSION_B64``, and only
  after the replacement is decoded, structurally validated, shown to be a
  different authorization, and itself accepted by Telegram. The rejected
  file is kept beside it as ``<name>.session.rejected-<UTC time>``. Nothing
  else in the directory is touched.

Output carries no secret: no session bytes, no API hash, no phone number —
only short SHA-256 fingerprints, which identify a value without revealing
it, so the same session or credentials can be recognised on two machines.

    python telegram_session.py ensure         # in the container (reads env)
    python telegram_session.py fingerprint    # locally: fingerprints from .env
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import gzip
import hashlib
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

OK, FATAL, INCONCLUSIVE = 0, 3, 4


# ----------------------------------------------------------- fingerprints

def api_fingerprint(api_id: Any, api_hash: str) -> str:
    """Identifies an API id/hash pair without revealing it (the 128-bit hash
    makes the pair infeasible to recover from the digest)."""
    return hashlib.sha256(f"{str(api_id).strip()}:{api_hash.strip()}".encode()).hexdigest()[:12]


def _key_fingerprint(key: bytes) -> str:
    return hashlib.sha256(b"telegram-auth-key:" + key).hexdigest()[:12]


# --------------------------------------------------------- session files

@dataclass(frozen=True)
class SessionFile:
    exists: bool
    sqlite: bool = False
    has_auth_key: bool = False
    dc_id: Optional[int] = None
    fingerprint: Optional[str] = None       # of the authorization key — stable across runs

    @property
    def usable_structure(self) -> bool:
        return self.exists and self.sqlite and self.has_auth_key


def inspect_file(path: Path) -> SessionFile:
    """Read-only structural check of a Telethon SQLite session."""
    path = Path(path)
    if not path.is_file():
        return SessionFile(exists=False)
    with path.open("rb") as fh:
        if not fh.read(16).startswith(b"SQLite format 3"):
            return SessionFile(exists=True)
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT dc_id, auth_key FROM sessions LIMIT 1").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return SessionFile(exists=True, sqlite=True)
    if not row or not row[1]:
        return SessionFile(exists=True, sqlite=True, dc_id=row[0] if row else None)
    return SessionFile(exists=True, sqlite=True, has_auth_key=True, dc_id=row[0],
                       fingerprint=_key_fingerprint(bytes(row[1])))


def decode_b64(value: str) -> bytes:
    """base64 -> optional gunzip, tolerant of pasted line breaks (as the
    entrypoint's ``decode_secret``). Raises ValueError on garbage."""
    try:
        raw = base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("not valid base64") from exc
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise ValueError("gzip data is corrupt") from exc
    return raw


# ---------------------------------------------------------------- probing

@dataclass(frozen=True)
class Probe:
    status: str          # authorized / unauthorized / api_rejected / inconclusive
    reason: str


def _default_client(prefix: str, api_id: int, api_hash: str):
    from telethon import TelegramClient
    return TelegramClient(prefix, api_id, api_hash, connection_retries=2, retry_delay=2,
                          request_retries=1, timeout=15, flood_sleep_threshold=0,
                          receive_updates=False)


async def probe(prefix: str, api_id: int, api_hash: str, *,
                client_factory: Optional[Callable[..., Any]] = None, timeout: float = 60.0) -> Probe:
    """Ask Telegram whether the session ``<prefix>.session`` is authorised."""
    info = inspect_file(Path(prefix + ".session"))
    if not info.exists:
        return Probe("unauthorized", "no session file")
    if not info.sqlite:
        return Probe("unauthorized", "not a Telethon session (not an SQLite file)")
    if not info.has_auth_key:
        return Probe("unauthorized", "empty session (no authorization key) — never logged in")
    from telethon import errors
    from telethon.tl.functions.updates import GetStateRequest
    rejected = (errors.AuthKeyUnregisteredError, errors.SessionRevokedError, errors.SessionExpiredError,
                errors.UserDeactivatedError, errors.UserDeactivatedBanError,
                errors.AuthKeyDuplicatedError, errors.AuthKeyInvalidError, errors.UnauthorizedError)
    client = (client_factory or _default_client)(prefix, api_id, api_hash)
    try:
        await asyncio.wait_for(client.connect(), timeout)
        await asyncio.wait_for(client(GetStateRequest()), timeout)
        return Probe("authorized", "Telegram accepted the session")
    except errors.ApiIdInvalidError:
        return Probe("api_rejected", "Telegram rejected TELEGRAM_API_ID / TELEGRAM_API_HASH (ApiIdInvalidError)")
    except rejected as exc:
        return Probe("unauthorized", f"Telegram rejected the session ({type(exc).__name__})")
    except errors.RPCError as exc:
        return Probe("inconclusive", f"Telegram answered {type(exc).__name__}")
    except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
        return Probe("inconclusive", f"could not reach Telegram ({type(exc).__name__})")
    finally:
        with contextlib.suppress(Exception):
            await client.disconnect()
        with contextlib.suppress(Exception):
            client.session.close()


# ----------------------------------------------------------------- policy

@dataclass(frozen=True)
class Outcome:
    code: int            # OK / FATAL / INCONCLUSIVE
    action: str          # kept / replaced / missing / unauthorized / api_rejected / inconclusive
    detail: str


ProbeFn = Callable[[str, int, str], Awaitable[Probe]]


def _remove(path: Path) -> None:
    for p in (path, Path(str(path) + "-journal")):
        with contextlib.suppress(FileNotFoundError):
            p.unlink()


async def ensure(prefix: str, b64: Optional[str], api_id: int, api_hash: str, *,
                 probe_fn: Optional[ProbeFn] = None, now: Optional[datetime] = None) -> Outcome:
    """Keep a working session; replace a REJECTED one only with a verified one."""
    probe_fn = probe_fn or probe
    target = Path(prefix + ".session")
    if not target.exists():
        return Outcome(FATAL, "missing", f"no session at {target}")
    current = await probe_fn(prefix, api_id, api_hash)
    if current.status == "authorized":
        return Outcome(OK, "kept", f"{target} is authorised — kept unchanged")
    if current.status == "api_rejected":
        return Outcome(FATAL, "api_rejected", f"{current.reason}; replacing the session cannot fix this — "
                                              "check the API variables. Session kept unchanged")
    if current.status != "unauthorized":
        return Outcome(INCONCLUSIVE, "inconclusive", f"could not verify {target} ({current.reason}) — "
                                                     "kept unchanged")
    if not b64:
        return Outcome(FATAL, "unauthorized", f"{target}: {current.reason}; no TELEGRAM_SESSION_B64 to "
                                              "recover from")
    try:
        data = decode_b64(b64)
    except ValueError as exc:
        return Outcome(FATAL, "unauthorized", f"{target}: {current.reason}; TELEGRAM_SESSION_B64 is unusable "
                                              f"({exc}) — nothing changed")
    candidate = target.parent / f".session-candidate-{os.getpid()}.session"
    try:
        candidate.write_bytes(data)
        with contextlib.suppress(OSError):
            candidate.chmod(0o600)
        info = inspect_file(candidate)
        if not info.usable_structure:
            return Outcome(FATAL, "unauthorized", f"{target}: {current.reason}; TELEGRAM_SESSION_B64 is not "
                                                  "a logged-in Telethon session — nothing changed")
        if info.fingerprint == inspect_file(target).fingerprint:
            return Outcome(FATAL, "unauthorized", f"{target}: {current.reason}; TELEGRAM_SESSION_B64 holds "
                                                  "this same rejected login — create a new session locally "
                                                  "and update the variable. Nothing changed")
        verdict = await probe_fn(str(candidate)[: -len(".session")], api_id, api_hash)
        if verdict.status != "authorized":
            return Outcome(FATAL, "unauthorized", f"{target}: {current.reason}; the TELEGRAM_SESSION_B64 "
                                                  f"replacement is not usable either ({verdict.reason}) — "
                                                  "nothing changed")
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        backup = target.with_name(f"{target.name}.rejected-{stamp}")
        shutil.copy2(target, backup)                       # the rejected login, kept for inspection
        os.replace(candidate, target)                      # atomic: the target always exists
        _remove(Path(str(target) + "-journal"))
        with contextlib.suppress(OSError):
            target.chmod(0o600)
        return Outcome(OK, "replaced", f"{target} was rejected ({current.reason}); replaced by the verified "
                                       f"TELEGRAM_SESSION_B64 session; rejected file kept as {backup.name}")
    finally:
        _remove(candidate)


# -------------------------------------------------------------------- CLI

def _say(msg: str) -> None:
    print(f"[session] {msg}", file=sys.stderr, flush=True)


def main(argv: Optional[list] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    command = args[0] if args else "ensure"
    if command == "fingerprint":
        from config import Settings
        s = Settings.load(".env")
        info = inspect_file(Path(s.session_path + ".session"))
        _say(f"local session {s.session_path}.session: exists={info.exists} logged-in structure="
             f"{info.usable_structure} dc={info.dc_id} auth key fp={info.fingerprint}")
        _say(f"api fp={api_fingerprint(s.api_id, s.api_hash)}")
        return OK if info.usable_structure else FATAL
    if command != "ensure":
        _say(f"unknown command {command!r}")
        return 2
    prefix = os.getenv("TELEGRAM_SESSION", "").strip() or "signal_monitor"
    api_id_raw, api_hash = os.getenv("TELEGRAM_API_ID", "").strip(), os.getenv("TELEGRAM_API_HASH", "").strip()
    if not api_id_raw.isdigit() or not api_hash:
        _say("TELEGRAM_API_ID / TELEGRAM_API_HASH missing or malformed")
        return FATAL
    before = inspect_file(Path(prefix + ".session"))
    _say(f"checking {prefix}.session with Telegram (auth key fp={before.fingerprint}, "
         f"api fp={api_fingerprint(api_id_raw, api_hash)})")
    try:
        outcome = asyncio.run(ensure(prefix, os.getenv("TELEGRAM_SESSION_B64"), int(api_id_raw), api_hash))
    except Exception as exc:  # noqa: BLE001 - a checker bug must never block startup
        _say(f"check failed unexpectedly ({type(exc).__name__}) — session left unchanged")
        return INCONCLUSIVE
    after = inspect_file(Path(prefix + ".session"))
    _say(f"{outcome.action}: {outcome.detail} (auth key fp now={after.fingerprint})")
    return outcome.code


if __name__ == "__main__":
    sys.exit(main())
