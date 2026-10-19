"""Telegram session check and recovery: never trust file existence, never
overwrite a working login, replace a rejected one only with a verified one."""

from __future__ import annotations

import asyncio
import base64
import gzip
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon import errors

import telegram_session as ts

KEY_A, KEY_B = b"\xa1" * 256, b"\xb2" * 256
API_HASH = "0123456789abcdef" * 2          # a placeholder, built at runtime
NOW = datetime(2026, 10, 19, 8, 30, tzinfo=timezone.utc)


def _session(path: Path, key=KEY_A, dc=4) -> Path:
    """A minimal Telethon-format SQLite session."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE version (version INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE sessions (dc_id INTEGER PRIMARY KEY, server_address TEXT, port INTEGER, "
                "auth_key BLOB, takeout_id INTEGER)")
    con.execute("INSERT INTO version VALUES (7)")
    con.execute("INSERT INTO sessions VALUES (?, '149.154.167.91', 443, ?, NULL)", (dc, key))
    con.commit()
    con.close()
    return path


def _b64(path: Path, compress=True) -> str:
    data = path.read_bytes()
    return base64.b64encode(gzip.compress(data) if compress else data).decode()


# ----------------------------------------------------------------- files

def test_a_logged_in_session_is_recognised_and_fingerprinted(tmp_path):
    info = ts.inspect_file(_session(tmp_path / "s.session"))
    assert info.usable_structure and info.dc_id == 4
    assert info.fingerprint == ts.inspect_file(_session(tmp_path / "copy.session")).fingerprint
    assert info.fingerprint != ts.inspect_file(_session(tmp_path / "b.session", KEY_B)).fingerprint


def test_existence_alone_is_not_a_login(tmp_path):
    assert not ts.inspect_file(tmp_path / "missing.session").exists
    (tmp_path / "junk.session").write_bytes(b"not sqlite at all")
    assert not ts.inspect_file(tmp_path / "junk.session").usable_structure
    empty = _session(tmp_path / "empty.session", key=None)
    assert ts.inspect_file(empty).exists and not ts.inspect_file(empty).has_auth_key


def test_the_variable_decodes_with_or_without_gzip_and_line_breaks(tmp_path):
    p = _session(tmp_path / "s.session")
    wrapped = "\n".join(_b64(p)[i:i + 60] for i in range(0, len(_b64(p)), 60))
    assert ts.decode_b64(wrapped) == p.read_bytes() == ts.decode_b64(_b64(p, compress=False))
    with pytest.raises(ValueError):
        ts.decode_b64("@@@ not base64 @@@")


def test_fingerprints_never_contain_the_secret():
    fp = ts.api_fingerprint(1234567, API_HASH)
    assert len(fp) == 12 and API_HASH not in fp and "1234567" not in fp
    assert fp == ts.api_fingerprint("1234567", API_HASH) != ts.api_fingerprint(1234567, API_HASH[::-1])


# ----------------------------------------------------------------- probe

class _Client:
    def __init__(self, exc=None):
        self.exc, self.connected, self.closed = exc, False, False
        self.session = SimpleNamespace(close=lambda: setattr(self, "closed", True))

    async def connect(self):
        self.connected = True

    async def __call__(self, request):
        if self.exc:
            raise self.exc
        return object()

    async def disconnect(self):
        pass


@pytest.mark.parametrize("exc,status", [
    (None, "authorized"),
    (errors.AuthKeyUnregisteredError(request=None), "unauthorized"),
    (errors.SessionRevokedError(request=None), "unauthorized"),
    (errors.AuthKeyDuplicatedError(request=None), "unauthorized"),
    (errors.ApiIdInvalidError(request=None), "api_rejected"),
    (errors.FloodWaitError(request=None, capture=30), "inconclusive"),
    (ConnectionError("down"), "inconclusive"),
])
def test_telegrams_answer_is_classified(tmp_path, exc, status):
    _session(tmp_path / "s.session")
    client = _Client(exc)
    result = asyncio.run(ts.probe(str(tmp_path / "s"), 1, API_HASH, client_factory=lambda *a: client))
    assert result.status == status and client.closed


def test_an_empty_session_is_rejected_without_connecting(tmp_path):
    """Connecting an empty session would mint a fresh key into the file."""
    _session(tmp_path / "s.session", key=None)
    made = []
    result = asyncio.run(ts.probe(str(tmp_path / "s"), 1, API_HASH,
                                  client_factory=lambda *a: made.append(a) or _Client()))
    assert result.status == "unauthorized" and "empty" in result.reason and made == []


# ---------------------------------------------------------------- ensure

def _volume(tmp_path):
    """A /data-like volume with everything the bot keeps there."""
    data = tmp_path / "data"
    _session(data / "signal_monitor.session", KEY_A)
    (data / "signals.db").write_bytes(b"database bytes")
    (data / "claude").mkdir()
    (data / "claude" / ".credentials.json").write_text("{\"secret\": true}")
    (data / "logs").mkdir()
    (data / "logs" / "bot.log").write_text("log line\n")
    return data


def _snapshot(directory: Path) -> dict:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def _probe_by_key(verdicts):
    """A fake probe answering by the session's authorization key."""
    async def fake(prefix, api_id, api_hash):
        info = ts.inspect_file(Path(prefix + ".session"))
        return verdicts[info.fingerprint]
    return fake


def _fp(key):
    return ts._key_fingerprint(key)


def _run(data, b64, verdicts):
    return asyncio.run(ts.ensure(str(data / "signal_monitor"), b64, 1, API_HASH,
                                 probe_fn=_probe_by_key(verdicts), now=NOW))


def test_a_working_session_is_never_replaced(tmp_path):
    data = _volume(tmp_path)
    before = _snapshot(data)
    other = _b64(_session(tmp_path / "new.session", KEY_B))
    out = _run(data, other, {_fp(KEY_A): ts.Probe("authorized", "ok"), _fp(KEY_B): ts.Probe("authorized", "ok")})
    assert out.code == ts.OK and out.action == "kept"
    assert _snapshot(data) == before


def test_an_unverifiable_session_is_kept(tmp_path):
    data = _volume(tmp_path)
    before = _snapshot(data)
    other = _b64(_session(tmp_path / "new.session", KEY_B))
    out = _run(data, other, {_fp(KEY_A): ts.Probe("inconclusive", "could not reach Telegram"),
                             _fp(KEY_B): ts.Probe("authorized", "ok")})
    assert out.code == ts.INCONCLUSIVE and _snapshot(data) == before


def test_rejected_api_credentials_never_trigger_a_replacement(tmp_path):
    data = _volume(tmp_path)
    before = _snapshot(data)
    other = _b64(_session(tmp_path / "new.session", KEY_B))
    out = _run(data, other, {_fp(KEY_A): ts.Probe("api_rejected", "ApiIdInvalidError"),
                             _fp(KEY_B): ts.Probe("authorized", "ok")})
    assert out.code == ts.FATAL and out.action == "api_rejected" and _snapshot(data) == before


def test_a_rejected_session_is_replaced_by_a_verified_one_and_kept_as_backup(tmp_path):
    data = _volume(tmp_path)
    before = _snapshot(data)
    fresh = _session(tmp_path / "new.session", KEY_B)
    out = _run(data, _b64(fresh), {_fp(KEY_A): ts.Probe("unauthorized", "AuthKeyUnregisteredError"),
                                   _fp(KEY_B): ts.Probe("authorized", "ok")})
    after = _snapshot(data)
    assert out.code == ts.OK and out.action == "replaced"
    assert after["signal_monitor.session"] == fresh.read_bytes()
    assert after["signal_monitor.session.rejected-20261019T083000Z"] == before["signal_monitor.session"]
    # Database, Claude credentials and logs untouched; no candidate left behind.
    for name in ("signals.db", "claude/.credentials.json", "logs/bot.log"):
        assert after[name] == before[name]
    assert set(after) == set(before) | {"signal_monitor.session.rejected-20261019T083000Z"}


def test_a_replacement_that_telegram_also_rejects_changes_nothing(tmp_path):
    data = _volume(tmp_path)
    before = _snapshot(data)
    other = _b64(_session(tmp_path / "new.session", KEY_B))
    out = _run(data, other, {_fp(KEY_A): ts.Probe("unauthorized", "SessionRevokedError"),
                             _fp(KEY_B): ts.Probe("unauthorized", "AuthKeyUnregisteredError")})
    assert out.code == ts.FATAL and "nothing changed" in out.detail and _snapshot(data) == before


def test_the_same_rejected_login_in_the_variable_is_refused_without_a_second_probe(tmp_path):
    data = _volume(tmp_path)
    before = _snapshot(data)
    same = _b64(data / "signal_monitor.session")
    out = _run(data, same, {_fp(KEY_A): ts.Probe("unauthorized", "AuthKeyUnregisteredError")})
    assert out.code == ts.FATAL and "same rejected login" in out.detail and _snapshot(data) == before


@pytest.mark.parametrize("value", [None, "", "@@@ not base64 @@@", base64.b64encode(b"not a session").decode()])
def test_a_missing_or_unusable_variable_changes_nothing(tmp_path, value):
    data = _volume(tmp_path)
    before = _snapshot(data)
    out = _run(data, value, {_fp(KEY_A): ts.Probe("unauthorized", "AuthKeyUnregisteredError")})
    assert out.code == ts.FATAL and _snapshot(data) == before


def test_the_cli_prints_fingerprints_never_secrets(tmp_path, monkeypatch, capsys):
    data = _volume(tmp_path)
    monkeypatch.setenv("TELEGRAM_SESSION", str(data / "signal_monitor"))
    monkeypatch.setenv("TELEGRAM_API_ID", "1234567")
    monkeypatch.setenv("TELEGRAM_API_HASH", API_HASH)
    b64 = _b64(_session(tmp_path / "new.session", KEY_B))
    monkeypatch.setenv("TELEGRAM_SESSION_B64", b64)

    async def rejected(prefix, api_id, api_hash):
        return ts.Probe("unauthorized" if _fp(KEY_A) == ts.inspect_file(Path(prefix + ".session")).fingerprint
                        else "authorized", "fake")
    monkeypatch.setattr(ts, "probe", rejected)
    assert ts.main(["ensure"]) == ts.OK
    err = capsys.readouterr().err
    assert "[session] replaced:" in err and f"api fp={ts.api_fingerprint('1234567', API_HASH)}" in err
    assert API_HASH not in err and b64[:40] not in err and KEY_B.hex()[:16] not in err


# ------------------------------------------------- the app's own diagnosis

class _TelethonFake:
    """Stands in for telethon.TelegramClient inside telegram_client."""
    behaviour = {}

    def __init__(self, session, api_id, api_hash, **kw):
        self.session = SimpleNamespace(auth_key=self.behaviour.get("auth_key"))

    async def connect(self):
        pass

    async def is_user_authorized(self):
        return False

    async def __call__(self, request):
        raise self.behaviour["error"]


@pytest.mark.parametrize("auth_key,error,expect", [
    (None, None, "held no login"),
    (b"k", errors.AuthKeyUnregisteredError(request=None), "AuthKeyUnregisteredError"),
    (b"k", errors.ApiIdInvalidError(request=None), "TELEGRAM_API_ID / TELEGRAM_API_HASH"),
])
def test_the_app_names_the_real_reason(tmp_path, monkeypatch, auth_key, error, expect):
    import telegram_client as tc
    monkeypatch.setattr(tc, "TelegramClient", _TelethonFake)
    _TelethonFake.behaviour = {"auth_key": auth_key, "error": error}
    monkeypatch.setattr(tc.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    settings = SimpleNamespace(session_path=str(tmp_path / "signal_monitor"), api_id=1, api_hash=API_HASH,
                               phone="", target_chat="x", analyse_edits=False)
    monitor = tc.TelegramMonitor(settings, asyncio.Queue())
    with pytest.raises(tc.TelegramError) as info:
        asyncio.run(monitor._authorize())
    message = str(info.value)
    assert "No authorised session found and stdin is not a terminal" in message and expect in message
    assert str(tmp_path) in message or expect != "held no login"
    assert API_HASH not in message


# ------------------------------------------------------------ entrypoint

BASH = shutil.which("bash")


def _entrypoint(tmp_path, *, rc=0, check=True, existing: bytes = None):
    root = Path(__file__).resolve().parent.parent
    (tmp_path / "docker-entrypoint.sh").write_bytes(
        (root / "docker-entrypoint.sh").read_bytes().replace(b"\r\n", b"\n"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_bytes(
        b'#!/usr/bin/env bash\necho "$TELEGRAM_SESSION" > "$STUB_SEEN"\nexit "$STUB_RC"\n')
    fresh = _session(tmp_path / "fresh.session", KEY_B)
    if existing is not None:
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "signal_monitor.session").write_bytes(existing)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TELEGRAM_")}
    env.update(DATA_DIR="./data", ANTHROPIC_API_KEY="unused-in-test", TELEGRAM_SESSION_B64=_b64(fresh),
               STUB_RC=str(rc), STUB_SEEN="./seen.txt", TELEGRAM_SESSION_CHECK="1" if check else "0")
    proc = subprocess.run(
        [BASH, "-c", 'chmod +x bin/python; export PATH="$PWD/bin:$PATH"; '
                     'bash ./docker-entrypoint.sh bash -c "echo STARTED session=\\$TELEGRAM_SESSION"'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    return proc, fresh


pytestmark_bash = pytest.mark.skipif(BASH is None, reason="bash not available")


@pytestmark_bash
def test_the_entrypoint_exports_the_prefix_it_seeded(tmp_path):
    proc, fresh = _entrypoint(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "STARTED session=./data/signal_monitor" in proc.stdout       # the app opens THIS file
    assert (tmp_path / "seen.txt").read_text().strip() == "./data/signal_monitor"
    assert (tmp_path / "data" / "signal_monitor.session").read_bytes() == fresh.read_bytes()


@pytestmark_bash
def test_the_entrypoint_stops_when_the_session_is_unusable(tmp_path):
    proc, _ = _entrypoint(tmp_path, rc=3)
    assert proc.returncode == 1 and "STARTED" not in proc.stdout and "cannot be used" in proc.stderr


@pytestmark_bash
def test_the_entrypoint_starts_when_the_check_is_inconclusive(tmp_path):
    proc, _ = _entrypoint(tmp_path, rc=4)
    assert proc.returncode == 0 and "STARTED" in proc.stdout and "could not be verified" in proc.stderr


@pytestmark_bash
def test_the_check_can_be_switched_off_and_an_existing_volume_session_is_not_reseeded(tmp_path):
    existing = _session(tmp_path / "old.session", KEY_A).read_bytes()
    proc, _ = _entrypoint(tmp_path, check=False, existing=existing)
    assert proc.returncode == 0 and "STARTED" in proc.stdout and "check skipped" in proc.stderr
    assert not (tmp_path / "seen.txt").exists()
    assert (tmp_path / "data" / "signal_monitor.session").read_bytes() == existing
