#!/usr/bin/env bash
#
# Seeds runtime credentials onto the Railway volume, then starts the app.
#
# Two things this project needs cannot be supplied as ordinary environment
# variables, because both are *files* produced by an interactive login:
#
#   1. The Telethon session (`<TELEGRAM_SESSION>.session`) — created by a
#      one-time phone + code login. telegram_client.py refuses to prompt when
#      stdin isn't a TTY (which it never is on Railway), so the session must
#      already exist before the app starts.
#   2. The Claude Code CLI OAuth credentials (~/.claude/.credentials.json) —
#      created by `claude auth login`, a browser-based flow.
#
# Railway has no "secret file" upload (unlike some other platforms), so each
# is passed gzipped+base64-encoded in an environment variable and decoded here:
#
#   TELEGRAM_SESSION_B64     <TELEGRAM_SESSION>.session
#   CLAUDE_CREDENTIALS_B64   ~/.claude/.credentials.json
#
# Use `python railway_secrets.py` to generate both values.
#
# Both are written to the persistent volume and are then *self-maintaining*:
# on every later boot the volume copy wins, so Telethon's session updates and
# the Claude CLI's refreshed OAuth tokens survive restarts and redeploys
# rather than being reverted to whatever was first uploaded.
set -euo pipefail

# Railway injects RAILWAY_VOLUME_MOUNT_PATH automatically for the attached
# volume; fall back to /data so the image also runs locally with `docker run`.
DATA_DIR="${DATA_DIR:-${RAILWAY_VOLUME_MOUNT_PATH:-/data}}"

log() { echo "[entrypoint] $*" >&2; }

# base64 -> optional gunzip -> stdout.
#
# Tolerates the line wrapping and stray whitespace that copy-pasting into a
# dashboard field tends to introduce. railway_secrets.py gzips before
# encoding (a raw session is ~38 KB of base64, vs ~3 KB compressed), but an
# uncompressed value is still accepted: the gzip magic bytes are detected
# rather than assumed.
decode_secret() {
    local tmp
    tmp="$(mktemp)"
    if ! printf '%s' "$1" | tr -d '[:space:]' | base64 -d > "$tmp" 2>/dev/null; then
        rm -f "$tmp"
        return 1
    fi
    if [ "$(head -c 2 "$tmp" | od -An -tx1 | tr -d ' \n')" = "1f8b" ]; then
        gzip -dc "$tmp" || { rm -f "$tmp"; return 1; }
    else
        cat "$tmp"
    fi
    rm -f "$tmp"
}

# --- Persistent volume --------------------------------------------------------
mkdir -p "$DATA_DIR"
if [ ! -w "$DATA_DIR" ]; then
    log "FATAL: $DATA_DIR is not writable."
    log "       Attach a Railway volume and mount it at $DATA_DIR (Service ->"
    log "       Settings -> Volumes), or set DATA_DIR to a writable path."
    exit 1
fi
log "Persistent data directory: $DATA_DIR"

# --- Telethon session ---------------------------------------------------------
# TELEGRAM_SESSION is a path *prefix*; Telethon appends '.session' to it.
SESSION_PREFIX="${TELEGRAM_SESSION:-$DATA_DIR/signal_monitor}"
SESSION_TARGET="${SESSION_PREFIX}.session"

if [ -f "$SESSION_TARGET" ]; then
    # Already on the volume. Never re-seed over it: Telethon rewrites this
    # file as it runs, so the volume copy is newer than the uploaded one.
    log "Using existing Telegram session at $SESSION_TARGET"
elif [ -n "${TELEGRAM_SESSION_B64:-}" ]; then
    mkdir -p "$(dirname "$SESSION_TARGET")"
    if ! decode_secret "$TELEGRAM_SESSION_B64" > "$SESSION_TARGET"; then
        log "FATAL: TELEGRAM_SESSION_B64 is not valid base64."
        log "       Regenerate it with 'python railway_secrets.py'."
        rm -f "$SESSION_TARGET"
        exit 1
    fi
    chmod 600 "$SESSION_TARGET"
    # A Telethon session is a SQLite database; a truncated paste would fail
    # later with a confusing Telegram error instead of here.
    if ! head -c 16 "$SESSION_TARGET" | grep -q "SQLite format"; then
        log "FATAL: decoded session at $SESSION_TARGET is not a SQLite database."
        log "       The TELEGRAM_SESSION_B64 value was probably truncated when"
        log "       pasted. Regenerate with 'python railway_secrets.py'."
        rm -f "$SESSION_TARGET"
        exit 1
    fi
    log "Seeded Telegram session -> $SESSION_TARGET ($(wc -c < "$SESSION_TARGET") bytes)"
else
    log "FATAL: no Telegram session available."
    log "       Expected either $SESSION_TARGET on the volume, or the"
    log "       TELEGRAM_SESSION_B64 environment variable."
    log "       Generate it locally with 'python railway_secrets.py'."
    log "       The app cannot log in here: Telegram's phone-code prompt needs"
    log "       an interactive terminal, which a Railway worker does not have."
    exit 1
fi

# --- Claude Code CLI credentials ---------------------------------------------
# ~/.claude is redirected onto the volume so the CLI's own token refreshes are
# persisted. Without this the refreshed access token would live only inside the
# container and be lost on every restart.
CLAUDE_STATE_DIR="$DATA_DIR/claude"
CLAUDE_HOME="${HOME:-/root}/.claude"

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    log "ANTHROPIC_API_KEY is set — the Claude CLI will use API billing."
else
    mkdir -p "$CLAUDE_STATE_DIR"
    # Point ~/.claude at the volume (idempotent: safe to re-run every boot).
    if [ ! -L "$CLAUDE_HOME" ]; then
        rm -rf "$CLAUDE_HOME"
        ln -s "$CLAUDE_STATE_DIR" "$CLAUDE_HOME"
    fi
    # Without a working symlink the CLI would read a container-local path:
    # auth would appear fine on this boot, then silently fail to persist the
    # refreshed token. Better to stop here than to discover that days later.
    if [ ! -L "$CLAUDE_HOME" ]; then
        log "FATAL: could not symlink $CLAUDE_HOME -> $CLAUDE_STATE_DIR."
        log "       Claude credentials would not persist across restarts."
        exit 1
    fi

    if [ -f "$CLAUDE_STATE_DIR/.credentials.json" ]; then
        log "Using existing Claude credentials on the volume (token refreshes preserved)"
    elif [ -n "${CLAUDE_CREDENTIALS_B64:-}" ]; then
        if ! decode_secret "$CLAUDE_CREDENTIALS_B64" > "$CLAUDE_STATE_DIR/.credentials.json"; then
            log "FATAL: CLAUDE_CREDENTIALS_B64 is not valid base64."
            rm -f "$CLAUDE_STATE_DIR/.credentials.json"
            exit 1
        fi
        chmod 600 "$CLAUDE_STATE_DIR/.credentials.json"
        log "Seeded Claude CLI credentials -> $CLAUDE_STATE_DIR/.credentials.json"
    else
        log "FATAL: no Claude authentication available."
        log "       Provide ONE of:"
        log "         - CLAUDE_CREDENTIALS_B64 (base64 of ~/.claude/.credentials.json"
        log "           from a machine where 'claude auth login' has been run), or"
        log "         - ANTHROPIC_API_KEY (switches to metered API billing)."
        log "       Generate the first with 'python railway_secrets.py'."
        exit 1
    fi
fi

log "Starting: $*"
exec "$@"
