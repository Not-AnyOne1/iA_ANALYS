# syntax=docker/dockerfile:1
#
# Image for the Telegram signal monitor (Railway worker service).
#
# Two runtimes are needed, not one:
#   - Python 3.11 runs the application itself.
#   - Node.js is required because claude_client.py/decision_engine.py shell
#     out to the Claude Code CLI (`claude -p ...`), which is an npm package,
#     not a pip one. Without it, ClaudeAnalyzer's constructor raises
#     ConfigError("Could not find the Claude Code CLI ... on PATH").
#
# Matches the locally verified interpreter (Python 3.11.5).
FROM python:3.11-slim

# --- OS-level dependencies ---------------------------------------------------
# curl/ca-certificates are needed to fetch the NodeSource setup script; they're
# removed again afterwards so they don't linger in the final image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @anthropic-ai/claude-code \
 && apt-get purge -y --auto-remove curl \
 && rm -rf /var/lib/apt/lists/*

# --- Python dependencies -----------------------------------------------------
# Copied and installed before the source so this layer stays cached whenever
# only application code changes.
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Application source ------------------------------------------------------
# .dockerignore keeps .env, *.session, tests/ and local caches out of this.
COPY . .

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# PYTHONUNBUFFERED is what makes logs show up live in Railway's log viewer:
# without it Python block-buffers stdout/stderr when they aren't a TTY (which
# they never are on Railway), so log lines would appear minutes late or only
# when the process exits. logging_setup.py writes to stderr; Railway captures
# both streams.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Fallback persistent-data location. On Railway the entrypoint prefers
# RAILWAY_VOLUME_MOUNT_PATH, which the platform injects for the attached
# volume; this default only applies when running the image elsewhere
# (e.g. `docker run -v ...` locally).
ENV DATA_DIR=/data

# The container runs as root deliberately: Railway mounts volumes root-owned,
# and a non-root user would need a chown on every boot to write the SQLite
# database and session file to the volume. Keeping root avoids that failure
# mode; the container is the only thing in its sandbox.

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# A worker has no HTTP port and no health-check endpoint — liveness is "the
# process is still running", and Railway's restart policy (railway.json)
# brings it back if this command exits. See README: Restarts and reconnection.
CMD ["python", "main.py"]
