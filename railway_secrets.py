"""Generate the base64 environment variables Railway needs.

Railway has no secret-file upload, so the two file-based credentials this
project requires are passed as base64 environment variables and decoded by
``docker-entrypoint.sh`` at boot:

    TELEGRAM_SESSION_B64     the Telethon session (created by an interactive
                             phone + code login on this machine)
    CLAUDE_CREDENTIALS_B64   ~/.claude/.credentials.json (created by
                             `claude auth login`)

Usage:

    python railway_secrets.py              # print both values
    python railway_secrets.py --out .railway-secrets.txt

This is a deployment helper only — nothing in the running application imports
it, and it is excluded from the deployed image by .dockerignore.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import os
import sys
from pathlib import Path


def _encode(path: Path) -> str:
    """gzip, then base64.

    The Telethon session is a ~28 KB SQLite database, which is ~38 KB as plain
    base64 — big enough to run into environment-variable size limits. It
    compresses to about 8% of that, so gzipping first keeps the variable small
    (~3 KB). docker-entrypoint.sh detects gzip by magic bytes, so an
    uncompressed value still works.
    """
    return base64.b64encode(gzip.compress(path.read_bytes(), 9)).decode("ascii")


def _session_path() -> Path:
    """Resolve the Telethon session the same way config.py does.

    TELEGRAM_SESSION is a path *prefix*; Telethon appends '.session'.
    """
    # Read .env without importing config.py, which would demand every other
    # required variable just to locate one file.
    prefix = os.getenv("TELEGRAM_SESSION", "").strip()
    if not prefix:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("TELEGRAM_SESSION=") and not line.startswith("#"):
                    prefix = line.split("=", 1)[1].strip()
                    break
    return Path(f"{prefix or 'signal_monitor'}.session")


def _credentials_path() -> Path:
    return Path.home() / ".claude" / ".credentials.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="railway_secrets",
        description="Generate the base64 env-var values for a Railway deployment.",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="write to FILE instead of stdout (add it to .gitignore!)",
    )
    args = parser.parse_args(argv)

    session = _session_path()
    credentials = _credentials_path()

    problems: list[str] = []
    if not session.exists():
        problems.append(
            f"Telegram session not found at '{session}'.\n"
            "  Create it by logging in once interactively:  python main.py --list-chats"
        )
    if not credentials.exists():
        problems.append(
            f"Claude credentials not found at '{credentials}'.\n"
            "  Create them with:  claude auth login\n"
            "  (Skip this if you plan to set ANTHROPIC_API_KEY on Railway instead.)"
        )

    # A missing Claude credentials file is survivable (the API-key route);
    # a missing session is not.
    if not session.exists():
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    for problem in problems:
        print(f"WARNING: {problem}\n", file=sys.stderr)

    lines = [
        "# Railway environment variables — paste into:",
        "#   Service -> Variables -> Raw Editor",
        "#",
        "# TREAT THIS OUTPUT AS A PASSWORD. The session grants full access to",
        "# your Telegram account; the credentials grant access to your Claude",
        "# account. Never commit it, paste it in a chat, or share a screenshot.",
        "",
        f"TELEGRAM_SESSION_B64={_encode(session)}",
    ]
    if credentials.exists():
        lines.append(f"CLAUDE_CREDENTIALS_B64={_encode(credentials)}")

    text = "\n".join(lines) + "\n"

    if args.out:
        out = Path(args.out)
        out.write_text(text, encoding="utf-8")
        try:
            out.chmod(0o600)
        except OSError:
            pass  # best-effort; Windows ACLs don't map cleanly
        print(f"Wrote {out} ({len(text)} bytes).", file=sys.stderr)
        print("Delete it once the values are in Railway.", file=sys.stderr)
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
