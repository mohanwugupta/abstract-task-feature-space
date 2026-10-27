"""Load named API credentials from the shell or a project-local .env file.

Only explicit live execution calls this module's loader. Values are literal,
never evaluated as shell commands, and never returned for logging/serialization.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex


def load_api_credentials(config):
    names = []
    for role in ("generator", "judge"):
        name = config[role].get("api_key_env")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ):
            raise ValueError("Configure a valid API-key environment variable name")
        if name not in names:
            names.append(name)

    missing = [name for name in names if not os.environ.get(name)]
    if not missing:
        return

    path = Path(config["root"]) / ".env"
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        lines = []
    except (OSError, UnicodeError):
        raise ValueError("Cannot read project .env as UTF-8") from None

    values = {}
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(
            r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line
        )
        if not match or match[1] not in missing:
            continue
        name, text = match.groups()
        try:
            parts = shlex.split(text, comments=True, posix=True)
        except ValueError:
            raise ValueError(
                f"Invalid .env value for {name} at line {number}"
            ) from None
        if len(parts) > 1:
            raise ValueError(f"Invalid .env value for {name} at line {number}")
        values[name] = parts[0] if parts else ""

    unresolved = [name for name in missing if not values.get(name)]
    if unresolved:
        raise ValueError(
            f"Missing API key: {', '.join(unresolved)}. "
            "Set it in the project .env file or export it in your shell."
        )
    for name in missing:
        os.environ[name] = values[name]
