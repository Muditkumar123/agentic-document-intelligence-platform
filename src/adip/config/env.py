"""Small environment-file loader for local demos.

The project keeps real secrets out of tracked files. This loader supports a
minimal `.env` format so local API keys can be read without adding a runtime
dependency.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

DEFAULT_ENV_PATH = Path(".env")
ENV_VAR_NAME = "ADIP_ENV_FILE"
ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_project_env(path: Path | None = None, override: bool = False) -> dict[str, str]:
    env_path = path
    if env_path is None and os.getenv(ENV_VAR_NAME):
        env_path = Path(os.environ[ENV_VAR_NAME])
    if env_path is None:
        env_path = DEFAULT_ENV_PATH
    return load_env_file(env_path, override=override)


def load_env_file(path: Path, override: bool = False) -> dict[str, str]:
    env_path = path.expanduser()
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()

    key, separator, raw_value = line.partition("=")
    if not separator:
        return None
    key = key.strip()
    if not ENV_KEY_PATTERN.match(key):
        return None

    try:
        values = shlex.split(raw_value, comments=True, posix=True)
    except ValueError:
        values = [raw_value.strip().strip("'\"")]
    value = values[0] if values else ""
    return key, value
