"""Fingerprint helpers for reproducible MLOps records."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_fingerprint(path: Path) -> dict[str, str]:
    """Return a stable mapping of relative file paths to SHA-256 checksums."""
    root = path.expanduser().resolve()
    if not root.exists():
        return {}
    if root.is_file():
        return {root.name: file_sha256(root)}

    fingerprints: dict[str, str] = {}
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file():
            fingerprints[str(candidate.relative_to(root))] = file_sha256(candidate)
    return fingerprints


def git_commit(cwd: Path | None = None) -> str | None:
    """Return the current git commit when available."""
    command = ["git", "rev-parse", "HEAD"]
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_dirty_status(cwd: Path | None = None) -> str | None:
    """Return short git status output when available."""
    command = ["git", "status", "--short"]
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
