"""Fail closed when a public snapshot contains common private material."""

from __future__ import annotations

import re
import sys
from pathlib import Path


PATTERNS = {
    "credential": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret(?:[_-]?key)?|password)"
        r"\s*[:=]\s*['\"]?[^\s'\"]+"
    ),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "internal_host": re.compile(r"(?i)(?:brainpp\.cn|shaipower\.com|basemind\.com|rjob\b|juicefs\+s3://)"),
    "private_mount": re.compile(r"/(?:mnt|home)/(?:ws-jfs|step2_alignment_jfs|i-[a-z0-9_-]+)"),
}
SKIP = {".git", ".pytest_cache", "__pycache__", ".venv", "dist", "build"}


def main(root: str = ".") -> int:
    root_path = Path(root).resolve()
    scanner_path = Path(__file__).resolve()
    findings: list[str] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path == scanner_path or any(part in SKIP for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(root_path)}:{line}: {name}")
    if findings:
        print("Public snapshot check failed:")
        print("\n".join(sorted(findings)))
        return 1
    print("Public snapshot check passed: no blocked credential, host, or mount patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
