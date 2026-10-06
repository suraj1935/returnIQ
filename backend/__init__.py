"""Loads <repo>/.env into os.environ on first import (existing environment variables win)."""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path | None = None) -> None:
    f = path or Path(__file__).resolve().parent.parent / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip().strip('"').strip("'")
        if k.strip() and v:
            os.environ.setdefault(k.strip(), v)


load_env()
