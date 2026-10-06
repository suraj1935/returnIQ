"""Shared DB access. Writers use DATABASE_URL; the Copilot uses a read-only connection."""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "returniq_dev.db"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB}")


def get_engine() -> Engine:
    return create_engine(database_url(), future=True)


def get_readonly_engine() -> Engine:
    """Read-only engine for the Copilot.
    Postgres: set DATABASE_URL_READONLY to a role with SELECT on analytics_* only.
    SQLite (dev): opened with mode=ro so writes fail at the driver level."""
    url = os.environ.get("DATABASE_URL_READONLY")
    if url:
        return create_engine(url, future=True)
    base = database_url()
    if base.startswith("sqlite:///"):
        path = base.removeprefix("sqlite:///")
        return create_engine(f"sqlite:///file:{path}?mode=ro&uri=true", future=True)
    raise RuntimeError("Set DATABASE_URL_READONLY for non-SQLite databases")
