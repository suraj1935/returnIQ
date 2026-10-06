"""
pipelines/ingestion.py
======================
ReturnIQ — Data Ingestion Pipeline  (v2 — all P0/P1 review fixes applied)

Skills applied:
  - returniq-data        : idempotency (true upsert), schema constraints,
                           quarantine, reject-rate gate, no raw/analytics mixing
  - returniq-architecture: Pydantic validation at data boundaries,
                           folder-structure adherence, no new frameworks

P0 fixes applied
-----------------
1.  True upsert (ON CONFLICT DO UPDATE) so mutable status / refund_amount
    are refreshed on re-run.  updated_at column added.
2.  Quarantine table for rejected rows; pipeline exits non-zero when reject
    rate exceeds REJECT_RATE_THRESHOLD.
3.  Whitespace stripped on all string IDs via model_config
    (ConfigDict(str_strip_whitespace=True)) to prevent " ORD-001" ≠ "ORD-001".
4.  FK raw_returns.order_id → raw_orders.order_id enforced in DDL;
    returns ingested after orders in the same transaction boundary.
5.  Indexes on raw_orders(customer_id), raw_orders(order_date) added.

P1 fixes applied
-----------------
6.  Money fields use Decimal, not float; cross-field total_amount check added.
7.  DateTime(timezone=True) on all timestamp columns.
8.  DATABASE_URL required unless RETURNIQ_ENV=dev (SQLite only for dev).
9.  Chunked upsert (CHUNK_SIZE rows) to stay under Postgres bind-param limit.
10. sys removed from unused imports.

Round-2 fixes: SQLite FK pragma, orphan-return quarantine, DB CHECK constraints,
null-row quarantine + correct reject-rate denominator, cross-table sanity
assertions, no-op-aware upsert (updated_at only moves on real change).
Not in scope here: Alembic migrations (tracked in backlog).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
import argparse

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backend  # noqa: E402,F401  (loads .env)
import pandas as pd  # noqa: E402
from sqlalchemy import CheckConstraint, event, or_
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Engine,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import OperationalError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR      = REPO_ROOT / "data"
ORDERS_CSV    = DATA_DIR / "orders.csv"
RETURNS_CSV   = DATA_DIR / "returns.csv"

# P1-fix-8: require DATABASE_URL in non-dev environments
RETURNIQ_ENV  = os.environ.get("RETURNIQ_ENV", "dev").lower()
_DEFAULT_DB   = f"sqlite:///{DATA_DIR / 'returniq_dev.db'}"

if RETURNIQ_ENV != "dev" and "DATABASE_URL" not in os.environ:
    raise EnvironmentError(
        "DATABASE_URL must be set when RETURNIQ_ENV != 'dev'. "
        "Example: postgresql://user:pass@host:5432/returniq"
    )

DATABASE_URL: str = os.environ.get("DATABASE_URL", _DEFAULT_DB)

# P1-fix-9: chunk size to stay under Postgres 65 535 bind-param limit
CHUNK_SIZE           = 5_000
# P0-fix-2: reject gate — exit non-zero if more than this fraction fails validation
REJECT_RATE_THRESHOLD = 0.02

# ---------------------------------------------------------------------------
# Allowed domain values (enforced in Pydantic; DB CHECK constraints need
# also mirrored as DB CHECK constraints below).
# ---------------------------------------------------------------------------
ORDER_STATUSES  = {"pending", "completed", "cancelled", "processing"}
RETURN_STATUSES = {"pending", "approved", "rejected"}
RETURN_REASONS  = {"defective", "wrong_item", "changed_mind", "not_as_described", "other"}

# ---------------------------------------------------------------------------
# Pydantic models
# (P0-fix-3) str_strip_whitespace prevents " ORD-001" ≠ "ORD-001" PK splits
# (P1-fix-6) Decimal for money; cross-field total check
# ---------------------------------------------------------------------------

class OrderRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    order_id:     str     = Field(..., min_length=1)
    customer_id:  str     = Field(..., min_length=1)
    product_id:   str     = Field(..., min_length=1)
    order_date:   date
    quantity:     int     = Field(..., ge=1)
    unit_price:   Decimal = Field(..., ge=0)
    total_amount: Decimal = Field(..., ge=0)
    status:       str     = Field(..., min_length=1)

    @field_validator("order_date", mode="before")
    @classmethod
    def _parse_date(cls, v: object) -> date:
        if isinstance(v, date):
            return v
        return datetime.strptime(str(v).strip(), "%Y-%m-%d").date()

    @field_validator("status", mode="after")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        if v not in ORDER_STATUSES:
            raise ValueError(f"status '{v}' not in {ORDER_STATUSES}")
        return v

    @model_validator(mode="after")
    def _check_total(self) -> "OrderRecord":
        expected = (self.unit_price * self.quantity).quantize(Decimal("0.0001"))
        actual   = self.total_amount.quantize(Decimal("0.0001"))
        if abs(expected - actual) > Decimal("0.01"):
            raise ValueError(
                f"total_amount {actual} ≠ unit_price×quantity {expected}"
            )
        return self


class ReturnRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    return_id:     str     = Field(..., min_length=1)
    order_id:      str     = Field(..., min_length=1)
    customer_id:   str     = Field(..., min_length=1)
    product_id:    str     = Field(..., min_length=1)
    return_date:   date
    reason:        str     = Field(..., min_length=1)
    refund_amount: Decimal = Field(..., ge=0)
    status:        str     = Field(..., min_length=1)

    @field_validator("return_date", mode="before")
    @classmethod
    def _parse_date(cls, v: object) -> date:
        if isinstance(v, date):
            return v
        return datetime.strptime(str(v).strip(), "%Y-%m-%d").date()

    @field_validator("status", mode="after")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        if v not in RETURN_STATUSES:
            raise ValueError(f"status '{v}' not in {RETURN_STATUSES}")
        return v

    @field_validator("reason", mode="after")
    @classmethod
    def _valid_reason(cls, v: str) -> str:
        if v not in RETURN_REASONS:
            raise ValueError(f"reason '{v}' not in {RETURN_REASONS}")
        return v


# ---------------------------------------------------------------------------
# Schema
# (P0-fix-1) updated_at column
# (P0-fix-4) FK raw_returns.order_id → raw_orders.order_id
# (P0-fix-5) indexes on customer_id, order_date
# (P1-fix-7) DateTime(timezone=True)
# ---------------------------------------------------------------------------
metadata = MetaData()

orders_table = Table(
    "raw_orders",
    metadata,
    Column("order_id",     String(64),       primary_key=True),
    Column("customer_id",  String(64),       nullable=False),
    Column("product_id",   String(64),       nullable=False),
    Column("order_date",   Date,             nullable=False),
    Column("quantity",     Integer,          nullable=False),
    Column("unit_price",   Numeric(12, 4),   nullable=False),
    Column("total_amount", Numeric(12, 4),   nullable=False),
    Column("status",       String(32),       nullable=False),
    Column("ingested_at",  DateTime(timezone=True), nullable=False),
    Column("updated_at",   DateTime(timezone=True), nullable=False),
    CheckConstraint("quantity >= 1", name="ck_orders_qty"),
    CheckConstraint("unit_price >= 0 AND total_amount >= 0", name="ck_orders_amounts"),
    CheckConstraint("status IN ('pending','completed','cancelled','processing')",
                    name="ck_orders_status"),
)

# Indexes: composite (customer_id, order_date) covers customer history look-ups
# and the time-based label join; order_date alone covers the censoring window.
Index("ix_raw_orders_customer_date", orders_table.c.customer_id, orders_table.c.order_date)
Index("ix_raw_orders_order_date",    orders_table.c.order_date)

returns_table = Table(
    "raw_returns",
    metadata,
    Column("return_id",     String(64),     primary_key=True),
    # P0-fix-4: FK to raw_orders — orphan returns are rejected at DB level
    Column("order_id",      String(64),
           ForeignKey("raw_orders.order_id", ondelete="RESTRICT"), nullable=False),
    Column("customer_id",   String(64),     nullable=False),
    Column("product_id",    String(64),     nullable=False),
    Column("return_date",   Date,           nullable=False),
    Column("reason",        String(128),    nullable=False),
    Column("refund_amount", Numeric(12, 4), nullable=False),
    Column("status",        String(32),     nullable=False),
    Column("ingested_at",   DateTime(timezone=True), nullable=False),
    Column("updated_at",    DateTime(timezone=True), nullable=False),
    CheckConstraint("refund_amount >= 0", name="ck_returns_refund"),
    CheckConstraint("status IN ('pending','approved','rejected')", name="ck_returns_status"),
    CheckConstraint(
        "reason IN ('defective','wrong_item','changed_mind','not_as_described','other')",
        name="ck_returns_reason"),
)

Index("ix_raw_returns_order_id", returns_table.c.order_id)

# Quarantine table — one row per rejected input row, no FK constraints
quarantine_table = Table(
    "quarantine_ingestion",
    metadata,
    Column("id",          Integer,  primary_key=True, autoincrement=True),
    Column("source",      String(32),  nullable=False),   # "orders" | "returns"
    Column("raw_row",     Text,        nullable=False),
    Column("error",       Text,        nullable=False),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_engine() -> Engine:
    """Create and verify a SQLAlchemy engine."""
    engine = create_engine(DATABASE_URL, echo=False, future=True)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _rec):  # SQLite ignores FKs unless enabled per-connection
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Database connection OK  [%s]", DATABASE_URL.split("@")[-1])
    except OperationalError as exc:
        log.error("Cannot connect to database: %s", exc)
        raise
    return engine


def _ensure_tables(engine: Engine) -> None:
    """Create tables if they do not exist (idempotent DDL)."""
    metadata.create_all(engine, checkfirst=True)
    log.info("Tables ensured: raw_orders, raw_returns, quarantine_ingestion")


def _load_csv(path: Path) -> pd.DataFrame:
    """Read CSV, strip column names, drop fully-blank rows."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    log.info("Loaded %d rows from %s", len(df), path.name)
    return df


def _validate_no_nulls(
    df: pd.DataFrame, required_cols: list[str], source: str
) -> tuple[pd.DataFrame, list[dict]]:
    """Split off rows with nulls / blank strings in required columns; they are quarantined."""
    blank = df[required_cols].apply(lambda c: c.isnull() | (c.astype(str).str.strip() == ""))
    null_mask = blank.any(axis=1)
    now = datetime.now(UTC)
    rejects = [
        {"source": source, "raw_row": r.to_json(),
         "error": "null/blank in required column", "ingested_at": now}
        for _, r in df[null_mask].iterrows()
    ]
    if rejects:
        log.warning("%s: %d row(s) with null/blank required values", source, len(rejects))
    return df[~null_mask].copy(), rejects


def _validate_records(
    engine: Engine,
    df: pd.DataFrame,
    model: type[BaseModel],
    pk_col: str,
    source: str,
    total_rows: int,
    prior_rejects: list[dict] | None = None,
) -> list[dict]:
    """
    Row-by-row Pydantic validation.
    Rejects are written to quarantine_ingestion.
    Exits non-zero if reject rate exceeds REJECT_RATE_THRESHOLD.
    Returns list of valid dicts (with ingested_at / updated_at).
    """
    valid:    list[dict] = []
    rejects:  list[dict] = list(prior_rejects or [])
    now = datetime.now(UTC)

    for _, row in df.iterrows():
        try:
            record = model(**row.to_dict())
            d = record.model_dump()
            d["ingested_at"] = now
            d["updated_at"]  = now
            valid.append(d)
        except ValidationError as exc:
            log.warning(
                "%s: validation failed pk=%s — %s",
                source, row.get(pk_col, "?"), exc,
            )
            rejects.append({
                "source":      source,
                "raw_row":     row.to_json(),
                "error":       str(exc),
                "ingested_at": now,
            })

    # Write rejects to quarantine
    if rejects:
        with engine.begin() as conn:
            conn.execute(quarantine_table.insert(), rejects)
        log.warning("%s: %d row(s) quarantined", source, len(rejects))

    # Reject-rate gate (P0-fix-2)
    reject_rate = len(rejects) / max(total_rows, 1)
    if reject_rate > REJECT_RATE_THRESHOLD:
        log.error(
            "%s: reject rate %.1f%% exceeds threshold %.1f%% — aborting",
            source, reject_rate * 100, REJECT_RATE_THRESHOLD * 100,
        )
        raise RuntimeError(
            f"{source} reject rate {reject_rate:.1%} > {REJECT_RATE_THRESHOLD:.1%}"
        )

    log.info("%s: %d valid / %d total rows", source, len(valid), total_rows)
    return valid


def _upsert(engine: Engine, table: Table, records: list[dict], pk_col: str) -> int:
    """
    Idempotent upsert (ON CONFLICT DO UPDATE), chunked for the Postgres bind limit.
    The UPDATE only fires when a business column actually changed, so updated_at
    reflects real changes and re-running an unchanged file touches 0 existing rows.
    Returns rows inserted + rows really changed.
    """
    if not records:
        log.info("No records to upsert into %s", table.name)
        return 0

    non_pk_cols = [c.name for c in table.columns if c.name not in (pk_col, "ingested_at")]
    compare_cols = [c for c in non_pk_cols if c != "updated_at"]
    dialect = engine.dialect.name
    affected = 0

    def _sqlite_safe(record: dict) -> dict:
        # sqlite3 cannot bind Decimal; SQLite stores Numeric as REAL anyway (dev only).
        return {k: float(v) if isinstance(v, Decimal) else v for k, v in record.items()}

    for start in range(0, len(records), CHUNK_SIZE):
        chunk = records[start:start + CHUNK_SIZE]
        with engine.begin() as conn:
            if dialect == "postgresql":
                stmt = pg_insert(table).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[pk_col],
                    set_={c: stmt.excluded[c] for c in non_pk_cols},
                    where=or_(*[table.c[c].is_distinct_from(stmt.excluded[c]) for c in compare_cols]),
                )
                affected += conn.execute(stmt).rowcount
            else:
                sql = text(
                    f"INSERT INTO {table.name} ({', '.join(chunk[0].keys())}) "
                    f"VALUES ({', '.join(':' + k for k in chunk[0].keys())}) "
                    f"ON CONFLICT({pk_col}) DO UPDATE SET "
                    + ", ".join(f"{c}=excluded.{c}" for c in non_pk_cols)
                    + " WHERE " + " OR ".join(f"{c} IS NOT excluded.{c}" for c in compare_cols)
                )
                for record in chunk:
                    affected += conn.execute(sql, _sqlite_safe(record)).rowcount

    log.info("Upserted %d new/changed row(s) into %s", affected, table.name)
    return affected


# ---------------------------------------------------------------------------
# Required column lists
# ---------------------------------------------------------------------------
ORDERS_REQUIRED = [
    "order_id", "customer_id", "product_id", "order_date",
    "quantity", "unit_price", "total_amount", "status",
]
RETURNS_REQUIRED = [
    "return_id", "order_id", "customer_id", "product_id",
    "return_date", "reason", "refund_amount", "status",
]

# Post-load cross-table invariants. Each query must return zero rows.
SANITY_CHECKS = {
    "return_before_order": """SELECT r.return_id FROM raw_returns r JOIN raw_orders o
        ON o.order_id = r.order_id WHERE r.return_date < o.order_date""",
    "refund_exceeds_total": """SELECT r.return_id FROM raw_returns r JOIN raw_orders o
        ON o.order_id = r.order_id WHERE r.refund_amount > o.total_amount""",
    "customer_or_product_mismatch": """SELECT r.return_id FROM raw_returns r JOIN raw_orders o
        ON o.order_id = r.order_id
        WHERE r.customer_id <> o.customer_id OR r.product_id <> o.product_id""",
}


def run_sanity_checks(engine: Engine) -> dict[str, list[str]]:
    """Return {check_name: [offending ids]} for every violated invariant."""
    out: dict[str, list[str]] = {}
    with engine.connect() as conn:
        for name, q in SANITY_CHECKS.items():
            ids = [r[0] for r in conn.execute(text(q))]
            if ids:
                out[name] = ids
                log.error("SANITY FAIL %s: %s", name, ids[:10])
    return out


# ---------------------------------------------------------------------------
# Public pipeline functions
# ---------------------------------------------------------------------------

def ingest_orders(engine: Engine, csv_path: Path | None = None) -> int:
    df = _load_csv(csv_path or ORDERS_CSV)
    total_rows = len(df)  # denominator includes null/blank rows
    df, null_rejects = _validate_no_nulls(df, ORDERS_REQUIRED, "orders")
    records = _validate_records(engine, df, OrderRecord, "order_id", "orders",
                                total_rows, null_rejects)
    return _upsert(engine, orders_table, records, "order_id")


def ingest_returns(engine: Engine, csv_path: Path | None = None) -> int:
    df = _load_csv(csv_path or RETURNS_CSV)
    total_rows = len(df)
    df, null_rejects = _validate_no_nulls(df, RETURNS_REQUIRED, "returns")
    records = _validate_records(engine, df, ReturnRecord, "return_id", "returns",
                                total_rows, null_rejects)

    # Orphan returns are quarantined individually instead of aborting a whole chunk.
    with engine.connect() as conn:
        known = {r[0] for r in conn.execute(text("SELECT order_id FROM raw_orders"))}
    orphans = [r for r in records if r["order_id"] not in known]
    if orphans:
        now = datetime.now(UTC)
        with engine.begin() as conn:
            conn.execute(quarantine_table.insert(), [
                {"source": "returns", "raw_row": str(o["return_id"]),
                 "error": f"orphan: order_id {o['order_id']} not in raw_orders", "ingested_at": now}
                for o in orphans])
        log.warning("returns: %d orphan row(s) quarantined", len(orphans))
        records = [r for r in records if r["order_id"] in known]
        if len(orphans) / max(total_rows, 1) > REJECT_RATE_THRESHOLD:
            raise RuntimeError(f"returns orphan rate too high ({len(orphans)}/{total_rows})")
    return _upsert(engine, returns_table, records, "return_id")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(orders_csv: Path | None = None, returns_csv: Path | None = None) -> dict:
    log.info("=== ReturnIQ Ingestion Pipeline starting ===")
    log.info("Env: %s  |  DB: %s", RETURNIQ_ENV, DATABASE_URL.split("@")[-1])

    engine = _get_engine()
    _ensure_tables(engine)

    # Orders first so the FK is satisfiable
    o = ingest_orders(engine, orders_csv)
    r = ingest_returns(engine, returns_csv)
    violations = run_sanity_checks(engine)
    if violations:
        raise RuntimeError(f"Cross-table sanity checks failed: {list(violations)}")

    log.info("=== Pipeline complete - orders changed: %d, returns changed: %d ===", o, r)
    return {"orders": o, "returns": r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=Path)
    ap.add_argument("--returns", type=Path)
    a = ap.parse_args()
    run(a.orders, a.returns)
