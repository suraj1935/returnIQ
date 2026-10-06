"""
ai/tools.py - the ONLY way the Copilot touches data.

Design rules (anti-hallucination):
  * No free-form SQL. Each tool has a fixed, parameterised statement over analytics_* tables.
  * Read-only engine; hard row cap; every call returns a ToolResult with a query_id,
    the params used and the data-as-of date, so answers can cite their source.
  * Empty result => ToolResult.empty=True (never silently 0).
  * Rates use mature labels only (label_mature=1), so right-censored orders do not deflate them.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, text

from backend.db import get_readonly_engine

MAX_ROWS = 50
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_PROD_RE = re.compile(r"^PROD-\d{1,6}$")

DATA_DICTIONARY = {
    "analytics_order_features": "One row per non-cancelled order. Order-time features plus returned_30d "
        "(1 if a non-rejected return was requested within 30 days of order_date) and label_mature "
        "(1 if the 30-day window is fully observed). Rates in answers use label_mature=1 only.",
    "analytics_returns_fact": "One row per return: reason, status, refund_amount, days_to_return. "
        "Post-event data used for reporting only, never for model features.",
    "analytics_risk_scores": "Model risk_score (0-1) and high_risk flag per order. All data is SYNTHETIC.",
    "return_rate": "returns / mature non-cancelled orders in the period.",
}


@dataclass
class ToolResult:
    tool: str
    query_id: str
    params: dict
    rows: list[dict]
    row_count: int
    empty: bool
    data_as_of: str | None
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class ToolError(ValueError):
    """Invalid arguments - surfaced to the model/user, never swallowed."""


def _d(v: str | None, name: str) -> str | None:
    if v is None:
        return None
    try:
        return date.fromisoformat(v).isoformat()
    except ValueError as e:
        raise ToolError(f"{name} must be YYYY-MM-DD, got {v!r}") from e


def _prod(v: str | None) -> str | None:
    if v is not None and not _PROD_RE.match(v):
        raise ToolError(f"product_id must look like PROD-0001, got {v!r}")
    return v


def _period(start_date, end_date, product_id=None) -> dict:
    return {"s": _d(start_date, "start_date") or "0000-01-01",
            "e": _d(end_date, "end_date") or "9999-12-31",
            "p": _prod(product_id)}


class Toolbox:
    def __init__(self, engine: Engine | None = None):
        self.engine = engine or get_readonly_engine()

    def _as_of(self) -> str | None:
        with self.engine.connect() as c:
            r = c.execute(text("SELECT snapshot_date FROM analytics_meta LIMIT 1")).fetchone()
        return r[0] if r else None

    def _run(self, tool: str, sql: str, params: dict, note: str = "") -> ToolResult:
        with self.engine.connect() as c:
            rows = [dict(r._mapping) for r in c.execute(text(sql + f" LIMIT {MAX_ROWS}"), params)]
        shown = {k: v for k, v in params.items() if v is not None and v not in ("0000-01-01", "9999-12-31")}
        return ToolResult(tool, uuid.uuid4().hex[:8], shown, rows, len(rows), len(rows) == 0,
                          self._as_of(), note)

    @staticmethod
    def _empty_if_zero(r: ToolResult, key: str) -> ToolResult:
        if r.rows and not r.rows[0][key]:
            r.rows, r.row_count, r.empty = [], 0, True
        return r

    # ---- tools ----
    def return_rate(self, start_date: str | None = None, end_date: str | None = None,
                    product_id: str | None = None) -> ToolResult:
        r = self._run("return_rate", """
            SELECT COUNT(*) AS mature_orders, SUM(returned_30d) AS returned_orders,
                   ROUND(1.0 * SUM(returned_30d) / COUNT(*), 4) AS return_rate
            FROM analytics_order_features
            WHERE label_mature = 1 AND order_date BETWEEN :s AND :e AND (:p IS NULL OR product_id = :p)""",
            _period(start_date, end_date, product_id),
            "30-day return rate over mature, non-cancelled orders.")
        return self._empty_if_zero(r, "mature_orders")

    def top_return_reasons(self, start_date: str | None = None, end_date: str | None = None,
                           product_id: str | None = None, limit: int = 5) -> ToolResult:
        p = _period(start_date, end_date, product_id)
        return self._run("top_return_reasons", f"""
            SELECT reason, COUNT(*) AS returns,
                   ROUND(1.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS share
            FROM analytics_returns_fact
            WHERE status <> 'rejected' AND order_date BETWEEN :s AND :e AND (:p IS NULL OR product_id = :p)
            GROUP BY reason ORDER BY returns DESC, reason
            LIMIT {max(1, min(int(limit), 20))} --""", p)

    def top_products_by_return_rate(self, start_date: str | None = None, end_date: str | None = None,
                                    min_orders: int = 50, limit: int = 5) -> ToolResult:
        p = _period(start_date, end_date)
        p["m"] = max(1, int(min_orders))
        return self._run("top_products_by_return_rate", f"""
            SELECT product_id, COUNT(*) AS mature_orders, SUM(returned_30d) AS returned_orders,
                   ROUND(1.0 * SUM(returned_30d) / COUNT(*), 4) AS return_rate
            FROM analytics_order_features
            WHERE label_mature = 1 AND order_date BETWEEN :s AND :e
            GROUP BY product_id HAVING COUNT(*) >= :m
            ORDER BY return_rate DESC, product_id
            LIMIT {max(1, min(int(limit), 20))} --""", p,
            f"Only products with at least {p['m']} mature orders are ranked.")

    def refund_total(self, start_date: str | None = None, end_date: str | None = None,
                     product_id: str | None = None) -> ToolResult:
        r = self._run("refund_total", """
            SELECT COUNT(*) AS returns_counted, ROUND(SUM(refund_amount), 2) AS total_refund_amount
            FROM analytics_returns_fact
            WHERE status <> 'rejected' AND return_date BETWEEN :s AND :e AND (:p IS NULL OR product_id = :p)""",
            _period(start_date, end_date, product_id),
            "Filtered by return_date; excludes rejected returns.")
        return self._empty_if_zero(r, "returns_counted")

    def high_risk_orders(self, limit: int = 10) -> ToolResult:
        return self._run("high_risk_orders", f"""
            SELECT s.order_id, f.product_id, f.order_date, s.risk_score
            FROM analytics_risk_scores s JOIN analytics_order_features f ON f.order_id = s.order_id
            WHERE f.label_mature = 0
            ORDER BY s.risk_score DESC, s.order_id
            LIMIT {max(1, min(int(limit), 20))} --""", {},
            "Recent orders whose 30-day outcome is not yet observed, ranked by model risk score (synthetic data).")

    def model_metrics(self) -> ToolResult:
        f = MODELS_DIR / "metrics.json"
        rows = []
        if f.exists():
            rep = json.loads(f.read_text())
            sel = rep["selected"]
            rows = [{"model": sel, "synthetic_data": rep.get("synthetic_data", True), **rep["models"][sel]["test"]}]
        return ToolResult("model_metrics", uuid.uuid4().hex[:8], {}, rows, len(rows), not rows, self._as_of(),
                          "Held-out TEST metrics of the selected model on SYNTHETIC data.")

    def data_dictionary(self) -> ToolResult:
        rows = [{"name": k, "description": v} for k, v in DATA_DICTIONARY.items()]
        return ToolResult("data_dictionary", uuid.uuid4().hex[:8], {}, rows, len(rows), False, self._as_of())


_DATES = {"start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
          "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive"}}
TOOL_SCHEMAS = [
    {"name": "return_rate", "description": "30-day return rate and counts for a period and optional product.",
     "input_schema": {"type": "object", "properties": {**_DATES, "product_id": {"type": "string"}}}},
    {"name": "top_return_reasons", "description": "Most common return reasons with counts and shares.",
     "input_schema": {"type": "object", "properties": {**_DATES, "product_id": {"type": "string"},
                                                       "limit": {"type": "integer"}}}},
    {"name": "top_products_by_return_rate", "description": "Products ranked by return rate (min order volume applies).",
     "input_schema": {"type": "object", "properties": {**_DATES, "min_orders": {"type": "integer"},
                                                       "limit": {"type": "integer"}}}},
    {"name": "refund_total", "description": "Total refunded amount and number of returns.",
     "input_schema": {"type": "object", "properties": {**_DATES, "product_id": {"type": "string"}}}},
    {"name": "high_risk_orders", "description": "Open orders with the highest predicted return risk.",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "model_metrics", "description": "Held-out evaluation metrics of the return-risk model.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "data_dictionary", "description": "Definitions of tables and metrics.",
     "input_schema": {"type": "object", "properties": {}}},
]
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}


def call_tool(box: Toolbox, name: str, args: dict[str, Any]) -> ToolResult:
    if name not in TOOL_NAMES:
        raise ToolError(f"unknown tool {name!r}")
    return getattr(box, name)(**args)
