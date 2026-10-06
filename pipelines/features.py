"""
pipelines/features.py
Builds the analytics layer from raw_* (never writes to raw_*):

  analytics_order_features : order-time features + label. FEATURES ONLY use information
                             available strictly before order_date (return_date < order_date).
  analytics_returns_fact   : post-event facts (reason, refund, days_to_return) for reporting
                             and the Copilot. NEVER joined into model features.

Label  : returned_30d = a non-rejected return with return_date <= order_date + 30d.
Censor : an order is label_mature only if order_date + 30d <= snapshot_date (max order_date);
         immature rows are kept for scoring but excluded from training/evaluation.
Cancelled orders are excluded (cannot be returned).
"""
from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.db import get_engine  # noqa: E402

log = logging.getLogger("features")
WINDOW_DAYS = 30
SMOOTH_K = 5  # pseudo-observations for Bayesian-smoothed rates

FEATURE_COLS = [
    "unit_price", "quantity", "total_amount", "order_dow", "order_month",
    "cust_prior_orders", "cust_prior_returns", "cust_prior_return_rate",
    "cust_days_since_last_order", "cust_tenure_days",
    "prod_prior_orders", "prod_prior_return_rate",
]


def _count_before(group_keys: pd.Series, dates: pd.Series, ev_keys: pd.Series, ev_dates: pd.Series) -> np.ndarray:
    """For each (key, date): number of events of same key with event_date strictly < date."""
    ev = pd.DataFrame({"k": ev_keys.values, "d": ev_dates.values.astype("datetime64[ns]")})
    by_key = {k: np.sort(g["d"].values) for k, g in ev.groupby("k")}
    out = np.zeros(len(dates), dtype=int)
    dvals = dates.values.astype("datetime64[ns]")
    for i, (k, d) in enumerate(zip(group_keys.values, dvals)):
        arr = by_key.get(k)
        if arr is not None:
            out[i] = np.searchsorted(arr, d, side="left")
    return out


def build(engine=None) -> dict:
    engine = engine or get_engine()
    orders = pd.read_sql("SELECT * FROM raw_orders", engine, parse_dates=["order_date"])
    returns = pd.read_sql("SELECT * FROM raw_returns", engine, parse_dates=["return_date"])
    for c in ("unit_price", "total_amount"):
        orders[c] = orders[c].astype(float)
    snapshot = orders["order_date"].max()
    log.info("snapshot_date=%s orders=%d returns=%d", snapshot.date(), len(orders), len(returns))

    base = orders[orders["status"] != "cancelled"].sort_values(["order_date", "order_id"]).reset_index(drop=True)

    # ---- label (post-event; kept in its own columns, never in FEATURE_COLS) ----
    valid_ret = returns[returns["status"] != "rejected"][["order_id", "return_date"]]
    m = base[["order_id", "order_date"]].merge(valid_ret, on="order_id", how="left")
    m["in_window"] = (m["return_date"] - m["order_date"]).dt.days.between(0, WINDOW_DAYS)
    lab = m.groupby("order_id")["in_window"].any()
    base["returned_30d"] = base["order_id"].map(lab).fillna(False).astype(int)
    base["label_mature"] = (base["order_date"] + timedelta(days=WINDOW_DAYS) <= snapshot).astype(int)

    # ---- as-of features: only data with date strictly < order_date ----
    # Prior orders: all earlier orders (incl. cancelled -> they are known at order time).
    base["cust_prior_orders"] = _count_before(base.customer_id, base.order_date, orders.customer_id, orders.order_date)
    base["prod_prior_orders"] = _count_before(base.product_id, base.order_date, orders.product_id, orders.order_date)
    # Prior returns: only returns already REQUESTED before this order (any status; status is a later fact).
    base["cust_prior_returns"] = _count_before(base.customer_id, base.order_date, returns.customer_id, returns.return_date)
    prod_prior_returns = _count_before(base.product_id, base.order_date, returns.product_id, returns.return_date)

    # Smoothed rates; prior = global rate over information before the earliest order is unknowable,
    # so use a fixed neutral prior (0.15) rather than a full-data mean (which would leak).
    prior = 0.15
    base["cust_prior_return_rate"] = (base.cust_prior_returns + SMOOTH_K * prior) / (base.cust_prior_orders + SMOOTH_K)
    base["prod_prior_return_rate"] = (prod_prior_returns + SMOOTH_K * prior) / (base.prod_prior_orders + SMOOTH_K)

    # Recency / tenure from earlier orders only
    allo = orders.sort_values(["customer_id", "order_date"])
    first = allo.groupby("customer_id")["order_date"].min()
    base["cust_tenure_days"] = (base.order_date - base.customer_id.map(first)).dt.days
    prev = (allo.assign(prev_date=allo.groupby("customer_id")["order_date"].shift(1))
            .drop_duplicates("order_id").set_index("order_id")["prev_date"])
    base["cust_days_since_last_order"] = (base.order_date - base.order_id.map(prev)).dt.days.fillna(-1)

    base["order_dow"] = base.order_date.dt.dayofweek
    base["order_month"] = base.order_date.dt.month

    feats = base[["order_id", "customer_id", "product_id", "order_date", *FEATURE_COLS,
                  "returned_30d", "label_mature"]].copy()
    feats["order_date"] = feats["order_date"].dt.date.astype(str)

    # ---- reporting fact table (post-event) ----
    fact = returns.merge(orders[["order_id", "order_date", "total_amount"]], on="order_id")
    fact["days_to_return"] = (fact["return_date"] - fact["order_date"]).dt.days
    fact["order_date"] = fact["order_date"].dt.date.astype(str)
    fact["return_date"] = fact["return_date"].dt.date.astype(str)
    fact["refund_amount"] = fact["refund_amount"].astype(float)
    fact = fact[["return_id", "order_id", "customer_id", "product_id", "order_date", "return_date",
                 "days_to_return", "reason", "status", "refund_amount", "total_amount"]]

    with engine.begin() as conn:
        for t in ("analytics_order_features", "analytics_returns_fact", "analytics_meta"):
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
        feats.to_sql("analytics_order_features", conn, index=False)
        fact.to_sql("analytics_returns_fact", conn, index=False)
        pd.DataFrame([{"snapshot_date": str(snapshot.date()), "window_days": WINDOW_DAYS,
                       "built_at": pd.Timestamp.utcnow().isoformat()}]).to_sql("analytics_meta", conn, index=False)
        conn.execute(text("CREATE INDEX ix_af_order ON analytics_order_features(order_id)"))
        conn.execute(text("CREATE INDEX ix_af_date ON analytics_order_features(order_date)"))
    mature = feats[feats.label_mature == 1]
    log.info("features=%d mature=%d positive_rate(mature)=%.3f", len(feats), len(mature), mature.returned_30d.mean())
    return {"rows": len(feats), "mature": len(mature), "positive_rate": float(mature.returned_30d.mean())}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(build())
