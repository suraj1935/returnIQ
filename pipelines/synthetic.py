"""
pipelines/synthetic.py
Seeded synthetic data generator. Output is SYNTHETIC: any metric computed from it
demonstrates the pipeline, not real-world model performance.

Planted signal (so evaluation can be sanity-checked):
  - latent customer propensity (heavy tailed)  -> recoverable via as-of customer history
  - latent product defect rate                 -> recoverable via as-of product history
  - price (high price -> more returns), quantity>1 (bracketing), Nov/Dec seasonality
  - cancelled orders are never returned
Returns are truncated at SNAPSHOT_DATE (an extract cannot contain future returns), which
creates realistic right-censoring for late orders.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

START, SNAPSHOT_DATE = date(2023, 1, 1), date(2024, 12, 31)
REASONS = ["defective", "wrong_item", "changed_mind", "not_as_described", "other"]


def generate(n_customers=3000, n_products=150, n_orders=25000, seed=42):
    rng = np.random.default_rng(seed)
    cust_prop = rng.gamma(shape=0.8, scale=0.6, size=n_customers)          # mean ~0.5
    prod_defect = rng.beta(1.2, 12, size=n_products)                       # mean ~0.09
    prod_price = np.round(np.exp(rng.normal(3.5, 0.8, n_products)), 2).clip(4, 600)

    days = (SNAPSHOT_DATE - START).days
    order_day = np.sort(rng.integers(0, days + 1, n_orders))
    cust = rng.integers(0, n_customers, n_orders)
    prod = rng.integers(0, n_products, n_orders)
    qty = rng.choice([1, 2, 3, 4], n_orders, p=[0.72, 0.18, 0.07, 0.03])
    price = np.round(prod_price[prod] * rng.uniform(0.85, 1.0, n_orders), 2)
    month = np.array([(START + timedelta(int(d))).month for d in order_day])

    logit = (-2.6 + 0.9 * np.log1p(cust_prop[cust] * 3) + 6.0 * prod_defect[prod]
             + 0.35 * np.log(price / 30) + 0.25 * (qty > 1) + 0.3 * np.isin(month, [11, 12])
             + rng.normal(0, 0.3, n_orders))
    p_ret = 1 / (1 + np.exp(-logit))
    cancelled = rng.random(n_orders) < 0.04
    returned = (rng.random(n_orders) < p_ret) & ~cancelled

    orders = pd.DataFrame({
        "order_id": [f"ORD-{i + 1:06d}" for i in range(n_orders)],
        "customer_id": [f"CUST-{c + 1:05d}" for c in cust],
        "product_id": [f"PROD-{p + 1:04d}" for p in prod],
        "order_date": [(START + timedelta(int(d))).isoformat() for d in order_day],
        "quantity": qty, "unit_price": price,
        "total_amount": np.round(price * qty, 2),
        "status": np.where(cancelled, "cancelled", "completed"),
    })

    idx = np.where(returned)[0]
    lag = np.minimum(1 + rng.gamma(2.0, 5.0, len(idx)).astype(int), 60)
    ret_day = order_day[idx] + lag
    keep = ret_day <= days                                                 # truncate at snapshot
    idx, ret_day = idx[keep], ret_day[keep]
    # defect-driven reasons
    reasons = np.where(rng.random(len(idx)) < np.clip(prod_defect[prod[idx]] * 4, 0, 0.8),
                       "defective", rng.choice(REASONS[1:], len(idx), p=[.2, .5, .2, .1]))
    status = rng.choice(["approved", "rejected", "pending"], len(idx), p=[.86, .09, .05])
    frac = np.where(rng.random(len(idx)) < 0.9, 1.0, rng.uniform(0.3, 0.9, len(idx)))
    returns = pd.DataFrame({
        "return_id": [f"RET-{i + 1:06d}" for i in range(len(idx))],
        "order_id": orders.order_id.values[idx],
        "customer_id": orders.customer_id.values[idx],
        "product_id": orders.product_id.values[idx],
        "return_date": [(START + timedelta(int(d))).isoformat() for d in ret_day],
        "reason": reasons,
        "refund_amount": np.round(orders.total_amount.values[idx] * frac, 2),
        "status": status,
    })
    return orders, returns


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "synthetic")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    o, r = generate(seed=a.seed)
    o.to_csv(a.out / "orders.csv", index=False)
    r.to_csv(a.out / "returns.csv", index=False)
    print(f"orders={len(o)} returns={len(r)} return_rate={len(r) / len(o):.3f} -> {a.out}")
