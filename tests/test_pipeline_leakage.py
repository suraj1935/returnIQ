"""Leakage / integrity guards on the analytics layer."""
import pandas as pd
from sqlalchemy import text

from backend.db import get_engine
from pipelines.features import FEATURE_COLS, WINDOW_DAYS

FORBIDDEN = {"reason", "refund_amount", "return_date", "status", "returned_30d", "days_to_return", "label_mature"}


def _df(q):
    return pd.read_sql(q, get_engine())


def test_no_post_event_columns_in_features():
    assert not FORBIDDEN & set(FEATURE_COLS)


def test_customer_prior_returns_are_as_of():
    """Recompute independently for a sample and compare."""
    f = _df("SELECT order_id, customer_id, order_date, cust_prior_returns FROM analytics_order_features LIMIT 300")
    r = _df("SELECT customer_id, return_date FROM raw_returns")
    for row in f.itertuples():
        exp = ((r.customer_id == row.customer_id) & (r.return_date < row.order_date)).sum()
        assert row.cust_prior_returns == exp, row.order_id


def test_immature_orders_flagged():
    f = _df("SELECT order_date, label_mature FROM analytics_order_features")
    snap = _df("SELECT snapshot_date FROM analytics_meta").iloc[0, 0]
    cutoff = (pd.Timestamp(snap) - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    assert (f[f.order_date > cutoff].label_mature == 0).all()
    assert (f[f.order_date <= cutoff].label_mature == 1).all()


def test_cancelled_orders_excluded():
    n = _df("SELECT COUNT(*) c FROM analytics_order_features f JOIN raw_orders o ON o.order_id=f.order_id "
            "WHERE o.status='cancelled'").c[0]
    assert n == 0


def test_time_split_ordered():
    from ml.train import TEST_START, VAL_START
    assert VAL_START < TEST_START


def test_raw_tables_hold_no_analytics_columns():
    with get_engine().connect() as c:
        cols = {r[1] for r in c.execute(text("PRAGMA table_info(raw_orders)"))}
    assert "returned_30d" not in cols and "risk_score" not in cols
