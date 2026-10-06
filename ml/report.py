"""
ml/report.py - save the README charts to docs/figures/ (matplotlib only).

Reads analytics_order_features and models/ (model.joblib, feature_importance.json). Does not retrain anything.
Charts: monthly return rate, precision-recall curve, calibration plot, feature importance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.db import get_engine  # noqa: E402
from ml.train import MODEL_DIR, TEST_START, VAL_START  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
NOTE = "synthetic data"
MIN_MONTH_ORDERS = 100  # months with fewer mature orders are too noisy to plot


def _finish(fig, ax, title: str, xlabel: str, ylabel: str, name: str) -> Path:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.text(0.99, 0.01, NOTE, ha="right", va="bottom", fontsize=8, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def monthly_return_rate(mature: pd.DataFrame) -> Path:
    m = mature.assign(month=pd.to_datetime(mature.order_date).dt.to_period("M").dt.to_timestamp())
    g = m.groupby("month").returned_30d
    rate = g.mean()[g.size() >= MIN_MONTH_ORDERS]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rate.index, rate.values * 100, marker="o")
    for label, start in (("val starts", VAL_START), ("test starts", TEST_START)):
        ax.axvline(pd.Timestamp(start), color="gray", linestyle="--")
        ax.text(pd.Timestamp(start), ax.get_ylim()[1], f" {label}", va="top", fontsize=8)
    ax.text(rate.index[0], ax.get_ylim()[1], "train ", va="top", fontsize=8)
    fig.text(0.01, 0.01, f"months with fewer than {MIN_MONTH_ORDERS} mature orders omitted", ha="left", va="bottom",
             fontsize=8, color="gray")
    return _finish(fig, ax, "Monthly 30-day return rate (mature orders)", "Order month", "Return rate (%)",
                   "monthly_return_rate.png")


def precision_recall(y: pd.Series, p: np.ndarray, name: str) -> Path:
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rec, prec, label=name)
    ax.axhline(y.mean(), color="gray", linestyle="--", label=f"prevalence {y.mean():.3f}")
    ax.legend()
    return _finish(fig, ax, "Precision-recall curve (test set)", "Recall", "Precision", "precision_recall.png")


def calibration(y: pd.Series, p: np.ndarray) -> Path:
    bins = pd.qcut(pd.Series(p), 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y.values}).groupby(bins, observed=True).mean()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect calibration")
    ax.plot(g.p, g.y, marker="o", label="model (10 bins)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return _finish(fig, ax, "Calibration (test set)", "Predicted return probability", "Observed return rate",
                   "calibration.png")


def feature_importance() -> Path:
    imp = json.loads((MODEL_DIR / "feature_importance.json").read_text())
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([k for k, _ in items], [v for _, v in items])
    return _finish(fig, ax, "Feature importance (mean absolute SHAP value)", "Mean |SHAP|", "Feature",
                   "feature_importance.png")


def build() -> list[Path]:
    df = pd.read_sql("SELECT * FROM analytics_order_features", get_engine())
    mature = df[df.label_mature == 1]
    test = mature[mature.order_date >= TEST_START]
    art = joblib.load(MODEL_DIR / "model.joblib")
    p = art["model"].predict_proba(test[art["features"]].astype(float))[:, 1]
    return [monthly_return_rate(mature), precision_recall(test.returned_30d, p, art["name"]),
            calibration(test.returned_30d, p), feature_importance()]


if __name__ == "__main__":
    for path in build():
        print(path)
