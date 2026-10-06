"""
ml/train.py - return-risk model: LR baseline -> HistGradientBoosting, time-based split.

Split (by order_date, mature labels only):  train < VAL_START <= val < TEST_START <= test
Threshold is chosen on VAL (max F1) and frozen before touching TEST.
Metrics: PR-AUC (primary), ROC-AUC, Brier, precision/recall/F1 @ threshold, vs prevalence baseline.
Artifacts: models/model.joblib, models/metrics.json, models/feature_importance.json (SHAP).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.db import get_engine  # noqa: E402
from pipelines.features import FEATURE_COLS  # noqa: E402

log = logging.getLogger("train")
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
VAL_START, TEST_START = "2024-04-01", "2024-08-01"
SEED = 42


def _metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "n": int(len(y)), "prevalence": round(float(y.mean()), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "threshold": round(float(thr), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
    }


def _best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return float(th[int(np.argmax(f1))])


def train() -> dict:
    eng = get_engine()
    df = pd.read_sql("SELECT * FROM analytics_order_features", eng)
    mature = df[df.label_mature == 1].copy()
    tr = mature[mature.order_date < VAL_START]
    va = mature[(mature.order_date >= VAL_START) & (mature.order_date < TEST_START)]
    te = mature[mature.order_date >= TEST_START]
    assert len(tr) and len(va) and len(te), "empty split"
    assert tr.order_date.max() < va.order_date.min() <= va.order_date.max() < te.order_date.min()
    X = lambda d: d[FEATURE_COLS].astype(float)

    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))
    lr.fit(X(tr), tr.returned_30d)

    sw = compute_sample_weight("balanced", tr.returned_30d)
    gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                        early_stopping=True, random_state=SEED)
    gb.fit(X(tr), tr.returned_30d, sample_weight=sw)

    results, models = {}, {"logistic_regression": lr, "hist_gradient_boosting": gb}
    for name, m in models.items():
        pv = m.predict_proba(X(va))[:, 1]
        thr = _best_f1_threshold(va.returned_30d, pv)
        pt = m.predict_proba(X(te))[:, 1]
        results[name] = {"val": _metrics(va.returned_30d, pv, thr), "test": _metrics(te.returned_30d, pt, thr)}
        log.info("%s test: %s", name, results[name]["test"])

    best = max(results, key=lambda k: results[k]["val"]["pr_auc"])  # select on VAL, not test
    model, thr = models[best], results[best]["val"]["threshold"]

    # SHAP explainability on a test sample
    imp = {}
    try:
        import shap
        sample = X(te).sample(min(500, len(te)), random_state=SEED)
        bg = X(tr).sample(100, random_state=SEED)
        explainer = shap.Explainer(lambda a: model.predict_proba(pd.DataFrame(a, columns=FEATURE_COLS))[:, 1], bg)
        sv = explainer(sample, silent=True).values
        imp = dict(sorted(zip(FEATURE_COLS, np.abs(sv).mean(0).round(5).tolist()), key=lambda kv: -kv[1]))
        imp = {k: float(v) for k, v in imp.items()}
    except Exception as e:  # keep training usable if shap breaks
        log.warning("SHAP failed (%s); falling back to permutation importance", e)
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X(te), te.returned_30d, scoring="average_precision",
                                   n_repeats=3, random_state=SEED)
        imp = {k: float(v) for k, v in sorted(zip(FEATURE_COLS, r.importances_mean.round(5)), key=lambda kv: -kv[1])}

    MODEL_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    joblib.dump({"model": model, "features": FEATURE_COLS, "threshold": thr, "name": best, "version": version},
                MODEL_DIR / "model.joblib")
    report = {"version": version, "selected": best, "synthetic_data": True,
              "split": {"train": [tr.order_date.min(), tr.order_date.max(), len(tr)],
                        "val": [va.order_date.min(), va.order_date.max(), len(va)],
                        "test": [te.order_date.min(), te.order_date.max(), len(te)]},
              "models": results}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    (MODEL_DIR / "feature_importance.json").write_text(json.dumps(imp, indent=2))

    # Persist scores for every order (incl. immature) for the API / Copilot
    allp = model.predict_proba(X(df))[:, 1]
    scores = pd.DataFrame({"order_id": df.order_id, "risk_score": allp.round(5),
                           "high_risk": (allp >= thr).astype(int), "model_version": version})
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS analytics_risk_scores"))
        scores.to_sql("analytics_risk_scores", conn, index=False)
        conn.execute(text("CREATE INDEX ix_rs_order ON analytics_risk_scores(order_id)"))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = train()
    print(json.dumps({k: r[k] for k in ("selected", "split")}, indent=1))
    for k, v in r["models"].items():
        print(k, "TEST", v["test"])
    print(open(MODEL_DIR / "feature_importance.json").read())
