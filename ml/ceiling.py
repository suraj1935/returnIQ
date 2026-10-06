"""
ml/ceiling.py - how good could ANY model be on this synthetic data?

Scores the held-out test orders with the generator's TRUE return probability (information no real
model has) and reports ROC-AUC / PR-AUC / Brier against the same label the models are judged on.
Because labels are noisy by construction (a Bernoulli draw, rejected returns and the 30-day window
are removed from the label), this is an upper bound for ml/train.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.db import get_engine  # noqa: E402
from ml.train import MODEL_DIR, TEST_START  # noqa: E402
from pipelines.synthetic import generate  # noqa: E402


def ceiling(seed: int = 42) -> dict:
    _, _, truth = generate(seed=seed, return_truth=True)
    f = pd.read_sql("SELECT order_id, order_date, returned_30d, label_mature FROM analytics_order_features", get_engine())
    te = f[(f.label_mature == 1) & (f.order_date >= TEST_START)].copy()
    te["true_p"] = te.order_id.map(truth)
    y, p = te.returned_30d, te.true_p
    out = {"n_test": int(len(te)), "prevalence": round(float(y.mean()), 4),
           "oracle_roc_auc": round(float(roc_auc_score(y, p)), 4),
           "oracle_pr_auc": round(float(average_precision_score(y, p)), 4),
           "oracle_brier": round(float(brier_score_loss(y, p)), 4)}
    (MODEL_DIR / "ceiling.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(ceiling(), indent=2))
