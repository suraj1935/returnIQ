# ReturnIQ

**Problem.** E-commerce returns teams need two things that usually fail in practice: (1) a risk score they can
trust *before* a return happens, and (2) answers about return metrics that are never invented by an LLM.
ReturnIQ is a small end-to-end platform for that: validated ingestion -> leakage-safe features -> return-risk
model -> a Copilot that can only answer from SQL tool results.

**For whom.** Returns / operations analysts who want to ask "what is our return rate for X in Q3?" and get a
sourced number, and data/ML teams who want a reference for leakage-safe evaluation and grounded agents.

> All data is **synthetic** (seeded generator). Metrics demonstrate the pipeline, not real-world performance.

## Results (held-out test set: orders from 2024-08-01, mature labels only, n = 3,984, prevalence 23.6%)

| | PR-AUC | ROC-AUC | Brier | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Constant guess (prevalence) | 0.236 | 0.500 | 0.1804 | - | - | - |
| **Logistic regression (selected)** | **0.351** | **0.636** | **0.1730** | 0.308 | 0.654 | 0.419 |
| HistGradientBoosting | 0.333 | 0.607 | 0.1805 | 0.267 | 0.771 | 0.396 |
| Oracle (true generating probability) | 0.427 | 0.691 | 0.1641 | - | - | - |

**How to read this.** The labels are noisy by construction, so even a perfect model tops out near 0.69 ROC-AUC
(`ml/ceiling.py` scores the generator's true probability on the same test orders). Against that ceiling, logistic regression recovers about
71% of the achievable ROC-AUC lift, (0.636 - 0.5) / (0.691 - 0.5), and about 60% of the achievable PR-AUC lift,
(0.351 - 0.236) / (0.427 - 0.236), with the current seed. PR-AUC is the primary metric, so 60% is the figure to
quote. Gradient boosting does not beat logistic regression, so the remaining gap looks like mostly irreducible
noise rather than model capacity. Probabilities are
**not** re-weighted (class weighting was removed because it pushed Brier above the constant-guess baseline); the
decision threshold is chosen on the validation window, never on test.

## Pipeline

```
synthetic.py -> ingestion.py -> raw_orders / raw_returns -> features.py -> analytics_* -> train.py -> models/
                                                                    \-> ai/tools.py -> ai/copilot.py -> backend/main.py + frontend/
```

```bash
pip install -r requirements.txt
cp .env.example .env                                # add OLLAMA_API_KEY (or set RETURNIQ_LLM=rules)
python pipelines/synthetic.py                       # seeded data -> data/synthetic/
python pipelines/ingestion.py --orders data/synthetic/orders.csv --returns data/synthetic/returns.csv
python pipelines/features.py                        # analytics_* tables
python ml/train.py && python ml/ceiling.py          # model + metrics + oracle ceiling -> models/*.json
uvicorn backend.main:app --reload                   # UI at /, API docs at /docs
python -m pytest tests -q                           # unit + API + leakage + Playwright E2E (offline, rules engine)
```

## Guarantees
- **Ingestion**: idempotent upsert (only genuinely changed rows are touched), FK + CHECK constraints, quarantine
  table, reject-rate gate, post-load cross-table checks.
- **Leakage**: features use only data dated strictly before `order_date`; orders whose 30-day window is not fully
  observed (`label_mature=0`) are excluded from training and from every rate; time-based split
  (train < 2024-04 <= val < 2024-08 <= test). Return reason / refund live in a reporting table the model never reads.
- **Copilot**: fixed, parameterised SQL tools over `analytics_*` on a read-only connection; no free-form SQL.
  Before an answer is shown, `verify_grounding` checks that every figure is the right *kind* of value from a tool
  result (counts match integer fields, percentages match rates), that every year/date/id appears in the query
  parameters or rows, and that nothing is spelled out ("three products" vs 2 rows). Failures are withheld, empty
  results are reported as "cannot be computed" (never 0), and each answer cites `tool [query_id] as of <date>`
  plus a `Scope:` line with the filters actually queried. Any LLM failure falls back to the deterministic engine.
  Remaining limit: numbers are checked by kind and value but not bound to a specific field, so a correct count
  attached to the wrong label can still pass.
- All tools filter on the same **order-date cohort** basis, so a "Q3" answer never mixes populations.

## LLM configuration (`.env`, see `.env.example`)
`RETURNIQ_LLM=ollama|nvidia|anthropic|gemini|rules`. Default is Ollama Cloud (`OLLAMA_URL=https://ollama.com/v1`,
`OLLAMA_API_KEY`, model `RETURNIQ_OLLAMA_MODEL=gemma4:31b`, optional `RETURNIQ_OLLAMA_FALLBACKS`). Real environment
variables override `.env`. Compare models with `python ai/eval_llm.py <model> [<model> ...]`.

Postgres: set `DATABASE_URL` (+ `RETURNIQ_ENV=prod`) and `DATABASE_URL_READONLY` (role with SELECT on `analytics_*` only).
Only SQLite has been exercised so far.
