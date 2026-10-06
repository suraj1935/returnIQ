# ReturnIQ

Returns intelligence: ingestion -> leakage-safe features -> return-risk model -> grounded Copilot. All data is **synthetic**.

```bash
pip install -r requirements.txt
python pipelines/synthetic.py                       # seeded data -> data/synthetic/
python pipelines/ingestion.py --orders data/synthetic/orders.csv --returns data/synthetic/returns.csv
python pipelines/features.py                        # analytics_* tables (as-of features, 30d label, censoring)
python ml/train.py                                  # LR baseline vs HistGradientBoosting, SHAP, models/*.json
uvicorn backend.main:app --reload                   # http://127.0.0.1:8000  (UI at /, docs at /docs)
python -m pytest tests -q                           # unit + API + leakage + Playwright E2E
```

## Guarantees
- **Ingestion**: idempotent upsert, FK + CHECK constraints, quarantine table, reject-rate gate, cross-table sanity checks.
- **Leakage**: features use only data with date strictly before `order_date`; orders whose 30-day window is not fully observed (`label_mature=0`) are excluded from training and from every rate; split is by time (train < 2024-04 <= val < 2024-08 <= test); threshold is chosen on val, never test.
- **Copilot**: fixed parameterised SQL tools over `analytics_*` via a read-only connection; no free-form SQL. Every number in an answer must appear in a tool result (`verify_grounding`) or the answer is withheld; empty results are reported as "cannot be computed", never 0; each answer cites `tool [query_id] as of <date>`.
- Local/free: `RETURNIQ_LLM=ollama` (model `qwen3:8b`, override `RETURNIQ_OLLAMA_MODEL`) or `NVIDIA_API_KEY` -> NVIDIA NIM (`RETURNIQ_NVIDIA_MODEL`, default meta/llama-3.3-70b-instruct). Set `RETURNIQ_AUTO_OLLAMA=1` to auto-use a running Ollama.
- LLM engines (all behind the same grounding gate): `ANTHROPIC_API_KEY` -> Claude, `GEMINI_API_KEY` -> Gemini (`RETURNIQ_GEMINI_MODEL`, default gemini-2.5-flash). Force one with `RETURNIQ_LLM=anthropic|gemini|rules`. With no key, a deterministic rule engine is used.
- Postgres: set `DATABASE_URL` (+ `RETURNIQ_ENV=prod`) and `DATABASE_URL_READONLY` (role with SELECT on `analytics_*` only).
