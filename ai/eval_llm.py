""" ai/eval_llm.py - compare LLM engines/models on the same questions.

For each (model, question): did it call the expected tool, did the answer pass verify_grounding,
how long did it take.  Usage:  python ai/eval_llm.py gemma4:31b gpt-oss:120b gpt-oss:20b
Requires .env (OLLAMA_URL / OLLAMA_API_KEY).  Results are printed; nothing is persisted.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backend  # noqa: E402,F401  (loads .env)
import ai.copilot as cp  # noqa: E402
from ai.tools import Toolbox  # noqa: E402

# CASES: (question, expected_tools, expected_params)
CASES = [
    ("What was the return rate in 2024?", {"return_rate"}, {"return_rate": {"s": "2024-01-01", "e": "2024-12-31"}}),
    ("Return rate for PROD-0052 in Q3 2024?", {"return_rate"}, {"return_rate": {"s": "2024-07-01", "e": "2024-09-30", "p": "PROD-0052"}}),
    ("Top 3 return reasons for PROD-0052", {"top_return_reasons"}, {"top_return_reasons": {"p": "PROD-0052"}}),
    ("Which 3 products have the highest return rate?", {"top_products_by_return_rate"}, {"top_products_by_return_rate": {"m": 30}}),
    ("Total refunds in March 2024", {"refund_total"}, {"refund_total": {"s": "2024-03-01", "e": "2024-03-31"}}),
    ("How good is the risk model?", {"model_metrics"}, {}),
    ("What is the weather in Paris?", set(), {}),            # must refuse, no tools
    ("return rate last March", {"return_rate"}, {"return_rate": {"s": "2024-03-01", "e": "2024-03-31"}}),
    ("Q3 2024 return rate", {"return_rate"}, {"return_rate": {"s": "2024-07-01", "e": "2024-09-30"}}),
    ("return rate from 2024-06-01 to 2024-08-31", {"return_rate"}, {"return_rate": {"s": "2024-06-01", "e": "2024-08-31"}}),
    ("return rate for PROD-9999", {"return_rate"}, {"return_rate": {"p": "PROD-9999"}}),
    ("Write me a poem", set(), {}),
    ("What will next month's return rate be?", set(), {}),   # forecast: refuse, no citations, any engine
    ("Return rate and top return reasons for PROD-0052 in 2024", {"return_rate", "top_return_reasons"}, {"return_rate": {"s": "2024-01-01", "e": "2024-12-31", "p": "PROD-0052"}, "top_return_reasons": {"s": "2024-01-01", "e": "2024-12-31", "p": "PROD-0052"}}),
]

# Cases judged on the final answer only (no citations), whichever engine produced it.
ANY_ENGINE = {"What will next month's return rate be?"}


def params_match(citations: list[dict], expect_params: dict) -> bool:
    """True if every citation's stored params equal the expected params for its tool."""
    return all(c.get("params", {}) == expect_params.get(c["tool"], {}) for c in citations)


def run(models: list[str]) -> None:
    """Run evaluation.

    Parameters
    ----------
    models: list[str]
        List of model identifiers.
    """
    parser = argparse.ArgumentParser(description="Evaluate LLMs on return-rate questions.")
    parser.add_argument("models", nargs="*", help="Model identifiers (default: use current OLLAMA_MODEL)")
    parser.add_argument("--out", type=str, default=None, help="Path to JSON output file")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per case (default 3)")
    args = parser.parse_args(models)

    repeats = max(1, args.repeats)
    out_path = args.out

    box = Toolbox()
    os.environ["RETURNIQ_OLLAMA_FALLBACKS"] = ""

    table_rows = []
    results = []  # one entry per model, in run order

    for m in (args.models or [cp.OLLAMA_MODEL]):
        cp.OLLAMA_MODEL = m
        ok_tool = ok_ground = fell_back = 0
        times = []
        per_case = []
        total_cases = len(CASES) * repeats
        print(f"\n== {m}")
        for _ in range(repeats):
            for q, expect_tools, expect_params in CASES:
                t = time.time()
                a = cp.ask(q, box)
                dt = time.time() - t
                used = {c["tool"] for c in a.citations}
                used_llm = a.engine == "ollama"
                p_ok = params_match(a.citations, expect_params)
                tool_ok = (used_llm or q in ANY_ENGINE) and (used == expect_tools if expect_tools else not a.citations) and p_ok
                fell_back += not used_llm
                ok_tool += tool_ok
                ok_ground += used_llm and a.grounded
                times.append(dt)
                per_case.append({
                    "question": q,
                    "expected_tools": sorted(list(expect_tools)),
                    "expected_params": expect_params,
                    "tools_used": sorted(list(used)),
                    "params_match": p_ok,
                    "engine": a.engine,
                    "answer": a.answer,
                    "grounded": a.grounded,
                    "seconds": dt,
                })
                print(f"  {dt:5.1f}s engine={a.engine:15} tools_ok={tool_ok!s:5} grounded={a.grounded!s:5} {q}")
                if not used_llm:
                    print("        (fell back to rules)")
                elif not tool_ok:
                    actual = {c["tool"]: c.get("params", {}) for c in a.citations}
                    print(f"        expected tools={sorted(expect_tools)} params={expect_params} | actual tools={sorted(used)} params={actual}")
        median = sorted(times)[len(times) // 2] if times else 0.0
        max_lat = max(times) if times else 0.0
        print(f"  -> tool routing {ok_tool}/{total_cases}, grounded {ok_ground}/{total_cases}, fell back {fell_back}/{total_cases}, median {median:.1f}s, max {max_lat:.1f}s")
        table_rows.append({
            "model": m,
            "tool_routing": f"{ok_tool}/{total_cases}",
            "grounded": f"{ok_ground}/{total_cases}",
            "fallback": f"{fell_back}/{total_cases}",
            "median": round(median, 2),
            "max": round(max_lat, 2),
        })
        if out_path:
            result = {
                "model": m,
                "date": datetime.now(timezone.utc).isoformat(),
                "cases": total_cases,
                "tool_routing_pass": f"{ok_tool}/{total_cases}",
                "grounded_pass": f"{ok_ground}/{total_cases}",
                "fallback_count": f"{fell_back}/{total_cases}",
                "median_latency": median,
                "max_latency": max_lat,
                "per_case": per_case,
            }
            results.append(result)
            with open(out_path, "w", encoding="utf-8") as f:  # rewritten per model so a partial run keeps its results
                json.dump(results, f, indent=2)

    print("\n| Model | Tool routing | Grounded | Fell back to rules | Median latency (s) | Max latency (s) |")
    print("|---|---|---|---|---|---|")
    for row in table_rows:
        print(f"| {row['model']} | {row['tool_routing']} | {row['grounded']} | {row['fallback']} | {row['median']} | {row['max']} |")

if __name__ == "__main__":
    run(sys.argv[1:])
