"""
ai/eval_llm.py - compare LLM engines/models on the same questions.

For each (model, question): did it call the expected tool, did the answer pass verify_grounding,
how long did it take.  Usage:  python ai/eval_llm.py gemma4:31b gpt-oss:120b gpt-oss:20b
Requires .env (OLLAMA_URL / OLLAMA_API_KEY).  Results are printed; nothing is persisted.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backend  # noqa: E402,F401  (loads .env)
import ai.copilot as cp  # noqa: E402
from ai.tools import Toolbox  # noqa: E402

CASES = [
    ("What was the return rate in 2024?", {"return_rate"}),
    ("Return rate for PROD-0052 in Q3 2024?", {"return_rate"}),
    ("Top 3 return reasons for PROD-0052", {"top_return_reasons"}),
    ("Which 3 products have the highest return rate?", {"top_products_by_return_rate"}),
    ("Total refunds in March 2024", {"refund_total"}),
    ("How good is the risk model?", {"model_metrics"}),
    ("What is the weather in Paris?", set()),            # must refuse, no tools
]


def run(models: list[str]) -> None:
    box = Toolbox()
    os.environ["RETURNIQ_OLLAMA_FALLBACKS"] = ""
    for m in models:
        cp.OLLAMA_MODEL = m
        ok_tool = ok_ground = fell_back = 0
        times = []
        print(f"\n== {m}")
        for q, expect in CASES:
            t = time.time()
            a = cp.ask(q, box)
            dt = time.time() - t
            used = {c["tool"] for c in a.citations}
            used_llm = a.engine == "ollama"
            # Only the LLM's own answers are scored; a rules-engine fallback is counted separately.
            tool_ok = used_llm and (used == expect if expect else not a.citations)
            fell_back += not used_llm
            ok_tool += tool_ok
            ok_ground += used_llm and a.grounded
            times.append(dt)
            print(f"  {dt:5.1f}s engine={a.engine:15} tools_ok={tool_ok!s:5} grounded={a.grounded!s:5} {q}")
            if not used_llm:
                print("        (fell back to rules)")
        n = len(CASES)
        print(f"  -> tool routing {ok_tool}/{n}, grounded {ok_ground}/{n}, fell back {fell_back}/{n}, median {sorted(times)[n // 2]:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1:] or [cp.OLLAMA_MODEL])
