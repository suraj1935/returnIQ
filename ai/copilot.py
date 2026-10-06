"""
ai/copilot.py - Returns Intelligence Copilot.

Two interchangeable engines, same guarantees:
  * LLM engine (ANTHROPIC_API_KEY set): Claude tool-calling over ai.tools whitelisted tools.
  * Rule engine (default / offline): deterministic intent -> tool routing with templated answers.

Guarantees enforced IN CODE (not by prompt):
  1. Every number in the final answer must appear in a tool result (verify_grounding);
     otherwise the answer is replaced by a refusal.
  2. No tool result / empty result => explicit "cannot be answered" message.
  3. Every answer carries citations (tool, query_id, params, data_as_of).
"""
from __future__ import annotations

import calendar
import json
import os
import re
from dataclasses import dataclass, field

from ai.tools import TOOL_SCHEMAS, ToolError, ToolResult, Toolbox, call_tool

MODEL = os.environ.get("RETURNIQ_LLM_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.environ.get("RETURNIQ_GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("RETURNIQ_OLLAMA_MODEL", "qwen3:8b")
NVIDIA_URL = os.environ.get("NVIDIA_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.environ.get("RETURNIQ_NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
LLM_TIMEOUT_S = float(os.environ.get("RETURNIQ_LLM_TIMEOUT", "90"))
MAX_TOOL_ROUNDS = 4
CANNOT_ANSWER = ("I can't answer that from the available data. I can report return rates, top return reasons, "
                 "top products by return rate, refund totals, high-risk open orders, and model metrics.")

SYSTEM_PROMPT = """You are the ReturnIQ Returns Intelligence Copilot.
Rules:
- Every number you state MUST come verbatim (or as a percentage of a rate) from a tool result in this conversation.
- Never estimate, extrapolate, or compute new statistics. Quote tool values only.
- If a tool returns empty or there is no suitable tool, say the question cannot be answered from the data.
- Only pass date or product filters the user actually asked for. If no period is given, omit start_date/end_date.
- State the period you used. Do not speculate about causes; report only what the tools returned.
- If a tool returns fewer rows than the user asked for, say so.
- Be concise."""


@dataclass
class Answer:
    answer: str
    grounded: bool
    citations: list[dict] = field(default_factory=list)
    engine: str = "rules"


# ---------------------------------------------------------------- grounding
_NUM = re.compile(r"(?<![\w.-])-?\d[\d,]*\.?\d*%?")
_STRIP = re.compile(r"\b(?:PROD|ORD|CUST|RET)-\d+\b|\b\d{4}-\d{2}-\d{2}\b|\bQ[1-4]\b|\b30[- ]day\b|\b20\d{2}\b", re.I)


def _tool_numbers(results: list[ToolResult]) -> list[float]:
    out: list[float] = []

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    for r in results:
        walk(r.rows)
        walk(r.row_count)
        walk(r.params)
    return out


_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.I)


def verify_grounding(answer: str, results: list[ToolResult]) -> tuple[bool, list[str]]:
    """True iff every number in `answer` is explained by a tool value (exact, x100 as %, or rounded).
    Spelled-out numbers (zero..twelve) are checked too ("three products" vs 2 rows)."""
    allowed = _tool_numbers(results)
    bad: list[str] = []
    answer = re.sub(r"(?m)^\s*\d+[.)]\s", " ", answer)  # list markers are not figures
    answer = _WORD_RE.sub(lambda m: _WORDS[m.group(1).lower()], answer)
    for tok in _NUM.findall(_STRIP.sub(" ", answer)):
        raw = tok.rstrip("%").replace(",", "").rstrip(".")
        if not raw or raw == "-":
            continue
        val = float(raw)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        pct = tok.endswith("%")
        ok = False
        for a in allowed:
            for cand in ((a * 100,) if pct else (a, a * 100)):
                if round(cand, dec) == round(val, dec) or abs(cand - val) <= 0.5 * 10 ** -dec:
                    ok = True
                    break
            if ok:
                break
        if not ok:
            bad.append(tok)
    return (not bad), bad


def _cites(results: list[ToolResult]) -> list[dict]:
    return [{"tool": r.tool, "query_id": r.query_id, "params": r.params, "rows": r.row_count,
             "data_as_of": r.data_as_of} for r in results]


# ---------------------------------------------------------------- rule engine
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


def parse_period(q: str) -> tuple[str | None, str | None]:
    ql = q.lower()
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|-|and|through)\s*(\d{4}-\d{2}-\d{2})", ql)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"\bq([1-4])\s*(20\d{2})", ql)
    if m:
        qn, y = int(m.group(1)), int(m.group(2))
        sm, em = 3 * qn - 2, 3 * qn
        return f"{y}-{sm:02d}-01", f"{y}-{em:02d}-{calendar.monthrange(y, em)[1]:02d}"
    m = re.search(r"\b(" + "|".join(_MONTHS) + r")\s+(20\d{2})", ql)
    if m:
        mo, y = _MONTHS[m.group(1)], int(m.group(2))
        return f"{y}-{mo:02d}-01", f"{y}-{mo:02d}-{calendar.monthrange(y, mo)[1]:02d}"
    m = re.search(r"\b(20\d{2})\b", ql)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"
    return None, None


def _fmt_period(s, e):
    return f"{s} to {e}" if s else "the full data range"


def _rules_answer(box: Toolbox, q: str) -> tuple[str, list[ToolResult]]:
    ql = q.lower()
    s, e = parse_period(q)
    pm = re.search(r"\bPROD-\d{1,6}\b", q, re.I)
    prod = pm.group(0).upper() if pm else None
    per = _fmt_period(s, e)
    scope = f" for {prod}" if prod else ""

    if re.search(r"\b(risk|risky|likely to be returned|high[- ]risk)\b", ql) and "model" not in ql:
        r = box.high_risk_orders(10)
        if r.empty:
            return "No open orders with risk scores are available.", [r]
        lines = [f"{x['order_id']} ({x['product_id']}, {x['order_date']}): risk {x['risk_score']}" for x in r.rows]
        return ("Highest-risk open orders (model scores, synthetic data):\n- " + "\n- ".join(lines)), [r]
    if re.search(r"\b(model|pr-auc|precision|recall|f1|auc|accuracy)\b", ql):
        r = box.model_metrics()
        if r.empty:
            return "No model metrics are available. The model has not been trained yet.", [r]
        m = r.rows[0]
        return (f"Selected model {m['model']} (synthetic data) on the held-out test set: "
                f"PR-AUC {m['pr_auc']}, ROC-AUC {m['roc_auc']}, precision {m['precision']}, "
                f"recall {m['recall']}, F1 {m['f1']}, against a prevalence of {m['prevalence']}."), [r]
    if re.search(r"\b(reasons?|why)\b", ql):
        r = box.top_return_reasons(s, e, prod, 5)
        if r.empty:
            return f"No returns found{scope} for {per}.", [r]
        lines = [f"{x['reason']}: {x['returns']} returns ({x['share'] * 100:.1f}%)" for x in r.rows]
        return f"Top return reasons{scope} for {per}:\n- " + "\n- ".join(lines), [r]
    if re.search(r"\b(refund|refunded|refunds)\b", ql):
        r = box.refund_total(s, e, prod)
        if r.empty:
            return f"No refunds found{scope} for {per}.", [r]
        x = r.rows[0]
        return (f"Total refunded{scope} for {per}: {x['total_refund_amount']:.2f} "
                f"across {x['returns_counted']} non-rejected returns."), [r]
    if re.search(r"\b(top|worst|highest|which)\b.*\bproduct", ql) or re.search(r"\bproducts?\b.*\b(rate|return)", ql) and not prod:
        r = box.top_products_by_return_rate(s, e, 50, 5)
        if r.empty:
            return f"No products meet the minimum volume for {per}.", [r]
        lines = [f"{x['product_id']}: {x['return_rate'] * 100:.1f}% ({x['returned_orders']} of {x['mature_orders']} orders)"
                 for x in r.rows]
        return (f"Products with the highest 30-day return rate for {per} ({r.note}):\n- " + "\n- ".join(lines)), [r]
    if re.search(r"\b(return rate|rate of returns|how many returns|percent|percentage)\b", ql):
        r = box.return_rate(s, e, prod)
        if r.empty:
            return f"No mature orders found{scope} for {per}, so a return rate can't be computed.", [r]
        x = r.rows[0]
        return (f"The 30-day return rate{scope} for {per} is {x['return_rate'] * 100:.2f}% "
                f"({x['returned_orders']} returned of {x['mature_orders']} mature orders)"
                + (" - small sample, interpret with caution." if x['mature_orders'] < 100 else ".")), [r]
    if re.search(r"\b(define|definition|dictionary|label_mature|returned_30d|mature)\b", ql):
        r = box.data_dictionary()
        return "Data definitions:\n- " + "\n- ".join(f"{x['name']}: {x['description']}" for x in r.rows), [r]
    return CANNOT_ANSWER, []


# ---------------------------------------------------------------- LLM engine
def _llm_answer(box: Toolbox, q: str) -> tuple[str, list[ToolResult]]:
    import anthropic

    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": q}]
    results: list[ToolResult] = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(model=MODEL, max_tokens=800, system=SYSTEM_PROMPT,
                                      tools=TOOL_SCHEMAS, messages=msgs)
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            return "".join(b.text for b in resp.content if b.type == "text").strip(), results
        msgs.append({"role": "assistant", "content": resp.content})
        outs = []
        for u in uses:
            try:
                r = call_tool(box, u.name, dict(u.input))
                results.append(r)
                outs.append({"type": "tool_result", "tool_use_id": u.id, "content": r.to_json()})
            except (ToolError, TypeError) as ex:
                outs.append({"type": "tool_result", "tool_use_id": u.id, "is_error": True, "content": str(ex)})
        msgs.append({"role": "user", "content": outs})
    return "", results


def _gemini_answer(box: Toolbox, q: str) -> tuple[str, list[ToolResult]]:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    decls = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                       parameters_json_schema=t["input_schema"]) for t in TOOL_SCHEMAS]
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, tools=[types.Tool(function_declarations=decls)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    contents = [types.Content(role="user", parts=[types.Part(text=q)])]
    results: list[ToolResult] = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=cfg)
        calls = resp.function_calls or []
        if not calls:
            return (resp.text or "").strip(), results
        contents.append(resp.candidates[0].content)
        parts = []
        for fc in calls:
            try:
                r = call_tool(box, fc.name, dict(fc.args or {}))
                results.append(r)
                payload = {"result": json.loads(r.to_json())}
            except (ToolError, TypeError) as ex:
                payload = {"error": str(ex)}
            parts.append(types.Part.from_function_response(name=fc.name, response=payload))
        contents.append(types.Content(role="user", parts=parts))
    return "", results


def _openai_compat_answer(box: Toolbox, q: str, base_url: str, model: str,
                          api_key: str | None, extra: dict | None = None) -> tuple[str, list[ToolResult]]:
    """OpenAI-style /chat/completions tool calling; works for Ollama and NVIDIA NIM."""
    import httpx

    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                               "parameters": t["input_schema"]}} for t in TOOL_SCHEMAS]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    results: list[ToolResult] = []
    for _ in range(MAX_TOOL_ROUNDS):
        r = httpx.post(f"{base_url}/chat/completions", headers=headers, timeout=LLM_TIMEOUT_S,
                       json={"model": model, "messages": msgs, "tools": tools, "temperature": 0, **(extra or {})})
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            content = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S)
            return content.strip(), results
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            fn = c["function"]
            try:
                args = fn["arguments"]
                args = json.loads(args) if isinstance(args, str) else (args or {})
                res = call_tool(box, fn["name"], args)
                results.append(res)
                out = res.to_json()
            except (ToolError, TypeError, json.JSONDecodeError) as ex:
                out = json.dumps({"error": str(ex)})
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "name": fn["name"], "content": out})
    return "", results


def _ollama_answer(box, q):
    # Thinking mode takes minutes on small GPUs and adds nothing to tool routing.
    return _openai_compat_answer(box, q, OLLAMA_URL, OLLAMA_MODEL, None, {"reasoning_effort": "none"})


def _nvidia_answer(box, q):
    return _openai_compat_answer(box, q, NVIDIA_URL, NVIDIA_MODEL, os.environ["NVIDIA_API_KEY"])


def _ollama_up() -> bool:
    try:
        import httpx
        return httpx.get(OLLAMA_URL.removesuffix("/v1") + "/api/tags", timeout=0.5).status_code == 200
    except Exception:
        return False


def _pick_engine() -> str:
    forced = os.environ.get("RETURNIQ_LLM", "").lower()  # anthropic|gemini|nvidia|ollama|rules
    if forced in ("anthropic", "gemini", "nvidia", "ollama", "rules"):
        return forced
    if os.environ.get("NVIDIA_API_KEY"):
        return "nvidia"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "ollama" if os.environ.get("RETURNIQ_AUTO_OLLAMA") == "1" and _ollama_up() else "rules"


# ---------------------------------------------------------------- entry
def ask(question: str, box: Toolbox | None = None, use_llm: bool | None = None) -> Answer:
    q = question.strip()
    if not q:
        return Answer(CANNOT_ANSWER, True)
    box = box or Toolbox()
    engine = _pick_engine() if use_llm is None else ("anthropic" if use_llm else "rules")
    llm = engine != "rules"
    runner = {"anthropic": _llm_answer, "gemini": _gemini_answer, "nvidia": _nvidia_answer,
              "ollama": _ollama_answer, "rules": _rules_answer}[engine]
    try:
        text, results = runner(box, q)
        if llm and (not text or not results):  # model skipped the tools -> deterministic path decides
            engine, text, results = "rules(fallback)", *_rules_answer(box, q)
    except ToolError as ex:
        return Answer(f"Invalid request: {ex}", True, engine=engine)
    except Exception:  # timeout, connection refused, bad API key, malformed tool call...
        engine = "rules(fallback)"
        text, results = _rules_answer(box, q)
    if results and all(r.tool == "data_dictionary" for r in results):
        return Answer(text, True, _cites(results), engine)  # static definitions, no metrics
    ok, bad = verify_grounding(text, results)
    if not ok:
        return Answer("I could not verify these figures against the data, so I won't state them "
                      f"(unverified: {', '.join(bad)}).", False, _cites(results), engine)
    if engine in ("anthropic", "gemini", "nvidia", "ollama") and results:
        # Filters are chosen by the model, so always disclose what was actually queried.
        text += "\n\nScope: " + "; ".join(
            f"{r.tool}({', '.join(f'{k}={v}' for k, v in r.params.items()) or 'no filters'}) -> {r.row_count} row(s)"
            for r in results)
    return Answer(text, True, _cites(results), engine)
