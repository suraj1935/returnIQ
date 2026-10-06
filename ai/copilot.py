"""
ai/copilot.py - Returns Intelligence Copilot.

Interchangeable engines, same guarantees (pick with RETURNIQ_LLM or by available API keys):
  * ollama (local or Ollama Cloud), nvidia (NIM), anthropic, gemini: LLM tool-calling over ai.tools.
  * rules (default / offline / fallback): deterministic intent -> tool routing with templated answers.

Guarantees enforced IN CODE (not by prompt):
  1. Every number, year, date and id in the final answer must match a tool result of the right
     kind (verify_grounding); otherwise the answer is replaced by a refusal.
  2. No tool result / empty result => explicit "cannot be answered" message.
  3. Every answer carries citations (tool, query_id, params, data_as_of).
"""
from __future__ import annotations

import calendar
import json
import os
import re
from dataclasses import dataclass, field

from sqlalchemy import text

from ai.tools import MIN_ORDERS, TOOL_SCHEMAS, ToolError, ToolResult, Toolbox, call_tool

MODEL = os.environ.get("RETURNIQ_LLM_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.environ.get("RETURNIQ_GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("RETURNIQ_OLLAMA_MODEL", "gemma4:31b")
NVIDIA_URL = os.environ.get("NVIDIA_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.environ.get("RETURNIQ_NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
LLM_TIMEOUT_S = float(os.environ.get("RETURNIQ_LLM_TIMEOUT", "90"))
MAX_TOOL_ROUNDS = 4
CANNOT_ANSWER = ("I can't answer that from the available data. I can report return rates, top return reasons, "
                 "top products by return rate, refund totals, high-risk open orders, and model metrics.")

FORECAST_REFUSAL = ("ReturnIQ reports historical return metrics and per-order risk scores. "
                    "It does not forecast aggregate future return rates.")
_RISK_RE = r"\b(risk|risky|likely to be returned|high[- ]risk)\b"
_FORECAST_RE = (r"\b(will|forecast|predict\w*|projections?|next month|next quarter|next year|"
                r"going to be|expected to be)\b")

SYSTEM_PROMPT = """You are the ReturnIQ Returns Intelligence Copilot.
Rules:
- Every number you state MUST come verbatim (or as a percentage of a rate) from a tool result in this conversation.
- Never estimate, extrapolate, or compute new statistics. Quote tool values only.
- If a tool returns empty or there is no suitable tool, say the question cannot be answered from the data.
- Only pass date or product filters the user actually asked for. If no period is given, omit start_date/end_date.
- State the period you used. Do not speculate about causes; report only what the tools returned.
- If a tool returns fewer rows than the user asked for, say so.
- Be concise."""


def _data_window(box: Toolbox) -> tuple[str, str] | None:
    with box.engine.connect() as c:
        start = c.execute(text("SELECT MIN(order_date) FROM analytics_order_features")).scalar()
        end = c.execute(text("SELECT snapshot_date FROM analytics_meta LIMIT 1")).scalar()
    return (str(start), str(end)) if start and end else None


def _system_prompt(box: Toolbox) -> str:
    """SYSTEM_PROMPT plus the data window, so relative dates resolve against the snapshot, not today."""
    try:
        win = _data_window(box)
    except Exception:
        win = None
    if not win:
        return SYSTEM_PROMPT
    start, end = win
    return (f"{SYSTEM_PROMPT}\nThe data covers {start} to {end}. Resolve relative dates such as 'last March' or "
            f"'last month' against {end}, not today's date. If a requested period is outside this window, "
            f"say the data does not cover it.")


@dataclass
class Answer:
    answer: str
    grounded: bool
    citations: list[dict] = field(default_factory=list)
    engine: str = "rules"


# ---------------------------------------------------------------- grounding
# A figure is "grounded" only if it is the RIGHT KIND of value from a tool result:
#   * integer token   -> must equal an integer-valued tool field (counts), never a rounded rate
#   * decimal token   -> must equal a tool value at that precision, or a rate (0..1) x100
#   * percent token   -> must equal a rate (0..1) x100
#   * years / ISO dates / entity ids -> must appear in tool params or rows (data_as_of only
#     validates an exact ISO date, never a bare year, so "in 2023" cannot ride on 2024 data)
_NUM = re.compile(r"(?<![\w.-])-?\d[\d,]*\.?\d*%?")
_ID = re.compile(r"\b(?:PROD|ORD|CUST|RET)-\d+\b", re.I)
_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_STRIP = re.compile(r"\bQ[1-4]\b|\b30[- ]day\b", re.I)
_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.I)


# LLMs emit typographic variants (non-breaking hyphen, narrow no-break space, minus sign...).
# Fold them to ASCII BEFORE matching, otherwise "2024<U+2011>01<U+2011>01" slips past the date check
# and "1<U+202F>234" is read as two numbers.
_FOLD = {**{ord(c): "-" for c in map(chr, (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212))},
         **{ord(c): " " for c in map(chr, (0x00A0, 0x2009, 0x202F))}}
_DIGIT_GROUP = re.compile(r"(?<=\d) (?=\d{3}\b)")


def _normalise(t: str) -> str:
    return _DIGIT_GROUP.sub("", t.translate(_FOLD))


# Day/month mentions next to a month name ("March 1", "1 March", "March 1-31", "31st March"). They are
# checked against real dates instead of the integer pool, so "31 orders" cannot ride on a 03-31 param.
_MON = (r"(?:(?i:january|february|march|april|june|july|august|september|october|november|december"
        r"|sept)|May|(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?)")
_ORD = r"(?:st|nd|rd|th)?"
_MD = re.compile(r"(?<![\w.,-])(" + _MON + r")\s+(\d{1,2})" + _ORD + r"(?:\s*(?:-|to|through)\s*(\d{1,2})" + _ORD
                 + r")?(?![\d,]\d|\.\d)")
_DM = re.compile(r"(?<![\w.,-])(\d{1,2})" + _ORD + r"\s+(?:of\s+)?(" + _MON + r")(?!\w)")
_MDY = re.compile(r"(?<![\w.,-])(" + _MON + r")\s+(\d{1,2})" + _ORD + r"[,\s]+((?:19|20)\d{2})\b")
_DMY = re.compile(r"(?<![\w.,-])(\d{1,2})" + _ORD + r"\s+(?:of\s+)?(" + _MON + r")[,\s]+((?:19|20)\d{2})\b")
_OF = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*)\s+(?:out\s+)?of\s+(\d+(?:,\d{3})*)(?![\d.]\d)", re.I)


def _month_no(name: str) -> int:
    n = name.lower().rstrip(".")[:3]
    return next(i for i, m in enumerate(calendar.month_abbr) if m.lower() == n)


@dataclass
class _Facts:
    ints: set
    nums: list
    strings: set
    as_of: set
    dates: set        # exact ISO dates appearing in tool rows/params
    ranges: list      # (lo, hi) ISO bounds from each tool call's queried params


def _date_ok(f: "_Facts", month: int, day: int) -> bool:
    """True if month/day is an exact date in the tool data or falls inside a queried range."""
    from datetime import date, timedelta
    if any(int(d[5:7]) == month and int(d[8:10]) == day for d in f.dates):
        return True
    if any(int(d[5:7]) == month and int(d[8:10]) == day for d in f.as_of):  # written full date matching data_as_of
        return True
    for lo, hi in f.ranges:
        try:
            d0, d1 = date.fromisoformat(lo), date.fromisoformat(hi)
        except ValueError:
            continue
        for k in range(min((d1 - d0).days, 3660) + 1):
            d = d0 + timedelta(days=k)
            if d.month == month and d.day == day:
                return True
    return False


def _check_dates(text_: str, f: "_Facts", bad: list[str]) -> str:
    """Validate and strip 'Month D' / 'D Month' mentions; whatever is not a real date is reported."""
    def one(mon, day, raw):
        m, d = _month_no(mon), int(day)
        if not (1 <= d <= 31 and _date_ok(f, m, d)):
            bad.append(raw)

    def md(m):
        for d in (m.group(2), m.group(3)):
            if d:
                one(m.group(1), d, f"{m.group(1)} {d}")
        return " "

    def dm(m):
        one(m.group(2), m.group(1), f"{m.group(1)} {m.group(2)}")
        return " "

    return _DM.sub(dm, _MD.sub(md, text_))


def _facts(results: list[ToolResult]) -> _Facts:
    f = _Facts(set(), [], set(), set(), set(), [])

    def walk(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, int):
            f.ints.add(v)
            f.nums.append(float(v))
        elif isinstance(v, float):
            f.nums.append(v)
            if v.is_integer() and abs(v) > 1:
                f.ints.add(int(v))  # e.g. SUM() returned as 54909.0
        elif isinstance(v, str):
            f.strings.add(v.upper())
            if _ISO.fullmatch(v):
                f.dates.add(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    for r in results:
        walk(r.rows)
        walk(r.params)
        iso = sorted(v for v in r.params.values() if isinstance(v, str) and _ISO.fullmatch(v))
        if iso:
            f.ranges.append((iso[0], iso[-1]))
        f.ints.add(int(r.row_count))
        f.nums.append(float(r.row_count))
        if r.data_as_of:
            f.as_of.add(str(r.data_as_of))
    return f


def verify_grounding(answer: str, results: list[ToolResult]) -> tuple[bool, list[str]]:
    """(ok, unverified_tokens). See the block comment above for the matching rules."""
    f = _facts(results)
    bad: list[str] = []
    text_ = re.sub(r"(?m)^\s*\d+[.)]\s", " ", _normalise(answer))  # list markers are not figures
    text_ = re.sub(r"(?m)^\s*\|\s*\d+\s*\|", "|", text_)            # markdown table rank cells
    text_ = re.sub(r"(?i)(\d)\s*percent\b", r"\1%", text_)
    text_ = _WORD_RE.sub(lambda m: _WORDS[m.group(1).lower()], text_)

    for a, b in _OF.findall(text_):  # "X of Y": the part cannot exceed the whole
        if int(a.replace(",", "")) > int(b.replace(",", "")):
            bad += [a, b]

    blob = " ".join(f.strings)
    for i in _ID.findall(text_):
        if i.upper() not in f.strings and i.upper() not in blob:
            bad.append(i)
    text_ = _ID.sub(" ", text_)

    iso_ok = {s for s in f.strings} | {s.upper() for s in f.as_of}
    for d in _ISO.findall(text_):
        if d not in iso_ok:
            bad.append(d)
    text_ = _ISO.sub(" ", text_)

    # Pre-scan: "Month Day, Year" / "Day Month Year" that exactly matches a data_as_of
    # → unlock that year so the full written date passes grounding
    _as_of_years: set[str] = set()
    for _m in _MDY.finditer(text_):
        _mon, _day, _yr = _month_no(_m.group(1)), int(_m.group(2)), _m.group(3)
        if any(d == f"{_yr}-{_mon:02d}-{_day:02d}" for d in f.as_of):
            _as_of_years.add(_yr)
    for _m in _DMY.finditer(text_):
        _day, _mon, _yr = int(_m.group(1)), _month_no(_m.group(2)), _m.group(3)
        if any(d == f"{_yr}-{_mon:02d}-{_day:02d}" for d in f.as_of):
            _as_of_years.add(_yr)
    text_ = _check_dates(text_, f, bad)

    years_ok = set(_YEAR.findall(blob)) | _as_of_years
    for y in _YEAR.findall(text_):
        if y not in years_ok:
            bad.append(y)
    text_ = _YEAR.sub(" ", text_)

    rates = [n for n in f.nums if 0 <= n <= 1]
    for tok in _NUM.findall(_STRIP.sub(" ", text_)):
        raw = tok.rstrip("%").replace(",", "").rstrip(".")
        if not raw or raw == "-":
            continue
        val = float(raw)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        if tok.endswith("%"):
            ok = any(round(r * 100, dec) == round(val, dec) for r in rates)
        elif dec == 0:
            ok = int(val) in f.ints
        else:
            ok = (any(round(n, dec) == round(val, dec) for n in f.nums)
                  or any(round(r * 100, dec) == round(val, dec) for r in rates))
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

    is_risk = re.search(_RISK_RE, ql) and "model" not in ql
    if not is_risk and re.search(_FORECAST_RE, ql):
        return FORECAST_REFUSAL, []

    if is_risk:
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
    if not prod and (re.search(r"\b(top|worst|highest|which)\b.*\bproduct", ql)
                     or re.search(r"\bproducts?\b.*\b(rate|return)", ql)):
        r = box.top_products_by_return_rate(s, e, MIN_ORDERS, 5)
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
        resp = client.messages.create(model=MODEL, max_tokens=800, system=_system_prompt(box),
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
        system_instruction=_system_prompt(box), tools=[types.Tool(function_declarations=decls)],
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
    msgs = [{"role": "system", "content": _system_prompt(box)}, {"role": "user", "content": q}]
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


def _is_local(url: str) -> bool:
    return any(h in url for h in ("localhost", "127.0.0.1", "[::1]"))


def _ollama_answer(box, q):
    """Local Ollama or Ollama Cloud (OpenAI-compatible /v1). Tries OLLAMA_MODEL, then each model in
    RETURNIQ_OLLAMA_FALLBACKS (comma separated); any failure falls through to the next, then to rules.
    A reply with no tool call and no digits is accepted as a refusal (ask() maps it to CANNOT_ANSWER);
    a tool-less reply containing digits is an ungrounded answer and moves on to the next model."""
    api_key = os.environ.get("OLLAMA_API_KEY") or None
    models = [OLLAMA_MODEL] + [m.strip() for m in os.environ.get("RETURNIQ_OLLAMA_FALLBACKS", "").split(",") if m.strip()]
    last: Exception | None = None
    for m in models:
        # Local small-GPU models: thinking mode takes minutes and adds nothing to tool routing.
        extra = {"reasoning_effort": "none"} if _is_local(OLLAMA_URL) and "gpt-oss" not in m else {}
        try:
            text, results = _openai_compat_answer(box, q, OLLAMA_URL, m, api_key, extra)
            if results:
                return text, results
            if text and not re.search(r"\d", text):  # no tools, no figures: a refusal, don't poll more models
                return text, results
        except ToolError:
            raise
        except Exception as ex:  # timeout / 5xx / malformed tool call -> next model
            last = ex
    if last:
        raise last
    return "", []


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
        if llm and text and not results and not re.search(r"\d", text):
            # LLM declined with no digits (clean refusal) → try rules; if rules finds data use it,
            # otherwise this is genuinely unanswerable: return CANNOT_ANSWER with the LLM engine name
            _fb_text, _fb_results = _rules_answer(box, q)
            if _fb_results:
                engine, text, results = "rules(fallback)", _fb_text, _fb_results
            else:
                return Answer(CANNOT_ANSWER, True, engine=engine)
        elif llm and (not text or not results):  # ungrounded numbers or exhausted rounds → rules decides
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
