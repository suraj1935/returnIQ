import pytest
from sqlalchemy import text

from ai.copilot import ask, parse_period, verify_grounding
from ai.tools import ToolError, Toolbox, ToolResult
from backend.db import get_readonly_engine


def _res(rows):
    return ToolResult("t", "q1", {}, rows, len(rows), not rows, "2024-12-31")


def test_grounding_rejects_fabricated_number():
    ok, bad = verify_grounding("The return rate is 31.5%.", [_res([{"return_rate": 0.2246}])])
    assert not ok and "31.5%" in bad


def test_grounding_accepts_percent_and_counts():
    rows = [{"return_rate": 0.2246, "a": 2481, "b": 11046}]
    ok, _ = verify_grounding("Rate 22.46% (2481 of 11046).", [_res(rows)])
    assert ok


def test_answer_numbers_match_database():
    a = ask("return rate in 2024")
    with get_readonly_engine().connect() as c:
        n, k = c.execute(text("SELECT COUNT(*), SUM(returned_30d) FROM analytics_order_features "
                              "WHERE label_mature=1 AND order_date BETWEEN '2024-01-01' AND '2024-12-31'")).one()
    assert a.grounded and f"{k} returned of {n} mature" in a.answer


def test_no_data_is_explicit_not_zero():
    a = ask("return rate in 2019")
    assert "can't be computed" in a.answer and "0.00%" not in a.answer


def test_unknown_product_no_data():
    assert "PROD-9999" in ask("return rate for PROD-9999").answer


def test_tool_rejects_bad_args():
    box = Toolbox()
    with pytest.raises(ToolError):
        box.return_rate(start_date="2024-13-45")
    with pytest.raises(ToolError):
        box.return_rate(product_id="1; DROP TABLE x")


def test_readonly_engine_cannot_write():
    with pytest.raises(Exception):
        with get_readonly_engine().begin() as c:
            c.execute(text("DELETE FROM raw_orders"))


def test_parse_period():
    assert parse_period("Q3 2024") == ("2024-07-01", "2024-09-30")
    assert parse_period("March 2024") == ("2024-03-01", "2024-03-31")
    assert parse_period("in 2023") == ("2023-01-01", "2023-12-31")
    assert parse_period("whenever") == (None, None)


def test_grounding_checks_spelled_out_numbers():
    two_rows = [_res([{"product_id": "A"}, {"product_id": "B"}])]
    ok, bad = verify_grounding("The three products with the highest rate.", two_rows)
    assert not ok and "3" in bad
    assert verify_grounding("The two products with the highest rate.", two_rows)[0]


RATE_2024 = ToolResult("return_rate", "q1", {"s": "2024-01-01", "e": "2024-12-31"},
                       [{"mature_orders": 1200, "returned_orders": 280, "return_rate": 0.2333}], 1, False, "2024-12-31")


@pytest.mark.parametrize("claim", [
    "In 2023 the return rate was 23.33%",        # wrong period
    "The rate across 23 orders was high",         # rate reused as a count
    "280 of 1200 for PROD-0001",                  # entity never queried
    "On 2024-06-01 the rate was 23.33%",          # date not in params
])
def test_grounding_rejects_right_number_wrong_context(claim):
    assert not verify_grounding(claim, [RATE_2024])[0]


@pytest.mark.parametrize("claim", [
    "In 2024 the rate was 23.33% (280 of 1200)",
    "The rate is 23.3 percent.",
    "Data as of 2024-12-31: 280 returned of 1200 mature orders.",
])
def test_grounding_accepts_correct_claims(claim):
    assert verify_grounding(claim, [RATE_2024])[0]


def test_year_cannot_ride_on_data_as_of():
    full_range = ToolResult("return_rate", "q2", {}, [{"return_rate": 0.2333}], 1, False, "2024-12-31")
    assert not verify_grounding("In 2024 the rate was 23.33%", [full_range])[0]


def test_top_rate_for_specific_product_is_not_routed_to_ranking():
    a = ask("top return rate for product PROD-0052")
    assert a.citations[0]["tool"] == "return_rate" and "PROD-0052" in a.answer


@pytest.mark.parametrize("val", ["ALL", "", "null", "None", " all "])
def test_tools_treat_nullish_filters_as_no_filter(val):
    box = Toolbox()
    assert box.return_rate(product_id=val).rows == box.return_rate().rows


def test_refund_total_uses_order_date_basis():
    box = Toolbox()
    assert "order" in box.refund_total("2024-01-01", "2024-03-31").note.lower()


# ---- date parts must not leak into the integer pool
MARCH = ToolResult("return_rate", "q1", {"s": "2024-03-01", "e": "2024-03-31"},
                   [{"mature_orders": 1200, "returned_orders": 280, "return_rate": 0.2333}], 1, False, "2024-12-31")


@pytest.mark.parametrize("claim", [
    "Only 3 returns were rejected",
    "31 orders were returned in March 2024",
    "On April 2 the rate was 23.33%",             # day outside the queried range
    "From March 1 to March 32, 2024 the rate was 23.33%",
])
def test_grounding_rejects_date_part_leak(claim):
    assert not verify_grounding(claim, [MARCH])[0]


@pytest.mark.parametrize("claim", [
    "From March 1 to March 31, 2024 the return rate was 23.33%",
    "From 1 March to 31st March 2024 the return rate was 23.33%",
    "March 1-31, 2024: 23.33%",
    "280 of 1200 mature orders were returned in March 2024",
])
def test_grounding_accepts_real_dates(claim):
    assert verify_grounding(claim, [MARCH])[0]


def test_grounding_rejects_swapped_x_of_y():
    ok, bad = verify_grounding("1200 of 280 orders were returned", [MARCH])
    assert not ok and {"1200", "280"} <= set(bad)
    assert not verify_grounding("1,200 out of 280 orders were returned", [MARCH])[0]
    assert verify_grounding("280 of 1200 orders were returned", [MARCH])[0]


# ---- Ollama refusals (no network: _openai_compat_answer is mocked)
def _patch_ollama(monkeypatch, replies):
    import ai.copilot as cp
    calls = []

    def fake(box, q, url, model, key, extra=None):
        calls.append(model)
        return replies[len(calls) - 1]

    monkeypatch.setattr(cp, "_openai_compat_answer", fake)
    monkeypatch.setenv("RETURNIQ_OLLAMA_FALLBACKS", "m2,m3")
    monkeypatch.setattr(cp, "OLLAMA_MODEL", "m1")
    monkeypatch.setenv("RETURNIQ_LLM", "ollama")
    return calls


def test_ollama_digitless_toolless_reply_is_a_refusal(monkeypatch):
    from ai.copilot import CANNOT_ANSWER
    calls = _patch_ollama(monkeypatch, [("I can only answer questions about returns.", [])])
    a = ask("What is the weather in Paris?")
    assert calls == ["m1"]
    assert a.engine == "ollama" and a.answer == CANNOT_ANSWER and not a.citations


def test_ollama_toolless_reply_with_digits_tries_next_model(monkeypatch):
    calls = _patch_ollama(monkeypatch, [("It is 18 degrees.", [])] * 3)
    a = ask("What is the weather in Paris?")
    assert calls == ["m1", "m2", "m3"]
    assert a.engine == "rules(fallback)"


def test_ollama_exception_moves_to_next_model(monkeypatch):
    import ai.copilot as cp
    seen = []

    def flaky(box, q, url, model, key, extra=None):
        seen.append(model)
        if model == "m1":
            raise TimeoutError
        return "I cannot help with that.", []

    monkeypatch.setattr(cp, "_openai_compat_answer", flaky)
    monkeypatch.setenv("RETURNIQ_OLLAMA_FALLBACKS", "m2,m3")
    monkeypatch.setattr(cp, "OLLAMA_MODEL", "m1")
    text, results = cp._ollama_answer(Toolbox(), "weather?")
    assert seen == ["m1", "m2"] and text and not results


def test_eval_llm_counts_fallbacks_separately(monkeypatch, capsys):
    import ai.copilot as cp
    import ai.eval_llm as ev
    from ai.copilot import Answer

    def fake_ask(q, box=None, use_llm=None):
        if "weather" in q:                       # fallback refusal: must not count as a pass
            return Answer(cp.CANNOT_ANSWER, True, [], "rules(fallback)")
        if q.startswith("What was the return rate"):                         # LLM answered correctly
            return Answer("x", True, [{"tool": "return_rate"}], "ollama")
        return Answer("x", True, [{"tool": "return_rate"}], "rules(fallback)")

    monkeypatch.setattr(cp, "ask", fake_ask)
    ev.run(["m1"])
    out = capsys.readouterr().out
    assert "tool routing 1/7, grounded 1/7, fell back 6/7" in out


# ---- Fix 1: unified LLM decline/fallback (all engines, mock _openai_compat_answer) -----------

def _patch_llm(monkeypatch, engine, reply_text):
    """Patch _openai_compat_answer to return a single text reply with no tool results.
    Works for ollama, nvidia (and anthropic/gemini use different functions, but we test
    the shared ask() logic via the openai-compat path for simplicity)."""
    import ai.copilot as cp
    calls = []

    def fake(box, q, url, model, key, extra=None):
        calls.append(model)
        return reply_text, []

    monkeypatch.setattr(cp, "_openai_compat_answer", fake)
    monkeypatch.setenv("RETURNIQ_LLM", engine)
    monkeypatch.setattr(cp, "OLLAMA_MODEL", "m1")
    monkeypatch.setattr(cp, "NVIDIA_MODEL", "m1")
    monkeypatch.delenv("RETURNIQ_OLLAMA_FALLBACKS", raising=False)
    return calls


def test_llm_digitless_decline_falls_back_to_rules_when_data_available(monkeypatch):
    """LLM says 'Roughly a quarter of orders.' (no digits) for a return-rate question
    → rules finds the real rate → answer comes back with engine 'rules(fallback)'."""
    calls = _patch_llm(monkeypatch, "ollama", "Roughly a quarter of orders.")
    a = ask("What was the return rate in 2024?")
    assert a.engine == "rules(fallback)"
    assert "return_rate" in a.citations[0]["tool"]
    # The answer must contain the actual rate figure from the DB
    assert any(c.isdigit() for c in a.answer), "Expected a numeric rate in the answer"
    assert calls == ["m1"]


def test_llm_digitless_decline_cannot_answer_when_no_rules_data(monkeypatch):
    """LLM says 'I can only answer questions about returns.' for a weather question
    → rules also finds nothing → CANNOT_ANSWER, no citations, LLM called exactly once."""
    from ai.copilot import CANNOT_ANSWER
    calls = _patch_llm(monkeypatch, "ollama", "I can only answer questions about returns.")
    a = ask("What is the weather in Paris?")
    assert a.answer == CANNOT_ANSWER
    assert not a.citations
    assert calls == ["m1"]


# ---- Fix 2: written full dates that equal a data_as_of value pass grounding ----------------

_FULL_RANGE = __import__('ai.tools', fromlist=['ToolResult']).ToolResult(
    "return_rate", "q1", {},
    [{"mature_orders": 10000, "returned_orders": 2226, "return_rate": 0.2226}],
    1, False, "2024-12-31",
)


@pytest.mark.parametrize("claim", [
    "as of December 31, 2024 the rate is 22.26%",
    "as of 31 December 2024 the rate is 22.26%",
])
def test_grounding_accepts_written_full_date_matching_as_of(claim):
    """Month-Day-Year written out should pass when it exactly equals a data_as_of date."""
    ok, bad = verify_grounding(claim, [_FULL_RANGE])
    assert ok, f"Expected pass but got bad={bad!r} for {claim!r}"


def test_grounding_rejects_bare_year_with_full_range_data():
    """'In 2024 the rate was ...' still fails when data has no year in params (data_as_of only)."""
    ok, bad = verify_grounding("In 2024 the rate was 22.26%", [_FULL_RANGE])
    assert not ok, "Bare year should still be rejected when no date params"


def test_grounding_rejects_wrong_day_in_written_date():
    """'December 30, 2024' does not match data_as_of '2024-12-31' → fails."""
    ok, bad = verify_grounding("as of December 30, 2024 the rate is 22.26%", [_FULL_RANGE])
    assert not ok, "Wrong day should fail"
