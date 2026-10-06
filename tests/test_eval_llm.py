import os
import json
import pathlib
import builtins
import pytest
from unittest import mock

# Import the run function from the updated eval script
from ai.eval_llm import run, CASES, params_match

# Helper to create a deterministic Answer object
from ai.copilot import Answer, CANNOT_ANSWER

def fake_answer(question, box=None, use_llm=None):
    """Return a deterministic Answer based on the question.
    - If the question contains "weather" -> refusal (rules fallback).
    - If the question contains "poem" -> refusal (rules fallback).
    - Otherwise return a successful answer with a dummy citation for the expected tool.
    """
    # Find expected entry from CASES
    for q, tools, params in CASES:
        if q == question:
            expect_tool = next(iter(tools)) if tools else None
            expect_p = params.get(expect_tool, {}) if tools else {}
            break
    else:
        expect_tool = None
        expect_p = {}

    if "weather" in question.lower() or "poem" in question.lower():
        # Simulate a rules fallback (no citations, engine "rules(fallback)")
        return Answer(CANNOT_ANSWER, True, [], "rules(fallback)")
    else:
        # Build a dummy citation matching the expected tool and params
        citation = {"tool": expect_tool, "params": dict(expect_p)}
        return Answer("dummy answer", True, [citation], "ollama")

@pytest.fixture(autouse=True)
def patch_copilot_ask(monkeypatch):
    """Patch ai.copilot.ask for all tests in this module."""
    import ai.copilot as cp
    monkeypatch.setattr(cp, "ask", fake_answer)
    # Ensure the fallback env var is empty
    os.environ["RETURNIQ_OLLAMA_FALLBACKS"] = ""

def test_eval_llm_json_output(tmp_path, capsys):
    # Path for the JSON output
    out_file = tmp_path / "eval.json"

    # Run the evaluation with a single repeat to keep it fast
    run(["dummy-model", "--out", str(out_file), "--repeats", "1"])

    # Capture printed markdown table (not strictly needed for this test)
    captured = capsys.readouterr().out
    assert "| Model | Tool routing" in captured

    # Verify JSON file exists and has the expected structure
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert isinstance(data, list) and len(data) == 1
    data = data[0]
    assert data["model"] == "dummy-model"
    assert data["cases"] == len(CASES)  # repeats=1
    # The per_case list should have one entry per case
    assert isinstance(data["per_case"], list)
    assert len(data["per_case"]) == len(CASES)
    # Each entry must contain the required keys
    for entry in data["per_case"]:
        for key in ["question", "expected_tools", "expected_params", "tools_used", "params_match", "engine", "grounded", "seconds"]:
            assert key in entry
        # For weather/poem questions we expect no tools used and a fallback engine
        if "weather" in entry["question"].lower() or "poem" in entry["question"].lower():
            assert entry["tools_used"] == []
            assert entry["engine"] == "rules(fallback)"
        else:
            # Otherwise we should have a tool and the dummy engine
            assert entry["engine"] == "ollama"
            assert entry["tools_used"]
            # Params should match (empty dict in our fake)
            assert entry["params_match"] is True


def test_params_match_q3_prod_0052():
    expect = next(p for q, _, p in CASES if q == "Return rate for PROD-0052 in Q3 2024?")
    good = [{"tool": "return_rate", "params": {"s": "2024-07-01", "e": "2024-09-30", "p": "PROD-0052"}}]
    bad = [{"tool": "return_rate", "params": {"s": "2024-07-01", "e": "2024-09-30", "p": "PROD-0001"}}]
    assert params_match(good, expect)
    assert not params_match(bad, expect)


def test_out_file_keeps_every_model_in_run_order(tmp_path):
    out_file = tmp_path / "eval.json"
    run(["model-a", "model-b", "--out", str(out_file), "--repeats", "1"])
    data = json.loads(out_file.read_text())
    assert [d["model"] for d in data] == ["model-a", "model-b"]
    assert all(len(d["per_case"]) == len(CASES) for d in data)
