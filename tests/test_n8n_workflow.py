import json
import os
from pathlib import Path

os.environ["RETURNIQ_LLM"] = "rules"

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

WORKFLOW = Path(__file__).resolve().parent.parent / "n8n" / "returniq_daily_alert.json"


def _load():
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def _nodes(wf, typ):
    return [n for n in wf["nodes"] if n["type"] == typ]


def _has_key(obj, key):
    if isinstance(obj, dict):
        return key in obj or any(_has_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_key(v, key) for v in obj)
    return False


def test_workflow_structure():
    wf = _load()
    assert len(wf["nodes"]) == 6
    names = {n["name"] for n in wf["nodes"]}
    for src, outs in wf["connections"].items():
        assert src in names
        for branch in outs["main"]:
            for target in branch:
                assert target["node"] in names


def test_telegram_nodes_have_placeholder_and_no_credentials():
    wf = _load()
    telegram = _nodes(wf, "n8n-nodes-base.telegram")
    assert len(telegram) == 2
    assert all(n["parameters"]["chatId"] == "PASTE_YOUR_CHAT_ID" for n in telegram)
    assert not _has_key(wf, "credentials")


def test_http_body_gets_grounded_high_risk_answer():
    wf = _load()
    http = _nodes(wf, "n8n-nodes-base.httpRequest")[0]
    body = json.loads(http["parameters"]["jsonBody"])
    r = client.post("/copilot/ask", json=body)
    assert r.status_code == 200
    b = r.json()
    assert b["grounded"] is True
    assert "answer" in b and "engine" in b
    assert any(c["tool"] == "high_risk_orders" for c in b["citations"])
    assert all("tool" in c and "data_as_of" in c for c in b["citations"])


def test_alert_text_joins_with_char_code_newline():
    wf = _load()
    alert = next(n for n in wf["nodes"] if n["name"] == "Send alert to Telegram")
    assert "String.fromCharCode(10)" in alert["parameters"]["text"]
