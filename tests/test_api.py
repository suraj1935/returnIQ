import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_return_rate_ok():
    r = client.get("/metrics/return-rate", params={"start_date": "2024-01-01", "end_date": "2024-06-30"})
    assert r.status_code == 200
    b = r.json()
    assert 0 < b["return_rate"] < 1 and b["returned_orders"] <= b["mature_orders"]
    assert abs(b["return_rate"] - b["returned_orders"] / b["mature_orders"]) < 1e-3


@pytest.mark.parametrize("params", [{"start_date": "not-a-date"},
                                    {"product_id": "x'; DROP TABLE raw_orders;--"},
                                    {"start_date": "2024-06-01", "end_date": "2024-01-01"}])
def test_return_rate_invalid_inputs(params):
    assert client.get("/metrics/return-rate", params=params).status_code == 422


def test_return_rate_no_data_is_404_not_zero():
    r = client.get("/metrics/return-rate", params={"start_date": "2019-01-01", "end_date": "2019-12-31"})
    assert r.status_code == 404


def test_order_risk_found_and_missing():
    ok = client.get("/orders/ORD-000010/risk")
    assert ok.status_code == 200 and 0 <= ok.json()["risk_score"] <= 1
    assert client.get("/orders/NOPE/risk").status_code == 404


def test_copilot_validation():
    assert client.post("/copilot/ask", json={"question": ""}).status_code == 422
    assert client.post("/copilot/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/copilot/ask", json={}).status_code == 422


def test_copilot_grounded_and_cited():
    r = client.post("/copilot/ask", json={"question": "What is the return rate in 2024?"}).json()
    assert r["grounded"] and r["citations"] and r["citations"][0]["tool"] == "return_rate"


def test_copilot_refuses_out_of_scope():
    r = client.post("/copilot/ask", json={"question": "what is the weather"}).json()
    assert "cannot" in r["answer"].lower() or "can't" in r["answer"].lower()
    assert r["citations"] == []
