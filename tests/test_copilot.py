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
