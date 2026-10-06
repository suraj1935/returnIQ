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
