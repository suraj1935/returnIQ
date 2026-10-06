"""Ingestion behaviour on an isolated SQLite DB."""
import importlib
from pathlib import Path

import pytest
from sqlalchemy import text

ORD_HDR = "order_id,customer_id,product_id,order_date,quantity,unit_price,total_amount,status\n"
RET_HDR = "return_id,order_id,customer_id,product_id,return_date,reason,refund_amount,status\n"


@pytest.fixture()
def ing(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.setenv("RETURNIQ_ENV", "dev")
    import pipelines.ingestion as m
    return importlib.reload(m)


def _w(p: Path, s: str) -> Path:
    p.write_text(s)
    return p


def test_idempotent_and_updates_change(ing, tmp_path):
    o = _w(tmp_path / "o.csv", ORD_HDR + "O1,C1,P1,2024-01-01,2,10,20,completed\n")
    r = _w(tmp_path / "r.csv", RET_HDR + "R1,O1,C1,P1,2024-01-05,defective,20,pending\n")
    assert ing.run(o, r) == {"orders": 1, "returns": 1}
    assert ing.run(o, r) == {"orders": 0, "returns": 0}          # idempotent, no spurious updates
    r = _w(tmp_path / "r.csv", RET_HDR + "R1,O1,C1,P1,2024-01-05,defective,20,approved\n")
    assert ing.run(o, r)["returns"] == 1                          # status change applied
    with ing.create_engine(ing.DATABASE_URL).connect() as c:
        assert c.execute(text("SELECT status FROM raw_returns")).scalar() == "approved"


def test_orphan_return_quarantined(ing, tmp_path):
    o = _w(tmp_path / "o.csv", ORD_HDR + "".join(f"O{i},C1,P1,2024-01-01,1,10,10,completed\n" for i in range(60)))
    r = _w(tmp_path / "r.csv", RET_HDR + "".join(
        f"R{i},O{i},C1,P1,2024-01-05,other,10,approved\n" for i in range(59)) + "RX,NOPE,C1,P1,2024-01-05,other,10,approved\n")
    ing.run(o, r)
    with ing.create_engine(ing.DATABASE_URL).connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM quarantine_ingestion WHERE error LIKE 'orphan%'")).scalar() == 1


def test_high_reject_rate_aborts(ing, tmp_path):
    o = _w(tmp_path / "o.csv", ORD_HDR + "O1,C1,P1,2024-01-01,1,10,10,completed\nO2,C1,P1,bad-date,1,10,10,completed\n")
    r = _w(tmp_path / "r.csv", RET_HDR)
    with pytest.raises(RuntimeError):
        ing.run(o, r)


def test_null_rows_count_toward_reject_rate(ing, tmp_path):
    o = _w(tmp_path / "o.csv", ORD_HDR + "O1,C1,P1,2024-01-01,1,10,10,completed\nO2,,P1,2024-01-01,1,10,10,completed\n")
    with pytest.raises(RuntimeError):
        ing.run(o, _w(tmp_path / "r.csv", RET_HDR))


def test_whitespace_ids_do_not_duplicate(ing, tmp_path):
    o = _w(tmp_path / "o.csv", ORD_HDR + "O1,C1,P1,2024-01-01,1,10,10,completed\n O1 ,C1,P1,2024-01-01,1,10,10,completed\n")
    ing.run(o, _w(tmp_path / "r.csv", RET_HDR))
    with ing.create_engine(ing.DATABASE_URL).connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM raw_orders")).scalar() == 1


def test_total_amount_mismatch_rejected(ing, tmp_path):
    rows = "".join(f"O{i},C1,P1,2024-01-01,1,10,10,completed\n" for i in range(60))
    o = _w(tmp_path / "o.csv", ORD_HDR + rows + "OBAD,C1,P1,2024-01-01,2,10,99,completed\n")
    ing.run(o, _w(tmp_path / "r.csv", RET_HDR))
    with ing.create_engine(ing.DATABASE_URL).connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM raw_orders WHERE order_id='OBAD'")).scalar() == 0
