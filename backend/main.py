"""ReturnIQ API.  Run: uvicorn backend.main:app --reload"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

from ai.copilot import ask
from ai.tools import ToolError, Toolbox
from backend.db import get_readonly_engine
from backend.schemas import AskRequest, AskResponse, ReturnRateResponse, RiskResponse

app = FastAPI(title="ReturnIQ", version="0.1.0")


def get_box() -> Toolbox:
    return Toolbox(get_readonly_engine())


@app.exception_handler(OperationalError)
@app.exception_handler(ProgrammingError)
async def _db_unavailable(_, exc):
    return JSONResponse(status_code=503,
                        content={"detail": "Analytics tables unavailable. Run the pipelines first."})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics/return-rate", response_model=ReturnRateResponse)
def return_rate(start_date: date | None = None, end_date: date | None = None,
                product_id: str | None = Query(None, pattern=r"^PROD-\d{1,6}$"),
                box: Toolbox = Depends(get_box)):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(422, "start_date must be <= end_date")
    try:
        r = box.return_rate(start_date.isoformat() if start_date else None,
                            end_date.isoformat() if end_date else None, product_id)
    except ToolError as e:
        raise HTTPException(422, str(e))
    if r.empty:
        raise HTTPException(404, "No mature orders for this filter; return rate cannot be computed.")
    return ReturnRateResponse(query_id=r.query_id, start_date=start_date, end_date=end_date,
                              product_id=product_id, data_as_of=r.data_as_of, **r.rows[0])


@app.get("/orders/{order_id}/risk", response_model=RiskResponse)
def order_risk(order_id: str, box: Toolbox = Depends(get_box)):
    with box.engine.connect() as c:
        row = c.execute(text("""SELECT s.order_id, f.product_id, f.order_date, s.risk_score, s.high_risk,
                                f.label_mature, s.model_version
                                FROM analytics_risk_scores s JOIN analytics_order_features f
                                  ON f.order_id = s.order_id WHERE s.order_id = :o"""), {"o": order_id}).fetchone()
    if row is None:
        raise HTTPException(404, f"No risk score for order {order_id!r} (unknown or cancelled).")
    d = dict(row._mapping)
    return RiskResponse(**{**d, "high_risk": bool(d["high_risk"]), "label_mature": bool(d["label_mature"])})


@app.get("/model/metrics")
def model_metrics(box: Toolbox = Depends(get_box)):
    r = box.model_metrics()
    if r.empty:
        raise HTTPException(404, "Model not trained yet.")
    return {"query_id": r.query_id, "note": r.note, "metrics": r.rows[0]}


@app.post("/copilot/ask", response_model=AskResponse)
def copilot_ask(req: AskRequest, box: Toolbox = Depends(get_box)):
    a = ask(req.question, box)
    return AskResponse(answer=a.answer, grounded=a.grounded, engine=a.engine, citations=a.citations)


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")
