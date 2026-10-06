from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class Citation(BaseModel):
    tool: str
    query_id: str
    params: dict
    rows: int
    data_as_of: str | None = None


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    engine: str
    citations: list[Citation]


class ReturnRateResponse(BaseModel):
    query_id: str
    start_date: date | None = None
    end_date: date | None = None
    product_id: str | None = None
    mature_orders: int
    returned_orders: int
    return_rate: float
    data_as_of: str | None = None


class RiskResponse(BaseModel):
    order_id: str
    product_id: str
    order_date: str
    risk_score: float
    high_risk: bool
    label_mature: bool
    model_version: str
