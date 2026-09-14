"""FastAPI P_seq scoring service."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from prod.scorer import (
    DEFAULT_CKPT,
    SchemaError,
    Scorer,
    load_scorer,
    score_dataframe,
)

_scorer: Scorer | None = None


def get_scorer() -> Scorer:
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scorer
    _scorer = load_scorer(DEFAULT_CKPT)
    yield
    _scorer = None


app = FastAPI(
    title="P_seq Realtime Scorer",
    description="LSTM–Transformer v5 privilege-escalation sequence scores",
    version="5.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    schema_version: str
    vocab_size: int
    device: str
    model_id: str
    thr_alert: float
    thr_triage: float


class ModelInfoResponse(BaseModel):
    model_id: str
    schema_version: str
    config: dict[str, Any]
    thr_alert: float
    thr_triage: float
    test_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    base_feature_cols: list[str]
    n_bag_models: int


class ScoreEvent(BaseModel):
    username: str
    timestamp: str
    event_name_idx: int | None = None
    event_name: str | None = None
    label: int | None = 0
    model_config = {"extra": "allow"}


class ScoreJsonRequest(BaseModel):
    events: list[dict[str, Any]] = Field(..., min_length=1)


class ScoreRow(BaseModel):
    username: str
    window_start: str
    window_end: str
    raw_len: int
    P_seq: float
    pred_triage: int
    pred_alert: int
    thr_triage: float
    thr_alert: float
    schema_version: str
    model_id: str
    window_label: int | None = None


class ScoreResponse(BaseModel):
    n_events: int
    n_windows: int
    windows: list[ScoreRow]


def _df_to_response(df_events: pd.DataFrame, scored: pd.DataFrame) -> ScoreResponse:
    rows = [ScoreRow(**r) for r in scored.to_dict(orient="records")]
    return ScoreResponse(
        n_events=int(len(df_events)),
        n_windows=int(len(scored)),
        windows=rows,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    s = get_scorer()
    return HealthResponse(
        status="ok",
        schema_version=s.schema_version,
        vocab_size=s.vocab_size,
        device=str(s.device),
        model_id=s.model_id,
        thr_alert=s.thr_alert,
        thr_triage=s.thr_triage,
    )


@app.get("/model", response_model=ModelInfoResponse)
def model_info():
    s = get_scorer()
    return ModelInfoResponse(
        model_id=s.model_id,
        schema_version=s.schema_version,
        config=s.config,
        thr_alert=s.thr_alert,
        thr_triage=s.thr_triage,
        test_metrics=s.test_metrics,
        val_metrics=s.val_metrics,
        base_feature_cols=s.base_feature_cols,
        n_bag_models=len(s.models),
    )


@app.post("/score/csv", response_model=ScoreResponse)
async def score_csv(file: UploadFile = File(...)):
    s = get_scorer()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}") from e
    try:
        scored = score_dataframe(df, s)
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _df_to_response(df, scored)


@app.post("/score/json", response_model=ScoreResponse)
def score_json(body: ScoreJsonRequest):
    s = get_scorer()
    df = pd.DataFrame(body.events)
    try:
        scored = score_dataframe(df, s)
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _df_to_response(df, scored)
