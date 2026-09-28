"""Production P_seq scorer — LSTM–Transformer v5 / v6 (max-pool events → windows)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import train_lstm_transformer as tlt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "artifacts" / "lstm_transformer" / "temporal_lstm_transformer.pt"
V6_CKPT = ROOT / "artifacts" / "lstm_transformer_v6" / "temporal_lstm_transformer_v6.pt"
V6_VOCAB_PATH = ROOT / "data" / "lstm" / "event_name_vocab_v6.json"
MODEL_ID = "LSTMTransformerV5"
SCHEMA_VERSION = "lstm_transformer_v5.0"

REQUIRED_COLS = {"username", "timestamp"}


class SchemaError(ValueError):
    """Invalid input schema for scoring."""


@dataclass
class Scorer:
    model: Any
    models: list
    ckpt: dict[str, Any]
    device: torch.device
    base_feature_cols: list[str]
    vocab: dict[str, int]
    thr_alert: float
    thr_triage: float
    schema_version: str
    config: dict[str, Any]
    test_metrics: dict[str, Any]
    val_metrics: dict[str, Any]

    @property
    def model_id(self) -> str:
        return str(self.config.get("model", MODEL_ID))

    @property
    def vocab_size(self) -> int:
        return int(self.config.get("vocab_size", len(self.vocab)))


def _vocab_fallback(ckpt_path: Path) -> dict[str, int]:
    import json

    key = ckpt_path.as_posix().replace("\\", "/")
    if "lstm_transformer_v6" in key or ckpt_path.name.endswith("_v6.pt"):
        if V6_VOCAB_PATH.exists():
            return json.loads(V6_VOCAB_PATH.read_text(encoding="utf-8"))
    if tlt.VOCAB_PATH.exists():
        return json.loads(tlt.VOCAB_PATH.read_text(encoding="utf-8"))
    return {}


def load_scorer(
    ckpt_path: Path | str = DEFAULT_CKPT,
    device: torch.device | None = None,
    thr_triage: float | None = None,
) -> Scorer:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {path}. "
            "Train with train_lstm_transformer.py (v5) or train_lstm_transformer_v6.py (v6)."
        )
    model, ckpt, device = tlt.load_checkpoint(path, device=device)
    cfg = dict(ckpt.get("config") or {})
    feature_cols = list(ckpt.get("feature_cols") or [])
    base_feats = [c for c in feature_cols if c not in tlt.PE_CONTEXT_COLS]
    vocab = dict(ckpt.get("event_name_vocab") or {})
    if not vocab:
        vocab = _vocab_fallback(path)

    evt_thr = float(ckpt.get("event_threshold", 0.625))
    win_thr = float(ckpt.get("threshold", 0.85))
    schema = str(ckpt.get("schema_version") or cfg.get("schema_version") or SCHEMA_VERSION)
    return Scorer(
        model=model,
        models=[model],
        ckpt=ckpt,
        device=device,
        base_feature_cols=base_feats,
        vocab=vocab,
        thr_alert=win_thr,
        thr_triage=float(thr_triage) if thr_triage is not None else evt_thr,
        schema_version=schema,
        config=cfg,
        test_metrics=dict(ckpt.get("test_event_metrics") or ckpt.get("test_metrics") or {}),
        val_metrics={},
    )


def prepare_dataframe(df: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    try:
        return tlt.prepare_score_frame(df, scorer.vocab, list(scorer.ckpt["feature_cols"]))
    except ValueError as e:
        raise SchemaError(str(e)) from e


def score_dataframe(df: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    """Validate → per-event score → P_seq = max(P_event) on 10-min / stride-2 windows."""
    try:
        scored = tlt.score_events_to_windows(df, scorer.model, scorer.ckpt, scorer.device)
    except ValueError as e:
        raise SchemaError(str(e)) from e
    out = scored.copy()
    out["pred_triage"] = (out["P_seq"] >= scorer.thr_triage).astype(int)
    out["pred_alert"] = (out["P_seq"] >= scorer.thr_alert).astype(int)
    out["thr_triage"] = scorer.thr_triage
    out["thr_alert"] = scorer.thr_alert
    out["schema_version"] = scorer.schema_version
    out["model_id"] = scorer.model_id
    keep = [
        "username",
        "window_start",
        "window_end",
        "raw_len",
        "P_seq",
        "pred_triage",
        "pred_alert",
        "thr_triage",
        "thr_alert",
        "schema_version",
        "model_id",
        "window_label",
    ]
    return out[[c for c in keep if c in out.columns]]
