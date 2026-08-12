"""Production P_seq scorer — load v2 ckpt, validate schema, score windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from prod.model import (
    BATCH_SIZE,
    TemporalSeqModelV2,
    WindowDataset,
    build_windows,
    predict_bag,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "artifacts" / "temporal_lstm_v2.pt"
THR_TRIAGE_DEFAULT = 0.55
MODEL_ID = "temporal_lstm_v2"

REQUIRED_COLS = {"username", "timestamp"}


class SchemaError(ValueError):
    """Invalid input schema for scoring."""


@dataclass
class Scorer:
    models: list[TemporalSeqModelV2]
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
        return MODEL_ID

    @property
    def vocab_size(self) -> int:
        return int(self.config.get("vocab_size", 0))


def load_scorer(
    ckpt_path: Path | str = DEFAULT_CKPT,
    device: torch.device | None = None,
    thr_triage: float = THR_TRIAGE_DEFAULT,
) -> Scorer:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {path}. Train with train_temporal_lstm_v2.py"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    models: list[TemporalSeqModelV2] = []
    for state in ckpt["state_dicts"]:
        m = TemporalSeqModelV2(
            vocab_size=int(cfg["vocab_size"]),
            n_features=int(cfg["n_features"]),
            embed_dim=int(cfg["embed_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg.get("lstm_layers", 1)),
            dropout=float(cfg["dropout"]),
        ).to(device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    base_feats = ckpt.get("base_feature_cols")
    if not base_feats:
        base_feats = [c for c in ckpt.get("feature_cols", []) if c != "delta_t_log1p"]

    thr_alert = float(ckpt.get("threshold", 0.7))
    return Scorer(
        models=models,
        ckpt=ckpt,
        device=device,
        base_feature_cols=list(base_feats),
        vocab=dict(ckpt.get("event_name_vocab") or {}),
        thr_alert=thr_alert,
        thr_triage=float(thr_triage),
        schema_version=str(ckpt.get("schema_version", "unknown")),
        config=cfg,
        test_metrics=dict(ckpt.get("test_metrics") or {}),
        val_metrics=dict(ckpt.get("val_metrics") or {}),
    )


def map_event_names(df: pd.DataFrame, vocab: dict[str, int]) -> pd.DataFrame:
    """Map event_name → idx with OOV→0; else clamp event_name_idx."""
    out = df.copy()
    if "event_name" in out.columns and vocab:
        out["event_name_idx"] = out["event_name"].map(
            lambda x: int(vocab.get(str(x), 0))
        )
    elif "event_name_idx" in out.columns:
        vmax = max(vocab.values()) if vocab else int(out["event_name_idx"].max())
        out["event_name_idx"] = (
            pd.to_numeric(out["event_name_idx"], errors="coerce").fillna(0).astype(int)
        )
        out.loc[out["event_name_idx"] > vmax, "event_name_idx"] = 0
        out.loc[out["event_name_idx"] < 0, "event_name_idx"] = 0
    else:
        raise SchemaError(
            "Input must include event_name_idx or event_name for API token mapping"
        )
    return out


def prepare_dataframe(df: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise SchemaError("Empty input: no events to score")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SchemaError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if out["timestamp"].isna().any():
        raise SchemaError("timestamp contains unparseable values")
    if out["username"].isna().any():
        raise SchemaError("username contains nulls")

    out = map_event_names(out, scorer.vocab)
    for c in scorer.base_feature_cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    if "label" not in out.columns:
        out["label"] = 0
    return out


def score_dataframe(df: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    """Validate → window → bag inference → dual-threshold tags."""
    prepared = prepare_dataframe(df, scorer)
    windows = build_windows(prepared, scorer.base_feature_cols)
    if not windows:
        raise SchemaError("No windows built from input (need ≥1 event per user)")

    loader = DataLoader(
        WindowDataset(windows), batch_size=BATCH_SIZE, shuffle=False
    )
    _, probs = predict_bag(scorer.models, loader, scorer.device)

    rows = []
    for w, p in zip(windows, probs):
        p = float(p)
        rows.append(
            {
                "username": w.username,
                "window_start": w.start.isoformat(),
                "window_end": w.end.isoformat(),
                "raw_len": w.length,
                "P_seq": p,
                "pred_triage": int(p >= scorer.thr_triage),
                "pred_alert": int(p >= scorer.thr_alert),
                "thr_triage": scorer.thr_triage,
                "thr_alert": scorer.thr_alert,
                "schema_version": scorer.schema_version,
                "model_id": scorer.model_id,
                "window_label": w.label,
            }
        )
    return pd.DataFrame(rows)
