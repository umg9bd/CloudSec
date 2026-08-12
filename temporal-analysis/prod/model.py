"""Production Temporal LSTM v2 model + windowing (no training deps)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

EMBED_DIM = 32
HIDDEN_DIM = 64
LSTM_LAYERS = 1
DROPOUT = 0.4
ATTN_DROPOUT = 0.1
SEQ_LEN = 32
WINDOW_MINUTES = 10
STRIDE_MINUTES = 2
BATCH_SIZE = 64
META_COLS = {"log_id", "username", "timestamp", "label", "event_name_idx"}


@dataclass
class Window:
    username: str
    start: pd.Timestamp
    end: pd.Timestamp
    event_idxs: np.ndarray
    feats: np.ndarray
    length: int
    label: int
    event_idx_list: list[int]


class WindowDataset(Dataset):
    def __init__(self, windows: list[Window]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        w = self.windows[i]
        return (
            torch.from_numpy(w.event_idxs),
            torch.from_numpy(w.feats),
            torch.tensor(w.length, dtype=torch.long),
            torch.tensor(w.label, dtype=torch.float32),
        )


class TemporalSeqModelV2(nn.Module):
    """Masked BiLSTM + attention pooling over valid timesteps."""

    def __init__(
        self,
        vocab_size: int,
        n_features: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = LSTM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim + n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        attn_in = hidden_dim * 2
        self.attn = nn.Linear(attn_in, 1)
        self.attn_drop = nn.Dropout(ATTN_DROPOUT)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(attn_in, 1)

    def forward(
        self,
        event_idx: torch.Tensor,
        feats: torch.Tensor,
        lengths: torch.Tensor,
        return_attn: bool = False,
    ):
        emb = self.embedding(event_idx)
        x = torch.cat([emb, feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=event_idx.size(1)
        )
        scores = self.attn(self.attn_drop(out)).squeeze(-1)
        T = event_idx.size(1)
        arange = torch.arange(T, device=event_idx.device).unsqueeze(0)
        mask = arange < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(out * weights.unsqueeze(-1), dim=1)
        logits = self.head(self.dropout(pooled)).squeeze(-1)
        if return_attn:
            return logits, weights
        return logits


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_minutes: int = WINDOW_MINUTES,
    stride_minutes: int = STRIDE_MINUTES,
    seq_len: int = SEQ_LEN,
) -> list[Window]:
    """Event-covering stride grid + right-aligned content with left PAD."""
    window_td = pd.Timedelta(minutes=window_minutes)
    stride_td = pd.Timedelta(minutes=stride_minutes)
    stride_ns = int(stride_td / pd.Timedelta(nanoseconds=1))
    window_ns = int(window_td / pd.Timedelta(nanoseconds=1))
    windows: list[Window] = []

    for username, g in df.groupby("username", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            continue
        ts = g["timestamp"]
        starts: set[pd.Timestamp] = set()
        for t in ts:
            aligned = (int(t.value) // stride_ns) * stride_ns
            k = 0
            while k * stride_ns < window_ns:
                starts.add(pd.Timestamp(aligned - k * stride_ns, tz="UTC"))
                k += 1

        for start in sorted(starts):
            end = start + window_td
            chunk = g.loc[(ts >= start) & (ts < end)]
            if chunk.empty:
                continue

            idxs = chunk["event_name_idx"].to_numpy(dtype=np.int64)
            feats = chunk[feature_cols].to_numpy(dtype=np.float32)
            times = chunk["timestamp"].astype("int64").to_numpy()
            deltas = np.diff(times, prepend=times[0]) / 1e9
            deltas = np.clip(deltas, 0, 3600)
            delta_feat = np.log1p(deltas).astype(np.float32).reshape(-1, 1)
            feats = np.concatenate([feats, delta_feat], axis=1)

            y = 1 if "label" in chunk.columns and int(chunk["label"].max()) == 1 else 0
            raw_len = len(idxs)
            if raw_len > seq_len:
                idxs = idxs[-seq_len:]
                feats = feats[-seq_len:]
                raw_len = seq_len
            pad = seq_len - raw_len
            if pad > 0:
                idxs = np.concatenate([np.zeros(pad, dtype=np.int64), idxs])
                feats = np.concatenate(
                    [np.zeros((pad, feats.shape[1]), dtype=np.float32), feats], axis=0
                )

            windows.append(
                Window(
                    username=str(username),
                    start=start,
                    end=end,
                    event_idxs=idxs,
                    feats=feats,
                    length=raw_len,
                    label=y,
                    event_idx_list=chunk["event_name_idx"].astype(int).tolist(),
                )
            )
    return windows


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all, y_all = [], []
    for event_idx, feats, lengths, y in loader:
        event_idx = event_idx.to(device)
        feats = feats.to(device)
        lengths = lengths.to(device)
        logits = model(event_idx, feats, lengths)
        logits_all.append(logits.cpu().numpy())
        y_all.append(y.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_all)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return y_true, probs


@torch.no_grad()
def predict_bag(models: list[nn.Module], loader: DataLoader, device: torch.device):
    all_probs = []
    y_true = None
    for model in models:
        y, p = predict(model, loader, device)
        all_probs.append(p)
        y_true = y
    return y_true, np.mean(np.stack(all_probs, axis=0), axis=0)
