"""
Head-only fine-tune of LSTM + Transformer v6.3 on real CloudTrail (real_dataset_dev.csv).

The embedding, BiLSTM and Transformer stay frozen at their v6.3 weights; only the three linear scoring
heads (seq_head, tab_head, secret_head) are refit on real dev event labels, with an L2 pull back to the
v6.3 head weights (L2-SP) so 71 attack sessions cannot drag them far. Because the frozen part never
changes, its per-event outputs are computed once and every fit is a full-batch logistic fit.

  - 5-fold cross-validation over dev SESSIONS (stratified by data_source x session label): every event
    is scored by heads that never saw its session -> out-of-fold (OOF) session metrics, same as v6.3's
    (session AUC-PR > session best F1 > event AUC-PR). The L2-SP strength is picked on those.
  - the final heads are fit on all of dev; their alert thresholds come from the OOF scores (an all-dev
    threshold would be tuned on the events the heads were fit on).
  - --test scores real_dataset_test.csv ONCE with the frozen thresholds: v6.3, v6.3-ft and the live
    clean LSTM. Nothing is chosen on test.

From the repo root:
  python temporal-analysis/finetune_lstm_v6_3.py           # CV + fit -> artifacts/lstm_transformer_v6_3_ft/
  python temporal-analysis/finetune_lstm_v6_3.py --test    # ... then the one real-test evaluation
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_lstm_transformer as v5  # noqa: E402
import train_lstm_transformer_v6_3 as t63  # noqa: E402

BASE_DIR = ROOT / "artifacts" / "lstm_transformer_v6_3"
BASE_CKPT = BASE_DIR / "temporal_lstm_transformer.pt"
OUT_DIR = ROOT / "artifacts" / "lstm_transformer_v6_3_ft"
TEST_FEATURES_DIR = ROOT / "artifacts" / "real_test_features"
REAL_TEST_CSV = t63.DATA / "real_dataset_test.csv"
SEED = 42
N_FOLDS = 5
L2SP_GRID = (1e-3, 1e-2, 1e-1, 1.0)
STEPS = 300
LR = 5e-3
HEAD_KEYS = ("seq_head.1", "tab_head.1", "secret_head.1")


def real_split(csv: Path, features_dir: Path, vocab, feature_cols) -> t63.RealDev:
    """Featurise a real split exactly as v6.3 featurises dev (t63 reads REAL_DEV_CSV / OUT_DIR at call time)."""
    saved = t63.REAL_DEV_CSV, t63.OUT_DIR
    t63.REAL_DEV_CSV, t63.OUT_DIR = csv, features_dir
    try:
        return t63.RealDev.build(t63.real_dev_features(), vocab, feature_cols)
    finally:
        t63.REAL_DEV_CSV, t63.OUT_DIR = saved


@torch.no_grad()
def frozen_parts(model: v5.LSTMTransformerModel, seqs, device) -> dict[str, torch.Tensor]:
    """LSTMTransformerModel.forward up to the heads (eval mode, no augmentation)."""
    model.eval()
    out = {k: [] for k in ("z", "f", "e", "m")}
    for event_idx, feats, lengths, _ in t63.make_loader(seqs):
        event_idx, feats, lengths = event_idx.to(device), feats.to(device), lengths.to(device)
        emb = model.embedding(event_idx)
        packed = nn.utils.rnn.pack_padded_sequence(torch.cat([emb, feats], -1), lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        h, _ = nn.utils.rnn.pad_packed_sequence(model.lstm(packed)[0], batch_first=True, total_length=event_idx.size(1))
        h = model.lstm_norm(h)
        tf = model.transformer(h, src_key_padding_mask=~v5.valid_mask(lengths, event_idx.size(1)))
        z = h + torch.sigmoid(model.len_gate(torch.log1p(lengths.float()).unsqueeze(-1))).unsqueeze(1) * tf
        last = (lengths - 1).clamp(min=0)
        b = torch.arange(z.size(0), device=device)
        out["z"].append(torch.cat([z[b, last], emb[b, last]], -1).cpu())
        out["f"].append(feats[b, last].cpu())
        out["e"].append(emb[b, last].cpu())
        out["m"].append(model.secret_id_mask[event_idx[b, last].long()].float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


class Heads(nn.Module):
    def __init__(self, model: v5.LSTMTransformerModel):
        super().__init__()
        self.seq = copy.deepcopy(model.seq_head[-1])
        self.tab = copy.deepcopy(model.tab_head[-1])
        self.sec = copy.deepcopy(model.secret_head[-1])

    def forward(self, p: dict[str, torch.Tensor], idx=slice(None)) -> torch.Tensor:
        f = p["f"][idx]
        return (self.tab(f) + self.seq(p["z"][idx])).squeeze(-1) + p["m"][idx] * self.sec(
            torch.cat([f, p["e"][idx]], -1)).squeeze(-1)

    def state(self) -> dict[str, torch.Tensor]:
        return {f"{k}.{n}": v.detach().clone() for k, mod in zip(HEAD_KEYS, (self.seq, self.tab, self.sec))
                for n, v in mod.state_dict().items()}


def fit_heads(model, parts, y: np.ndarray, idx: np.ndarray, lam: float) -> Heads:
    torch.manual_seed(SEED)
    heads = Heads(model)
    p0 = [q.detach().clone() for q in heads.parameters()]
    yt = torch.as_tensor(y[idx], dtype=torch.float32)
    n_pos = float(yt.sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([np.sqrt((len(yt) - n_pos) / max(n_pos, 1.0))]))
    opt = torch.optim.Adam(heads.parameters(), lr=LR)
    it = torch.as_tensor(idx)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = crit(heads(parts, it), yt) + lam * sum(((q - q0) ** 2).sum() for q, q0 in zip(heads.parameters(), p0))
        loss.backward()
        opt.step()
    return heads


@torch.no_grad()
def probs(heads: Heads, parts, idx=slice(None)) -> np.ndarray:
    return torch.sigmoid(heads(parts, idx)).numpy()


def session_folds(dev: t63.RealDev, log_ids: list[str], by: str = "session") -> np.ndarray:
    """Fold id per event; folds are whole sessions.
    by="session": stratified by data_source x session label.
    by="technique": leave one attack tactic out -- each fold holds ALL attack sessions of one
    attack_technique (plus a random 1/k of benign sessions), so the heads never saw that tactic."""
    raw = pd.read_csv(t63.REAL_DEV_CSV, usecols=["session_id", "data_source", "attack_technique", "session_label"])
    fold_of = {}
    if by == "technique":
        tech = (raw[raw["session_label"] == 1].dropna(subset=["attack_technique"])
                .groupby("session_id")["attack_technique"].agg(lambda s: s.mode().iloc[0]))
        names = sorted(tech.unique())
        rng = np.random.default_rng(SEED)
        for sid, lab in dev.sessions.items():
            fold_of[sid] = names.index(tech[sid]) if lab == 1 and sid in tech.index else int(rng.integers(len(names)))
        return np.array([fold_of[dev.event_sid[i]] for i in log_ids]), names
    raw = raw.drop_duplicates("session_id")
    src = raw.set_index("session_id")["data_source"].reindex(dev.sessions.index).fillna("unknown")
    strata = src.astype(str) + "_" + dev.sessions.astype(str)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(strata)), strata.to_numpy())):
        fold_of.update({sid: k for sid in dev.sessions.index[te]})
    return np.array([fold_of[dev.event_sid[i]] for i in log_ids]), [f"fold{k}" for k in range(N_FOLDS)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="after CV + fit, score real_dataset_test.csv once")
    ap.add_argument("--technique-cv", action="store_true",
                    help="diagnostic only: leave-one-attack-tactic-out CV (does the fine-tune generalise to an unseen tactic?)")
    ap.add_argument("--l2sp", type=float, default=None,
                    help="fix the L2-SP strength instead of picking it on session-CV (e.g. from --technique-cv)")
    args = ap.parse_args()
    os.chdir(t63.REPO)  # feature_engine9's paths are repo-relative
    device = torch.device("cpu")
    model, ckpt, _ = v5.load_checkpoint(BASE_CKPT, device=device)
    vocab, fc = dict(ckpt["event_name_vocab"]), list(ckpt["feature_cols"])

    dev = real_split(t63.REAL_DEV_CSV, BASE_DIR, vocab, fc)
    log_ids = [s.log_id for s in dev.seqs]
    y = np.array([s.label for s in dev.seqs], dtype=int)
    parts = frozen_parts(model, dev.seqs, device)
    base_p = probs(Heads(model), parts)
    ref = t63.score_seqs(model, dev.seqs, device)["P_event"].to_numpy()
    assert np.abs(base_p - ref).max() < 1e-4, f"frozen replay != model forward ({np.abs(base_p - ref).max():.2e})"
    folds, _ = session_folds(dev, log_ids)
    print(f"real dev: {len(y)} events ({y.sum()} attack), {len(dev.sessions)} sessions "
          f"({int(dev.sessions.sum())} attack); {N_FOLDS} session folds; frozen replay matches v6.3 forward", flush=True)
    if args.technique_cv:
        tfolds, names = session_folds(dev, log_ids, by="technique")
        sid = np.array([dev.event_sid[i] for i in log_ids])
        sess_y = dev.sessions
        print("leave-one-tactic-out: session AUC-PR on the held-out tactic's attack sessions + that fold's benign")
        for lam in L2SP_GRID:
            cells = []
            for k, name in enumerate(names):
                te = np.flatnonzero(tfolds == k)
                p_ft = probs(fit_heads(model, parts, y, np.flatnonzero(tfolds != k), lam), parts, torch.as_tensor(te))
                s_ids = pd.unique(sid[te])
                ys = sess_y.reindex(s_ids).to_numpy()
                agg = lambda v: pd.Series(v, index=sid[te]).groupby(level=0).max().reindex(s_ids).to_numpy()
                cells.append(f"{name}: {average_precision_score(ys, agg(base_p[te])):.3f}->"
                             f"{average_precision_score(ys, agg(p_ft)):.3f} (n_atk={int(ys.sum())})")
            print(f"  L2-SP {lam:g}: " + " | ".join(cells), flush=True)
        return

    base_m = t63.score_dev(dev, log_ids, base_p, y)
    rows = {"v6.3 (no fine-tune)": base_m}
    oof = {}
    for lam in L2SP_GRID:
        p = np.zeros(len(y))
        for k in range(N_FOLDS):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            p[te] = probs(fit_heads(model, parts, y, tr, lam), parts, torch.as_tensor(te))
        oof[lam] = p
        rows[f"head fine-tune, L2-SP {lam:g} (OOF)"] = t63.score_dev(dev, log_ids, p, y)
    for name, m in rows.items():
        print(f"  {name:36s} session AUC-PR={m['session']['auc_pr']:.4f} AUC={m['session']['auc_roc']:.4f} "
              f"best F1={m['session']['best_f1']:.4f} | event AUC-PR={m['event']['auc_pr']:.4f}", flush=True)

    best_lam = max(L2SP_GRID, key=lambda lam: t63.selection_key(rows[f"head fine-tune, L2-SP {lam:g} (OOF)"]))
    if args.l2sp is not None:
        assert args.l2sp in L2SP_GRID, f"--l2sp must be one of {L2SP_GRID}"
        best_lam = args.l2sp
    best_m = rows[f"head fine-tune, L2-SP {best_lam:g} (OOF)"]
    wins = t63.selection_key(best_m) > t63.selection_key(base_m)
    print(f"best L2-SP={best_lam:g}: OOF key {t63.selection_key(best_m)} vs v6.3 {t63.selection_key(base_m)} -> "
          f"{'fine-tune wins' if wins else 'v6.3 stays'}", flush=True)

    heads = fit_heads(model, parts, y, np.arange(len(y)), best_lam)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state.update(heads.state())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ft_ckpt = {**ckpt, "state_dict": state, "schema_version": "lstm_transformer_v6.3_ft",
               "threshold": best_m["session"]["threshold"], "event_threshold": best_m["event"]["threshold"],
               "dev_metrics": best_m, "test_metrics": {}, "test_event_metrics": {},
               "config": {**ckpt["config"], "model": "LSTMTransformerV6_3_FT", "fine_tune": {
                   "base": str(BASE_CKPT.relative_to(t63.REPO)).replace("\\", "/"), "trainable": list(HEAD_KEYS),
                   "frozen": "embedding, BiLSTM, LayerNorm, Transformer, len_gate", "data": "real_dataset_dev.csv",
                   "l2sp": best_lam, "l2sp_source": "--l2sp (leave-one-tactic-out CV)" if args.l2sp is not None
                   else "session-CV OOF key", "steps": STEPS, "lr": LR, "folds": N_FOLDS,
                   "thresholds_from": "out-of-fold session / event scores", "real_test_scored": False}}}
    torch.save(ft_ckpt, OUT_DIR / "temporal_lstm_transformer.pt")
    ft_model, _, _ = v5.load_checkpoint(OUT_DIR / "temporal_lstm_transformer.pt", device=device)
    reload_p = t63.score_seqs(ft_model, dev.seqs, device)["P_event"].to_numpy()
    assert np.abs(reload_p - probs(heads, parts)).max() < 1e-4, "saved checkpoint does not reproduce the fitted heads"
    report = {"selection": "OOF session AUC-PR > session best F1 > event AUC-PR", "best_l2sp": best_lam,
              "fine_tune_wins_cv": bool(wins), "dev_cv": rows}
    print(f"Wrote {OUT_DIR / 'temporal_lstm_transformer.pt'} (reload check ok)", flush=True)

    if args.test:
        # thresholds: v6.3-ft from its OOF dev scores; v6.3 and the live LSTM tuned on dev now (same rule)
        test_rows = {}
        for name, path in (("v6.3", BASE_CKPT), ("v6.3-ft", OUT_DIR / "temporal_lstm_transformer.pt"),
                           ("live clean LSTM", t63.CLEAN_CKPT)):
            m_, ck, _ = v5.load_checkpoint(path, device=device)
            voc, cols = dict(ck["event_name_vocab"]), list(ck["feature_cols"])
            if name == "v6.3-ft":
                evt_thr, ses_thr = best_m["event"]["threshold"], best_m["session"]["threshold"]
            else:
                dd = real_split(t63.REAL_DEV_CSV, BASE_DIR, voc, cols)
                ev = t63.score_seqs(m_, dd.seqs, device)
                dm = t63.score_dev(dd, ev["log_id"], ev["P_event"], ev["label"])
                evt_thr, ses_thr = dm["event"]["threshold"], dm["session"]["threshold"]
            d = real_split(REAL_TEST_CSV, TEST_FEATURES_DIR, voc, cols)
            ev = t63.score_seqs(m_, d.seqs, device)
            test_rows[name] = t63.score_dev(d, ev["log_id"], ev["P_event"], ev["label"], evt_thr=evt_thr, ses_thr=ses_thr)
        print("=== REAL TEST (scored once; thresholds frozen from dev) ===", flush=True)
        for name, m in test_rows.items():
            s = m["session"]
            print(f"  {name:16s} session P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
                  f"AUC-PR={s['auc_pr']:.4f} AUC={s['auc_roc']:.4f} (thr {s['threshold']:.3f}) | "
                  f"event AUC-PR={m['event']['auc_pr']:.4f}", flush=True)
        report["real_test"] = test_rows
    (OUT_DIR / "finetune_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
