"""
LSTM + Transformer v6.3 -- v6.2 retrained on the team's feature_engine9 data, selected on real dev sessions.
LSTM + Transformer only: no graph model is used for training or selection.

Loads in pipeline.py / prod.scorer unchanged (same checkpoint keys as lstm_transformer_clean):
  - train: datasets/privilege-escalation/cloudtrail_temporal.csv, the feature_engine9 default run
    of the current generator -- the same features, vocab and risk priors the live pipeline computes
  - synthetic split: whole campaign families (splits/campaign_family_seed42.csv, campaign_split.py)
  - real dev: real_dataset_dev.csv featurised the way Pipeline.featurize does it (frozen vocab and
    priors, fresh per-principal state), cached in OUT_DIR; sessions from its session_id.
    real_dataset_test.csv is never read here.
  - carried over from v6.2: label-leaking features dropped (see --audit), <UNK> trained on the
    current event, epoch picked on real dev (session AUC-PR, then session best F1, then event
    AUC-PR; epochs >= MIN_EPOCHS), exact best-F1 thresholds with a recall-side margin
  - new in v6.3 (ablated on real dev; v6.2 as-is scored session AUC-PR 0.534 on this data):
      campaign relabel (v5): secrets / AssumeRole / CreateSecret within 10 min after a PE write -> 1,
        synthetic training labels only (real dev labels are never relabelled)
      secret positives weighted x4 in the sampler (v6.2: x2)
      read-name <UNK>: read-only, non-secret, non-PE names hidden at any position, whatever the label --
        ~30% of benign real-dev events are reads the synthetic vocab never saw, and v6.2 scored
        "unseen read in a busy session" as an attack

From the repo root:
  python campaign_split.py --out splits/campaign_family_seed42.csv
  python temporal-analysis/train_lstm_transformer_v6_3.py --audit     # feature AUC: synthetic vs real dev
  python temporal-analysis/train_lstm_transformer_v6_3.py             # train -> artifacts/lstm_transformer_v6_3/
  python temporal-analysis/train_lstm_transformer_v6_3.py --compare   # v6.3 vs clean LSTM vs RF on real dev
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = REPO / "datasets" / "privilege-escalation"
for _p in (REPO, DATA, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_lstm_transformer as v5  # noqa: E402
from train_lstm_transformer import (  # noqa: E402
    BATCH_SIZE,
    PE_CONTEXT_COLS,
    EventDataset,
    LSTMTransformerModel,
    attach_pe_context,
    build_event_sequences,
    metrics_dict,
    predict,
    prepare_score_frame,
    score_seqs,
)

TRAIN_CSV = DATA / "cloudtrail_temporal.csv"
REAL_DEV_CSV = DATA / "real_dataset_dev.csv"
SPLIT_FILE = REPO / "splits" / "campaign_family_seed42.csv"
CLEAN_CKPT = ROOT / "artifacts" / "lstm_transformer_clean" / "temporal_lstm_transformer.pt"
OUT_DIR = ROOT / "artifacts" / "lstm_transformer_v6_3"
CKPT_PATH = OUT_DIR / "temporal_lstm_transformer.pt"

SEED = 42
SCHEMA_VERSION = "lstm_transformer_v6.3"
META_COLS = {"log_id", "username", "timestamp", "label", "event_name_idx", "event_name", "split"}
# From --audit on the current generator: separate the classes on synthetic data (AUC 0.87-0.94) but
# not on real dev (0.42-0.54); params_length_normalized even flips direction.
DROP_FEATURES: tuple[str, ...] = ("params_length_normalized", "no_mfa", "mfa_absent")
SECRET_NAMES = v5.SECRET_NAMES | {"GetParameters", "GetParameter", "BatchGetSecretValue"}
SELECTION = "epochs >= MIN_EPOCHS: real-dev session AUC-PR > session best F1 > event AUC-PR"
UNK_LAST_P = 0.15
PRIOR_WEIGHT = 15  # feature_engine9.FeatureEngineer's AdaptiveRiskPrior prior_weight
UNK_READ_P = 0.0  # --unk-read: helped HGT+LSTM, not LSTM alone (dev: 0 -> 0.852, 0.6 -> 0.848, 1.0 -> 0.822)
# recipe -> (last-event <UNK> rate, secret-positive weight, campaign relabel, read-name <UNK> rate)
RECIPES = {"v6.3": (UNK_LAST_P, 4.0, True, UNK_READ_P), "v6.2": (UNK_LAST_P, 2.0, False, 0.0),
           "v5": (0.0, 4.0, True, 0.0)}
THRESHOLD_MARGIN = 0.1
MAX_EPOCHS = 20
PATIENCE = 6
MIN_EPOCHS = 4
LR = 8e-4
WEIGHT_DECAY = 5e-3


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fe9_vocab() -> dict[str, int]:
    import feature_engine9 as fe9

    vocab = {str(k): int(v) for k, v in json.loads(Path(fe9.EVENT_NAME_VOCAB_FILE).read_text(encoding="utf-8")).items()}
    assert vocab.get("<UNK>") == 0, "feature_engine9 vocab must keep <UNK> at 0 (the LSTM's padding id)"
    return vocab


def serve_priors(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the CSV's action_risk_prior / principal_type_prior_risk with the values the live pipeline
    serves: feature_engine9's frozen prior files (the same files Pipeline.featurize and real dev use).

    The training CSV holds STREAMED priors: each row got the shrinkage estimate from the rows before it,
    so attack names start near the base rate and climb. Serving uses the final counts. On this data
    attack rows average 0.395 streamed vs 0.500 served, so live attack names looked riskier than
    anything seen in training. With the served values train == serve exactly.

    Not target leakage: a frozen prior is one constant per event name / principal type, so it tells
    the model nothing the name itself does not, and real dev / test never enter the prior files.
    (Out-of-fold priors were tried and are wrong here: leave-one-user-out sends benign AssumeRole --
    1,206 rows from 6 users -- from 0.11 to 0.65, because removing a heavy user leaves mostly attacks.)"""
    import feature_engine9 as fe9

    def frozen(path):
        return fe9.AdaptiveRiskPrior(priors={}, default=fe9.AdaptiveRiskPrior.BASE_RATE,
                                     prior_weight=PRIOR_WEIGHT, path=path, frozen=True)

    act, prin = frozen(fe9.ACTION_PRIOR_FILE), frozen(fe9.PRINCIPAL_PRIOR_FILE)
    ptypes = list(fe9.FIXED_PRINCIPAL_TYPES)
    out = df.copy()
    names = df["event_name"].astype(str)
    out["action_risk_prior"] = names.map({n: act.score(n) for n in names.unique()}).to_numpy()
    ptype = df["principal_type_idx"].astype(int).map(lambda i: ptypes[i] if 0 < i < len(ptypes) else "unknown")
    out["principal_type_prior_risk"] = ptype.map({p: prin.score(p) for p in ptype.unique()}).to_numpy()
    return out


def load_train(drop: tuple[str, ...], priors: str = "serve") -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Synthetic feature_engine9 features + campaign-family split. Vocab = <UNK> + names seen here.
    priors='serve' swaps the streamed label priors for the frozen values the pipeline serves (serve_priors)."""
    import campaign_split

    df = pd.read_csv(TRAIN_CSV, dtype={"log_id": str})
    assert df["log_id"].is_unique, "duplicate log_ids in the training features"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed").astype("datetime64[ns, UTC]")
    df["username"] = df["username"].astype(str)
    inv = {v: k for k, v in fe9_vocab().items()}
    df["event_name"] = df["event_name_idx"].astype(int).map(inv)
    assert df["event_name"].notna().all() and int(df["event_name_idx"].min()) >= 1
    split = campaign_split.read_split_file(str(SPLIT_FILE))
    df["split"] = df["log_id"].map(split)
    assert df["split"].notna().all(), f"{SPLIT_FILE} does not cover {TRAIN_CSV.name}; rebuild it with campaign_split.py"
    users = df.groupby("username")["split"].nunique()
    assert (users == 1).all(), "users straddle splits; rebuild the split file with campaign_split.py"
    if priors == "serve":
        streamed = df["action_risk_prior"].to_numpy()
        df = serve_priors(df)
        atk = df["label"].to_numpy() == 1
        print(f"priors: served (frozen feature_engine9 files). action_risk_prior on attack rows: streamed "
              f"{streamed[atk].mean():.3f} -> served {df['action_risk_prior'].to_numpy()[atk].mean():.3f}", flush=True)
    pairs = df[["event_name", "event_name_idx"]].drop_duplicates()
    vocab = {"<UNK>": 0, **{str(n): int(i) for n, i in zip(pairs["event_name"], pairs["event_name_idx"])}}
    feats = [c for c in df.columns if c not in META_COLS and c not in drop]
    print(
        f"train csv={TRAIN_CSV.name} rows={len(df)} attacks={int(df['label'].sum())} users={df['username'].nunique()} "
        f"features={len(feats)} vocab={len(vocab)} split={df['split'].value_counts().to_dict()}",
        flush=True,
    )
    return df, feats, vocab


def real_dev_features() -> pd.DataFrame:
    """real_dataset_dev.csv through feature_engine9 as pipeline.Pipeline.featurize does: frozen training
    vocab and priors, fresh per-principal state. Cached; rebuilt whenever an input file changes."""
    import feature_engine9 as fe9

    inputs = [REAL_DEV_CSV, Path(fe9.EVENT_NAME_VOCAB_FILE), Path(fe9.ACTION_PRIOR_FILE), Path(fe9.PRINCIPAL_PRIOR_FILE)]
    key = "|".join(fe9.file_fingerprint(str(p)) for p in inputs)
    cache, key_file = OUT_DIR / "real_dev_features.csv", OUT_DIR / "real_dev_features.key"
    if cache.exists() and key_file.exists() and key_file.read_text(encoding="utf-8") == key:
        df = pd.read_csv(cache, dtype={"log_id": str})
    else:
        records = []
        with tempfile.TemporaryDirectory() as tmp:
            engine = fe9.FeatureEngineer(
                event_name_vocab_path=fe9.EVENT_NAME_VOCAB_FILE,
                state_tracker_path=os.path.join(tmp, "state_tracker.json"),
                graph_state_path=os.path.join(tmp, "graph_node_state.json"),
                identity_state_path=os.path.join(tmp, "identity_state.json"),
                action_prior_path=fe9.ACTION_PRIOR_FILE,
                principal_prior_path=fe9.PRINCIPAL_PRIOR_FILE,
                freeze_vocab=True,
                freeze_priors=True,
            )
            for idx, row in enumerate(fe9.iter_input_rows(str(REAL_DEV_CSV))):
                try:
                    engine.get_structural_data(row)  # same call order as the pipeline (shared state)
                    temporal = engine.get_temporal_features(row)
                except ValueError:
                    continue
                rec = {
                    "log_id": f"{REAL_DEV_CSV.name}:{idx}",
                    "timestamp": row.get("timestamp"),
                    "username": row.get("username") or "unknown_user",
                    "event_name": row.get("event_name"),
                    "label": row.get("label", 0),
                }
                rec.update(dict(zip(fe9.TEMPORAL_COLS, temporal)))
                records.append(rec)
        df = pd.DataFrame(records)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        key_file.write_text(key, encoding="utf-8")
        print(f"built {cache.name}: {len(df)} events (feature_engine9, frozen vocab + priors)", flush=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed").astype("datetime64[ns, UTC]")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["username"] = df["username"].astype(str)
    return df


@dataclass
class RealDev:
    frame: pd.DataFrame
    seqs: list
    sessions: pd.Series  # session_id -> session_label
    event_sid: dict[str, object]  # log_id -> session_id

    @classmethod
    def build(cls, dev_df: pd.DataFrame, vocab: dict[str, int], feature_cols: list[str]) -> "RealDev":
        import feature_engine9 as fe9

        cols = ["log_id", "username", "timestamp", "event_name", "label"] + [c for c in fe9.TEMPORAL_COLS if c in dev_df]
        frame = prepare_score_frame(dev_df[cols], vocab, feature_cols)
        raw = pd.read_csv(REAL_DEV_CSV, usecols=["session_id", "session_label"])
        rows = frame["log_id"].str.rsplit(":", n=1).str[1].astype(int).to_numpy()
        sessions = raw.drop_duplicates("session_id").set_index("session_id")["session_label"].astype(int)
        return cls(frame, build_event_sequences(frame, feature_cols), sessions,
                   dict(zip(frame["log_id"], raw["session_id"].to_numpy()[rows])))

    def session_scores(self, log_ids, probs) -> np.ndarray:
        s = pd.Series(np.asarray(probs, float), index=[self.event_sid[i] for i in log_ids]).groupby(level=0).max()
        return s.reindex(self.sessions.index).fillna(0.0).to_numpy()


def unk_rate(dev: RealDev) -> dict[str, float]:
    unk = dev.frame["event_name_idx"] == 0
    return {"all": float(unk.mean()), "attack": float(unk[dev.frame["label"] == 1].mean())}


def tune_threshold(y_true, probs) -> float:
    """Exact best-F1 cut; ties -> widest score gap; THRESHOLD_MARGIN into that gap from the low side."""
    y, p = np.asarray(y_true), np.asarray(probs, dtype=float)
    cuts = np.unique(p)
    if y.sum() == 0 or len(np.unique(y)) < 2 or len(cuts) < 2:
        return 0.5
    if len(cuts) > 400:
        cuts = np.unique(np.quantile(p, np.linspace(0, 1, 400)))
    f1s = np.array([f1_score(y, (p >= t).astype(int), zero_division=0) for t in cuts])
    tied = np.flatnonzero(f1s >= f1s.max() - 1e-12)
    lower = np.concatenate([[0.0], cuts[:-1]])
    i = tied[int(np.argmax(cuts[tied] - lower[tied]))]
    return float(lower[i] + THRESHOLD_MARGIN * (cuts[i] - lower[i]))


def score_dev(dev: RealDev, log_ids, probs, labels, evt_thr=None, ses_thr=None) -> dict:
    """Event + session metrics on real dev. Thresholds are tuned here when not given."""
    y_e, p_e = np.asarray(labels), np.asarray(probs, float)
    s, y_s = dev.session_scores(log_ids, p_e), dev.sessions.to_numpy()
    et = tune_threshold(y_e, p_e) if evt_thr is None else float(evt_thr)
    st = tune_threshold(y_s, s) if ses_thr is None else float(ses_thr)
    return {
        "event": metrics_dict(y_e, p_e, et),
        "session": {**metrics_dict(y_s, s, st), "best_f1": float(max(
            f1_score(y_s, (s >= t).astype(int), zero_division=0) for t in np.unique(np.quantile(s, np.linspace(0, 1, 300)))))},
    }


def selection_key(m: dict) -> tuple[float, float, float]:
    r = lambda x: round(0.0 if x is None or math.isnan(x) else float(x), 4)
    return r(m["session"]["auc_pr"]), r(m["session"]["best_f1"]), r(m["event"]["auc_pr"])


def make_loader(seqs, weighted: bool = False, secret_ids: set[int] | None = None,
                secret_weight: float = 2.0) -> DataLoader:
    ds = EventDataset(seqs)
    if not weighted:
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    secret_ids = secret_ids or set()
    user_n = Counter(s.username for s in seqs)
    weights = []
    for s in seqs:
        w = 1.0 / math.sqrt(user_n[s.username])
        if s.label:
            w *= 3.0 * (secret_weight if s.last_idx in secret_ids else 1.0)
        weights.append(w)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(seqs), replacement=True)
    return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)


def unk_last(event_idx: torch.Tensor, lengths: torch.Tensor, p: float = UNK_LAST_P) -> torch.Tensor:
    """Hide the current event's name (-> <UNK>) so the heads learn to score from features alone."""
    hit = torch.rand(event_idx.size(0)) < p
    if p <= 0 or not bool(hit.any()):
        return event_idx
    rows = torch.nonzero(hit).squeeze(1)
    out = event_idx.clone()
    out[rows, (lengths[rows] - 1).clamp(min=0)] = 0
    return out


def unk_reads(event_idx: torch.Tensor, read_ids: torch.Tensor | None, p: float) -> torch.Tensor:
    """Hide read-only, non-secret, non-PE event names (-> <UNK>) at any position with rate p, whatever the
    label: real traffic is full of read APIs the synthetic vocab never saw, and an unseen read must look
    like "some read", not like an attack."""
    if read_ids is None or p <= 0:
        return event_idx
    hit = torch.isin(event_idx, read_ids) & (torch.rand(event_idx.shape) < p)
    return event_idx.masked_fill(hit, 0)


def seqs_ap(model, seqs, device) -> float:
    y, p = predict(model, make_loader(seqs), device)
    return float(average_precision_score(y, p)) if y.sum() else float("nan")


def train_model(train_s, val_s, dev: RealDev, vocab_size, n_features, device, risk_idx, secret_ids,
                unk_p: float = UNK_LAST_P, secret_weight: float = 2.0,
                read_ids: torch.Tensor | None = None, unk_read_p: float = 0.0):
    set_seed(SEED)
    loader = make_loader(train_s, weighted=True, secret_ids=secret_ids, secret_weight=secret_weight)
    n_pos = sum(s.label for s in train_s)
    pos_weight = torch.tensor([math.sqrt((len(train_s) - n_pos) / max(n_pos, 1))], device=device)
    model = LSTMTransformerModel(vocab_size=vocab_size, n_features=n_features, risk_idx=risk_idx,
                                 secret_ids=secret_ids).to(device)
    print(f"train_events={len(train_s)} pos={n_pos} params={sum(p.numel() for p in model.parameters()):,} "
          f"unk_last_p={unk_p} secret_weight={secret_weight}", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=3)
    history, best_key, best_state, best_epoch, patience_left = [], None, None, 0, PATIENCE
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total, n_b = 0.0, 0
        for event_idx, feats, lengths, y in loader:
            if int(lengths.min()) < 1:
                continue
            event_idx = unk_reads(unk_last(event_idx, lengths, unk_p), read_ids, unk_read_p)
            optim.zero_grad()
            loss = criterion(model(event_idx.to(device), feats.to(device), lengths.to(device)), y.to(device) * 0.9 + 0.05)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total, n_b = total + loss.item(), n_b + 1
        ev = score_seqs(model, dev.seqs, device)
        m = score_dev(dev, ev["log_id"], ev["P_event"], ev["label"])
        key = selection_key(m)
        row = {"epoch": epoch, "train_loss": total / max(n_b, 1), "lr": optim.param_groups[0]["lr"],
               "syn_val_auc_pr": seqs_ap(model, val_s, device), "dev_session_auc_pr": m["session"]["auc_pr"],
               "dev_session_best_f1": m["session"]["best_f1"], "dev_event_auc_pr": m["event"]["auc_pr"],
               "dev_session_auc_roc": m["session"]["auc_roc"], "select_key": json.dumps(key)}
        history.append(row)
        print(f"epoch {epoch:03d} loss={row['train_loss']:.4f} syn_val_ap={row['syn_val_auc_pr']:.4f} "
              f"dev_sess_ap={row['dev_session_auc_pr']:.4f} dev_sess_f1={row['dev_session_best_f1']:.4f} "
              f"dev_sess_auc={row['dev_session_auc_roc']:.4f} dev_evt_ap={row['dev_event_auc_pr']:.4f}", flush=True)
        scheduler.step(float(np.mean(key)))
        if epoch < MIN_EPOCHS:
            continue  # an undertrained model can spike on dev
        if best_key is None or key > best_key:
            best_key, best_epoch, patience_left = key, epoch, PATIENCE
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop @ {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    print(f"selected epoch {best_epoch} key={best_key} ({SELECTION})", flush=True)
    return model, history, best_epoch


def audit(df: pd.DataFrame, feats: list[str], dev_df: pd.DataFrame) -> pd.DataFrame:
    """Single-feature AUC on synthetic vs real dev events: large synthetic-only separation = shortcut."""
    rows = []
    for c in feats:
        if c not in dev_df:
            continue
        a = roc_auc_score(df["label"], df[c]) if df[c].nunique() > 1 else 0.5
        b = roc_auc_score(dev_df["label"], dev_df[c]) if dev_df[c].nunique() > 1 else 0.5
        rows.append({"feature": c, "auc_synthetic": round(a, 3), "auc_real_dev": round(b, 3),
                     "gap": round(abs(a - 0.5) - abs(b - 0.5), 3),
                     "flips_direction": bool((a - 0.5) * (b - 0.5) < 0 and abs(a - 0.5) > 0.1)})
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


def rf_dev_scores(df: pd.DataFrame, feats: list[str], dev: RealDev) -> np.ndarray:
    cols = feats + ["event_name_idx"]
    train = df[df["split"] == "train"]
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=SEED).fit(train[cols].to_numpy(np.float32), train["label"])
    return rf.predict_proba(dev.frame[cols].to_numpy(np.float32))[:, 1]


def compare(df, feats, vocab, dev_df, device) -> dict:
    """v6.3 vs the committed clean LSTM (what the pipeline serves) vs RF, all on real dev sessions.
    df must already carry the PE context columns."""
    out = {}
    for name, path in (("lstm_v6.3", CKPT_PATH), ("lstm_clean (current pipeline)", CLEAN_CKPT)):
        model, ckpt, dev_device = v5.load_checkpoint(path, device=device)
        d = RealDev.build(dev_df, dict(ckpt["event_name_vocab"]), list(ckpt["feature_cols"]))
        ev = score_seqs(model, d.seqs, dev_device)
        out[name] = {**score_dev(d, ev["log_id"], ev["P_event"], ev["label"]), "unk_rate": unk_rate(d)}
    d = RealDev.build(dev_df, vocab, feats + PE_CONTEXT_COLS)
    p = rf_dev_scores(df, feats + PE_CONTEXT_COLS, d)
    out["random_forest"] = score_dev(d, d.frame["log_id"], p, d.frame["label"])
    return out


def maybe_pe_ids(vocab: dict[str, int]) -> tuple[set[int], set[int]]:
    pe = {int(vocab[n]) for n in v5.PE_WRITE_NAMES if n in vocab}
    sec = {int(vocab[n]) for n in SECRET_NAMES if n in vocab}
    return pe, sec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true", help="print feature AUCs, synthetic vs real dev, and exit")
    ap.add_argument("--compare", action="store_true", help="compare the saved v6.3 with the clean LSTM and RF on dev")
    ap.add_argument("--drop", default=None, help="comma list of features to drop (default: DROP_FEATURES)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: artifacts/lstm_transformer_v6_3)")
    ap.add_argument("--recipe", choices=list(RECIPES), default="v6.3",
                    help="v6.2 = the old recipe (no relabel, secret x2, no read <UNK>); v5 = the live model's recipe "
                         "(relabel, secret x4, no <UNK>). Real dev labels are never relabelled.")
    ap.add_argument("--unk-p", type=float, default=None, help="override the recipe's last-event <UNK> rate")
    ap.add_argument("--secret-weight", type=float, default=None, help="override the recipe's secret-positive weight")
    ap.add_argument("--relabel", choices=["on", "off"], default=None, help="override the recipe's campaign relabel")
    ap.add_argument("--unk-read", type=float, default=None, help="override the recipe's read-name <UNK> rate")
    ap.add_argument("--priors", choices=["serve", "streamed"], default="serve",
                    help="label priors: serve = the frozen values the pipeline serves (default); streamed = as in the CSV")
    args = ap.parse_args()
    global OUT_DIR, CKPT_PATH
    if args.out_dir:
        OUT_DIR = Path(args.out_dir).resolve()
        CKPT_PATH = OUT_DIR / "temporal_lstm_transformer.pt"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)  # feature_engine9's paths are repo-relative
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drop = tuple(c for c in args.drop.split(",") if c) if args.drop is not None else DROP_FEATURES

    df, feats, vocab = load_train(drop, args.priors)
    dev_df = real_dev_features()
    if args.audit:
        table = audit(df, feats, dev_df)
        print(table.to_string(index=False))
        return
    pe_ids, sec_ids = maybe_pe_ids(vocab)
    df = attach_pe_context(df, pe_ids)
    feature_cols = feats + PE_CONTEXT_COLS
    if args.compare:
        res = compare(df, feats, vocab, dev_df, device)
        (OUT_DIR / "dev_comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        for name, m in res.items():
            print(f"{name:32s} session AUC-PR={m['session']['auc_pr']:.4f} AUC={m['session']['auc_roc']:.4f} "
                  f"best F1={m['session']['best_f1']:.4f} | event AUC-PR={m['event']['auc_pr']:.4f}")
        print(f"wrote {OUT_DIR / 'dev_comparison.json'}")
        return

    seqs_df = df
    unk_p, secret_weight, relabel, unk_read = RECIPES[args.recipe]
    unk_p = unk_p if args.unk_p is None else args.unk_p
    secret_weight = secret_weight if args.secret_weight is None else args.secret_weight
    relabel = relabel if args.relabel is None else args.relabel == "on"
    unk_read = unk_read if args.unk_read is None else args.unk_read
    if relabel:
        sec5 = {int(vocab[n]) for n in v5.SECRET_NAMES if n in vocab}
        extra5 = {int(vocab[n]) for n in v5.CAMPAIGN_EXTRA_NAMES if n in vocab}
        seqs_df = v5.relabel_campaign(df, sec5, extra5)
    print(f"recipe={args.recipe} unk_p={unk_p} secret_weight={secret_weight} relabel={relabel} unk_read={unk_read}",
          flush=True)
    seqs = build_event_sequences(seqs_df, feature_cols)
    split_of = dict(zip(df["log_id"], df["split"]))
    parts = {k: [s for s in seqs if split_of[s.log_id] == k] for k in ("train", "val", "test")}
    print("synthetic events/pos " + " ".join(f"{k}={len(v)}/{sum(s.label for s in v)}" for k, v in parts.items()), flush=True)
    dev = RealDev.build(dev_df, vocab, feature_cols)
    print(f"real dev events={len(dev.seqs)} sessions={len(dev.sessions)} attack sessions={int(dev.sessions.sum())} "
          f"UNK={unk_rate(dev)}", flush=True)

    vocab_size = max(vocab.values()) + 1
    risk_idx = feature_cols.index("action_risk_prior") if "action_risk_prior" in feature_cols else None
    wr = df.groupby("event_name_idx")["is_write_action"].mean()
    read_list = sorted(int(i) for i, v in wr.items() if v < 0.5 and int(i) not in sec_ids | pe_ids and int(i) != 0)
    read_ids = torch.as_tensor(read_list, dtype=torch.long) if unk_read > 0 else None
    print(f"read-name <UNK> rate {unk_read} over {len(read_list)} read-only non-secret names", flush=True)
    model, history, best_epoch = train_model(parts["train"], parts["val"], dev, vocab_size, seqs[0].feats.shape[1],
                                             device, risk_idx, sec_ids, unk_p=unk_p, secret_weight=secret_weight,
                                             read_ids=read_ids, unk_read_p=unk_read)
    ev = score_seqs(model, dev.seqs, device)
    dev_m = score_dev(dev, ev["log_id"], ev["P_event"], ev["label"])
    evt_thr, ses_thr = dev_m["event"]["threshold"], dev_m["session"]["threshold"]
    syn = {}
    for k in ("val", "test"):
        y, p = predict(model, make_loader(parts[k]), device)
        syn[f"syn_{k}_event"] = metrics_dict(y, p, evt_thr)
    print("=== real DEV session ===", dev_m["session"], flush=True)
    print("=== real DEV event ===", dev_m["event"], flush=True)
    print("=== synthetic unseen-family TEST event ===", syn["syn_test_event"], flush=True)

    split_users = {k: sorted({s.username for s in v}) for k, v in parts.items()}
    config = {
        "model": "LSTMTransformerV6_3", "schema_version": SCHEMA_VERSION,
        "architecture": "BiLSTM + 1-layer Transformer (v5 model class)",
        "dataset": str(TRAIN_CSV.relative_to(REPO)).replace("\\", "/"),
        "split": f"campaign families ({SPLIT_FILE.name}); real dev = real_dataset_dev.csv sessions",
        "split_users": split_users, "vocab_size": vocab_size, "n_features": int(seqs[0].feats.shape[1]),
        "seq_len": v5.SEQ_LEN, "window_minutes": v5.WINDOW_MINUTES, "stride_minutes": v5.STRIDE_MINUTES,
        "secret_ids": sorted(sec_ids), "dropped_features": list(drop), "unk_last_p": unk_p, "unk_read_p": unk_read,
        "priors": args.priors,
        "recipe": args.recipe, "campaign_relabel": relabel, "secret_weight": secret_weight,
        "selection": SELECTION, "selected_epoch": best_epoch, "threshold_margin": THRESHOLD_MARGIN,
        "real_test_scored": False,
    }
    torch.save({"schema_version": SCHEMA_VERSION, "state_dict": model.state_dict(), "feature_cols": feature_cols,
                "threshold": ses_thr, "event_threshold": evt_thr, "event_name_vocab": vocab,
                "dev_metrics": dev_m, "test_metrics": {}, "test_event_metrics": {}, "config": config}, CKPT_PATH)
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    (OUT_DIR / "metrics.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "recipe": args.recipe,
        "selected_epoch": best_epoch,
        "dropped_features": list(drop), "dev": dev_m, "unk_rate_dev": unk_rate(dev), **syn,
        "event_threshold": evt_thr, "session_threshold": ses_thr, "real_test_scored": False}, indent=2), encoding="utf-8")
    print(f"Wrote {CKPT_PATH} (real test not scored)", flush=True)


if __name__ == "__main__":
    main()
