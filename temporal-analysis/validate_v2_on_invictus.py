"""
Transfer-check temporal_lstm_v2 on official Invictus (no retrain, no merge).

Two scoring modes:
  A) integer event_name_idx as stored (IDs > vocab clamped to UNK) — what happens
     if Invictus CSV is fed as-is. Mapping is NOT comparable across datasets.
  B) recover event_name via log_id → invictus_enriched.csv, then map through the
     v2 vocab (OOV→UNK). Fairer test of whether the trained model recognizes
     Invictus API sequences it actually saw names for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prod.scorer import load_scorer, score_dataframe
from train_temporal_lstm_v2 import metrics_dict, tune_threshold

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts"
CKPT = OUT_DIR / "temporal_lstm_v2.pt"
INVICTUS_CSV = ROOT / "data" / "lstm" / "invictus_temporal.csv"
ENRICHED = ROOT / "invictus_enriched.csv"
OUT_JSON = OUT_DIR / "validate_v2_on_invictus.json"


def attach_event_name(temporal: pd.DataFrame, enriched_path: Path) -> tuple[pd.DataFrame, dict]:
    if not enriched_path.exists():
        raise SystemExit(f"Missing {enriched_path}")
    enr = pd.read_csv(enriched_path)
    if "event_name" not in enr.columns:
        raise SystemExit(f"{enriched_path} has no event_name column")

    parsed = temporal["log_id"].astype(str).str.rsplit(":", n=1)
    row_idx = parsed.str[-1].astype(int)
    ts = pd.to_datetime(temporal["timestamp"], utc=True)
    enr_ts = pd.to_datetime(enr["timestamp"], utc=True)

    def match_rate(offset: int) -> float:
        j = row_idx + offset
        ok = (j >= 0) & (j < len(enr))
        if not bool(ok.all()):
            return 0.0
        return float((enr_ts.iloc[j.to_numpy()].to_numpy() == ts.to_numpy()).mean())

    # log_id uses pandas row index (0-based). Try 0 then -1 (header-line numbering).
    rates = {0: match_rate(0), -1: match_rate(-1)}
    offset = max(rates, key=rates.get)
    if rates[offset] < 0.99:
        raise SystemExit(f"log_id join failed, timestamp match rates={rates}")

    j = (row_idx + offset).to_numpy()
    out = temporal.copy()
    out["event_name"] = enr["event_name"].iloc[j].to_numpy()
    info = {
        "enriched_path": str(enriched_path),
        "join_offset": offset,
        "timestamp_match_rate": rates[offset],
        "n_unique_event_names": int(pd.Series(out["event_name"]).nunique()),
    }
    return out, info


def summarize(scored: pd.DataFrame, alert_thr: float, triage_thr: float) -> dict:
    y = scored["window_label"].to_numpy(dtype=float)
    p = scored["P_seq"].to_numpy(dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    oracle = tune_threshold(y, p) if y.sum() > 0 else 0.5
    return {
        "n_windows": int(len(scored)),
        "n_pos_windows": int(y.sum()),
        "score_gap_pos_minus_neg": float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan"),
        "pos_score_mean": float(pos.mean()) if len(pos) else float("nan"),
        "neg_score_mean": float(neg.mean()) if len(neg) else float("nan"),
        "alert_threshold": metrics_dict(y, p, threshold=alert_thr),
        "triage_threshold": metrics_dict(y, p, threshold=triage_thr),
        "oracle_f1_threshold": metrics_dict(y, p, threshold=oracle),
    }


def attacker_slice(scored: pd.DataFrame, alert_thr: float) -> dict:
    out = {}
    for user, g in scored.groupby("username"):
        if int(g["window_label"].sum()) == 0:
            continue
        y = g["window_label"].to_numpy(dtype=float)
        p = g["P_seq"].to_numpy(dtype=float)
        out[str(user)] = metrics_dict(y, p, threshold=alert_thr)
    return out


def main() -> None:
    if not CKPT.exists():
        raise SystemExit(f"Missing {CKPT}")
    if not INVICTUS_CSV.exists():
        raise SystemExit(f"Missing {INVICTUS_CSV}")

    scorer = load_scorer(CKPT)
    df = pd.read_csv(INVICTUS_CSV)
    vmax = max(scorer.vocab.values()) if scorer.vocab else int(scorer.vocab_size) - 1

    n_over = int((df["event_name_idx"] > vmax).sum())
    n_events = int(len(df))
    n_pos_events = int(df["label"].sum())

    print("=== Checkpoint ===")
    print(f"path={CKPT}")
    print(f"schema={scorer.schema_version} vocab_size={scorer.vocab_size} alert_thr={scorer.thr_alert}")
    print("=== Invictus events ===")
    print(f"rows={n_events} pos={n_pos_events} users={df['username'].nunique()}")
    print(f"event_name_idx max={int(df['event_name_idx'].max())}  ids>vocab={n_over}/{n_events}")

    # A) as-is integer IDs (clamp in scorer)
    scored_a = score_dataframe(df, scorer)
    summary_a = summarize(scored_a, scorer.thr_alert, scorer.thr_triage)
    print("\n=== A) Raw Invictus event_name_idx (clamp OOV->UNK) ===")
    print(json.dumps(summary_a, indent=2))

    # B) name-mapped through v2 vocab
    named, join_info = attach_event_name(df, ENRICHED)
    in_vocab = named["event_name"].astype(str).isin(scorer.vocab)
    named_b = named.drop(columns=["event_name_idx"])
    scored_b = score_dataframe(named_b, scorer)
    summary_b = summarize(scored_b, scorer.thr_alert, scorer.thr_triage)
    oov_names = sorted(set(named.loc[~in_vocab, "event_name"].astype(str)))
    print("\n=== B) event_name mapped via v2 vocab (OOV->UNK) ===")
    print("join:", json.dumps(join_info, indent=2))
    print(
        f"name coverage: {int(in_vocab.sum())}/{n_events} in vocab  "
        f"OOV names={len(oov_names)}"
    )
    print(json.dumps(summary_b, indent=2))
    print("\n=== B) attacker windows @ alert thr ===")
    print(json.dumps(attacker_slice(scored_b, scorer.thr_alert), indent=2))

    report = {
        "checkpoint": str(CKPT),
        "dataset": str(INVICTUS_CSV),
        "caveat_A": (
            "Invictus event_name_idx (1-260) is a different mapping than v2 vocab (1-67). "
            "IDs above 67 become UNK; IDs 1-67 may mean different APIs. Not in-distribution."
        ),
        "caveat_B": (
            "event_name recovered from invictus_enriched via log_id, then mapped with v2 vocab. "
            "Unknown Invictus APIs → UNK(0). Fairer transfer check; still not a merge."
        ),
        "checkpoint_vocab_size": scorer.vocab_size,
        "alert_threshold": scorer.thr_alert,
        "triage_threshold": scorer.thr_triage,
        "invictus_events": {
            "n": n_events,
            "n_pos": n_pos_events,
            "n_users": int(df["username"].nunique()),
            "n_idx_gt_vocab": n_over,
        },
        "mode_A_raw_idx": summary_a,
        "mode_B_name_mapped": {
            **summary_b,
            "join": join_info,
            "n_events_in_vocab": int(in_vocab.sum()),
            "n_oov_event_names": len(oov_names),
            "oov_event_names_sample": oov_names[:30],
            "attacker_windows": attacker_slice(scored_b, scorer.thr_alert),
        },
        "v2_in_distribution_test_ref": {
            "auc_pr": 0.4165,
            "auc_roc": 0.9390,
            "precision": 0.4881,
            "recall": 0.5190,
            "threshold": 0.7,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
