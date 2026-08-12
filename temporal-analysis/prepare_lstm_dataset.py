"""
Prepare privilege-escalation temporal data for LSTM training.

Merges official Invictus temporal events with fe-final CloudTrail:
recover event_name via log_id, build a union vocab (keep fe-final IDs 1-67),
remap Invictus indices, prefix usernames so windows never mix sources.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "CloudSec" / "datasets" / "privilege-escalation"
OUT = ROOT / "data" / "lstm"

INVICTUS_TEMPORAL = ROOT / "invictus_temporal.csv"
INVICTUS_ENRICHED = ROOT / "invictus_enriched.csv"
SYNTHETIC_CLOUDTRAIL = SRC / "synthetic_cloudtrail.csv"
FE_VOCAB_SRC = SRC / ".event_name_vocab.json"

META_COLS = {"log_id", "username", "timestamp", "label", "event_name_idx"}
EXPECTED_FEATURES = [
    "no_mfa",
    "mfa_absent",
    "principal_type_prior_risk",
    "principal_type_idx",
    "has_access_key",
    "action_velocity",
    "is_new_action",
    "session_duration_normalized",
    "events_per_minute_normalized",
    "time_sin",
    "time_cos",
    "is_weekend",
    "is_off_hours",
    "action_risk_prior",
    "event_source_idx",
    "is_write_action",
    "read_only_absent",
    "has_error",
    "is_access_denied",
    "is_iam_event",
    "is_recon_action",
    "is_defense_evasion",
    "is_get_caller_identity",
    "is_malicious_user_agent",
    "is_public_ip",
    "params_length_normalized",
    "targets_sensitive_resource",
    "is_non_default_region",
    "is_create_key",
    "is_secrets_or_kms",
    "is_permission_modification",
    "policy_statement_count_normalized",
    "has_wildcard_action",
    "has_wildcard_resource",
    "privileged_action_reach",
]
WINDOW_MINUTES = 10
STRIDE_MINUTES = 2
SEQ_LEN = 128
JOIN_MATCH_MIN = 0.99

ORDERED_COLS = [
    "log_id",
    "username",
    "timestamp",
    "no_mfa",
    "mfa_absent",
    "principal_type_prior_risk",
    "principal_type_idx",
    "has_access_key",
    "action_velocity",
    "is_new_action",
    "session_duration_normalized",
    "events_per_minute_normalized",
    "time_sin",
    "time_cos",
    "is_weekend",
    "is_off_hours",
    "action_risk_prior",
    "event_name_idx",
    "event_source_idx",
    "is_write_action",
    "read_only_absent",
    "has_error",
    "is_access_denied",
    "is_iam_event",
    "is_recon_action",
    "is_defense_evasion",
    "is_get_caller_identity",
    "is_malicious_user_agent",
    "is_public_ip",
    "params_length_normalized",
    "targets_sensitive_resource",
    "is_non_default_region",
    "is_create_key",
    "is_secrets_or_kms",
    "is_permission_modification",
    "policy_statement_count_normalized",
    "has_wildcard_action",
    "has_wildcard_resource",
    "privileged_action_reach",
    "label",
]


def ensure_sources() -> None:
    required = [
        SRC / "cloudtrail_temporal.csv",
        FE_VOCAB_SRC,
        SYNTHETIC_CLOUDTRAIL,
        INVICTUS_TEMPORAL,
        INVICTUS_ENRICHED,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing source files for Invictus + fe-final merge:\n  " + "\n  ".join(missing))


def copy_assets() -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    copied = {}
    pairs = [
        (SRC / "cloudtrail_temporal.csv", OUT / "cloudtrail_temporal.csv"),
        (SRC / "cloudtrail_structural.csv", OUT / "cloudtrail_structural.csv"),
        (INVICTUS_TEMPORAL, OUT / "invictus_temporal.csv"),
        (FE_VOCAB_SRC, OUT / "event_name_vocab_fe_final.json"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)
            copied[dst.name] = str(dst)
    return copied


def validate_temporal(df: pd.DataFrame, name: str) -> None:
    required = META_COLS | set(EXPECTED_FEATURES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if df.shape[1] != 40:
        raise ValueError(f"{name}: expected 40 columns, got {df.shape[1]}")
    ts_na = int(pd.to_datetime(df["timestamp"], utc=True).isna().sum())
    if ts_na:
        raise ValueError(f"{name}: NaNs in timestamp: {ts_na}")
    for col in ["username", "event_name_idx", "label"]:
        n = int(df[col].isna().sum())
        if n:
            raise ValueError(f"{name}: NaNs in {col}: {n}")
    feature_cols = [c for c in df.columns if c not in META_COLS]
    if set(feature_cols) != set(EXPECTED_FEATURES):
        raise ValueError(f"{name}: feature column set does not match LSTM recipe")


def coerce_temporal(df: pd.DataFrame) -> pd.DataFrame:
    out = df[ORDERED_COLS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for c in EXPECTED_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["event_name_idx"] = out["event_name_idx"].astype(int)
    out["label"] = out["label"].astype(int)
    out["username"] = out["username"].astype(str)
    out["log_id"] = out["log_id"].astype(str)
    return out


def parse_log_id(log_id: str) -> tuple[str, int]:
    file, idx = str(log_id).rsplit(":", 1)
    return Path(file).name, int(idx)


def attach_event_name(
    temporal: pd.DataFrame,
    source_csv: Path,
    expected_stem: str,
) -> tuple[pd.DataFrame, dict]:
    """Join temporal rows to a name-bearing CSV via log_id row index."""
    enr = pd.read_csv(source_csv)
    if "event_name" not in enr.columns:
        raise SystemExit(f"{source_csv} has no event_name column")

    files = temporal["log_id"].map(lambda x: parse_log_id(x)[0])
    bad = sorted(set(files.unique()) - {expected_stem})
    if bad:
        raise SystemExit(f"Unexpected log_id stems (want {expected_stem}): {bad[:8]}")

    row_idx = temporal["log_id"].map(lambda x: parse_log_id(x)[1]).astype(int)
    ts = pd.to_datetime(temporal["timestamp"], utc=True)
    enr_ts = pd.to_datetime(enr["timestamp"], utc=True)

    def match_rate(offset: int) -> float:
        j = row_idx + offset
        ok = (j >= 0) & (j < len(enr))
        if not bool(ok.all()):
            return 0.0
        return float((enr_ts.iloc[j.to_numpy()].to_numpy() == ts.to_numpy()).mean())

    rates = {0: match_rate(0), -1: match_rate(-1)}
    offset = max(rates, key=lambda k: rates[k])
    if rates[offset] < JOIN_MATCH_MIN:
        raise SystemExit(
            f"log_id join failed for {expected_stem}: timestamp match rates={rates}"
        )

    j = (row_idx + offset).to_numpy()
    out = temporal.copy()
    out["event_name"] = enr["event_name"].iloc[j].astype(str).to_numpy()

    user_match = None
    if "username" in enr.columns:
        user_match = float(
            (
                enr["username"].iloc[j].astype(str).to_numpy()
                == out["username"].astype(str).to_numpy()
            ).mean()
        )
        if user_match < JOIN_MATCH_MIN:
            print(
                f"WARN: {expected_stem} username match={user_match:.4f} "
                "(join still accepted; timestamp match is the hard check)"
            )

    info = {
        "source_csv": str(source_csv),
        "expected_stem": expected_stem,
        "join_offset": int(offset),
        "timestamp_match_rate": rates[offset],
        "username_match_rate": user_match,
        "n_unique_event_names": int(pd.Series(out["event_name"]).nunique()),
    }
    return out, info


def build_union_vocab(fe_vocab: dict, invictus_names: set[str]) -> dict[str, int]:
    vocab = {str(k): int(v) for k, v in fe_vocab.items()}
    vocab["<UNK>"] = 0
    nxt = max(vocab.values()) + 1
    extra = sorted(n for n in invictus_names if n not in vocab)
    for name in extra:
        vocab[name] = nxt
        nxt += 1
    return vocab


def build_window_stats(df: pd.DataFrame) -> dict:
    window_td = pd.Timedelta(minutes=WINDOW_MINUTES)
    stride_td = pd.Timedelta(minutes=STRIDE_MINUTES)
    stride_ns = int(stride_td / pd.Timedelta(nanoseconds=1))
    window_ns = int(window_td / pd.Timedelta(nanoseconds=1))
    n_windows = 0
    n_pos = 0
    lengths: list[int] = []
    pos_users: dict[str, int] = {}

    for username, g in df.groupby("username", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if g.empty:
            continue
        ts = g["timestamp"]
        starts: set[pd.Timestamp] = set()
        for t in ts:
            t_ns = int(t.value)
            aligned = (t_ns // stride_ns) * stride_ns
            k = 0
            while k * stride_ns < window_ns:
                starts.add(pd.Timestamp(aligned - k * stride_ns, tz="UTC"))
                k += 1

        for start in sorted(starts):
            end = start + window_td
            chunk = g.loc[(ts >= start) & (ts < end)]
            if len(chunk) == 0:
                continue
            y = 1 if int(chunk["label"].max()) == 1 else 0
            n_windows += 1
            n_pos += y
            lengths.append(len(chunk))
            if y == 1:
                pos_users[str(username)] = pos_users.get(str(username), 0) + 1

    lengths_arr = np.asarray(lengths, dtype=np.int64) if lengths else np.array([0])
    top_attackers = sorted(pos_users.items(), key=lambda x: (-x[1], x[0]))[:10]
    return {
        "n_windows": n_windows,
        "n_pos_windows": n_pos,
        "n_neg_windows": n_windows - n_pos,
        "raw_len_min": int(lengths_arr.min()),
        "raw_len_median": float(np.median(lengths_arr)),
        "raw_len_max": int(lengths_arr.max()),
        "n_users_with_pos_windows": len(pos_users),
        "loao_candidates": [u for u, _c in top_attackers if _c >= 1][:5],
        "top_pos_window_users": [
            {"username": u, "pos_windows": c} for u, c in top_attackers
        ],
        "windowing": "event-covering stride grid (skip empty spans)",
    }


def main() -> None:
    ensure_sources()
    copied = copy_assets()

    fe_vocab = json.loads(FE_VOCAB_SRC.read_text(encoding="utf-8"))
    fe_vocab = {str(k): int(v) for k, v in fe_vocab.items()}
    fe_vocab["<UNK>"] = 0
    fe_max = max(v for k, v in fe_vocab.items() if k != "<UNK>")

    fe_raw = pd.read_csv(OUT / "cloudtrail_temporal.csv")
    inv_raw = pd.read_csv(OUT / "invictus_temporal.csv")
    validate_temporal(fe_raw, "cloudtrail_temporal")
    validate_temporal(inv_raw, "invictus_temporal")
    fe = coerce_temporal(fe_raw)
    inv = coerce_temporal(inv_raw)

    fe_named, fe_join = attach_event_name(fe, SYNTHETIC_CLOUDTRAIL, "synthetic_cloudtrail.csv")
    mapped = fe_named["event_name"].map(lambda n: int(fe_vocab.get(str(n), -1)))
    if not bool((mapped.to_numpy() == fe_named["event_name_idx"].to_numpy()).all()):
        n_bad = int((mapped.to_numpy() != fe_named["event_name_idx"].to_numpy()).sum())
        raise SystemExit(f"fe-final event_name_idx does not match vocab for {n_bad} rows")
    fe_idx_before = fe["event_name_idx"].to_numpy().copy()

    inv_named, inv_join = attach_event_name(inv, INVICTUS_ENRICHED, "invictus_enriched.csv")
    inv_names = set(inv_named["event_name"].astype(str))
    vocab = build_union_vocab(fe_vocab, inv_names)
    n_extra = sum(1 for k, v in vocab.items() if k != "<UNK>" and v > fe_max)

    inv_named["event_name_idx"] = (
        inv_named["event_name"].astype(str).map(lambda n: int(vocab.get(n, 0))).astype(int)
    )
    n_unk = int((inv_named["event_name_idx"] == 0).sum())
    if n_unk:
        raise SystemExit(f"Invictus remap produced {n_unk} UNK rows; expected 0")

    fe["username"] = "fe:" + fe["username"].astype(str)
    inv_named["username"] = "inv:" + inv_named["username"].astype(str)

    if not np.array_equal(fe["event_name_idx"].to_numpy(), fe_idx_before):
        raise SystemExit("fe-final event_name_idx changed during merge; abort")

    if len(fe) != 9711 or len(inv_named) != 2900:
        raise SystemExit(
            f"Unexpected source sizes: fe={len(fe)} (want 9711), inv={len(inv_named)} (want 2900)"
        )
    merged = pd.concat([fe[ORDERED_COLS], inv_named[ORDERED_COLS]], ignore_index=True)
    if merged.shape[1] != 40:
        raise SystemExit(f"Merged table must stay 40 cols, got {merged.shape[1]}")
    if int(len(merged)) != 12611:
        raise SystemExit(f"Merged row count {len(merged)} != 12611")

    prefixes = merged["username"].astype(str).str.split(":", n=1).str[0]
    if set(prefixes.unique()) != {"fe", "inv"}:
        raise SystemExit(f"Username prefixes unexpected: {sorted(prefixes.unique())}")

    prepared_path = OUT / "train_temporal.csv"
    merged.to_csv(prepared_path, index=False)

    vocab_path = OUT / "event_name_vocab.json"
    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    idx_max = int(merged["event_name_idx"].max())
    vocab_max = max(vocab.values())
    if idx_max != vocab_max:
        raise SystemExit(f"vocab max id {vocab_max} != CSV event_name_idx max {idx_max}")
    vocab_size = idx_max + 1

    window_stats = build_window_stats(merged)
    feature_cols = [c for c in ORDERED_COLS if c not in META_COLS]

    manifest = {
        "source_branch": "fe-final + official Invictus",
        "source_repo": "umg9bd/CloudSec",
        "sources": {
            "fe_final_temporal": "data/lstm/cloudtrail_temporal.csv",
            "invictus_temporal": "data/lstm/invictus_temporal.csv",
            "fe_final_names": str(SYNTHETIC_CLOUDTRAIL.relative_to(ROOT))
            if SYNTHETIC_CLOUDTRAIL.is_relative_to(ROOT)
            else str(SYNTHETIC_CLOUDTRAIL),
            "invictus_names": "invictus_enriched.csv",
        },
        "prepared_csv": str(prepared_path.relative_to(ROOT)),
        "vocab_json": "data/lstm/event_name_vocab.json",
        "n_rows": int(len(merged)),
        "n_rows_fe_final": int(len(fe)),
        "n_rows_invictus": int(len(inv_named)),
        "n_cols": int(merged.shape[1]),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_pos_events": int(merged["label"].sum()),
        "n_neg_events": int((merged["label"] == 0).sum()),
        "n_users": int(merged["username"].nunique()),
        "event_name_idx_min": int(merged["event_name_idx"].min()),
        "event_name_idx_max": idx_max,
        "vocab_size": vocab_size,
        "fe_final_vocab_max": fe_max,
        "n_invictus_only_event_names": n_extra,
        "username_prefix": {"fe_final": "fe:", "invictus": "inv:"},
        "join": {"fe_final": fe_join, "invictus": inv_join},
        "window_minutes": WINDOW_MINUTES,
        "stride_minutes": STRIDE_MINUTES,
        "seq_len": SEQ_LEN,
        "label_rule": "any(label==1)",
        "time_range": [
            merged["timestamp"].min().isoformat(),
            merged["timestamp"].max().isoformat(),
        ],
        "window_stats": window_stats,
        "copied_files": copied,
        "unmerged_copies": {
            "cloudtrail_temporal.csv": "data/lstm/cloudtrail_temporal.csv",
            "invictus_temporal.csv": "data/lstm/invictus_temporal.csv",
        },
        "notes": (
            "train_temporal.csv = official Invictus + fe-final CloudTrail. "
            "fe-final event_name_idx 1-67 frozen; Invictus names remapped via union vocab. "
            "Usernames prefixed fe:/inv: so windowing does not mix sources. "
            "VOCAB_SIZE must be event_name_idx_max+1."
        ),
    }

    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== LSTM merged dataset prepared ===")
    print(f"rows={manifest['n_rows']} (fe={len(fe)} + inv={len(inv_named)})")
    print(f"features={manifest['n_features']} vocab_size={vocab_size} extra_names={n_extra}")
    print(
        f"events pos/neg={manifest['n_pos_events']}/{manifest['n_neg_events']} "
        f"users={manifest['n_users']}"
    )
    print(
        f"windows={window_stats['n_windows']} "
        f"pos={window_stats['n_pos_windows']} neg={window_stats['n_neg_windows']}"
    )
    print(f"fe join ts={fe_join['timestamp_match_rate']} inv join ts={inv_join['timestamp_match_rate']}")
    print(f"loao_candidates={window_stats['loao_candidates']}")
    print(f"wrote {prepared_path}")
    print(f"wrote {vocab_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
