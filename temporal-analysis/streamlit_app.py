"""Streamlit tester for LSTM–Transformer v5 / v6 (P_seq).

Run from temporal-analysis/:

    pip install -r requirements-ui.txt
    python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prod.scorer import SchemaError, load_scorer, score_dataframe

DATA = ROOT / "data" / "lstm"
MODELS = {
    "v5 — bert-jan specialist": {
        "kind": "v5",
        "ckpt": ROOT / "artifacts" / "lstm_transformer" / "temporal_lstm_transformer.pt",
        "metrics": ROOT / "artifacts" / "lstm_transformer" / "test_metrics.json",
        "caption": (
            "LSTM–Transformer v5 — held-out bert-jan; per-event history, then "
            "`P_seq = max(P_event)` on 10-minute / stride-2 windows"
        ),
        "train_hint": "python train_lstm_transformer.py",
    },
    "v6 — general (user-disjoint)": {
        "kind": "v6",
        "ckpt": ROOT / "artifacts" / "lstm_transformer_v6" / "temporal_lstm_transformer_v6.pt",
        "metrics": ROOT / "artifacts" / "lstm_transformer_v6" / "test_metrics.json",
        "caption": (
            "LSTM–Transformer v6 — same architecture as v5, user-disjoint 70/15/15 "
            "on cloudtrail_temporal_final (no bert-jan lock). "
            "`P_seq = max(P_event)` on 10-minute / stride-2 windows"
        ),
        "train_hint": "python train_lstm_transformer_v6.py",
    },
}

COLUMN_ALIASES = {
    "userName": "username",
    "user_name": "username",
    "UserName": "username",
    "eventTime": "timestamp",
    "event_time": "timestamp",
    "EventTime": "timestamp",
    "eventName": "event_name",
    "EventName": "event_name",
    "eventname": "event_name",
}

BUILTIN = {
    "Merged + syn chains (v5 train)": DATA / "train_temporal_aug.csv",
    "Merged train (no syn)": DATA / "train_temporal.csv",
    "Invictus (2.9k events)": DATA / "invictus_temporal.csv",
    "fe-final CloudTrail (9.7k)": DATA / "cloudtrail_temporal.csv",
    "CloudTrail final (v6 train, deduped)": DATA / "cloudtrail_temporal_final.csv",
}

SAMPLE_CUSTOM = pd.DataFrame(
    [
        {"username": "demo-attacker", "timestamp": "2024-06-01T12:00:00Z", "event_name": "CreateRole", "label": 1},
        {"username": "demo-attacker", "timestamp": "2024-06-01T12:00:15Z", "event_name": "AttachRolePolicy", "label": 1},
        {"username": "demo-attacker", "timestamp": "2024-06-01T12:00:30Z", "event_name": "AssumeRole", "label": 1},
        {"username": "demo-attacker", "timestamp": "2024-06-01T12:00:45Z", "event_name": "GetSecretValue", "label": 1},
        {"username": "demo-attacker", "timestamp": "2024-06-01T12:01:00Z", "event_name": "CreateAccessKey", "label": 1},
        {"username": "demo-benign", "timestamp": "2024-06-01T12:00:00Z", "event_name": "DescribeInstances", "label": 0},
        {"username": "demo-benign", "timestamp": "2024-06-01T12:00:20Z", "event_name": "ListBuckets", "label": 0},
        {"username": "demo-benign", "timestamp": "2024-06-01T12:00:40Z", "event_name": "GetCallerIdentity", "label": 0},
    ]
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {c: COLUMN_ALIASES[c] for c in out.columns if c in COLUMN_ALIASES and COLUMN_ALIASES[c] not in out.columns}
    if rename:
        out = out.rename(columns=rename)
    return out


def decision(p: float, thr_triage: float, thr_alert: float) -> str:
    if p >= thr_alert:
        return "ALERT"
    if p >= thr_triage:
        return "TRIAGE"
    return "OK"


def apply_thresholds(scored: pd.DataFrame, thr_triage: float, thr_alert: float) -> pd.DataFrame:
    out = scored.copy()
    out["pred_triage"] = (out["P_seq"] >= thr_triage).astype(int)
    out["pred_alert"] = (out["P_seq"] >= thr_alert).astype(int)
    out["decision"] = [decision(p, thr_triage, thr_alert) for p in out["P_seq"]]
    return out


def clf_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(y), 1)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@st.cache_resource
def get_scorer(ckpt_str: str):
    return load_scorer(ckpt_str)


def _metric_line(block: dict) -> str:
    auc = block.get("auc_pr", float("nan"))
    f1 = block.get("f1", float("nan"))
    prec = block.get("precision", float("nan"))
    rec = block.get("recall", float("nan"))
    n = block.get("n", "?")
    return f"AUC-PR {auc:.3f} · F1 {f1:.3f} · P {prec:.2f} · R {rec:.2f} · n={n}"


def render_saved_metrics(kind: str, metrics_path: Path, tm: dict) -> None:
    if tm:
        st.markdown("Held-out test events (from checkpoint)")
        st.write(_metric_line(tm))
    if not metrics_path.exists():
        return
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    if kind == "v5":
        bj = saved.get("test_bertjan_campaign") or {}
        if bj:
            st.markdown("bert-jan campaign (report)")
            st.write(
                f"P {bj.get('precision', float('nan')):.2f} · "
                f"R {bj.get('recall', float('nan')):.2f} · "
                f"F1 {bj.get('f1', float('nan')):.2f} · "
                f"AUC-PR {bj.get('auc_pr', float('nan')):.2f}"
            )
        return
    te = saved.get("test_event") or {}
    tw = saved.get("test_window") or {}
    if te:
        st.markdown("v6 test events (user-disjoint)")
        st.write(_metric_line(te))
    if tw:
        st.markdown("v6 test windows (`P_seq`)")
        st.write(_metric_line(tw))


@st.cache_data
def load_builtin(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def vocab_event_names(scorer) -> list[str]:
    names = sorted(k for k, v in scorer.vocab.items() if v != 0)
    return names


def describe_input(df: pd.DataFrame, scorer) -> None:
    n_users = int(df["username"].nunique()) if "username" in df.columns else 0
    st.caption(f"{len(df):,} events · {n_users} users")
    missing = [c for c in scorer.base_feature_cols if c not in df.columns]
    if missing:
        st.info(
            f"{len(missing)} numeric features are missing and will be filled with 0. "
            "Full temporal CSVs (same schema as train) score closest to training. "
            "Custom rows with only `event_name` still run — signal comes mostly from the API sequence."
        )
    if "event_name" in df.columns:
        mapped = df["event_name"].map(lambda x: int(scorer.vocab.get(str(x), 0)))
        n_oov = int((mapped == 0).sum())
        if n_oov:
            st.warning(f"{n_oov} event names are unknown (OOV → idx 0). Check spelling against the vocab.")
    st.dataframe(df.head(15), use_container_width=True)


def score_events(df: pd.DataFrame, scorer) -> pd.DataFrame:
    df = normalize_columns(df)
    return score_dataframe(df, scorer)


def render_results(scored: pd.DataFrame, n_events: int, scorer) -> None:
    st.subheader("Scores")
    c1, c2 = st.columns(2)
    with c1:
        thr_triage = st.slider("Triage threshold", 0.0, 1.0, float(scorer.thr_triage), 0.01)
    with c2:
        thr_alert = st.slider("Alert threshold", 0.0, 1.0, float(scorer.thr_alert), 0.01)
    if thr_alert < thr_triage:
        st.caption("Alert is below triage — both flags can fire independently.")

    out = apply_thresholds(scored, thr_triage, thr_alert)
    n_triage = int(out["pred_triage"].sum())
    n_alert = int(out["pred_alert"].sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Events", f"{n_events:,}")
    k2.metric("Windows", f"{len(out):,}")
    k3.metric("Mean P_seq", f"{out['P_seq'].mean():.3f}")
    k4.metric("Triage", n_triage)
    k5.metric("Alert", n_alert)

    has_labels = "window_label" in out.columns and int(out["window_label"].nunique()) > 1
    if has_labels:
        y = out["window_label"].to_numpy()
        m_t = clf_metrics(y, out["pred_triage"])
        m_a = clf_metrics(y, out["pred_alert"])
        st.markdown("**Window metrics** (needs `label` on events)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Triage F1", f"{m_t['f1']:.3f}")
        m2.metric("Triage precision / recall", f"{m_t['precision']:.2f} / {m_t['recall']:.2f}")
        m3.metric("Alert F1", f"{m_a['f1']:.3f}")
        m4.metric("Alert precision / recall", f"{m_a['precision']:.2f} / {m_a['recall']:.2f}")
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            st.caption(
                f"AUC-PR {average_precision_score(y, out['P_seq']):.3f} · "
                f"AUC-ROC {roc_auc_score(y, out['P_seq']):.3f}"
            )
        except Exception:
            pass

    plot = out.copy()
    plot["window_start"] = pd.to_datetime(plot["window_start"], utc=True)
    users = sorted(plot["username"].unique().tolist())
    pick = users
    if len(users) > 6:
        pick = st.multiselect("Plot users", users, default=users[:6])
        if not pick:
            pick = users[:6]
    vis = plot[plot["username"].isin(pick)]

    left, right = st.columns(2)
    with left:
        st.caption("P_seq over window start")
        st.line_chart(vis, x="window_start", y="P_seq", color="username", use_container_width=True)
    with right:
        st.caption("P_seq distribution")
        bins = pd.cut(out["P_seq"], bins=np.linspace(0, 1, 21), include_lowest=True)
        hist = bins.value_counts().sort_index()
        hist.index = [f"{i.left:.2f}" for i in hist.index]
        st.bar_chart(hist, use_container_width=True)

    show = out.sort_values("P_seq", ascending=False)
    st.markdown("**Windows** (highest P_seq first)")
    display_cols = [
        "username",
        "window_start",
        "window_end",
        "raw_len",
        "P_seq",
        "decision",
        "pred_triage",
        "pred_alert",
        "window_label",
    ]
    display_cols = [c for c in display_cols if c in show.columns]
    st.dataframe(
        show[display_cols].style.format({"P_seq": "{:.4f}"}),
        use_container_width=True,
        height=420,
    )
    csv_bytes = show.to_csv(index=False).encode("utf-8")
    st.download_button("Download scores CSV", csv_bytes, "P_seq_scores.csv", "text/csv")


def builtin_events(scorer) -> pd.DataFrame | None:
    choice = st.selectbox("Dataset", list(BUILTIN.keys()))
    path = BUILTIN[choice]
    if not path.exists():
        st.error(f"Missing file: {path}")
        return None
    df = load_builtin(str(path))
    users = sorted(df["username"].astype(str).unique().tolist())
    subset = st.multiselect("Filter users (optional)", users, default=[])
    n = max(int(len(df)), 1)
    max_rows = st.slider("Max events", 1, n, min(n, 3000))
    if subset:
        df = df[df["username"].astype(str).isin(subset)]
    df = df.head(max_rows)
    describe_input(df, scorer)
    return df


def upload_events(scorer) -> pd.DataFrame | None:
    st.markdown(
        "Upload a CSV. **Required:** `username`, `timestamp`, and `event_name` *or* `event_name_idx`. "
        "CloudTrail-style names (`userName`, `eventTime`, `eventName`) are remapped. "
        "Optional `label` (0/1) enables metrics. Other numeric features default to 0. "
        "A ready example is `data/lstm/sample_custom_events.csv`."
    )
    tmpl = SAMPLE_CUSTOM.to_csv(index=False).encode("utf-8")
    st.download_button("Download template CSV", tmpl, "custom_events_template.csv", "text/csv")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None
    df = normalize_columns(df)
    describe_input(df, scorer)
    return df


def builder_events(scorer) -> pd.DataFrame | None:
    st.markdown(
        "Edit events below, or paste JSON. Minimal columns: "
        "`username`, `timestamp`, `event_name`, optional `label`."
    )
    names = vocab_event_names(scorer)
    if st.button("Reset to PE-chain example"):
        st.session_state["builder_df"] = SAMPLE_CUSTOM.copy()
        st.rerun()

    paste = st.text_area(
        "Or paste JSON list of events",
        height=90,
        placeholder='[{"username":"alice","timestamp":"2024-06-01T12:00:00Z","event_name":"AssumeRole"}]',
    )

    if paste.strip():
        try:
            df = normalize_columns(pd.DataFrame(json.loads(paste)))
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            return None
    else:
        if "builder_df" not in st.session_state:
            st.session_state["builder_df"] = SAMPLE_CUSTOM.copy()
        df = st.data_editor(
            st.session_state["builder_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="custom_event_editor",
            column_config={
                "event_name": st.column_config.SelectboxColumn(
                    "event_name", options=names, required=True
                ),
                "label": st.column_config.NumberColumn("label", min_value=0, max_value=1, step=1),
            },
        )

    if df is None or df.empty:
        return None
    df = normalize_columns(df)
    missing = [c for c in ("username", "timestamp") if c not in df.columns]
    if missing:
        st.error(f"Custom events need columns: {missing}")
        return None
    n_oov = 0
    if "event_name" in df.columns:
        mapped = df["event_name"].map(lambda x: int(scorer.vocab.get(str(x), 0)))
        n_oov = int((mapped == 0).sum())
    st.caption(
        f"{len(df)} events · {df['username'].nunique()} users"
        + (f" · {n_oov} unknown event names (OOV)" if n_oov else "")
        + " · missing numeric features filled with 0"
    )
    return df


def main() -> None:
    st.set_page_config(page_title="P_seq tester", layout="wide")

    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox("Checkpoint", list(MODELS.keys()))

    spec = MODELS[model_name]
    ckpt = spec["ckpt"]
    st.title("P_seq tester")
    st.caption(spec["caption"])

    if not ckpt.exists():
        st.error(f"Checkpoint not found: {ckpt}. Run `{spec['train_hint']}` first.")
        st.stop()

    if st.session_state.get("_loaded_ckpt") != str(ckpt):
        st.session_state.pop("last_scored", None)
        st.session_state.pop("last_events_n", None)
        st.session_state["_loaded_ckpt"] = str(ckpt)

    try:
        scorer = get_scorer(str(ckpt))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    with st.sidebar:
        st.write(f"**{scorer.model_id}**")
        st.write(f"schema `{scorer.schema_version}`")
        st.write(f"device `{scorer.device}` · vocab {scorer.vocab_size}")
        st.write("heads: IAM + secrets · `P_seq = max(P_event)`")
        st.write(f"default triage **{scorer.thr_triage:.2f}** · alert **{scorer.thr_alert:.2f}**")
        render_saved_metrics(spec["kind"], spec["metrics"], scorer.test_metrics or {})
        st.divider()
        st.markdown(
            "Windows are **10 min / stride 2 min**, per username. "
            "`P_seq` is the **max** event probability in the window "
            "(not a bagged LSTM average)."
        )

    source = st.radio(
        "Data source",
        ["Built-in dataset", "Upload custom CSV", "Custom events"],
        horizontal=True,
    )

    df = None
    if source == "Built-in dataset":
        df = builtin_events(scorer)
    elif source == "Upload custom CSV":
        df = upload_events(scorer)
    else:
        df = builder_events(scorer)

    run = st.button("Score model", type="primary", disabled=df is None or df.empty)
    if run and df is not None:
        with st.spinner("Scoring windows…"):
            try:
                scored = score_events(df, scorer)
            except SchemaError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                st.stop()
        st.session_state["last_events_n"] = int(len(df))
        st.session_state["last_scored"] = scored

    if "last_scored" in st.session_state:
        render_results(st.session_state["last_scored"], st.session_state.get("last_events_n", 0), scorer)


if __name__ == "__main__":
    main()
