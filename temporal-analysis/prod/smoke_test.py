"""
Production smoke / readiness checks.

Run before demo:
  python -m prod.smoke_test
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "lstm" / "train_temporal.csv"
CKPT = ROOT / "artifacts" / "lstm_transformer" / "temporal_lstm_transformer.pt"
N_ROWS = 500


def main() -> int:
    from fastapi.testclient import TestClient

    from prod.app import app
    from prod.scorer import SchemaError, load_scorer, score_dataframe

    errors: list[str] = []

    if not CKPT.exists():
        print(f"FAIL: missing {CKPT}")
        return 1
    if not CSV.exists():
        print(f"FAIL: missing {CSV}")
        return 1

    # --- cold load + batch score ---
    t0 = time.perf_counter()
    scorer = load_scorer(CKPT)
    load_s = time.perf_counter() - t0
    print(f"OK load ckpt in {load_s:.2f}s device={scorer.device} schema={scorer.schema_version}")

    df = pd.read_csv(CSV, nrows=N_ROWS)
    t1 = time.perf_counter()
    try:
        out = score_dataframe(df, scorer)
    except SchemaError as e:
        print(f"FAIL score: {e}")
        return 1
    score_s = time.perf_counter() - t1

    if out.empty:
        errors.append("empty score output")
    if out["P_seq"].isna().any():
        errors.append("NaN in P_seq")
    if not ((out["P_seq"] >= 0) & (out["P_seq"] <= 1)).all():
        errors.append("P_seq out of [0,1]")
    if "pred_triage" not in out.columns or "pred_alert" not in out.columns:
        errors.append("missing dual threshold columns")
    if not (out["pred_alert"] <= out["pred_triage"]).all() and scorer.thr_alert >= scorer.thr_triage:
        # alert implies triage when thr_alert >= thr_triage
        bad = out[out["pred_alert"] > out["pred_triage"]]
        if len(bad):
            errors.append("pred_alert set without pred_triage")

    print(
        f"OK score events={len(df)} windows={len(out)} "
        f"in {score_s:.2f}s p_mean={out['P_seq'].mean():.4f}"
    )

    # --- FastAPI health ---
    with TestClient(app) as client:
        r = client.get("/health")
        if r.status_code != 200:
            errors.append(f"/health status {r.status_code}")
        else:
            body = r.json()
            if body.get("status") != "ok":
                errors.append(f"/health body {body}")
            print(f"OK /health {body}")

        r2 = client.get("/model")
        if r2.status_code != 200:
            errors.append(f"/model status {r2.status_code}")
        else:
            print(f"OK /model model_id={r2.json().get('model_id')}")

        # tiny JSON score
        sample = df.head(20).copy()
        sample["timestamp"] = sample["timestamp"].astype(str)
        payload = {"events": sample.to_dict(orient="records")}
        r3 = client.post("/score/json", json=payload)
        if r3.status_code != 200:
            errors.append(f"/score/json status {r3.status_code}: {r3.text[:200]}")
        else:
            print(f"OK /score/json windows={r3.json().get('n_windows')}")

    if errors:
        print("SMOKE FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
