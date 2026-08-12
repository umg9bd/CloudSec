"""CLI batch scorer: python -m prod.cli --csv ... --out ..."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from prod.scorer import DEFAULT_CKPT, ROOT, SchemaError, load_scorer, score_dataframe
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Score temporal CloudTrail CSV → P_seq")
    ap.add_argument("--csv", type=Path, required=True, help="Input temporal events CSV")
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help="Path to temporal_lstm_v2.pt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "P_seq_prod.csv",
        help="Output scores CSV",
    )
    ap.add_argument("--thr-triage", type=float, default=0.55)
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"Missing CSV: {args.csv}")

    t0 = time.perf_counter()
    scorer = load_scorer(args.ckpt, thr_triage=args.thr_triage)
    df = pd.read_csv(args.csv)
    try:
        out = score_dataframe(df, scorer)
    except SchemaError as e:
        raise SystemExit(f"Schema error: {e}") from e

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    elapsed = time.perf_counter() - t0
    summary = {
        "rows_events": int(len(df)),
        "rows_windows": int(len(out)),
        "p_seq_mean": float(out["P_seq"].mean()),
        "triage_rate": float(out["pred_triage"].mean()),
        "alert_rate": float(out["pred_alert"].mean()),
        "thr_triage": scorer.thr_triage,
        "thr_alert": scorer.thr_alert,
        "schema_version": scorer.schema_version,
        "elapsed_sec": round(elapsed, 3),
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
