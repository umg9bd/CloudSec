"""
Thin CLI wrapper — prefer: python -m prod.cli

Kept for backward compatibility with earlier docs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Re-export via prod CLI
from prod.cli import main

if __name__ == "__main__":
    # Map legacy flags if invoked as infer_p_seq_v2.py --csv ...
    # prod.cli requires --csv; default old paths when missing
    if "--csv" not in sys.argv:
        root = Path(__file__).resolve().parent
        default_csv = root / "data" / "lstm" / "train_temporal.csv"
        default_out = root / "artifacts" / "P_seq_infer_v2.csv"
        sys.argv.extend(["--csv", str(default_csv), "--out", str(default_out)])
    main()
