"""
ingest.py
---------
Simple, dynamic ingestion for invictus_enriched.csv.

Features:
- Streams CSV in chunks via `pandas.read_csv(chunksize=...)`
- Maintains a watermark (max event timestamp) in a JSON state file
- Writes processed chunks to Parquet in an output directory
- Can be run repeatedly to only process new rows
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


DEFAULT_CSV = Path("datasets/privilege-escalation/invictus_enriched.csv")
DEFAULT_STATE = Path(".ingest_state.json")


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"max_event_time": None, "last_index": -1}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state))


def stream_csv(path: Path, chunksize: int = 10000, time_col: Optional[str] = "eventTime", state: Optional[dict] = None) -> Iterator[pd.DataFrame]:
    """Yield new DataFrame chunks from `path` filtered by watermark in `state`.

    If `time_col` is present, `state['max_event_time']` is used to drop older rows.
    Otherwise a simple index-based watermark `last_index` is used.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    parse_dates = [time_col] if time_col else None
    reader = pd.read_csv(path, chunksize=chunksize, parse_dates=parse_dates, low_memory=False)

    max_seen = state.get("max_event_time") if state else None
    last_index = state.get("last_index", -1) if state else -1

    for chunk in reader:
        if time_col and time_col in chunk.columns and max_seen is not None:
            # keep rows strictly newer than watermark
            try:
                chunk = chunk[chunk[time_col] > pd.to_datetime(max_seen)]
            except Exception:
                pass

        if chunk.empty:
            continue

        # update watermarks based on chunk
        if time_col and time_col in chunk.columns:
            col_max = chunk[time_col].max()
            if pd.notnull(col_max):
                max_seen = str(col_max)

        # index-based fallback
        last_index += len(chunk)

        # yield the filtered chunk
        yield chunk

        # after yielding, update state for downstream persistence
        if state is not None:
            state["max_event_time"] = max_seen
            state["last_index"] = last_index


def process_and_write(chunk: pd.DataFrame, out_dir: Path, seq: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ingested_chunk_{seq:06d}.parquet"
    # More transformation / feature extraction hooks can be added here
    chunk.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Stream invictus_enriched.csv and write Parquet chunks")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to invictus_enriched.csv")
    p.add_argument("--chunksize", type=int, default=10000)
    p.add_argument("--state", type=Path, default=DEFAULT_STATE, help="Path to ingestion state JSON")
    p.add_argument("--out", type=Path, default=Path("ingested"), help="Output directory for Parquet chunks")
    args = p.parse_args()

    state = load_state(args.state)
    seq = 0
    for chunk in stream_csv(args.csv, chunksize=args.chunksize, time_col="eventTime", state=state):
        path = process_and_write(chunk, args.out, seq)
        print(f"Wrote chunk {seq} -> {path} (rows: {len(chunk)})")
        seq += 1
        # persist state after each chunk
        save_state(args.state, state)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
