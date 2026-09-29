"""
feed_incoming.py -- replays a dataset into the pipeline's watched folder, batch by batch,
the way CloudTrail delivers log files: every few seconds a new file with the next N events.

    python feed_incoming.py                                   # real_dataset_test.csv, 200 events / 5 s
    python feed_incoming.py --dataset datasets/privilege-escalation/synthetic_cloudtrail.csv
    python feed_incoming.py --batch-size 50 --interval 2 --limit 2000

Run pipeline.py --watch incoming --show-events in another terminal to see every event scored.
Batches are written next to the folder and then moved in with one rename, so the watcher never
sees a half-written file. Needs no torch: runs natively as well as in the Docker image.
"""
import argparse
import csv
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join("datasets", "privilege-escalation", "real_dataset_test.csv")


def feed(dataset: str = DEFAULT_DATASET, incoming: str = "incoming", batch_size: int = 200,
         interval: float = 5.0, start: int = 0, limit: int | None = None) -> None:
    """Drops `dataset` into `incoming` as numbered files of `batch_size` events, one per `interval` s."""
    dataset = os.path.join(ROOT, dataset) if not os.path.isabs(dataset) else dataset
    incoming = os.path.join(ROOT, incoming) if not os.path.isabs(incoming) else incoming
    staging = os.path.join(os.path.dirname(incoming), ".incoming_staging")
    os.makedirs(incoming, exist_ok=True)
    os.makedirs(staging, exist_ok=True)

    with open(dataset, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    end = len(rows) if limit is None else min(len(rows), start + limit)
    rows = rows[start:end]
    stem = os.path.splitext(os.path.basename(dataset))[0]
    n_batches = -(-len(rows) // batch_size)
    print(f"[FEED] {len(rows)} events from {os.path.relpath(dataset, ROOT)} -> {n_batches} files of "
          f"{batch_size}, one every {interval:g}s", flush=True)
    for b in range(n_batches):
        chunk = rows[b * batch_size:(b + 1) * batch_size]
        name = f"{stem}_batch{b + 1:04d}.csv"
        tmp = os.path.join(staging, name)
        with open(tmp, "w", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=fields)
            w.writeheader()
            w.writerows(chunk)
        os.replace(tmp, os.path.join(incoming, name))
        print(f"[FEED] {name}: {len(chunk)} events ({min((b + 1) * batch_size, len(rows))}/{len(rows)})", flush=True)
        if b + 1 < n_batches:
            time.sleep(interval)
    print(f"[FEED] done: all {len(rows)} events dropped.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="A CloudTrail CSV in feature_engine9's input format (the real_dataset_*.csv files, "
                         "synthetic_cloudtrail.csv)")
    ap.add_argument("--incoming", default="incoming")
    ap.add_argument("--batch-size", type=int, default=200, help="events per dropped file")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between files")
    ap.add_argument("--start", type=int, default=0, help="first row of the dataset to send")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many events")
    args = ap.parse_args()
    try:
        feed(args.dataset, args.incoming, args.batch_size, args.interval, args.start, args.limit)
    except KeyboardInterrupt:
        print("\nStopped feeding.")


if __name__ == "__main__":
    main()
