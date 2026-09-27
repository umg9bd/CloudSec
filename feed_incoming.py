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

    dataset = os.path.join(ROOT, args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    incoming = os.path.join(ROOT, args.incoming) if not os.path.isabs(args.incoming) else args.incoming
    staging = os.path.join(os.path.dirname(incoming), ".incoming_staging")
    os.makedirs(incoming, exist_ok=True)
    os.makedirs(staging, exist_ok=True)

    with open(dataset, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    end = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
    rows = rows[args.start:end]
    stem = os.path.splitext(os.path.basename(dataset))[0]
    n_batches = -(-len(rows) // args.batch_size)
    print(f"Feeding {len(rows)} events from {args.dataset} into {args.incoming}/ as {n_batches} files of "
          f"{args.batch_size}, one every {args.interval:g}s (Ctrl+C to stop)", flush=True)

    try:
        for b in range(n_batches):
            chunk = rows[b * args.batch_size:(b + 1) * args.batch_size]
            name = f"{stem}_batch{b + 1:04d}.csv"
            tmp = os.path.join(staging, name)
            with open(tmp, "w", newline="", encoding="utf-8") as out:
                w = csv.DictWriter(out, fieldnames=fields)
                w.writeheader()
                w.writerows(chunk)
            os.replace(tmp, os.path.join(incoming, name))
            print(f"[FEED] {name}: {len(chunk)} events ({min((b + 1) * args.batch_size, len(rows))}/{len(rows)})",
                  flush=True)
            if b + 1 < n_batches:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped feeding.")
    else:
        print(f"[FEED] done: all {len(rows)} events dropped.")


if __name__ == "__main__":
    main()
