"""Command-line entry point for the real-time directory worker."""
from __future__ import annotations

import argparse
from typing import Optional

from .directory_worker import DirectoryWorker


def add_watch_parser(subparsers: object) -> None:
    # Kept separate from cli.py so the queue implementation can be unit-tested
    # and reused without importing the whole CLI's argument parser.
    watch = subparsers.add_parser("watch", help="process incoming request files continuously or once")
    watch.add_argument("--input-dir", default="incoming", help="directory containing request JSON files (default: incoming)")
    watch.add_argument("--output-dir", default="output", help="directory for paired response JSON files (default: output)")
    watch.add_argument("--processed-dir", default="processed", help="archive for processed request files (default: processed)")
    watch.add_argument("--failed-dir", default="failed", help="archive and diagnostics for file-level failures (default: failed)")
    watch.add_argument("--workers", type=int, default=4, help="parallel request workers (default: 4)")
    watch.add_argument("--poll-interval", type=float, default=0.5, help="seconds between inbox scans (default: 0.5)")
    watch.add_argument("--once", action="store_true", help="process the current queue once, then exit")


def run_watch(args: argparse.Namespace) -> int:
    if args.workers < 1:
        print("--workers must be at least 1")
        return 2
    if args.poll_interval <= 0:
        print("--poll-interval must be greater than 0")
        return 2
    worker = DirectoryWorker(
        incoming_dir=args.input_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        failed_dir=args.failed_dir,
        workers=args.workers,
    )
    if args.once:
        summary = worker.process_available()
        print(
            f"Queue pass finished: {summary.responses_written} response(s) written, "
            f"{summary.failures} file/worker failure(s)."
        )
        return 1 if summary.failures else 0
    try:
        worker.watch_forever(args.poll_interval)
    except KeyboardInterrupt:
        print("\nDirectory worker stopped. Any claimed but unfinished request is retained for recovery.")
    return 0
