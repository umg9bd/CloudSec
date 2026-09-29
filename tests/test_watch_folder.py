"""
test_watch_folder.py
====================
Guards pipeline.watch / scan_once -- the live detector's folder loop.

The failure it pins was real: a second pipeline watching the same incoming/
folder moved a batch away after the first had scored it. The first's move to
processed/ then raised FileNotFoundError, its error handler's move to failed/
raised again, and that second exception stopped the detector. Files are now
claimed by an atomic rename before scoring, so exactly one watcher gets each
file, and a file that vanishes is skipped with a warning instead of a crash.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import feature_engine9 as fe9
import pipeline
from pipeline import CLAIM_DIR, clear_unscored_feed, scan_once, watch


class StubPipeline:
    """Records what it scored; `fail` names raise, `interrupt` simulates Ctrl+C."""
    alert_dir, output_csv = "alerts", "out.csv"

    def __init__(self, fail=(), interrupt=False):
        self.scored, self.fail, self.interrupt = [], set(fail), interrupt

    def process_file(self, path):
        name = os.path.basename(path)
        if self.interrupt:
            raise KeyboardInterrupt
        if name in self.fail:
            raise ValueError("unreadable")
        with open(path, encoding="utf-8") as f:
            f.read()
        self.scored.append(name)


class WatchSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = os.path.join(self.tmp, "incoming")
        for d in (self.dir, os.path.join(self.dir, CLAIM_DIR),
                  os.path.join(self.dir, "processed"), os.path.join(self.dir, "failed")):
            os.makedirs(d)
        self.stable = mock.patch.object(fe9, "_file_is_stable", lambda p: os.path.exists(p))
        self.stable.start()

    def tearDown(self):
        self.stable.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def drop(self, *names):
        for n in names:
            with open(os.path.join(self.dir, n), "w", encoding="utf-8") as f:
                f.write("timestamp,event_name\n")

    def listing(self, sub=""):
        return sorted(os.listdir(os.path.join(self.dir, sub))) if sub else sorted(
            n for n in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, n)))


class TestScanOnce(WatchSandbox):
    def test_scored_and_failed_files_are_filed_away(self):
        self.drop("a.csv", "b.csv", "notes.txt")
        stub = StubPipeline(fail={"b.csv"})
        self.assertEqual(scan_once(self.dir, stub), 1)
        self.assertEqual(stub.scored, ["a.csv"])
        self.assertEqual(self.listing("processed"), ["a.csv"])
        self.assertEqual(self.listing("failed"), ["b.csv"])
        self.assertEqual(self.listing(), ["notes.txt"])          # not an input file: untouched
        self.assertEqual(self.listing(CLAIM_DIR), [])

    def test_file_taken_by_another_watcher_before_the_claim_is_skipped(self):
        self.drop("a.csv")

        def other_watcher_wins(path):
            os.remove(path)                                        # gone before our claim
            return True

        with mock.patch.object(fe9, "_file_is_stable", other_watcher_wins):
            stub = StubPipeline()
            self.assertEqual(scan_once(self.dir, stub), 0)
        self.assertEqual(stub.scored, [])

    def test_file_vanishing_after_scoring_does_not_stop_the_detector(self):
        """The exact failure from the live run: scored, then the move found no file."""
        self.drop("a.csv")
        stub = StubPipeline()
        original = stub.process_file

        def score_then_lose_it(path):
            original(path)
            os.remove(path)

        stub.process_file = score_then_lose_it
        self.assertEqual(scan_once(self.dir, stub), 1)             # warns, does not raise

    def test_two_watchers_on_one_folder_score_each_file_exactly_once(self):
        names = [f"batch{i:04d}.csv" for i in range(1, 9)]
        self.drop(*names)
        a, b = StubPipeline(), StubPipeline()
        original_replace = os.replace
        turn = {"n": 0}

        def interleave(src, dst):
            # before each claim by one watcher, let the other scan (and claim) first
            if dst.endswith(os.path.join(CLAIM_DIR, os.path.basename(src))) and turn["n"] == 0:
                turn["n"] += 1
                scan_once(self.dir, b)
            return original_replace(src, dst)

        with mock.patch.object(pipeline.os, "replace", interleave):
            scan_once(self.dir, a)
        scan_once(self.dir, a)
        self.assertEqual(sorted(a.scored + b.scored), names)
        self.assertEqual(len(a.scored + b.scored), len(set(a.scored + b.scored)))
        self.assertEqual(self.listing("processed"), names)


class TestRestartAndReset(WatchSandbox):
    def test_file_interrupted_mid_scoring_is_rescored_on_restart(self):
        self.drop("a.csv")
        watch(self.dir, StubPipeline(interrupt=True), poll_seconds=0)   # Ctrl+C while scoring a.csv
        self.assertEqual(self.listing(CLAIM_DIR), ["a.csv"])
        stub = StubPipeline()
        with mock.patch.object(pipeline.time, "sleep", side_effect=KeyboardInterrupt):
            watch(self.dir, stub, poll_seconds=0)
        self.assertEqual(stub.scored, ["a.csv"])
        self.assertEqual(self.listing("processed"), ["a.csv"])

    def test_reset_clears_unscored_and_claimed_files(self):
        self.drop("a.csv")
        open(os.path.join(self.dir, CLAIM_DIR, "b.csv"), "w").close()
        open(os.path.join(self.dir, "processed", "old.csv"), "w").close()
        self.assertEqual(clear_unscored_feed(self.dir), 2)
        self.assertEqual(self.listing(), [])
        self.assertEqual(self.listing(CLAIM_DIR), [])
        self.assertEqual(self.listing("processed"), ["old.csv"])


if __name__ == "__main__":
    unittest.main()
