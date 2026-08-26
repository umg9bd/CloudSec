"""
run_tests.py
============
One command to run the project's test suites.

    python run_tests.py            # fast suites only (~2s, no external deps)
    python run_tests.py --all      # adds the slow Neo4j-dependent suite (~10 min)

WHY TWO TIERS: the fast suites need nothing but the Python environment -- no
Neo4j, no Docker, no trained checkpoint -- so they can run on every commit and
in CI. test_incremental_updater.py is different: it reads
datasets/privilege-escalation/cloudtrail_structural.csv, builds the whole graph
twice (batch and streaming) and compares them, which takes ~10 minutes. It is
opt-in so the fast feedback loop stays fast.

WHAT IS NOT COVERED HERE: ensemble.py. It combines a blast-radius score with
the LSTM score keyed by username, never reads the GNN's attack probability, and
computes no metric against ground truth -- so there is nothing to assert about
it yet. It is out of scope until it is an ensemble of the two detectors and is
scored like one.

KNOWN FAILURES in the slow suite (3 of 13, reproducible, documented in
PROJECT_STATUS_REPORT.md section 6.9): the batch and streaming pipelines do not
currently produce identical graphs. These do NOT affect any reported result --
the evaluation path never imports incremental_updater -- but they do mean the
streaming path is not trustworthy yet. Do not "fix" them by deleting the
assertions.

Sets PYTHONPATH itself so cross-module imports resolve from any working
directory, which is otherwise a recurring setup papercut on Windows.
"""

import argparse
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))

# graph_construction/ and the repo root both need to be importable: several
# modules import `neo4j_graph_builder` and `privilege_features` as top-level
# names rather than as package members.
for path in (ROOT, os.path.join(ROOT, "graph_construction")):
    if path not in sys.path:
        sys.path.insert(0, path)

# Windows consoles default to cp1252, which crashes on the Unicode in several
# modules' print statements -- on an otherwise-passing run.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Grouped by what they protect, not by file. Both tracks train on synthetic and
# are evaluated on held-out real data; the suites below cover the places where
# that discipline has actually been broken before.
FAST_SUITES = [
    # ── graph track (GNN) ────────────────────────────────────────────────
    "test_data_loader",           # feature construction, scaler discipline, ordering
    "test_models",                # logit/label alignment contract
    "test_evaluation_integrity",  # edge->session join, guards, paired baseline
    # ── shared: the train/eval boundary both tracks must respect ─────────
    "test_leakage_guard",         # held-out detection, label-derived prior freeze
]

SLOW_SUITES = [
    "test_incremental_updater",   # batch vs. streaming equivalence (~10 min)
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true",
                   help="also run the slow Neo4j/CSV-dependent equivalence suite")
    p.add_argument("-v", "--verbose", action="store_true", help="per-test output")
    args = p.parse_args()

    names = list(FAST_SUITES) + (list(SLOW_SUITES) if args.all else [])
    if not args.all:
        print("Running fast suites only. Add --all for the batch/streaming "
              "equivalence suite (~10 min, 3 known failures -- see section 6.9).\n")

    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromNames(names)
    )
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
