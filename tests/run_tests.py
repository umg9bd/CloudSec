"""
run_tests.py
============
One command to run the project's test suites. Run from the repo root:

    python tests/run_tests.py            # fast suites only (~2s, no external deps)
    python tests/run_tests.py --all      # adds the slow Neo4j-dependent suite (~10 min)

WHY TWO TIERS: the fast suites need nothing but the Python environment -- no
Neo4j, no Docker, no trained checkpoint -- so they can run on every commit and
in CI. test_incremental_updater.py is different: it reads
datasets/privilege-escalation/cloudtrail_structural.csv, builds the whole graph
twice (batch and streaming) and compares them, which takes ~10 minutes. It is
opt-in so the fast feedback loop stays fast.

The ensemble is covered: it lives in pipeline.py (HGT + LSTM scores combined
per event), and test_pipeline.py checks it.

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

# This file lives in tests/, one level below the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The repo root (feature_engine9, leakage_guard), graph_construction/
# (data_loader, model_gat, ...) and temporal-analysis/ (train_lstm_transformer,
# used by the pipeline tests) all need to be importable: several modules import
# each other as top-level names rather than as package members. Adding
# temporal-analysis here lets the suite run natively, not only in the Docker image.
for path in (ROOT, os.path.join(ROOT, "graph_construction"), os.path.join(ROOT, "temporal-analysis")):
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
    "test_model_hgt",             # HGT output shape/order contract
    "test_hgt_attention_dropout", # attn_dropout really applied on PyG without the kwarg
    "test_neighbor_sampling",     # relation-aware sampler + sampled training view (GNN-final)
    "test_ensemble",              # HGT/SAGE/GAT ensemble coverage fallback (GNN-final)
    "test_explainability",        # feature names match edge_attr columns; explains the right edge
    "test_infer_checkpoint",      # sage/gat/hgt/ensemble checkpoints load back identically
    "test_sampled_training",      # --sampling trains on the sampled view, evaluates on the full graph
    "test_node_importance",       # importance ranks on the right columns; ties share a rank
    "test_loader_metrics_feeder", # load_offline == OfflineGraphLoader, AUPR, demo feeder
    # ── real-time pipeline ───────────────────────────────────────────────
    "test_pipeline",              # streaming == batch (LSTM), fallback, alert schema
    "test_watch_folder",          # one watcher per file, no crash when a file vanishes
    "test_ingestion_gate",        # benign/malicious gate: validate, route to Neo4j/LSTM stores vs template
    "test_agent_orchestrator",    # agent gate: locked security section, execution == approved plan
    "test_ensemble_explain",      # per-alert HGT + LSTM explanation: exact model shares, reasons
    "test_notify",                # email alerts: reach the admin; never break detection when misconfigured
    "test_lstm_scorer",           # v5/v6 checkpoints load with the right vocab; pipeline can feed both
    "test_lstm_v6_3",             # v6.3 scripts' paths resolve; v6.3 / v6.3-ft plug into the pipeline
    "test_lstm_explain",          # LSTM alert explanations (Nandan)
    "test_cloudtrail_input",      # every accepted input format; raw-JSON identity + target parsing
    # ── shared: the train/eval boundary both tracks must respect ─────────
    "test_leakage_guard",         # held-out detection, label-derived prior freeze
    "test_causal_features",       # temporal leakage: causal ordering + frozen-on-eval prior
    "test_feature_engine_cache",  # priors fit on train rows only; stale cached outputs refused
    "test_family_split",          # campaign-family holdout: no family or session crosses splits
    "test_campaign_split",        # one family assignment for both tracks; priors fit train families only
    # ── feature semantics ────────────────────────────────────────────────
    "test_identity_features",     # ground truth never a feature; handoffs, permission deltas
    "test_lineage_verification",  # inferred lineage == generator hop_id; dataset link checks
    "test_policy_features",       # policy-document features the LSTM consumes
    "test_iam_permissions",       # permission state from observed policy content
    "test_attack_taxonomy",       # tactic vs technique, ATT&CK-consistent pairs
]

SLOW_SUITES = [
    "test_incremental_updater",   # batch vs. streaming equivalence (~10 min)
]

# The 2026-10-22 test-coverage audit (opt-in: --coverage). They pin open defects as
# expectedFailure ("KNOWN DEFECT:" in the docstring), so a run stays green until one is fixed --
# then the test reports "unexpected success" and the marker should be removed.
COVERAGE_SUITES = [
    "test_full_pipeline_integration",  # CloudTrail -> graph/LSTM -> ensemble -> alert; Stage 1 -> 2 -> 3
    "test_system_e2e",                 # one watch-folder system run; malformed input; rejected prompts never planned
    "test_security_cross_component",   # no screening bypass, fail-closed, no forged approval, no secrets in logs
    "test_ensemble_evasion",           # padding / UA / timing / region perturbations vs the alert threshold
    "test_prompt_dataset_split",       # no exact/near-duplicate prompts or shared templates across splits
]

# --prompt: the Stage 1/2/3 packages keep their own runners (pytest from inside the tests dir,
# the hardening package's run_tests.py), so they run as subprocesses here, one report each.
PROMPT_RUNS = [
    ("prompt_injection_gate", [sys.executable, "-m", "pytest", "-q"], os.path.join(ROOT, "prompt_injection_gate", "tests")),
    ("prompt_hardening_realtime", [sys.executable, os.path.join("tests", "run_tests.py")],
     os.path.join(ROOT, "prompt_hardening_realtime")),
    ("log-generation", [sys.executable, "-m", "pytest", "-q", os.path.join("tests", "test_stage3.py")],
     os.path.join(ROOT, "log-generation")),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true",
                   help="also run the slow Neo4j/CSV-dependent equivalence suite")
    p.add_argument("--coverage", action="store_true", help="also run the audit coverage suites (~5 min)")
    p.add_argument("--prompt", action="store_true",
                   help="also run the prompt-injection gate, prompt-hardening and log-generation suites")
    p.add_argument("-v", "--verbose", action="store_true", help="per-test output")
    args = p.parse_args()

    names = list(FAST_SUITES) + (list(SLOW_SUITES) if args.all else [])
    names += list(COVERAGE_SUITES) if args.coverage else []
    if not args.all:
        print("Running fast suites only. Add --all for the batch/streaming "
              "equivalence suite (~10 min, 3 known failures -- see section 6.9).\n")

    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromNames(names)
    )
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    ok = result.wasSuccessful()
    if args.prompt:
        import subprocess
        for name, cmd, cwd in PROMPT_RUNS:
            print(f"\n=== {name} ===", flush=True)
            rc = subprocess.run(cmd, cwd=cwd).returncode
            print(f"=== {name}: {'OK' if rc == 0 else f'FAILED (exit {rc})'} ===", flush=True)
            ok = ok and rc == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
