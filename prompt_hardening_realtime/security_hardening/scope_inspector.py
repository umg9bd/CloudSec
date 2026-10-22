"""
scope_inspector.py
==================
READ-ONLY inspection of the project's EXISTING dataset and of the cloud-event
scope the pipeline actually supports.

WHY THIS EXISTS
---------------
The prompt-hardening module must stay inside the services the project really
handles. "Which services are those?" has two authoritative answers in this
repository, and this file reads both instead of guessing:

  1. The pipeline's own vocabulary: `FIXED_EVENT_SOURCES` in feature_engine9.py.
     Any event source outside it is mapped to <UNK> by the feature engine.
  2. The data: the `event_source` / `event_name` / `label` columns of the CSVs
     under datasets/privilege-escalation/.

Nothing is imported from the pipeline (no torch, no pandas): feature_engine9.py
is parsed with `ast`, and CSVs are streamed with the `csv` module. Every file is
opened read-only. The report it writes is a *description* of the dataset, never a
copy of it.

    python -m security_hardening.scope_inspector                    # print summary
    python -m security_hardening.scope_inspector --write            # refresh data/dataset_scope_report.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = "datasets/privilege-escalation"
REPORT_PATH = Path(__file__).resolve().parent / "data" / "dataset_scope_report.json"

# (file, role). Roles follow README.md ("trained on synthetic, validated on real")
# and the dev/test discipline described there.
PROFILED_DATASETS = (
    ("real_dataset_combined.csv", "real labelled CloudTrail (Stratus Red Team detonations + Invictus 2023), dev + test combined"),
    ("real_dataset_dev.csv", "real labelled CloudTrail, dev split (thresholds/weights tuned here)"),
    ("real_dataset_test.csv", "real labelled CloudTrail, held-out test split (also the demo feed)"),
    ("synthetic_cloudtrail.csv", "synthetic CloudTrail used for training (generate_synthetic_data.py)"),
)
DERIVED_DATASETS = (
    "cloudtrail_structural.csv", "cloudtrail_temporal.csv",
    "real_dataset_dev_structural.csv", "real_dataset_dev_temporal.csv",
    "real_dataset_test_structural.csv", "real_dataset_test_temporal.csv",
    "synthetic_campaign_annotations.csv",
)
_TRUE_LABELS = {"1", "1.0", "true", "True"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_assignments(source: str):
    """Yield (name, value_node) for simple top-level `NAME = value` assignments."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            yield node.targets[0].id, node.value


def read_pipeline_vocabulary(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """FIXED_EVENT_SOURCES and the fast-lane CRITICAL_ACTIONS, parsed (not imported) from feature_engine9.py."""
    path = repo_root / "feature_engine9.py"
    strings: Dict[str, str] = {}
    sources: List[str] = []
    fast_lane: Dict[str, str] = {}
    for name, value in _module_assignments(path.read_text(encoding="utf-8")):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            strings[name] = value.value
        elif name == "FIXED_EVENT_SOURCES" and isinstance(value, (ast.List, ast.Tuple)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    sources.append(elt.value)
                elif isinstance(elt, ast.Name):
                    sources.append(strings.get(elt.id, f"<{elt.id}>"))
        elif name == "CRITICAL_ACTIONS" and isinstance(value, ast.Dict):
            for k, v in zip(value.keys, value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    fast_lane[str(k.value)] = str(v.value)
    unk = strings.get("UNK_TOKEN")
    return {
        "file": "feature_engine9.py",
        "symbol": "FIXED_EVENT_SOURCES",
        "unk_token": unk,
        "event_sources": [s for s in sources if s != unk],
        "fast_lane_actions": fast_lane,
        "note": "Event sources outside this list are mapped to the <UNK> token by the feature engine.",
    }


def service_key(event_source: str) -> str:
    return (event_source or "").split(".")[0]


def profile_csv(path: Path, vocabulary_sources: set, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Stream one dataset CSV and summarise it. Opens the file read-only."""
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    rows = 0
    labels, tactics, principals, collectors, regions = Counter(), Counter(), Counter(), Counter(), Counter()
    src_rows, src_attack = Counter(), Counter()
    src_events: Dict[str, Counter] = defaultdict(Counter)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            src = row.get("event_source") or ""
            label = (row.get("label") or "").strip()
            labels[label] += 1
            src_rows[src] += 1
            src_events[src][row.get("event_name") or ""] += 1
            principals[row.get("principal_type") or ""] += 1
            regions[row.get("aws_region") or ""] += 1
            if "data_source" in row:
                collectors[row.get("data_source") or ""] += 1
            if label in _TRUE_LABELS:
                src_attack[src] += 1
                tactics[row.get("attack_technique") or ""] += 1
    sources: Dict[str, Any] = {}
    for src, count in src_rows.most_common():
        entry: Dict[str, Any] = {
            "rows": count,
            "attack_labelled_rows": src_attack.get(src, 0),
            "distinct_event_names": len(src_events[src]),
            "in_pipeline_vocabulary": src in vocabulary_sources,
        }
        if src in vocabulary_sources:  # full event list only for the services the pipeline encodes
            entry["event_names"] = dict(sorted(src_events[src].items(), key=lambda kv: (-kv[1], kv[0])))
        sources[src] = entry
    return {
        "path": str(path.relative_to(repo_root)).replace("\\", "/"),
        "format": "csv (utf-8, header row, one CloudTrail event per row)",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "columns": columns,
        "label_counts": dict(labels),
        "attack_tactic_counts_for_label_1": dict(tactics.most_common()),
        "principal_type_counts": dict(principals.most_common()),
        "collector_counts": dict(collectors.most_common()),
        "region_counts": dict(regions.most_common()),
        "distinct_event_sources": len(src_rows),
        "distinct_event_names": len({n for c in src_events.values() for n in c}),
        "event_sources": sources,
    }


def inspect_repository(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    vocab = read_pipeline_vocabulary(repo_root)
    vocab_set = set(vocab["event_sources"])
    ds_root = repo_root / DATASET_DIR
    if not ds_root.is_dir():
        raise FileNotFoundError(
            f"{DATASET_DIR} not found under {repo_root}. Search the repository before substituting anything: "
            "this module must not use another dataset."
        )
    profiled = []
    for name, role in PROFILED_DATASETS:
        path = ds_root / name
        if path.is_file():
            profile = profile_csv(path, vocab_set, repo_root)
            profile["role"] = role
            profiled.append(profile)
    derived = []
    for name in DERIVED_DATASETS:
        path = ds_root / name
        if path.is_file():
            with path.open("r", encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh), [])
            derived.append({"path": f"{DATASET_DIR}/{name}", "bytes": path.stat().st_size, "columns": header})
    raw_dir = ds_root / "stratus_own_runs" / "CloudTrail"
    raw_files = sum(1 for p in raw_dir.glob("*.json")) if raw_dir.is_dir() else 0

    seen_real = Counter()
    seen_synth = Counter()
    for p in profiled:
        bucket = seen_synth if p["path"].endswith("synthetic_cloudtrail.csv") else seen_real
        if p["path"].endswith("real_dataset_combined.csv") or bucket is seen_synth:
            for src, e in p["event_sources"].items():
                bucket[src] += e["rows"]
    summary = {
        "pipeline_vocabulary_services": [service_key(s) for s in vocab["event_sources"]],
        "vocabulary_services_present_in_real_data": sorted(service_key(s) for s in vocab_set if seen_real.get(s)),
        "vocabulary_services_present_in_synthetic_data": sorted(service_key(s) for s in vocab_set if seen_synth.get(s)),
        "services_in_data_but_outside_pipeline_vocabulary": {
            "real_combined_distinct": sum(1 for s in seen_real if s not in vocab_set),
            "synthetic": sorted(service_key(s) for s in seen_synth if s not in vocab_set),
        },
    }
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "security_hardening/scope_inspector.py (read-only)",
        "dataset_dir": DATASET_DIR,
        "pipeline_vocabulary": vocab,
        "datasets": profiled,
        "derived_tables": derived,
        "raw_cloudtrail_json_files": {"path": f"{DATASET_DIR}/stratus_own_runs/CloudTrail", "files": raw_files,
                                      "note": "gitignored in this repo; present only in working copies that collected them"},
        "summary": summary,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    ap.add_argument("--write", action="store_true", help=f"write {REPORT_PATH.relative_to(REPO_ROOT)}")
    args = ap.parse_args(argv)
    report = inspect_repository(Path(args.repo_root))
    if args.write:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=1, sort_keys=False) + "\n", encoding="utf-8")
        print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    vocab = report["pipeline_vocabulary"]["event_sources"]
    print(f"pipeline vocabulary ({len(vocab)}): {', '.join(service_key(s) for s in vocab)}")
    for d in report["datasets"]:
        print(f"{d['path']}: {d['rows']} rows, {len(d['columns'])} cols, labels={d['label_counts']}, sources={d['distinct_event_sources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
