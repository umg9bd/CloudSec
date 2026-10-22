"""
ingestion_gate.py
=================
A benign/malicious gate in front of the detector's stores.

For every incoming CloudTrail event it:
  1. VALIDATES the log (well-formed, has the fields the detector needs);
  2. CLASSIFIES it with the existing detector (HGT ensemble + LSTM -> risk),
     benign when risk is below the alert threshold and no fast-lane rule fired,
     malicious otherwise;
  3. if BENIGN:
       - stores the edge in the Privilege Propagation Graph (Neo4j write-through
         when a database is reachable, always also to a local JSONL so nothing is
         lost when Neo4j is down), and
       - appends the event to a per-principal temporal-history store the LSTM can
         warm-start from;
  4. if MALICIOUS:
       - does NOT pollute the clean baseline stores; instead generates a REVISED
         SECURITY TEMPLATE for the action (via the Stage 2 hardening service) and
         writes it out for review.

Nothing in the existing pipeline is modified. This reuses pipeline.Pipeline for
scoring, graph_construction.IncrementalGraphUpdater for the graph store, and
prompt_hardening_realtime (Stage 2) for the revised template. Each downstream
piece fails soft: a missing Neo4j, LSTM, or Stage 2 never drops an event on the
floor or crashes the gate -- it degrades and is reported in the summary.

Run (uses the repo's own interpreter/.venv, where torch/pandas live):
    python ingestion_gate.py --files incoming/real_dataset_test_batch0082.csv
    python ingestion_gate.py --watch incoming
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass, field

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "graph_construction"), os.path.join(ROOT, "temporal-analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Lightweight (no torch): safe to import at module load, so validate()/the store
# writers are importable and testable even where the model stack is absent.
import feature_engine9 as fe9                              # noqa: E402
import privilege_features as pf                            # noqa: E402
import incremental_updater as iu                           # noqa: E402
# pipeline (torch + the HGT/LSTM checkpoints) is imported lazily, only when a gate
# is actually constructed -- see IngestionGate.__init__ and main().
INPUT_SUFFIXES = (".csv", ".csv.gz", ".json", ".jsonl", ".ndjson", ".json.gz", ".jsonl.gz", ".ndjson.gz")

_SAFE = re.compile(r"[^A-Za-z0-9._-]")

# Neo4j connection the graph builder already uses (graph_construction/neo4j_graph_builder.py).
NEO4J_URI = os.environ.get("CLOUDSEC_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("CLOUDSEC_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("CLOUDSEC_NEO4J_PASSWORD", "test1234")

STAGE2_ROOT = os.path.join(ROOT, "prompt_hardening_realtime")


def _neo4j_session_factory():
    """A zero-arg `session` factory for IncrementalGraphUpdater, or None when Neo4j
    is not reachable (so the gate runs on the local JSONL store alone). A short socket
    pre-check keeps a down database from hanging the gate on connect."""
    host, port = NEO4J_URI.split("://")[-1].split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=1.5):
            pass
    except OSError:
        return None, None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver.session, driver
    except Exception as exc:          # driver present but auth/handshake failed
        print(f"[gate] Neo4j present but unusable ({type(exc).__name__}); using the file store.", flush=True)
        return None, None


@dataclass
class GateSummary:
    source: str
    validated: int = 0
    rejected: int = 0
    benign: int = 0
    malicious: int = 0
    neo4j_synced: int = 0
    templates_written: int = 0
    reasons: list = field(default_factory=list)


class IngestionGate:
    def __init__(self, store_dir: str = "gate_store", cfg=None):
        self.store_dir = store_dir if os.path.isabs(store_dir) else os.path.join(ROOT, store_dir)
        self.templates_dir = os.path.join(self.store_dir, "revised_templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        self.graph_edges_file = os.path.join(self.store_dir, "benign_graph_edges.jsonl")
        self.lstm_history_file = os.path.join(self.store_dir, "lstm_history.jsonl")

        # Scoring reuses the real detector; write_outputs=False so it never emails
        # or appends to the production risk_scores.csv -- the gate owns its stores.
        import pipeline as pl                               # torch + checkpoints; lazy
        self.pipeline = pl.Pipeline(cfg, write_outputs=False)

        # Benign-only Privilege Propagation Graph, with Neo4j write-through if up.
        self.neo4j_factory, self._driver = _neo4j_session_factory()
        self.neo4j_live = self.neo4j_factory is not None
        resolver = pf.ActionAccessLevelResolver()
        self.updater = iu.IncrementalGraphUpdater(
            pf.PrivilegePropagationGraph(resolver), iu.IncrementalUpdateConfig(),
            resolver=resolver, neo4j_session_factory=self.neo4j_factory)

        self._stage2 = None
        self._stage2_tried = False
        print(f"[gate] stores -> {os.path.relpath(self.store_dir, ROOT)}/ | "
              f"Neo4j: {'live at ' + NEO4J_URI if self.neo4j_live else 'offline (file store only)'}", flush=True)

    # ── 1. validation ────────────────────────────────────────────────────────
    @staticmethod
    def validate(row: dict) -> tuple[bool, str]:
        """A log is processable when it is an object carrying the detector's minimal
        fields: an event name, a timestamp, and some principal."""
        if not isinstance(row, dict):
            return False, "not a JSON object"
        if not (row.get("event_name") or row.get("eventName")):
            return False, "missing event_name"
        if not (row.get("timestamp") or row.get("eventTime")):
            return False, "missing timestamp"
        if not (row.get("principal_arn") or row.get("username") or row.get("userIdentity")):
            return False, "missing principal"
        return True, ""

    # ── 2/3/4. classify and route ──────────────────────────────────────────────
    def _store_benign(self, r) -> str:
        """Graph (Neo4j + JSONL) and LSTM temporal history. Returns the Neo4j sync status."""
        event = iu.CloudTrailEvent(log_id=str(r.log_id), source_node=r.source_node,
                                   target_node=r.target_node, edge_type=r.edge_type, label=0)
        res = self.updater.apply_event(event)
        with open(self.graph_edges_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "log_id": str(r.log_id), "source_node": r.source_node, "target_node": r.target_node,
                "edge_type": r.edge_type, "relation": res.relation, "risk_score": float(r.risk_score),
                "neo4j": res.sync_status.value,
            }) + "\n")
        rec = {"log_id": str(r.log_id), "username": r.username, "principal_arn": r.principal_arn,
               "timestamp": str(r.timestamp), "event_name": r.event_name, "risk_score": float(r.risk_score),
               "label": 0}
        rec.update({c: getattr(r, c) for c in fe9.TEMPORAL_COLS if hasattr(r, c)})
        with open(self.lstm_history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return res.sync_status.value

    def _revised_template(self, r) -> dict | None:
        """Stage 2 hardening for a malicious action: produce a tightened security template."""
        svc = self._stage2_service()
        if svc is None:
            return None
        principal = r.principal_arn or r.username or "an unknown principal"
        task = (f"An AWS principal ({principal}) performed the action {r.event_name} "
                f"on {r.target_node}. Write the security requirements and least-privilege "
                f"controls that must constrain this action so it cannot be used for privilege "
                f"escalation or unauthorized access.")
        try:
            return svc.handle({"request_id": f"gate-{_SAFE.sub('-', str(r.log_id))}",
                               "screening_status": "SAFE", "original_prompt": task})
        except Exception:
            return None

    def _store_malicious_template(self, r, response: dict) -> str:
        name = _SAFE.sub("-", str(r.log_id))[:150] + ".json"
        path = os.path.join(self.templates_dir, name)
        payload = {
            "log_id": str(r.log_id), "event_name": r.event_name, "principal_arn": r.principal_arn,
            "target_node": r.target_node, "risk_score": float(r.risk_score),
            "p_graph": None if pd.isna(r.p_graph) else float(r.p_graph), "p_sequence": float(r.p_sequence),
            "fast_lane": bool(r.fast_lane), "classification": "MALICIOUS",
            "revised_security_template": {
                "status": response.get("status"),
                "revised_prompt": response.get("revised_prompt"),
                "security_gaps": response.get("security_gaps", []),
                "added_controls": [{"control_id": c.get("control_id"), "clause": c.get("clause"),
                                    "verification_expectation": c.get("verification_expectation")}
                                   for c in response.get("added_controls", [])],
            } if response else {"status": "UNAVAILABLE",
                                "note": "Stage 2 hardening not importable in this environment"},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    def _stage2_service(self):
        if not self._stage2_tried:
            self._stage2_tried = True
            try:
                if STAGE2_ROOT not in sys.path:
                    sys.path.insert(0, STAGE2_ROOT)
                from security_hardening.integration import HardeningService  # type: ignore
                self._stage2 = HardeningService()
            except Exception as exc:
                print(f"[gate] Stage 2 hardening unavailable ({type(exc).__name__}); "
                      f"malicious events will be recorded without a revised template.", flush=True)
                self._stage2 = None
        return self._stage2

    # ── driver ────────────────────────────────────────────────────────────────
    def process_rows(self, raw_rows, source_name: str) -> GateSummary:
        s = GateSummary(source=source_name)
        valid = []
        for idx, row in raw_rows:
            ok, why = self.validate(row)
            if ok:
                valid.append((idx, row))
            else:
                s.rejected += 1
                s.reasons.append(f"row {idx}: {why}")
        s.validated = len(valid)
        if not valid:
            return s

        new = self.pipeline.featurize(valid, source_name)   # structural + temporal features
        if new.empty:
            return s
        scored = self.pipeline.score(new)                   # p_graph, p_sequence, risk, alert, fast_lane
        scored["malicious"] = scored["alert"] | scored["fast_lane"]

        for r in scored.itertuples():
            if r.malicious:
                s.malicious += 1
                resp = self._revised_template(r)
                self._store_malicious_template(r, resp)
                if resp is not None:
                    s.templates_written += 1
            else:
                s.benign += 1
                if self._store_benign(r) == "synced":
                    s.neo4j_synced += 1
        return s

    def process_file(self, path: str) -> GateSummary:
        name = os.path.basename(path)
        t0 = time.time()
        s = self.process_rows(enumerate(fe9.iter_input_rows(path)), name)
        print(f"[gate] {name}: {s.validated} validated ({s.rejected} rejected) | "
              f"{s.benign} benign -> stores ({s.neo4j_synced} to Neo4j) | "
              f"{s.malicious} malicious -> {s.templates_written} revised template(s) "
              f"[{time.time() - t0:.1f}s]", flush=True)
        return s

    def close(self):
        if self._driver is not None:
            self._driver.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", nargs="+", metavar="FILE", help="process these CloudTrail files once")
    src.add_argument("--watch", metavar="DIR", help="process every file now in DIR (one-shot)")
    ap.add_argument("--store-dir", default="gate_store")
    args = ap.parse_args()

    gate = IngestionGate(store_dir=args.store_dir)
    try:
        files = args.files or [os.path.join(args.watch, n) for n in sorted(os.listdir(args.watch))
                               if n.endswith(INPUT_SUFFIXES) and os.path.isfile(os.path.join(args.watch, n))]
        for f in files:
            gate.process_file(f)
    finally:
        gate.close()


if __name__ == "__main__":
    main()
