"""
test_full_pipeline_integration.py
=================================
Integration across the real component boundaries, with the real modules on both sides of each:

  (i)   CloudTrail rows -> feature_engine9 -> graph (gnn_scorer, 3-seed HGT) -> LSTM -> ensemble
        (pipeline.Pipeline.process_rows on a slice of the committed real_dataset_dev.csv, outputs
        written to a temp dir) -> the alert JSON schema the dashboard and notify.py read;
  (ii)  graph + sequence scores -> ensemble -> alerts: risk == w*p_graph + (1-w)*p_seq, an alert
        iff risk >= threshold, and only those groups reach the notifier;
  (iii) prompt screening (Stage 1) -> hardening (Stage 2) -> planning (Stage 3) -> dry-run scoring
        (Stage 4), through the real pipeline bridge and agent orchestrator;
  (iv)  model decision -> explanation -> dashboard CSV + alert files + audit trail;
  and the parent-pipeline integration that prompt_hardening_realtime's own suite skips
  (test_parent_pipeline_integration): Stage 2 driven through the parent repo's interfaces.

What is NOT real: the Stage-1 fine-tuned classifier checkpoint is not in this branch. Where a SAFE
screening is needed to reach Stage 2, a stub classifier that reports SAFE stands in for it; the
rule engine, normaliser, decision combiner and SAFE-only handoff schema stay real. The notifier is
replaced by a recorder so no email can be sent.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "graph_construction"), os.path.join(ROOT, "temporal-analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent_orchestrator as ao                                   # noqa: E402  (also puts Stage 2/3 on sys.path)
import feature_engine9 as fe9                                     # noqa: E402
import pipeline as pl                                             # noqa: E402
from prompt_injection_gate.classifier_interface import ClassifierResult  # noqa: E402
from prompt_injection_gate.config import GateConfig               # noqa: E402
from prompt_injection_gate.pipeline_bridge import PipelineBridge  # noqa: E402
from prompt_injection_gate.schemas import Decision, ScreeningRequest  # noqa: E402
from prompt_injection_gate.screening_service import ScreeningService  # noqa: E402
from security_hardening import integration as s2                  # noqa: E402
from security_hardening.models import PROCEED_STATUSES            # noqa: E402
from security_hardening.prompt_hardener import BLOCK_BEGIN, BLOCK_END  # noqa: E402
from security_hardening.schema_validation import validate_request, validate_response  # noqa: E402

DEV = os.path.join(ROOT, "datasets", "privilege-escalation", "real_dataset_dev.csv")
ATTACK = os.path.join(ROOT, "samples", "cloudtrail", "synthetic_attack_chain.json")
N_EVENTS, CHUNK = 400, 100
BENIGN_PROMPT = "Create a private S3 bucket named audit-logs-prod with versioning."
ALERT_KEYS = {"alert_id", "source_file", "principal", "principal_arn", "first_seen", "last_seen",
              "max_risk_score", "n_flagged_events", "fast_lane_reasons", "threshold", "events", "explanations"}
ALERT_EVENT_KEYS = {"log_id", "timestamp", "event_name", "target", "p_graph", "p_sequence", "risk_score"}
SMTP_ENV = ("CLOUDSEC_SMTP_USER", "CLOUDSEC_SMTP_PASSWORD", "CLOUDSEC_ADMIN_EMAIL")


class SafeStubClassifier:
    """Stands in for the fine-tuned Stage-1 checkpoint (absent from this branch): always SAFE.
    Only used where a test needs a prompt to reach Stage 2; rules + combiner still decide."""
    available, version = True, "stub-safe"

    def classify(self, text):
        return ClassifierResult("ok", Decision.SAFE, 0.01, model_version=self.version)


class SpyNotifier:
    """Records what the pipeline would email; never touches the network."""
    def __init__(self):
        self.alerts = []

    def submit(self, alert):
        self.alerts.append(alert)

    def flush(self, timeout=0):
        pass


def sandbox_pipeline(tmp, write_outputs=True, **overrides):
    """The deployed pipeline_config.json (checkpoints, w, threshold), state/outputs in tmp."""
    cfg = pl.PipelineConfig.load()
    cfg.state_dir = os.path.join(tmp, "state")
    cfg.alert_dir = os.path.join(tmp, "alerts")
    cfg.output_csv = os.path.join(tmp, "output", "risk_scores.csv")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return pl.Pipeline(cfg, write_outputs=write_outputs)


def sandbox_gate(tmp, pipe):
    """An AgentGate whose every folder (Stage 2 inbox/outbox, pipeline incoming, audit log) is in
    tmp, whose Stage-4 scorer is `pipe`, and which does not wait for a Stage 2 watcher.
    Returns (gate, patches) -- stop the patches when done."""
    tmp = Path(tmp)
    patches = [mock.patch.object(ao, name, tmp / sub) for name, sub in (
        ("STAGE2_INCOMING", "s2_incoming"), ("STAGE2_OUTPUT", "s2_output"), ("PIPELINE_INCOMING", "incoming"),
        ("PIPELINE_STAGING", ".incoming_staging"), ("AUDIT_LOG", "agent_audit.jsonl"))]
    for p in patches:
        p.start()
    gate = ao.AgentGate(stage2_wait=0)
    scorer = ao.DryRunScorer.__new__(ao.DryRunScorer)        # DryRunScorer() would load the repo's runtime/
    scorer.pl, scorer.p, scorer.cfg, scorer._lock = pl, pipe, pipe.cfg, threading.Lock()
    gate._scorer = scorer
    return gate, patches


# ─────────────────────────────────────────────────────────────────────────────
# (i) + (ii) + (iv): CloudTrail slice through the real detector
# ─────────────────────────────────────────────────────────────────────────────

class DetectorSliceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = mock.patch.dict(os.environ, {k: "" for k in SMTP_ENV})
        cls.env.start()
        cls.cwd = os.getcwd()
        os.chdir(ROOT)
        cls.tmp = tempfile.mkdtemp()
        cls.pipe = sandbox_pipeline(cls.tmp)
        cls.pipe.notifier = SpyNotifier()
        rows = list(enumerate(fe9.iter_input_rows(DEV)))[:N_EVENTS]
        cls.raw_by_log_id = {}
        parts = []
        for i in range(0, len(rows), CHUNK):
            name = f"dev_chunk{i // CHUNK}.csv"
            chunk = rows[i:i + CHUNK]
            cls.raw_by_log_id.update({f"{name}:{idx}": r for idx, r in chunk})
            parts.append(cls.pipe.process_rows(chunk, name))
            if i == 0:
                cls.first_graph_rows = cls.pipe._scored_inputs["graph_rows"].copy()
                cls.first_scored = parts[0].copy()
        cls.scored = pd.concat(parts, ignore_index=True)
        cls.csv = pd.read_csv(cls.pipe.output_csv)
        cls.alerts = []
        for name in sorted(os.listdir(cls.pipe.alert_dir)):
            with open(os.path.join(cls.pipe.alert_dir, name), encoding="utf-8") as f:
                cls.alerts.append(json.load(f))

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # (i) features + graph + sequence complete
    def test_every_event_featurised_and_scored_once(self):
        self.assertEqual(len(self.scored), N_EVENTS)
        self.assertEqual(self.scored["log_id"].nunique(), N_EVENTS)
        for c in fe9.TEMPORAL_COLS + pl.STRUCT_COLS:
            self.assertIn(c, self.scored.columns)
        self.assertFalse(self.scored[fe9.TEMPORAL_COLS].isna().any().any())
        self.assertTrue(self.scored["p_sequence"].between(0, 1).all())
        g = self.scored["p_graph"].dropna()
        self.assertGreater(len(g), 0.9 * N_EVENTS)
        self.assertTrue(g.between(0, 1).all())

    def test_graph_built_from_the_batch_scores_exactly_the_events_with_p_graph(self):
        data, order = self.pipe.gnn.build_graph(self.first_graph_rows[["log_id"] + pl.STRUCT_COLS + ["label"]])
        self.assertIsNotNone(data)
        scored_ids = set(self.first_scored.loc[self.first_scored["p_graph"].notna(), "log_id"])
        self.assertEqual(set(order), scored_ids)
        for t in data.edge_types:
            if hasattr(data[t], "log_id"):
                self.assertIn(t, self.pipe.gnn.trained_triples)    # untrained triples were dropped

    def test_seed_ensemble_is_the_mean_of_its_checkpoints(self):
        self.assertIsInstance(self.pipe.gnn, pl.EnsembleGNNScorer)
        rows = self.first_graph_rows[["log_id"] + pl.STRUCT_COLS + ["label"]]
        each = [s.score(rows).set_index("log_id")["gnn_prob"] for s in self.pipe.gnn.scorers]
        expect = pd.concat(each, axis=1).mean(axis=1, skipna=True)
        got = self.first_scored.set_index("log_id")["p_graph"]
        np.testing.assert_allclose(got.loc[expect.index].to_numpy(float), expect.to_numpy(float), atol=1e-6)

    def test_alert_files_follow_the_schema(self):
        self.assertTrue(self.alerts, "the dev slice contains attack sessions; expected alerts")
        for a in self.alerts:
            self.assertEqual(set(a), ALERT_KEYS)
            self.assertEqual(a["threshold"], self.pipe.cfg.alert_threshold)
            self.assertGreaterEqual(a["n_flagged_events"], 1)
            self.assertLessEqual(len(a["events"]), 25)
            for e in a["events"]:
                self.assertEqual(set(e), ALERT_EVENT_KEYS)
            self.assertEqual(a["max_risk_score"], max(e["risk_score"] for e in a["events"]))
            self.assertLessEqual(a["first_seen"], a["last_seen"])

    # (ii) ensemble -> alert consistency
    def test_risk_is_the_weighted_ensemble_and_alert_is_risk_at_threshold(self):
        w, thr = self.pipe.cfg.weight_graph, self.pipe.cfg.alert_threshold
        s = self.scored
        expect = np.where(s["p_graph"].isna(), s["p_sequence"], w * s["p_graph"] + (1 - w) * s["p_sequence"])
        np.testing.assert_allclose(s["risk"], expect, rtol=0, atol=1e-6)   # model outputs are float32
        np.testing.assert_array_equal(s["alert"].to_numpy(), (s["risk"] >= thr).to_numpy())
        np.testing.assert_allclose(s["risk_score"], (s["risk"] * 10).round(2))

    def flagged_for(self, alert):
        flagged = self.scored[self.scored["alert"] | self.scored["fast_lane"]]
        return flagged[(flagged["username"] == alert["principal"])
                       & flagged["log_id"].str.startswith(alert["source_file"] + ":")]

    def test_alerts_cover_exactly_the_flagged_events(self):
        flagged = self.scored[self.scored["alert"] | self.scored["fast_lane"]]
        groups = {(p, src) for p, src in zip(flagged["username"], flagged["log_id"].str.split(":").str[0])}
        self.assertEqual({(a["principal"], a["source_file"]) for a in self.alerts}, groups)
        for a in self.alerts:
            mine = self.flagged_for(a)
            self.assertEqual(a["n_flagged_events"], len(mine))
            self.assertAlmostEqual(a["max_risk_score"], float(mine["risk_score"].max()))

    def test_only_groups_at_or_above_threshold_reach_the_notifier(self):
        thr = self.pipe.cfg.alert_threshold
        expect = {a["alert_id"] for a in self.alerts if self.flagged_for(a)["risk"].max() >= thr}
        self.assertEqual({a["alert_id"] for a in self.pipe.notifier.alerts}, expect)

    # (iv) decision -> explanation -> dashboard
    def test_dashboard_csv_matches_scored_events(self):
        self.assertEqual(list(self.csv.columns), pl.OUTPUT_COLS)
        self.assertEqual(len(self.csv), N_EVENTS)
        m = self.csv.merge(self.scored[["log_id", "risk_score", "alert"]], on="log_id", suffixes=("", "_s"))
        self.assertEqual(len(m), N_EVENTS)
        np.testing.assert_allclose(m["risk_score"], m["risk_score_s"])
        self.assertTrue((m["alert"].astype(str).str.lower().eq("true") == m["alert_s"]).all())

    def test_explanations_agree_with_the_scores_they_explain(self):
        csv = self.csv.set_index("log_id")
        thr, n = self.pipe.cfg.alert_threshold, self.pipe.cfg.explain_top_events
        explained = 0
        for a in self.alerts:
            self.assertLessEqual(len(a["explanations"]), n + int(self.flagged_for(a)["fast_lane"].sum()))
            for e in a["explanations"]:
                explained += 1
                self.assertIn(e["log_id"], csv.index)                  # the dashboard joins on log_id
                self.assertIn(e["log_id"], {ev["log_id"] for ev in a["events"]})
                self.assertAlmostEqual(e["risk_score"], float(csv.loc[e["log_id"], "risk_score"]), delta=0.011)
                self.assertEqual(e["threshold"], thr)
                self.assertIn(e["driven_by"], ("graph", "sequence"))
                shares = e["models"]["graph"]["share"] + e["models"]["sequence"]["share"]
                self.assertAlmostEqual(shares, 1.0, delta=2e-3)
                self.assertIn(f"alert at {thr * 10:.2f}", e["summary"])
                self.assertEqual(e["principal"], a["principal"])
        self.assertGreater(explained, 0)


# ─────────────────────────────────────────────────────────────────────────────
# (iii) + (iv): prompt -> screening -> hardening -> planning -> dry-run scoring
# ─────────────────────────────────────────────────────────────────────────────

class PromptToPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        os.chdir(ROOT)
        cls.tmp = tempfile.mkdtemp()
        cls.pipe = sandbox_pipeline(os.path.join(cls.tmp, "scorer"), write_outputs=False)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def bridge(self, classifier=None):
        inbox = tempfile.mkdtemp(dir=self.tmp)
        cfg = GateConfig(classifier_dir=os.path.join(self.tmp, "__no_checkpoint__"),
                         stage2_incoming_dir=inbox, session_memory=False)
        b = PipelineBridge(cfg)
        if classifier is not None:
            b.service.classifier = classifier
        return b, Path(inbox)

    def test_safe_prompt_flows_through_stage2_and_stage3_contracts(self):
        b, inbox = self.bridge(SafeStubClassifier())
        out = b.run_end_to_end(BENIGN_PROMPT)
        self.assertEqual(out["stage1"]["decision"], "SAFE")
        delivered = json.loads(Path(out["stage1"]["incoming_path"]).read_text(encoding="utf-8"))
        self.assertEqual(validate_request(delivered), [])
        self.assertEqual(delivered["original_prompt"], BENIGN_PROMPT)      # byte-identical handoff
        s2resp = out["stage2"]
        self.assertEqual(validate_response(s2resp), [])
        self.assertIn(s2resp["status"], {s.value for s in PROCEED_STATUSES})
        planner = s2.to_planner_payload(s2resp)
        self.assertIsNotNone(planner)
        self.assertEqual(planner["prompt"], s2resp["revised_prompt"])
        s3 = out["stage3"]
        self.assertEqual(ao._status(s3["status"]), "GENERATED")
        names = [r["eventName"] for r in s3["cloudtrail_records"]["Records"]]
        self.assertEqual(names, s3["event_sequence"])
        self.assertIn("CreateBucket", names)
        # Stage 3's two downstream formats are consumable by the detector's two branches
        parsed = [fe9.normalize_cloudtrail_row(r) for r in s3["cloudtrail_records"]["Records"]]
        self.assertEqual([r["event_name"] for r in parsed], names)
        g = self.pipe.gnn.score(pd.DataFrame(s3["gnn_structural_rows"])[["log_id"] + pl.STRUCT_COLS + ["label"]])
        self.assertEqual(len(g), len(names))

    def test_non_safe_prompt_never_reaches_stage2_or_stage3(self):
        b, inbox = self.bridge()                     # no checkpoint -> fail closed
        with mock.patch.object(b, "run_stage2_inline") as s2spy, mock.patch.object(b, "run_stage3") as s3spy:
            out = b.run_end_to_end(BENIGN_PROMPT)
        self.assertEqual(out["stage1"]["decision"], "REVIEW_REQUIRED")
        s2spy.assert_not_called()
        s3spy.assert_not_called()
        self.assertEqual(list(inbox.glob("*.json")), [])

    def test_orchestrator_scores_exactly_the_planned_events(self):
        gate, patches = sandbox_gate(os.path.join(self.tmp, "gate"), self.pipe)
        try:
            gate.bridge.service.classifier = SafeStubClassifier()
            rnd = gate.run(BENIGN_PROMPT)
            self.assertIsNone(rnd.stopped_at, rnd.blockers)
            self.assertTrue(rnd.approved)
            planned = rnd.stage3["downstream_event_sequence"]
            a = rnd.assessment
            self.assertEqual([e["event_name"] for e in a["events"]], planned)
            w, thr = a["weight_graph"], a["threshold"]
            for e in a["events"]:
                expect = e["p_sequence"] if e["p_graph"] is None else w * e["p_graph"] + (1 - w) * e["p_sequence"]
                self.assertAlmostEqual(e["risk"], expect, delta=1e-3)
                self.assertEqual(e["alert"], e["risk"] >= thr)
            self.assertEqual(rnd.fingerprint, ao.plan_fingerprint(rnd.approved_records))
            # (iv) audit trail: one record per stage, hash only, never the prompt
            log = [json.loads(l) for l in ao.AUDIT_LOG.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["event"] for r in log], ["stage1", "stage2", "stage3", "stage4"])
            self.assertEqual({r["request_id"] for r in log}, {rnd.request_id})
            self.assertEqual(log[0]["prompt_sha"], hashlib.sha256(BENIGN_PROMPT.encode()).hexdigest()[:16])
            self.assertTrue(log[3]["approved"])
            self.assertNotIn(BENIGN_PROMPT, ao.AUDIT_LOG.read_text(encoding="utf-8"))
        finally:
            for p in patches:
                p.stop()

    def test_stage1_audit_record_has_hash_not_prompt(self):
        svc = ScreeningService(GateConfig(classifier_dir=os.path.join(self.tmp, "__none__")))
        req = ScreeningRequest(prompt=BENIGN_PROMPT)
        rec = svc.audit_record(req, svc.screen(req))
        self.assertEqual(rec["prompt_sha256"], hashlib.sha256(BENIGN_PROMPT.encode()).hexdigest())
        self.assertNotIn("raw_prompt", rec)
        self.assertNotIn(BENIGN_PROMPT, json.dumps(rec))


# ─────────────────────────────────────────────────────────────────────────────
# Parent-pipeline integration (the case test_security_hardening_integrity skips)
# ─────────────────────────────────────────────────────────────────────────────

class ParentPipelineHardeningIntegrationTests(unittest.TestCase):
    """prompt_hardening_realtime/tests/test_security_hardening_integrity.py skips
    test_parent_pipeline_integration because feature_engine9.py is not in its standalone ZIP. It is
    here: Stage 2's integration module is exercised through each parent-repo consumer
    (pipeline_bridge, agent_orchestrator, ingestion_gate) and its output is scored by pipeline.py."""

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        os.chdir(ROOT)
        cls.tmp = tempfile.mkdtemp()
        cls.pipe = sandbox_pipeline(cls.tmp, write_outputs=False)
        cls.service = s2.HardeningService()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_parent_repository_is_present(self):
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "feature_engine9.py")))
        self.assertTrue(self.service.ready)
        self.assertEqual(Path(s2.__file__).resolve().parents[1], ao.STAGE2_ROOT.resolve())

    def test_orchestrator_constants_match_stage2(self):
        self.assertEqual(ao.STAGE2_PROCEED, {s.value for s in PROCEED_STATUSES})
        self.assertTrue(BLOCK_BEGIN.startswith(ao.SECURITY_BLOCK_PREFIX))
        resp = self.service.handle({"request_id": "parent-1", "screening_status": "SAFE",
                                    "original_prompt": BENIGN_PROMPT})
        self.assertEqual(resp["status"], "HARDENED")
        request, block = ao.split_template(resp["revised_prompt"])
        self.assertEqual(request, BENIGN_PROMPT)
        self.assertTrue(block.startswith(ao.SECURITY_BLOCK_PREFIX) and block.endswith(BLOCK_END))

    def test_hardened_plan_is_scored_by_the_parent_pipeline(self):
        resp = self.service.handle({"request_id": "parent-2", "screening_status": "SAFE",
                                    "original_prompt": BENIGN_PROMPT})
        payload = s2.to_planner_payload(resp)
        import adapter as stage3_adapter          # log-generation, put on sys.path by agent_orchestrator
        s3 = stage3_adapter.Stage3PipelineAdapter().process_stage2_output(
            original_prompt=BENIGN_PROMPT, hardened_prompt=payload["prompt"], request_id="parent-2")
        records = ao.materialize_events(s3["downstream_lstm_records"]["Records"], ao.DEFAULT_PRINCIPAL)
        self.pipe.reload_state()
        new = self.pipe.featurize([(i, fe9.normalize_cloudtrail_row(r)) for i, r in enumerate(records)], "parent")
        scored = self.pipe.score(new)
        self.assertEqual(list(scored["event_name"]), s3["downstream_event_sequence"])
        self.assertTrue(scored["risk"].between(0, 1).all())
        self.assertFalse(scored["alert"].any(), scored[["event_name", "risk"]].to_dict("records"))

    def test_ingestion_gate_writes_stage2_templates_for_detected_events(self):
        import ingestion_gate as ig
        store = os.path.join(self.tmp, "gate_store")
        g = object.__new__(ig.IngestionGate)
        g.store_dir, g.templates_dir = store, os.path.join(store, "revised_templates")
        os.makedirs(g.templates_dir)
        g.graph_edges_file = os.path.join(store, "benign_graph_edges.jsonl")
        g.lstm_history_file = os.path.join(store, "lstm_history.jsonl")
        g.pipeline = self.pipe
        g.updater = mock.Mock()
        g.updater.apply_event.return_value = mock.Mock(relation="R", sync_status=mock.Mock(value="pending"))
        g._stage2, g._stage2_tried = None, False             # real Stage 2 import through the gate
        self.pipe.reload_state()
        summary = g.process_rows(enumerate(fe9.iter_input_rows(ATTACK)), "attack.json")
        self.assertGreater(summary.malicious, 0)
        self.assertEqual(summary.templates_written, summary.malicious)
        statuses = json.load(open(os.path.join(ROOT, "prompt_hardening_realtime", "security_hardening", "schemas",
                                               "hardening_response.schema.json")))["properties"]["status"]["enum"]
        for name in os.listdir(g.templates_dir):
            with open(os.path.join(g.templates_dir, name), encoding="utf-8") as f:
                t = json.load(f)
            self.assertEqual(t["classification"], "MALICIOUS")
            self.assertIn(t["revised_security_template"]["status"], statuses)


if __name__ == "__main__":
    unittest.main()
