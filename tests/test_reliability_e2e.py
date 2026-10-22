"""
test_reliability_e2e.py
=======================
Failure behaviour of the live system, end to end, with no Neo4j / Docker /
network (every external dependency is mocked):

  * Neo4j outage mid-run -- incremental graph write-through (incremental_updater),
    batch ingest (neo4j_graph_builder.build_graph) and graph-only inference
    (graph_construction/infer.InferenceEngine). The real-time pipeline.py has no Neo4j
    dependency at all (gnn_scorer builds graphs in memory).
  * missing / truncated / wrong-type checkpoints -- a clear error at startup, before
    any state is written.
  * interrupted scoring then restart, and re-delivery of a file -- no duplicate alerts
    or duplicated score rows; a clean restart resumes exactly.
  * invalid / incomplete files arriving while the watcher runs -- filed away, the
    watcher keeps going, and later files score exactly as on fresh state; several
    watchers on one folder under real (threaded) concurrency.
  * repeated runs -- identical log_ids and scores; alert ids never collide.

Already covered elsewhere and not repeated: notification failure never blocks
detection (test_notify), graph model failure at scoring time -> LSTM-only with
explanations (test_ensemble_explain), watcher claim races (test_watch_folder).
"""

import contextlib
import glob
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import networkx as nx
import numpy as np
import pandas as pd
import torch
from neo4j.exceptions import ServiceUnavailable

import feature_engine9 as fe9
import neo4j_graph_builder as nb
import pipeline as P
import privilege_features as pf
from incremental_updater import CloudTrailEvent, IncrementalGraphUpdater, IncrementalUpdateConfig, SyncStatus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "privilege-escalation")
DEV = os.path.join(DATA, "real_dataset_dev.csv")
STRUCTURAL = os.path.join(DATA, "cloudtrail_structural.csv")
ATTACK = os.path.join(ROOT, "samples", "cloudtrail", "synthetic_attack_chain.json")
CFG = P.PipelineConfig.load()
HGT_SEED = os.path.join(ROOT, CFG.hgt_checkpoints[0] if CFG.hgt_checkpoints else CFG.hgt_checkpoint)
LSTM_CKPT = os.path.join(ROOT, CFG.lstm_checkpoint)

_MODELS = {}


def setUpModule():
    logging.disable(logging.CRITICAL)


def tearDownModule():
    logging.disable(logging.NOTSET)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def fast_pipeline(tmp, write_outputs=True):
    """A real Pipeline on state in `tmp`; models are loaded once and shared (they are
    read-only), the email notifier is a mock."""
    cfg = P.PipelineConfig.load()
    cfg.state_dir, cfg.alert_dir = os.path.join(tmp, "state"), os.path.join(tmp, "alerts")
    cfg.output_csv, cfg.explain_top_events = os.path.join(tmp, "out", "scores.csv"), 0
    if not _MODELS:
        first = quiet(P.Pipeline, cfg, write_outputs=False)
        _MODELS.update(gnn=first.gnn, lstm=first.lstm)
    with mock.patch.object(P, "EnsembleGNNScorer", lambda paths: _MODELS["gnn"]), \
            mock.patch.object(P, "GNNScorer", lambda path: _MODELS["gnn"]), \
            mock.patch.object(P.lstm_scorer, "load_scorer", lambda path: _MODELS["lstm"]), \
            mock.patch.object(P.notify, "Notifier", mock.MagicMock):
        return P.Pipeline(cfg, write_outputs=write_outputs)


def alerts_in(tmp):
    out = []
    for path in sorted(glob.glob(os.path.join(tmp, "alerts", "alert_*.json"))):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def flagged_log_ids(alerts):
    return [e["log_id"] for a in alerts for e in a["events"]]


def dev_slice_csv(path, start, stop):
    pd.read_csv(DEV, low_memory=False).iloc[start:stop].to_csv(path, index=False)
    return path


# ── Neo4j outage ──────────────────────────────────────────────────────────────

class FlakySession:
    """Context-managed session whose run() raises ServiceUnavailable while `down()`."""

    def __init__(self, down, log):
        self.down, self.log = down, log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        if self.down():
            raise ServiceUnavailable("Couldn't connect to localhost:7687 (mocked outage)")
        self.log.append((query, params))
        return mock.MagicMock(data=lambda: [])


def structural_events(n):
    df = pd.read_csv(STRUCTURAL, nrows=n)
    return [CloudTrailEvent(str(r.log_id), r.source_node, r.target_node, r.edge_type, int(r.label))
            for r in df.itertuples()]


def new_updater(factory=None):
    resolver = pf.ActionAccessLevelResolver()
    return IncrementalGraphUpdater(pf.PrivilegePropagationGraph(resolver), IncrementalUpdateConfig(),
                                   resolver=resolver, neo4j_session_factory=factory)


def graph_state(g):
    return (sorted((n, sorted((k, sorted(v) if isinstance(v, set) else v) for k, v in d.items()))
                   for n, d in g.nodes(data=True)),
            sorted((u, v, sorted(d.items())) for u, v, d in g.edges(data=True)))


class TestNeo4jWriteThroughOutage(unittest.TestCase):
    N, DOWN = 60, range(20, 40)

    @classmethod
    def setUpClass(cls):
        cls.events = structural_events(cls.N)
        ref = new_updater()
        for ev in cls.events:
            ref.apply_event(ev)
        cls.reference = graph_state(ref.graph)

    def run_with_outage(self):
        i, log = {"n": 0}, []
        factory = lambda: FlakySession(lambda: i["n"] in self.DOWN, log)
        upd = new_updater(factory)
        results = []
        for ev in self.events:
            results.append(upd.apply_event(ev))      # must not raise
            i["n"] += 1
        return upd, results, log

    def test_outage_mid_run_is_flagged_per_event_and_never_raises(self):
        _, results, _ = self.run_with_outage()
        status = [r.sync_status for r in results]
        self.assertTrue(all(s == SyncStatus.FAILED for k, s in enumerate(status) if k in self.DOWN))
        self.assertTrue(all(s == SyncStatus.SYNCED for k, s in enumerate(status) if k not in self.DOWN))
        self.assertIn("mocked outage", results[self.DOWN[0]].sync_error)

    def test_in_memory_graph_is_unaffected_by_the_outage(self):
        """No partial / corrupt state: the graph detection reads is identical to a run
        that never had Neo4j at all, and writes resume after recovery."""
        upd, _, log = self.run_with_outage()
        self.assertEqual(graph_state(upd.graph), self.reference)
        written = {p.get("log_id") for _, p in log if "log_id" in p}
        self.assertEqual(written, {ev.log_id for k, ev in enumerate(self.events) if k not in self.DOWN})


class FakeDriver:
    def __init__(self, fail_after_edges):
        self.queries, self.closed, self.fail_after = [], False, fail_after_edges

    def session(self):
        drv = self

        class S(FlakySession):
            def run(s, query, **params):
                is_edge = lambda q: "]->(dst)" in q
                if is_edge(query) and sum(map(is_edge, drv.queries)) >= drv.fail_after:
                    raise ServiceUnavailable("connection lost mid-ingest (mocked)")
                drv.queries.append(query)
                return mock.MagicMock()
        return S(lambda: False, [])

    def close(self):
        self.closed = True


class TestNeo4jBatchIngestOutage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        csv = os.path.join(self.tmp, "slice_structural.csv")
        pd.read_csv(STRUCTURAL, nrows=40).to_csv(csv, index=False)
        self.csv = csv

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, fail_after):
        driver = FakeDriver(fail_after)
        with mock.patch.object(nb, "CSV_PATH", self.csv), \
                mock.patch("neo4j.GraphDatabase.driver", lambda *a, **k: driver):
            try:
                quiet(nb.build_graph)
                return driver, None
            except Exception as e:
                return driver, e

    def test_complete_ingest_writes_every_edge(self):
        driver, err = self.build(fail_after=10 ** 9)
        self.assertIsNone(err)
        self.assertEqual(sum("]->(dst)" in q for q in driver.queries), 40)
        self.assertTrue(driver.closed)

    def test_outage_mid_ingest_fails_loudly(self):
        _, err = self.build(fail_after=15)
        self.assertIsInstance(err, ServiceUnavailable)

    @unittest.expectedFailure
    def test_a_partial_ingest_is_not_stamped_as_a_valid_graph(self):
        """KNOWN DEFECT: build_graph writes the (:_GraphProvenance {source_csv}) marker
        right after 'MATCH (n) DETACH DELETE n', BEFORE ingesting nodes/edges, and nothing
        removes it on failure. A Neo4j outage mid-ingest therefore leaves a cleared,
        partially-loaded graph that carries a valid-looking provenance stamp, which the
        provenance check (evaluate_session_level.check_graph_provenance) accepts."""
        driver, err = self.build(fail_after=15)
        self.assertIsNotNone(err)
        self.assertFalse(any("_GraphProvenance" in q for q in driver.queries))

    @unittest.expectedFailure
    def test_driver_is_closed_when_ingest_fails(self):
        """KNOWN DEFECT: driver.close() is a plain statement after the `with
        driver.session()` block (no try/finally), so any exception mid-ingest leaks the
        driver and its connection pool."""
        driver, err = self.build(fail_after=15)
        self.assertIsNotNone(err)
        self.assertTrue(driver.closed)


class TestNeo4jOutageDuringGraphInference(unittest.TestCase):
    @unittest.expectedFailure
    def test_inference_engine_survives_neo4j_outage(self):
        """KNOWN DEFECT: infer.InferenceEngine.process_event tolerates a failed Neo4j
        write-through (step 2, SyncStatus.FAILED) and SubgraphLoader swallows errors in
        its diagnostic queries, but SubgraphLoader._fetch_neighbourhood (step 3) is not
        guarded: ServiceUnavailable propagates out of process_event -> process_file ->
        watch_directory, whose `finally: engine.close()` then stops the graph-only
        detector on the first outage."""
        import infer
        tmp = tempfile.mkdtemp()
        down = lambda: True
        driver = mock.MagicMock()
        driver.session.side_effect = lambda: FlakySession(down, [])
        eng = object.__new__(infer.InferenceEngine)
        eng.threshold, eng.device, eng.alert_dir = 0.5, torch.device("cpu"), os.path.join(tmp, "alerts")
        eng._log_id_counter = 0
        eng.model = mock.MagicMock(side_effect=AssertionError("must not be reached"))
        eng.updater = new_updater(lambda: FlakySession(down, []))
        eng.subgraph_loader = infer.SubgraphLoader(driver=driver, fit_artifacts={}, device=eng.device)
        eng.feature_engineer = fe9.FeatureEngineer(state_tracker_path=os.path.join(tmp, "t.json"),
                                                   graph_state_path=os.path.join(tmp, "g.json"))
        row = fe9.normalize_cloudtrail_row({"eventTime": "2024-01-01T10:00:00Z", "eventName": "CreateAccessKey",
                                            "eventSource": "iam.amazonaws.com",
                                            "userIdentity": {"type": "IAMUser", "userName": "u1",
                                                             "arn": "arn:aws:iam::111122223333:user/u1"}})
        out = quiet(eng.process_event, row)           # fail safe: no alert, no exception
        self.assertIsNone(out)


# ── checkpoints ───────────────────────────────────────────────────────────────

class TestCheckpointFailures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def truncated(self, src, name):
        dst = os.path.join(self.tmp, name)
        with open(src, "rb") as f:
            data = f.read()
        with open(dst, "wb") as f:
            f.write(data[: len(data) // 2])
        return dst

    def start(self, **over):
        cfg = P.PipelineConfig.load()
        cfg.state_dir = os.path.join(self.tmp, "state")
        for k, v in over.items():
            setattr(cfg, k, v)
        with mock.patch.object(P.notify, "Notifier", mock.MagicMock):
            return quiet(P.Pipeline, cfg, write_outputs=False)

    def assert_fails_before_any_state(self, exc_types, **over):
        with self.assertRaises(exc_types) as ctx:
            self.start(**over)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "state")), "startup left partial state behind")
        return ctx.exception

    def test_missing_checkpoints_name_the_missing_file(self):
        gone = os.path.join(self.tmp, "nope.pt")
        e = self.assert_fails_before_any_state(FileNotFoundError, lstm_checkpoint=gone)
        self.assertIn("nope.pt", str(e))
        e = self.assert_fails_before_any_state(FileNotFoundError, hgt_checkpoints=[gone])
        self.assertIn("nope.pt", str(e))

    def test_truncated_graph_checkpoint_fails_at_startup(self):
        bad = self.truncated(HGT_SEED, "hgt_truncated.pt")
        self.assert_fails_before_any_state(Exception, hgt_checkpoints=[bad])

    def test_one_corrupted_seed_fails_the_ensemble_instead_of_silently_dropping_it(self):
        bad = self.truncated(HGT_SEED, "seed_truncated.pt")
        self.assert_fails_before_any_state(Exception, hgt_checkpoints=[HGT_SEED, bad])

    def test_truncated_sequence_checkpoint_fails_at_startup(self):
        bad = self.truncated(LSTM_CKPT, "lstm_truncated.pt")
        self.assert_fails_before_any_state(Exception, lstm_checkpoint=bad)

    def test_sequence_checkpoint_as_graph_model_is_a_clear_error(self):
        e = self.assert_fails_before_any_state(ValueError, hgt_checkpoints=[LSTM_CKPT])
        self.assertIn("not an inference-ready checkpoint", str(e))

    @unittest.expectedFailure
    def test_graph_checkpoint_as_sequence_model_is_a_clear_error(self):
        """KNOWN DEFECT: pointing lstm_checkpoint at a graph (HGT) checkpoint fails inside
        train_lstm_transformer.load_checkpoint with a bare KeyError('vocab_size')
        (cfg = ckpt.get('config') or {}), naming neither the file nor the problem --
        unlike GNNScorer, which raises 'not an inference-ready checkpoint'."""
        self.assert_fails_before_any_state(ValueError, lstm_checkpoint=HGT_SEED)


# ── interruption, restart, re-delivery ────────────────────────────────────────

class TestRestartAndRedelivery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fast_pipeline(tempfile.mkdtemp())        # load the shared models once

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.incoming = os.path.join(self.tmp, "incoming")
        os.makedirs(self.incoming)
        self.stable = mock.patch.object(fe9, "_file_is_stable", lambda p: os.path.exists(p))
        self.stable.start()

    def tearDown(self):
        self.stable.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_restart_resumes_exactly(self):
        """Persisted state (trackers + event buffer) reproduces an uninterrupted run."""
        a = dev_slice_csv(os.path.join(self.tmp, "a.csv"), 0, 100)
        b = dev_slice_csv(os.path.join(self.tmp, "b.csv"), 100, 200)
        run1 = os.path.join(self.tmp, "run1")
        quiet(fast_pipeline(run1).process_file, a)
        restarted = quiet(fast_pipeline(run1).process_file, b)
        cont = fast_pipeline(os.path.join(self.tmp, "run2"))
        quiet(cont.process_file, a)
        continuous = quiet(cont.process_file, b)
        m = restarted.merge(continuous, on="log_id", suffixes=("", "_c"))
        self.assertEqual(len(m), 100)
        self.assertLess(float(np.abs(m["risk"] - m["risk_c"]).max()), 1e-9)

    def test_repeated_runs_give_identical_ids_and_scores_and_unique_alert_ids(self):
        runs = [quiet(fast_pipeline(os.path.join(self.tmp, f"r{i}")).process_file, ATTACK) for i in range(2)]
        self.assertEqual(list(runs[0]["log_id"]), list(runs[1]["log_id"]))
        for c in ("p_sequence", "p_graph", "risk", "alert"):
            np.testing.assert_allclose(runs[0][c].astype(float), runs[1][c].astype(float), atol=1e-9, err_msg=c)
        ids = [a["alert_id"] for i in range(2) for a in alerts_in(os.path.join(self.tmp, f"r{i}"))]
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))

    @unittest.expectedFailure
    def test_crash_after_alerting_then_restart_does_not_duplicate_alerts(self):
        """KNOWN DEFECT: Pipeline.process_file writes alert files (emit) before
        save_state(), and the watcher re-queues a claimed file on restart. A stop between
        the two (here: Ctrl+C during save_state) makes the restarted watcher score the
        file again and write a SECOND alert file, with a new alert_id, for the same
        log_ids -- alerts are not idempotent per event/file."""
        shutil.copy(ATTACK, os.path.join(self.incoming, "batch1.json"))
        crashing = fast_pipeline(self.tmp)
        crashing.save_state = mock.MagicMock(side_effect=KeyboardInterrupt)
        quiet(P.watch, self.incoming, crashing, poll_seconds=0)          # stopped mid-file
        first = flagged_log_ids(alerts_in(self.tmp))
        self.assertTrue(first)
        with mock.patch.object(P.time, "sleep", side_effect=KeyboardInterrupt):
            quiet(P.watch, self.incoming, fast_pipeline(self.tmp), poll_seconds=0)
        ids = flagged_log_ids(alerts_in(self.tmp))
        self.assertEqual(len(ids), len(set(ids)), "the same event was alerted twice")

    @unittest.expectedFailure
    def test_redelivered_file_is_not_double_counted(self):
        """KNOWN DEFECT: log_ids are '<file name>:<row>', and Pipeline keeps no record of
        files already scored. Re-delivering a file under the same name puts duplicate
        log_ids in the event buffer, and the merges in Pipeline.score fan out: 150 dev
        events come back as 242 scored rows, all appended
        to risk_scores.csv, with a second set of alerts."""
        f = dev_slice_csv(os.path.join(self.tmp, "batch.csv"), 0, 150)
        pipe = fast_pipeline(self.tmp)
        quiet(pipe.process_file, f)
        again = quiet(pipe.process_file, f)
        self.assertEqual(len(again), 150)
        out = pd.read_csv(os.path.join(self.tmp, "out", "scores.csv"))
        self.assertEqual(out["log_id"].nunique(), len(out))


# ── invalid input while the watcher runs ──────────────────────────────────────

class TestWatcherWithBadInput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fast_pipeline(tempfile.mkdtemp())

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.incoming = os.path.join(self.tmp, "incoming")
        for d in ("", P.CLAIM_DIR, "processed", "failed"):
            os.makedirs(os.path.join(self.incoming, d), exist_ok=True)
        self.stable = mock.patch.object(fe9, "_file_is_stable", lambda p: os.path.exists(p))
        self.stable.start()

    def tearDown(self):
        self.stable.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def drop(self, name, content, mtime):
        path = os.path.join(self.incoming, name)
        with open(path, "wb" if isinstance(content, bytes) else "w", **({} if isinstance(content, bytes)
                                                                         else {"encoding": "utf-8"})) as f:
            f.write(content)
        os.utime(path, (mtime, mtime))                                   # scan order = mtime order

    def test_bad_files_are_filed_away_and_later_files_score_as_on_fresh_state(self):
        with open(ATTACK, encoding="utf-8") as f:
            good = f.read()
        self.drop("01_truncated.json", good[: len(good) // 2], 1000)       # partial write
        self.drop("02_garbage.json", b"\x00\xff\x13 not json at all", 1001)
        self.drop("03_header_only.csv", "timestamp,event_name,username\n", 1002)
        self.drop("04_no_timestamps.csv", "event_name,username\nCreateUser,u1\nListRoles,u1\n", 1003)
        self.drop("05_empty.csv", "", 1004)
        self.drop("06_good.json", good, 1005)
        pipe = fast_pipeline(self.tmp)
        n = quiet(P.scan_once, self.incoming, pipe)                       # must not raise
        self.assertEqual(sorted(os.listdir(os.path.join(self.incoming, "failed"))),
                         ["01_truncated.json", "02_garbage.json"])
        self.assertEqual(sorted(os.listdir(os.path.join(self.incoming, "processed"))),
                         ["03_header_only.csv", "04_no_timestamps.csv", "05_empty.csv", "06_good.json"])
        self.assertEqual(n, 4)
        self.assertEqual(os.listdir(os.path.join(self.incoming, P.CLAIM_DIR)), [])
        out = pd.read_csv(os.path.join(self.tmp, "out", "scores.csv"))
        self.assertEqual(set(out["log_id"].str.rsplit(":", n=1).str[0]), {"06_good.json"})
        fresh = quiet(fast_pipeline(os.path.join(self.tmp, "fresh")).process_rows,
                      enumerate(fe9.iter_input_rows(ATTACK)), "06_good.json")
        np.testing.assert_allclose(out["risk_score"].to_numpy(), (fresh["risk"] * 10).round(2).to_numpy(), atol=1e-9)
        self.assertTrue(alerts_in(self.tmp))


class TestSharedFolderWatchers(unittest.TestCase):
    """Real concurrency (threads), not the scripted interleaving test_watch_folder uses."""

    class Reader:
        alert_dir, output_csv = "alerts", "out.csv"

        def __init__(self, log):
            self.log = log

        def process_file(self, path):
            with open(path, encoding="utf-8") as f:
                f.read()
            time.sleep(0.005)
            self.log.append(os.path.basename(path))

    @unittest.expectedFailure
    def test_concurrent_watchers_on_one_folder_score_each_file_exactly_once(self):
        """KNOWN DEFECT (Windows): scan_once's claim -- os.replace(incoming/x,
        .processing/x), relying on the loser getting FileNotFoundError -- is not exclusive
        under real concurrency on NTFS. With 4 watcher threads on 100 files, ~20 files are
        scored by two watchers and ~30 valid files land in failed/ (PermissionError on the
        claimed .processing path, then 'could not move ... cannot find the file'), so the
        docstring's 'exactly one wins each file' does not hold here."""
        tmp = tempfile.mkdtemp()
        inc = os.path.join(tmp, "incoming")
        for d in ("", P.CLAIM_DIR, "processed", "failed"):
            os.makedirs(os.path.join(inc, d), exist_ok=True)
        names = [f"b{i:04d}.csv" for i in range(100)]
        for n in names:
            with open(os.path.join(inc, n), "w", encoding="utf-8") as f:
                f.write("timestamp,event_name\n" * 50)
        log, stop = [], threading.Event()

        def watcher():
            me = self.Reader(log)
            while not stop.is_set():
                if P.scan_once(inc, me) == 0:
                    time.sleep(0.001)

        t0 = time.time()
        with mock.patch.object(fe9, "_file_is_stable", lambda p: os.path.exists(p)), \
                contextlib.redirect_stdout(io.StringIO()):
            threads = [threading.Thread(target=watcher) for _ in range(4)]
            for th in threads:
                th.start()
            while sum(len(os.listdir(os.path.join(inc, d))) for d in ("processed", "failed")) < 100 \
                    and time.time() - t0 < 60:
                time.sleep(0.05)
            time.sleep(0.3)
            stop.set()
            for th in threads:
                th.join()
        shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(sorted(log), names)


if __name__ == "__main__":
    unittest.main()
