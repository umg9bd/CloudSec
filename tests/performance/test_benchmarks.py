"""
test_benchmarks.py
==================
Repeatable performance benchmarks for the real-time pipeline, with the real
checkpoints from pipeline_config.json on CPU. Opt-in, so the fast suite stays fast:

    RUN_PERF=1 python -m unittest discover -v -s tests/performance -p test_benchmarks.py

Measured numbers go to tests/performance/baseline_results.json; each is checked
against tests/performance/thresholds.json (generous: ~3x the baseline for times,
~1/3 for throughputs), so a code change that makes something several times slower
fails here. Re-baseline by deleting thresholds.json, running once (it is not
asserted then), and setting new limits from baseline_results.json.

Measured: checkpoint-loading overhead; p50/p95 end-to-end latency per delivered file
and per event; throughput under concurrent file arrivals; watch-folder throughput
with 1/2/4 workers on one folder; graph construction + HGT inference at 3 window
sizes; Python-heap peak (tracemalloc) and RSS / CPU time (psutil).
"""

import contextlib
import io
import json
import logging
import os
import platform
import shutil
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "datasets", "privilege-escalation")
DEV = os.path.join(DATA, "real_dataset_dev.csv")
DEV_STRUCTURAL = os.path.join(DATA, "real_dataset_dev_structural.csv")
RESULTS = os.path.join(HERE, "baseline_results.json")
THRESHOLDS = os.path.join(HERE, "thresholds.json")
RUN = os.environ.get("RUN_PERF") == "1"

FILE_EVENTS = 50          # events per delivered file
N_FILES = 12              # sequential files for the latency run
GRAPH_SIZES = (100, 300, 1000)

results, _S = {}, {}


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def setUpModule():
    if not RUN:
        raise unittest.SkipTest("performance benchmarks are opt-in: set RUN_PERF=1")
    logging.disable(logging.CRITICAL)
    import feature_engine9 as fe9
    import pipeline as P
    _S.update(P=P, fe9=fe9)
    cfg = P.PipelineConfig.load()
    try:
        first = quiet(P.Pipeline, _cfg(tempfile.mkdtemp()), write_outputs=False)
    except Exception as e:                                       # pragma: no cover
        raise unittest.SkipTest(f"configured checkpoints do not load on this machine: {type(e).__name__}: {e}")
    _S.update(gnn=first.gnn, lstm=first.lstm, cfg=cfg)
    dev = pd.read_csv(DEV, low_memory=False)
    _S["dev"] = dev
    _S["rows"] = list(enumerate(fe9.iter_input_rows(DEV)))[: FILE_EVENTS * N_FILES]
    results["_meta"] = {"python": platform.python_version(), "machine": platform.machine(),
                        "processor": platform.processor(), "cpu_count": os.cpu_count(),
                        "file_events": FILE_EVENTS, "n_files": N_FILES,
                        "hgt_seeds": len(cfg.hgt_checkpoints) or 1, "lstm": cfg.lstm_checkpoint,
                        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def tearDownModule():
    logging.disable(logging.NOTSET)
    if RUN and len(results) > 1:
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True, default=float)
        print(f"\n  wrote {os.path.relpath(RESULTS, ROOT)}")


def _cfg(tmp, explain=0):
    P = _S.get("P") or __import__("pipeline")
    cfg = P.PipelineConfig.load()
    cfg.state_dir, cfg.alert_dir = os.path.join(tmp, "state"), os.path.join(tmp, "alerts")
    cfg.output_csv, cfg.explain_top_events = os.path.join(tmp, "out", "scores.csv"), explain
    return cfg


def pipeline(tmp, write_outputs=False, explain=0):
    """A real Pipeline on fresh state; the already-loaded models are shared."""
    P = _S["P"]
    with mock.patch.object(P, "EnsembleGNNScorer", lambda paths: _S["gnn"]), \
            mock.patch.object(P, "GNNScorer", lambda path: _S["gnn"]), \
            mock.patch.object(P.lstm_scorer, "load_scorer", lambda path: _S["lstm"]), \
            mock.patch.object(P.notify, "Notifier", mock.MagicMock):
        return P.Pipeline(_cfg(tmp, explain), write_outputs=write_outputs)


def load_thresholds():
    if not os.path.exists(THRESHOLDS):
        return None
    with open(THRESHOLDS, encoding="utf-8") as f:
        return json.load(f)


class Bench(unittest.TestCase):
    thresholds = None

    @classmethod
    def setUpClass(cls):
        cls.thresholds = load_thresholds()

    def record(self, name, value):
        results[name] = float(value)
        print(f"\n  {name} = {value:.4g}", end="")
        lim = (self.thresholds or {}).get(name)
        if lim is None:
            return
        if "max" in lim:
            self.assertLessEqual(value, lim["max"], f"{name} regressed: {value:.4g} > {lim['max']:.4g}")
        if "min" in lim:
            self.assertGreaterEqual(value, lim["min"], f"{name} regressed: {value:.4g} < {lim['min']:.4g}")

    def write_files(self, folder, n_files, start=0, prefix="batch"):
        dev = _S["dev"]
        names = []
        for i in range(n_files):
            name = f"{prefix}{start + i:04d}.csv"
            dev.iloc[(start + i) * FILE_EVENTS:(start + i + 1) * FILE_EVENTS].to_csv(
                os.path.join(folder, name), index=False)
            names.append(name)
        return names


class TestCheckpointLoading(Bench):
    def test_checkpoint_loading_overhead(self):
        import prod.scorer as lstm_scorer
        from gnn_scorer import GNNScorer
        cfg = _S["cfg"]
        seeds = [os.path.join(ROOT, p) for p in (cfg.hgt_checkpoints or [cfg.hgt_checkpoint])]
        t = []
        for p in seeds:
            t0 = time.perf_counter()
            GNNScorer(p)
            t.append(time.perf_counter() - t0)
        self.record("load_hgt_per_seed_s", float(np.mean(t)))
        t0 = time.perf_counter()
        lstm_scorer.load_scorer(os.path.join(ROOT, cfg.lstm_checkpoint))
        self.record("load_lstm_s", time.perf_counter() - t0)
        t0 = time.perf_counter()
        quiet(_S["P"].Pipeline, _cfg(tempfile.mkdtemp()), write_outputs=False)
        self.record("pipeline_init_in_process_s", time.perf_counter() - t0)


class TestEndToEndLatency(Bench):
    def test_sequential_file_latency_and_resources(self):
        import psutil
        proc = psutil.Process()
        pipe = pipeline(tempfile.mkdtemp())
        rows = _S["rows"]
        lat = []
        cpu0 = proc.cpu_times()
        for i in range(N_FILES):
            t0 = time.perf_counter()
            quiet(pipe.process_rows, rows[i * FILE_EVENTS:(i + 1) * FILE_EVENTS], "real_dataset_dev.csv")
            lat.append(time.perf_counter() - t0)
        cpu1 = proc.cpu_times()
        lat = np.array(lat[1:])                                   # first file warms caches
        self.record("e2e_file_latency_p50_s", np.percentile(lat, 50))
        self.record("e2e_file_latency_p95_s", np.percentile(lat, 95))
        self.record("e2e_event_latency_p50_ms", 1000 * np.percentile(lat, 50) / FILE_EVENTS)
        self.record("e2e_throughput_events_per_s", FILE_EVENTS * len(lat) / lat.sum())
        cpu = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
        self.record("cpu_ms_per_event", 1000 * cpu / (FILE_EVENTS * N_FILES))
        mem = proc.memory_info()
        self.record("rss_with_models_mb", mem.rss / 2 ** 20)
        if hasattr(mem, "peak_wset"):                            # Windows: peak working set
            self.record("rss_peak_mb", mem.peak_wset / 2 ** 20)

    def test_python_heap_peak_while_scoring(self):
        """tracemalloc slows Python several-fold, so it gets its own short run."""
        pipe = pipeline(tempfile.mkdtemp())
        rows = _S["rows"]
        tracemalloc.start()
        for i in range(3):
            quiet(pipe.process_rows, rows[i * FILE_EVENTS:(i + 1) * FILE_EVENTS], "real_dataset_dev.csv")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.record("scoring_python_heap_peak_mb", peak / 2 ** 20)

    def test_cold_start_in_a_fresh_process(self):
        """Imports + all checkpoints, as `python pipeline.py --watch` pays on startup."""
        import subprocess
        import sys
        code = ("import time,os,sys;t=time.perf_counter();import pipeline as P;c=P.PipelineConfig.load();"
                "c.state_dir=sys.argv[1];P.Pipeline(c,write_outputs=False);print(time.perf_counter()-t)")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [ROOT, os.path.join(ROOT, "graph_construction"), os.path.join(ROOT, "temporal-analysis")]))
        out = subprocess.run([sys.executable, "-W", "ignore", "-c", code, tempfile.mkdtemp()], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.record("pipeline_cold_start_process_s", float(out.stdout.strip().splitlines()[-1]))

    def test_alert_latency_with_explanations(self):
        fe9 = _S["fe9"]
        sample = os.path.join(ROOT, "samples", "cloudtrail", "synthetic_attack_chain.json")
        lat = []
        for _ in range(4):
            pipe = pipeline(tempfile.mkdtemp(), explain=3)
            t0 = time.perf_counter()
            scored = quiet(pipe.process_rows, enumerate(fe9.iter_input_rows(sample)), "attack.json")
            lat.append(time.perf_counter() - t0)
            self.assertTrue(scored["alert"].any())
        self.record("alert_file_with_explanations_p50_s", np.percentile(lat[1:], 50))


class TestConcurrency(Bench):
    def run_workers(self, n_workers, n_files, producers=0):
        """n_workers watcher threads in this process, each with its own Pipeline and its
        own incoming folder (files dealt round-robin). Several watchers on ONE folder is
        not used: its atomic-rename claim double-claims files on Windows (pinned in
        test_reliability_e2e.TestSharedFolderWatchers). With producers > 0 the files
        arrive from that many threads while the workers run; otherwise all are present
        at the start."""
        P = _S["P"]
        tmp = tempfile.mkdtemp()
        folders = [os.path.join(tmp, f"incoming{i}") for i in range(n_workers)]
        for inc in folders:
            for d in ("", P.CLAIM_DIR, "processed", "failed"):
                os.makedirs(os.path.join(inc, d), exist_ok=True)
        staging = os.path.join(tmp, "staging")
        os.makedirs(staging)
        names = self.write_files(staging, n_files)
        target = {n: folders[i % n_workers] for i, n in enumerate(names)}
        arrived, done = {}, {}
        if not producers:
            for n in names:
                os.replace(os.path.join(staging, n), os.path.join(target[n], n))
                arrived[n] = time.perf_counter()
        pipes = [pipeline(os.path.join(tmp, f"w{i}")) for i in range(n_workers)]
        for p in pipes:
            original = p.process_file

            def timed(path, _orig=original):
                out = _orig(path)
                done[os.path.basename(path)] = time.perf_counter()
                return out
            p.process_file = timed
        stop = threading.Event()

        def worker(p, inc):
            while not stop.is_set():
                if P.scan_once(inc, p) == 0:
                    time.sleep(0.01)

        def producer(mine):
            rng = np.random.default_rng(len(mine))
            for n in mine:
                time.sleep(float(rng.uniform(0.0, 0.3)))
                os.replace(os.path.join(staging, n), os.path.join(target[n], n))
                arrived[n] = time.perf_counter()

        def finished():
            return sum(len(os.listdir(os.path.join(inc, d))) for inc in folders for d in ("processed", "failed"))

        t0 = time.perf_counter()
        with mock.patch.object(_S["fe9"], "_file_is_stable", lambda p: os.path.exists(p)), \
                contextlib.redirect_stdout(io.StringIO()):          # once, not per thread: it is process-global
            threads = [threading.Thread(target=worker, args=(p, inc)) for p, inc in zip(pipes, folders)]
            threads += [threading.Thread(target=producer, args=(names[i::producers],)) for i in range(producers)]
            for th in threads:
                th.start()
            while finished() < n_files and time.perf_counter() - t0 < 600:
                time.sleep(0.02)
            stop.set()
            for th in threads:
                th.join()
        wall = time.perf_counter() - t0
        self.assertEqual(sorted(done), sorted(names), "every file scored exactly once")
        self.assertEqual(sorted(n for inc in folders for n in os.listdir(os.path.join(inc, "processed"))), sorted(names))
        shutil.rmtree(tmp, ignore_errors=True)
        lat = np.array([done[n] - arrived[n] for n in names])
        return n_files * FILE_EVENTS / wall, lat

    def test_watch_folder_throughput_by_worker_count(self):
        base = None
        for k in (1, 2, 4):
            tput, _ = self.run_workers(k, 12)
            self.record(f"watch_throughput_{k}w_events_per_s", tput)
            base = base or tput
            results[f"watch_speedup_{k}w"] = tput / base

    def test_concurrent_arrivals(self):
        tput, lat = self.run_workers(1, 12, producers=4)
        self.record("concurrent_arrivals_throughput_events_per_s", tput)
        self.record("concurrent_arrival_to_scored_p50_s", np.percentile(lat, 50))
        self.record("concurrent_arrival_to_scored_p95_s", np.percentile(lat, 95))


class TestGraphScaling(Bench):
    def test_graph_construction_and_inference_by_size(self):
        import torch
        gnn = _S["gnn"]
        seed = getattr(gnn, "scorers", [gnn])[0]
        st = pd.read_csv(DEV_STRUCTURAL)
        cols = ["log_id", "source_node", "target_node", "edge_type", "label"]
        for n in GRAPH_SIZES:
            df = st.head(n)[cols]
            build, infer, ens = [], [], []
            for _ in range(3):
                t0 = time.perf_counter()
                data, _ = seed.build_graph(df)
                t1 = time.perf_counter()
                with torch.no_grad():
                    seed.model(data)
                t2 = time.perf_counter()
                gnn.score(df)
                ens.append(time.perf_counter() - t2)
                build.append(t1 - t0)
                infer.append(t2 - t1)
            self.record(f"graph_build_{n}_edges_s", float(np.median(build)))
            self.record(f"hgt_forward_{n}_edges_s", float(np.median(infer)))
            self.record(f"ensemble_score_{n}_edges_s", float(np.median(ens)))


if __name__ == "__main__":
    unittest.main()
