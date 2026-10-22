"""
test_leakage_extra.py
=====================
Train/eval-boundary checks NOT already covered by test_leakage_guard,
test_causal_features, test_feature_engine_cache, test_campaign_split and
test_family_split (those cover the held-out-row guard, the frozen prior unit,
cached-output staleness and the split ALGORITHMS on fixtures). Here:

  * tuning provenance -- pipeline_config.json says dev; evaluate_pipeline.py's
    dev run never opens the test split, and --test reads it once, after the
    weight/threshold are frozen and saved;
  * served artifacts -- the risk priors / vocabulary the live pipeline reads hold
    no real-data keys and are never written by it; the GNN's fitted scalers are not
    refit while serving; the priors' provenance w.r.t. the committed campaign split;
  * streaming causality -- appending later events cannot change the features or
    scores of earlier events (sequence branch: holds; graph branch: see the
    expected failure);
  * the COMMITTED split files -- families / campaigns / background users / real
    sessions do not cross split boundaries;
  * two disclosed limitations, as skips.

real_dataset_test.csv: only its session_id / username columns are read (split
boundary checks); no model ever scores it here.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import os
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import campaign_split as cs
import feature_engine9 as fe9
import pipeline as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "privilege-escalation")
DEV = os.path.join(DATA, "real_dataset_dev.csv")
TEST = os.path.join(DATA, "real_dataset_test.csv")
SPLIT_FILE = os.path.join(ROOT, "splits", "campaign_family_seed42.csv")

_MODELS = {}


def setUpModule():
    logging.disable(logging.WARNING)     # graph loader logs every window at INFO


def tearDownModule():
    logging.disable(logging.NOTSET)


def fast_pipeline(tmp, write_outputs=False):
    """A real Pipeline on fresh state in `tmp`; the (read-only) models are loaded once
    per process and shared, so each extra instance costs nothing."""
    cfg = P.PipelineConfig.load()
    cfg.state_dir, cfg.alert_dir = os.path.join(tmp, "state"), os.path.join(tmp, "alerts")
    cfg.output_csv, cfg.explain_top_events = os.path.join(tmp, "out", "scores.csv"), 0
    if not _MODELS:
        with contextlib.redirect_stdout(io.StringIO()):
            first = P.Pipeline(cfg, write_outputs=False)
        _MODELS.update(gnn=first.gnn, lstm=first.lstm)
    with mock.patch.object(P, "EnsembleGNNScorer", lambda paths: _MODELS["gnn"]), \
            mock.patch.object(P, "GNNScorer", lambda path: _MODELS["gnn"]), \
            mock.patch.object(P.lstm_scorer, "load_scorer", lambda path: _MODELS["lstm"]), \
            mock.patch.object(P.notify, "Notifier", mock.MagicMock):
        return P.Pipeline(cfg, write_outputs=write_outputs)


def dev_rows(n):
    return list(enumerate(fe9.iter_input_rows(DEV)))[:n]


def run_quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── tuning provenance ─────────────────────────────────────────────────────────

def load_evaluate_pipeline():
    cwd = os.getcwd()
    try:
        spec = importlib.util.spec_from_file_location("evaluate_pipeline", os.path.join(DATA, "evaluate_pipeline.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # it chdirs to the repo root at import
        return mod
    finally:
        os.chdir(cwd)


class TestTuningProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = load_evaluate_pipeline()
        cls.dev = cls.ev.load_real_split("dev")

    def fake_events(self, split_name, rng):
        n = len(self.dev["raw"])
        return pd.DataFrame({"log_id": [f"real_dataset_{split_name}.csv:{i}" for i in range(n)],
                             "p_graph": rng.random(n), "p_sequence": rng.random(n), "risk": rng.random(n)})

    def run_main(self, argv):
        """evaluate_pipeline.main() with the multi-minute replay stubbed; every file
        opened and every split loaded is recorded, in order."""
        ev, calls, opened = self.ev, [], []
        rng = np.random.default_rng(0)
        real_read_csv, real_open = pd.read_csv, open

        def load_split(split):
            calls.append(("load", split))
            return self.dev if split == "dev" else dict(self.dev)    # test: never actually read

        def replay(split, cfg):
            calls.append(("replay", split, cfg.weight_graph, cfg.alert_threshold))
            return self.fake_events(split, rng), None

        def save(cfg, path=None):
            calls.append(("save", cfg.weight_graph, cfg.alert_threshold, cfg.tuned_on))

        def spy_read_csv(path, *a, **kw):
            opened.append(str(path))
            return real_read_csv(path, *a, **kw)

        def spy_open(path, *a, **kw):
            opened.append(str(path))
            return real_open(path, *a, **kw)

        mlb = sys.modules["evaluate_ml_baselines"]
        fake_run = lambda: {"sessions": self.dev["sessions_true"].index, "y": self.dev["y"],
                            "LogReg": np.zeros(len(self.dev["y"]), dtype=int)}
        with mock.patch.object(ev, "load_real_split", load_split), mock.patch.object(ev, "replay", replay), \
                mock.patch.object(ev, "check_against_batch", lambda *a: None), \
                mock.patch.object(ev.PipelineConfig, "save", save), mock.patch.object(mlb, "run", fake_run), \
                mock.patch.object(pd, "read_csv", spy_read_csv), mock.patch("builtins.open", spy_open), \
                mock.patch.object(sys, "argv", ["evaluate_pipeline.py"] + argv), \
                contextlib.redirect_stdout(io.StringIO()):
            ev.main()
        return calls, opened

    def test_config_was_tuned_on_dev(self):
        cfg = P.PipelineConfig.load()
        self.assertIn("real_dataset_dev.csv", cfg.tuned_on)
        self.assertIn("test never used for tuning", cfg.tuned_on)
        self.assertNotIn("real_dataset_test", cfg.tuned_on)

    def test_dev_tuning_run_never_reads_the_test_split(self):
        calls, opened = self.run_main([])
        self.assertEqual([c[1] for c in calls if c[0] in ("load", "replay")], ["dev", "dev"])
        self.assertFalse([p for p in opened if "real_dataset_test" in p], opened)
        saved = [c for c in calls if c[0] == "save"]
        self.assertEqual(len(saved), 1)
        self.assertIn("real_dataset_dev.csv", saved[0][3])

    def test_test_run_reads_test_once_after_the_config_is_frozen(self):
        calls, _ = self.run_main(["--test"])
        kinds = [(c[0], c[1]) if c[0] != "save" else ("save",) for c in calls]
        self.assertEqual(kinds, [("load", "dev"), ("replay", "dev"), ("save",), ("load", "test"), ("replay", "test")])
        save = next(c for c in calls if c[0] == "save")
        replay_test = calls[-1]
        self.assertEqual(replay_test[2:], save[1:3])     # test replayed with exactly the dev-chosen w / threshold


# ── served artifacts ──────────────────────────────────────────────────────────

class TestServedArtifacts(unittest.TestCase):
    def test_served_priors_hold_no_real_data_keys(self):
        """The priors are label-derived; a key that only occurs in real dev would mean
        they were fit on real labels."""
        syn = set(pd.read_csv(os.path.join(DATA, "synthetic_cloudtrail.csv"), usecols=["event_name"])["event_name"].dropna())
        dev = set(pd.read_csv(DEV, usecols=["event_name"])["event_name"].dropna())
        with open(os.path.join(ROOT, fe9.ACTION_PRIOR_FILE), encoding="utf-8") as f:
            prior = json.load(f)
        self.assertGreater(len(dev - syn), 0)
        self.assertEqual(set(prior) & (dev - syn), set())
        self.assertTrue(set(prior) <= syn)

    @unittest.expectedFailure
    def test_served_priors_were_fit_on_campaign_split_train_rows_only(self):
        """KNOWN DEFECT: .action_risk_prior.json / .principal_risk_prior.json (the
        label-derived priors behind action_risk_prior, principal_type_prior_risk and the
        structural source_historical_risk) were fitted on ALL 11,753 synthetic rows, i.e.
        including the 2,419 val + 1,389 test rows of the committed
        splits/campaign_family_seed42.csv. run_batch only withholds rows when the input
        carries a `split` column / --split-file, and no tagged prior file exists. Real
        dev/test are unaffected (priors are frozen there), but any within-synthetic
        validation, early stopping or campaign-family held-out number computed on these
        features has seen its own labels."""
        split = cs.read_split_file(SPLIT_FILE)
        n_train = sum(1 for s in split.values() if s == "train")
        for name in (fe9.ACTION_PRIOR_FILE, fe9.PRINCIPAL_PRIOR_FILE):
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                fitted = sum(t for _, t in json.load(f).values())
            self.assertEqual(fitted, n_train, name)

    def test_gnn_fit_artifacts_are_not_refit_while_serving(self):
        tmp = tempfile.mkdtemp()
        pipe = fast_pipeline(tmp)
        scorers = getattr(pipe.gnn, "scorers", [pipe.gnn])
        before = [pickle.dumps(s.fit_artifacts) for s in scorers]
        run_quiet(pipe.process_rows, dev_rows(150), "real_dataset_dev.csv")
        self.assertEqual([pickle.dumps(s.fit_artifacts) for s in scorers], before)

    def test_live_pipeline_never_writes_priors_or_vocabulary(self):
        """End to end, with outputs and state saving on: labelled rows must not move the
        training-fitted files the pipeline reads (frozen at construction)."""
        files = [os.path.join(ROOT, p) for p in (fe9.ACTION_PRIOR_FILE, fe9.PRINCIPAL_PRIOR_FILE,
                                                 fe9.EVENT_NAME_VOCAB_FILE)]
        before = [sha(p) for p in files]
        tmp = tempfile.mkdtemp()
        pipe = fast_pipeline(tmp, write_outputs=True)
        self.assertTrue(pipe.engine.freeze_priors and pipe.engine.event_name_vocab.frozen)
        dev = pd.read_csv(DEV, low_memory=False)
        dev.head(120).assign(label=1).to_csv(os.path.join(tmp, "labelled.csv"), index=False)  # all "attack"
        run_quiet(pipe.process_file, os.path.join(tmp, "labelled.csv"))
        self.assertEqual([sha(p) for p in files], before)


# ── streaming causality ───────────────────────────────────────────────────────

class TestStreamingCausality(unittest.TestCase):
    """The same first events, scored (a) alone and (b) together with later events in
    one delivery. Nothing about an event may depend on what happened after it."""
    N, EXTRA = 120, 120

    @classmethod
    def setUpClass(cls):
        rows = dev_rows(cls.N + cls.EXTRA)
        cls.alone = run_quiet(fast_pipeline(tempfile.mkdtemp()).process_rows, rows[:cls.N], "real_dataset_dev.csv")
        cls.joint = run_quiet(fast_pipeline(tempfile.mkdtemp()).process_rows, rows, "real_dataset_dev.csv")
        cls.m = cls.alone.merge(cls.joint, on="log_id", suffixes=("", "_later"))

    def test_future_events_do_not_change_earlier_features_or_sequence_scores(self):
        self.assertEqual(len(self.m), self.N)
        for c in fe9.TEMPORAL_COLS + P.STRUCT_COLS:
            self.assertTrue((self.m[c].astype(str) == self.m[f"{c}_later"].astype(str)).all(), c)
        self.assertLess(float(np.abs(self.m["p_sequence"] - self.m["p_sequence_later"]).max()), 1e-6)

    @unittest.expectedFailure
    def test_future_events_do_not_change_earlier_graph_scores(self):
        """KNOWN DEFECT: Pipeline.score builds ONE graph over the whole window, including
        events that arrived in the same file AFTER the event being scored, and the HGT's
        message passing / rank-normalised edge features use all of them. So an event's
        p_graph -- and its ensemble risk, at w=0.8 -- depends on how the stream was cut
        into files (observed: up to ~0.28 on the first 120 dev events). evaluate_pipeline
        replays in 250-row chunks, so its graph scores carry within-chunk look-ahead."""
        g = self.m[["p_graph", "p_graph_later"]].dropna()
        self.assertGreater(len(g), 0)
        self.assertLess(float(np.abs(g["p_graph"] - g["p_graph_later"]).max()), 1e-6)


# ── the committed split files ─────────────────────────────────────────────────

class TestCommittedSplitBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = cs.read_split_file(SPLIT_FILE)
        cls.ann = cs.load_annotations()
        cls.users = pd.read_csv(cs.DEFAULT_EVENTS, usecols=["username", "campaign_id"], dtype=str,
                                keep_default_na=False)
        cls.ann["split"] = cls.ann["log_id"].map(cls.split)
        cls.ann["username"] = cls.users["username"].to_numpy()

    def test_split_file_covers_every_synthetic_row(self):
        self.assertEqual(list(self.split), list(self.ann["log_id"]))

    def test_campaign_families_and_campaigns_stay_in_one_split(self):
        """Every row of a family -- label-0 context included -- and of a campaign_id."""
        fam = self.ann[self.ann["chain_name"] != ""]
        self.assertEqual(int(fam.groupby("chain_name")["split"].nunique().max()), 1)
        camp = self.ann[self.ann["campaign_id"] != ""]
        self.assertGreater(len(camp), 0)
        self.assertEqual(int(camp.groupby("campaign_id")["split"].nunique().max()), 1)
        self.assertEqual(set(self.ann.loc[self.ann["label"] == "1", "split"]), set(cs.SPLIT_NAMES))

    def test_background_users_stay_in_one_split(self):
        bg = self.ann[self.ann["chain_name"] == ""]
        self.assertEqual(int(bg.groupby("username")["split"].nunique().max()), 1)

    def test_split_file_is_what_campaign_split_regenerates(self):
        self.assertEqual(cs.build_assignment(), self.split)

    def test_real_dev_and_test_share_no_session(self):
        dev = set(pd.read_csv(DEV, usecols=["session_id"])["session_id"])
        test = set(pd.read_csv(TEST, usecols=["session_id"])["session_id"])
        self.assertEqual(dev & test, set())

    @unittest.expectedFailure
    def test_real_attack_principals_do_not_cross_dev_and_test(self):
        """KNOWN DEFECT: split_real_dev_test.py splits by session_id only, and one
        attacker identity runs several sessions, so attack principals seen (with labels)
        in dev reappear in test -- e.g. stratus-red-team-ec2-get-password-data-role,
        stratus-redteam1, nexus-user-1. The deployed LSTM (v6.3-ft) was fine-tuned on dev
        labels and w/threshold were tuned on dev, so the test sessions of these
        principals are not independent of what was tuned on. (AWS service principals,
        *.amazonaws.com, are excluded: they are shared infrastructure, not attackers.)"""
        dev = pd.read_csv(DEV, usecols=["username", "label"])
        attackers = {u for u in dev.loc[dev["label"] == 1, "username"].dropna() if not u.endswith("amazonaws.com")}
        test_users = set(pd.read_csv(TEST, usecols=["username"])["username"].dropna())
        self.assertEqual(sorted(attackers & test_users), [])


# ── disclosed limitations ─────────────────────────────────────────────────────

class TestDisclosedLimitations(unittest.TestCase):
    @unittest.skip("LIMITATION: real_dataset_test.csv was evaluated repeatedly during earlier structural "
                   "bug verification (docs/PROJECT_STATUS_REPORT.md sections 6.9-6.15) before the dev-only "
                   "discipline was adopted, so it is not a pristine single-touch hold-out. Disclosed; not "
                   "testable after the fact.")
    def test_real_test_split_was_accessed_during_earlier_debugging(self):
        pass

    @unittest.skip("LIMITATION: the deployed LSTM (lstm_transformer_v6_3_ft) was fine-tuned on real_dataset_dev "
                   "labels (finetune_report.json, OOF heads), so any dev-set LSTM score or explanation metric is "
                   "in-sample for that branch.")
    def test_lstm_branch_dev_metrics_are_out_of_sample(self):
        pass


if __name__ == "__main__":
    unittest.main()
