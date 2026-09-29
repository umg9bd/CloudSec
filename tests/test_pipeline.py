"""
test_pipeline.py
================
The real-time pipeline's promises that a refactor could silently break:

  - Streaming = batch for the sequence branch: scoring real dev in small file-sized chunks gives
    each event the same LSTM probability as one pass over all of it (history buffer, predecessor
    events, tie order). This broke three different ways while pipeline.py was being written.
  - The ensemble falls back to the sequence score when the graph branch has none.
  - An alert file is written for flagged events, with the schema alert consumers read.
"""
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import feature_engine9 as fe9
import train_lstm_transformer as tlt
from pipeline import Pipeline, PipelineConfig, ensemble_risk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(ROOT, "datasets", "privilege-escalation", "real_dataset_dev.csv")
N_EVENTS, CHUNK = 1500, 100


def _pipeline(tmp):
    return Pipeline(PipelineConfig(state_dir=os.path.join(tmp, "state"), alert_dir=os.path.join(tmp, "alerts"),
                                   output_csv=os.path.join(tmp, "out", "scores.csv")), write_outputs=False)


class TestEnsemble(unittest.TestCase):
    def test_graph_score_missing_falls_back_to_sequence(self):
        r = ensemble_risk([0.8, np.nan], [0.2, 0.3], 0.4)
        np.testing.assert_allclose(r, [0.4 * 0.8 + 0.6 * 0.2, 0.3])


class TestStreamingEqualsBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cwd = os.getcwd()
        os.chdir(ROOT)  # feature_engine9 paths are repo-relative
        try:
            cls.rows = list(enumerate(fe9.iter_input_rows(DEV)))[:N_EVENTS]
            with tempfile.TemporaryDirectory() as tmp:
                pipe = _pipeline(tmp)
                cls.events = pd.concat([pipe.process_rows(cls.rows[i:i + CHUNK], "real_dataset_dev.csv")
                                        for i in range(0, len(cls.rows), CHUNK)], ignore_index=True)
                frame = tlt.prepare_score_frame(
                    cls.events[["log_id", "username", "timestamp", "event_name", "label"] + fe9.TEMPORAL_COLS],
                    pipe.lstm.vocab, pipe.lstm_features)
                cls.batch = tlt.score_seqs(pipe.lstm.model, tlt.build_event_sequences(frame, pipe.lstm_features),
                                           pipe.lstm.device)
        finally:
            os.chdir(cwd)

    def test_every_event_scored_once(self):
        self.assertEqual(len(self.events), N_EVENTS)
        self.assertEqual(self.events["log_id"].nunique(), N_EVENTS)

    def test_lstm_streaming_matches_one_batch_pass(self):
        m = self.events[["log_id", "p_sequence"]].merge(self.batch[["log_id", "P_event"]], on="log_id")
        self.assertEqual(len(m), N_EVENTS)
        self.assertLess(float(np.abs(m["p_sequence"] - m["P_event"]).max()), 1e-5)

    def test_scores_are_probabilities(self):
        for c in ("p_sequence", "risk"):
            self.assertTrue(self.events[c].between(0, 1).all(), c)
        g = self.events["p_graph"].dropna()
        self.assertGreater(len(g), 0.9 * N_EVENTS)  # nearly every real event is in a trained triple
        self.assertTrue(g.between(0, 1).all())


class TestAlerts(unittest.TestCase):
    def test_attack_sample_raises_alert_with_schema(self):
        sample = os.path.join(ROOT, "samples", "cloudtrail", "synthetic_attack_chain.json")
        with tempfile.TemporaryDirectory() as tmp:
            pipe = _pipeline(tmp)
            pipe.write_outputs = True
            scored = pipe.process_file(sample)
            self.assertTrue(scored["alert"].any())
            files = os.listdir(os.path.join(tmp, "alerts"))
            self.assertTrue(files)
            with open(os.path.join(tmp, "alerts", files[0]), encoding="utf-8") as f:
                alert = json.load(f)
            for key in ("alert_id", "principal", "max_risk_score", "n_flagged_events", "events", "threshold"):
                self.assertIn(key, alert)
            self.assertTrue(os.path.exists(os.path.join(tmp, "out", "scores.csv")))


if __name__ == "__main__":
    unittest.main()
