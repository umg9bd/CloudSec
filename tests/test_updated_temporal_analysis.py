"""
test_updated_temporal_analysis.py
=================================
Runs Nandan's updated-temporal-analysis tests (test_lstm_explain.py) in their
own folder and process: that folder carries its own train_lstm_transformer.py,
which would shadow temporal-analysis/'s if both were on this runner's path.

Also checks the claim in updated-temporal-analysis/README.md that v6.3 and
v6.3-ft plug into the live pipeline unchanged: they load through
prod.scorer.load_scorer (what pipeline.py uses), their vocab fits their
embedding, and every feature they read is produced by feature_engine9 (plus the
PE-context columns the scorer derives).
"""

import os
import subprocess
import sys
import unittest

import torch

import feature_engine9 as fe
import train_lstm_transformer as tlt
from prod import scorer as sc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(ROOT, "updated-temporal-analysis")
MODELS = {
    "v6.3": os.path.join(FOLDER, "artifacts", "lstm_transformer_v6_3", "temporal_lstm_transformer.pt"),
    "v6.3-ft": os.path.join(FOLDER, "artifacts", "lstm_transformer_v6_3_ft", "temporal_lstm_transformer.pt"),
}


class TestNandansSuite(unittest.TestCase):
    def test_lstm_explain_suite_passes(self):
        if not os.path.isdir(FOLDER):
            self.skipTest("updated-temporal-analysis not present")
        run = subprocess.run([sys.executable, "-m", "unittest", "test_lstm_explain"], cwd=FOLDER,
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-3000:])


class TestV63PlugsIntoThePipeline(unittest.TestCase):
    def test_models_load_and_can_be_fed(self):
        for name, path in MODELS.items():
            with self.subTest(name):
                if not os.path.exists(path):
                    self.skipTest(f"missing {path}")
                s = sc.load_scorer(path, device=torch.device("cpu"))
                rows = s.model.state_dict()["embedding.weight"].shape[0]
                self.assertLess(max(s.vocab.values()), rows)
                missing = [c for c in s.ckpt["feature_cols"]
                           if c not in fe.TEMPORAL_COLS and c not in tlt.PE_CONTEXT_COLS]
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
