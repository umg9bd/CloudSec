"""
test_lstm_v6_3.py
=================
Guards Nandan's LSTM v6.3 work (developed as updated-temporal-analysis/ on
feature/Temporal-Analyst) in its realtime-pipeline location.

  * Every path its scripts depend on resolves. They were committed in a
    folder where they could not run: the trainer looked for the split file at
    <repo>/splits/ and for the live v5 under its own artifacts/, and the folder
    carried an older copy of train_lstm_transformer.py (without the pandas-3
    timestamp fix). They now live in temporal-analysis/ as written for.
  * v6.3 and v6.3-ft plug into the live pipeline unchanged: they load through
    prod.scorer.load_scorer (what pipeline.py uses), their vocab fits their
    embedding, and every feature they read is produced by feature_engine9 plus
    the PE-context columns the scorer derives.
  * The split file still covers the current training features, so the
    trainer's own coverage assert holds.
"""

import unittest

import pandas as pd
import torch

import feature_engine9 as fe
import finetune_lstm_v6_3 as ft
import lstm_explain as lx
import train_lstm_transformer as tlt
import train_lstm_transformer_v6_3 as t63
from prod import scorer as sc


class TestScriptPaths(unittest.TestCase):
    def test_trainer_inputs_exist(self):
        for path in (t63.TRAIN_CSV, t63.REAL_DEV_CSV, t63.SPLIT_FILE, t63.CLEAN_CKPT, t63.CKPT_PATH):
            self.assertTrue(path.exists(), path)

    def test_finetune_and_explain_models_exist(self):
        self.assertTrue(ft.BASE_CKPT.exists(), ft.BASE_CKPT)
        self.assertTrue((ft.OUT_DIR / "temporal_lstm_transformer.pt").exists())
        for name, path in lx.MODELS.items():
            self.assertTrue(path.exists(), (name, path))

    def test_scripts_use_the_shared_lstm_module(self):
        """One train_lstm_transformer: the live pipeline's, with the timestamp fix."""
        self.assertIs(t63.v5, tlt)
        self.assertTrue(hasattr(tlt, "_ns"))

    def test_split_file_covers_the_training_features(self):
        import campaign_split
        split = campaign_split.read_split_file(str(t63.SPLIT_FILE))
        log_ids = pd.read_csv(t63.TRAIN_CSV, usecols=["log_id"])["log_id"].astype(str)
        self.assertTrue(log_ids.isin(split.keys()).all())


class TestPlugsIntoThePipeline(unittest.TestCase):
    def test_v6_3_and_v6_3_ft_load_and_can_be_fed(self):
        for name, path in (("v6.3", t63.CKPT_PATH), ("v6.3-ft", ft.OUT_DIR / "temporal_lstm_transformer.pt")):
            with self.subTest(name):
                s = sc.load_scorer(path, device=torch.device("cpu"))
                self.assertLess(max(s.vocab.values()), s.model.state_dict()["embedding.weight"].shape[0])
                missing = [c for c in s.ckpt["feature_cols"]
                           if c not in fe.TEMPORAL_COLS and c not in tlt.PE_CONTEXT_COLS]
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
