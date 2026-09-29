"""
test_lstm_scorer.py
===================
Guards temporal-analysis/prod/scorer.py's loading of the v5 (live pipeline)
and v6 (user-disjoint general model) LSTM-Transformer checkpoints.

The scorer picks a vocabulary by looking at the checkpoint's file name when
the checkpoint does not embed one. A v6 checkpoint paired with the v5
vocabulary (281 names) would index past v6's 68-row embedding or, worse,
silently map event names to the wrong rows. These tests pin: which vocabulary
each checkpoint gets, that it fits the embedding, that the fallback file agrees
with what the checkpoint embeds, and that the live pipeline can feed either.
"""

import json
import os
import unittest
from pathlib import Path

import pandas as pd
import torch

import feature_engine9 as fe
import train_lstm_transformer as tlt
from prod import scorer as sc

ROOT = Path(__file__).resolve().parent.parent
TA = ROOT / "temporal-analysis"
V5 = TA / "artifacts" / "lstm_transformer_clean" / "temporal_lstm_transformer.pt"
V6 = TA / "artifacts" / "lstm_transformer_v6" / "temporal_lstm_transformer_v6.pt"
SMOKE_CSV = TA / "data" / "lstm" / "train_temporal.csv"


def embedding_rows(scorer):
    return scorer.model.state_dict()["embedding.weight"].shape[0]


class TestVocabFallback(unittest.TestCase):
    def test_v6_directory_gets_the_v6_vocab(self):
        vocab = sc._vocab_fallback(Path("anywhere/lstm_transformer_v6/model.pt"))
        self.assertEqual(vocab, json.loads(sc.V6_VOCAB_PATH.read_text(encoding="utf-8")))

    def test_v6_file_name_gets_the_v6_vocab(self):
        vocab = sc._vocab_fallback(Path("anywhere/some_model_v6.pt"))
        self.assertEqual(vocab, json.loads(sc.V6_VOCAB_PATH.read_text(encoding="utf-8")))

    def test_windows_separators_are_recognized(self):
        vocab = sc._vocab_fallback(Path("C:\\x\\lstm_transformer_v6\\model.pt"))
        self.assertEqual(len(vocab), len(json.loads(sc.V6_VOCAB_PATH.read_text(encoding="utf-8"))))

    def test_other_checkpoints_get_the_default_vocab(self):
        if not tlt.VOCAB_PATH.exists():
            self.skipTest("default vocab file not present")
        vocab = sc._vocab_fallback(Path("anywhere/temporal_lstm_transformer.pt"))
        self.assertEqual(vocab, json.loads(tlt.VOCAB_PATH.read_text(encoding="utf-8")))


class TestCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorers = {}
        for name, path in (("v5", V5), ("v6", V6)):
            if not path.exists():
                raise unittest.SkipTest(f"missing checkpoint {path}")
            cls.scorers[name] = sc.load_scorer(path, device=torch.device("cpu"))

    def test_schema_versions_identify_the_model(self):
        self.assertEqual(self.scorers["v5"].schema_version, "lstm_transformer_v5.0")
        # v6 is retrained in place (v6.0 -> v6.2 on the Temporal-Analyst branch); pin the family.
        self.assertTrue(self.scorers["v6"].schema_version.startswith("lstm_transformer_v6"),
                        self.scorers["v6"].schema_version)

    def test_vocab_fits_the_embedding(self):
        for name, s in self.scorers.items():
            self.assertLess(max(s.vocab.values()), embedding_rows(s), name)

    def test_models_have_different_vocabularies(self):
        self.assertNotEqual(len(self.scorers["v5"].vocab), len(self.scorers["v6"].vocab))

    def test_v6_fallback_file_agrees_with_the_embedded_vocab(self):
        embedded = self.scorers["v6"].vocab
        self.assertEqual(embedded, json.loads(sc.V6_VOCAB_PATH.read_text(encoding="utf-8")))

    def test_the_live_feature_engine_can_feed_both(self):
        """pipeline.py passes feature_engine9.TEMPORAL_COLS; the PE-context
        columns are derived inside the scorer. Anything else would be missing."""
        for name, s in self.scorers.items():
            missing = [c for c in s.ckpt["feature_cols"]
                       if c not in fe.TEMPORAL_COLS and c not in tlt.PE_CONTEXT_COLS]
            self.assertEqual(missing, [], name)

    def test_both_score_probabilities(self):
        if not SMOKE_CSV.exists():
            self.skipTest(f"missing {SMOKE_CSV}")
        df = pd.read_csv(SMOKE_CSV, nrows=300)
        for name, s in self.scorers.items():
            out = sc.score_dataframe(df, s)
            self.assertGreater(len(out), 0, name)
            self.assertFalse(out["P_seq"].isna().any(), name)
            self.assertTrue(((out["P_seq"] >= 0) & (out["P_seq"] <= 1)).all(), name)


class TestConfiguredLiveLstm(unittest.TestCase):
    """Whatever pipeline_config.json serves (v6.3-ft since 2026-09-29) must exist,
    load through the pipeline's scorer, and be fully fed by feature_engine9."""

    def test_configured_checkpoint_loads_and_can_be_fed(self):
        with open(ROOT / "pipeline_config.json", encoding="utf-8") as f:
            path = ROOT / json.load(f)["lstm_checkpoint"]
        self.assertTrue(path.exists(), path)
        s = sc.load_scorer(path, device=torch.device("cpu"))
        self.assertLess(max(s.vocab.values()), embedding_rows(s))
        missing = [c for c in s.ckpt["feature_cols"]
                   if c not in fe.TEMPORAL_COLS and c not in tlt.PE_CONTEXT_COLS]
        self.assertEqual(missing, [])

    def test_rollback_checkpoint_is_still_available(self):
        self.assertTrue(V5.exists(), V5)


if __name__ == "__main__":
    unittest.main()
