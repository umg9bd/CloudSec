"""
test_lstm_explain.py
====================
Guards temporal-analysis/lstm_explain.py (LSTM + Transformer explanations for the v5 and v6.3
checkpoints) on a small random model -- no trained checkpoint or data needed:
  - Integrated Gradients is complete: contributions sum to logit(x) - logit(baseline)
  - leave-one-event-out drops exactly one event and merges its time gap into the next one
  - explain_seq returns JSON-ready output with one entry per earlier event
"""

import json
import math
import unittest

import numpy as np
import torch

import lstm_explain as lx
import train_lstm_transformer as tlt

N_FEAT = 5  # 4 named features + the appended time-gap column


def make_model(seed=0):
    torch.manual_seed(seed)
    m = tlt.LSTMTransformerModel(vocab_size=12, n_features=N_FEAT, secret_ids={3})
    for p in m.parameters():  # heads start at zero; give every path a signal
        torch.nn.init.normal_(p, std=0.3)
    return m.eval()


def make_seq(length=6, seq_len=tlt.SEQ_LEN, seed=1):
    rng = np.random.default_rng(seed)
    idxs = rng.integers(1, 12, size=length).astype(np.int64)
    feats = rng.random((length, N_FEAT)).astype(np.float32)
    feats[:, -1] = np.log1p(rng.integers(1, 90, size=length)).astype(np.float32)
    feats[0, -1] = 0.0
    idxs, feats, n = tlt.pad_seq(idxs, feats, seq_len)
    return tlt.EventSeq(username="u", timestamp="2026-01-01T00:00:00Z", log_id="x:0", event_idxs=idxs,
                        feats=feats, length=n, label=1, last_idx=int(idxs[n - 1]), label_orig=1)


class TestLstmExplain(unittest.TestCase):
    def setUp(self):
        self.model, self.seq = make_model(), make_seq()
        self.names = ["f0", "f1", "f2", "f3", lx.DT_FEATURE]
        self.dev = torch.device("cpu")

    def test_ig_completeness(self):
        ig = lx.integrated_gradients(self.model, self.seq, self.dev, steps=64)
        gap = abs(ig["logit"] - ig["baseline_logit"])
        self.assertGreater(gap, 1e-3)
        self.assertLess(ig["completeness_error"], 0.05 * max(gap, 1.0))
        self.assertEqual(ig["feats"].shape, (self.seq.length, N_FEAT))

    def test_drop_event_merges_gap(self):
        L, dt = self.seq.length, N_FEAT - 1
        idx, feats = np.array(self.seq.event_idxs), np.array(self.seq.feats)
        new_idx, new_feats = lx.drop_event(idx, feats, L, 2, dt)
        self.assertEqual(list(new_idx[: L - 1]), [idx[k] for k in range(L) if k != 2])
        self.assertTrue((new_idx[L - 1:] == 0).all())
        merged = math.log1p(math.expm1(feats[2, dt]) + math.expm1(feats[3, dt]))
        self.assertAlmostEqual(float(new_feats[2, dt]), merged, places=5)
        _, first_dropped = lx.drop_event(idx, feats, L, 0, dt)
        self.assertEqual(float(first_dropped[0, dt]), 0.0)  # new first event has no previous gap

    def test_explain_seq_json(self):
        id2name = {i: f"E{i}" for i in range(12)}
        e = lx.explain_seq(self.model, self.seq, self.names, id2name, self.dev, top_k=50)
        json.dumps(e)
        self.assertEqual(len(e["top_events"]), self.seq.length - 1)
        self.assertEqual(len(e["top_features"]), N_FEAT + 1)  # + the event-name embedding
        self.assertEqual(e["event_name"], f"E{self.seq.last_idx}")
        p = float(torch.sigmoid(self.model(*lx._tensors(self.seq, self.dev)))[0])
        self.assertAlmostEqual(e["score"], p, places=3)

    def test_single_event_window_has_no_event_effects(self):
        seq = make_seq(length=1)
        self.assertEqual(lx.event_effects(self.model, seq, N_FEAT - 1, self.dev), [])


if __name__ == "__main__":
    unittest.main()
