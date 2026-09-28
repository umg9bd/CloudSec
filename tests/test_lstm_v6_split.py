"""
test_lstm_v6_split.py
=====================
Guards train_lstm_transformer_v6.py's campaign-family mode (--split
campaign_family): sequences follow the split of the event they score, every
event must be assigned, no user may straddle splits (window metrics select by
user), and the committed v6 artifacts are never the output location.
"""

import unittest
from types import SimpleNamespace

import train_lstm_transformer_v6 as v6


def seq(log_id, user, label=0):
    return SimpleNamespace(log_id=log_id, username=user, label=label)


SEQS = [seq("f:0", "u1", 1), seq("f:1", "u1", 0), seq("f:2", "u2", 1), seq("f:3", "u3", 1),
        seq("f:4", "u4", 0)]
SPLIT = {"f:0": "train", "f:1": "train", "f:2": "val", "f:3": "test", "f:4": "train"}


class TestCampaignSplitSeqs(unittest.TestCase):
    def test_sequences_follow_their_events_split(self):
        tr, va, te = v6.campaign_family_split_seqs(SEQS, SPLIT)
        self.assertEqual([s.log_id for s in tr], ["f:0", "f:1", "f:4"])
        self.assertEqual([s.log_id for s in va], ["f:2"])
        self.assertEqual([s.log_id for s in te], ["f:3"])

    def test_unassigned_event_is_refused(self):
        with self.assertRaises(SystemExit):
            v6.campaign_family_split_seqs(SEQS, {k: v for k, v in SPLIT.items() if k != "f:4"})

    def test_user_in_two_splits_is_refused(self):
        with self.assertRaises(SystemExit):
            v6.campaign_family_split_seqs(SEQS, dict(SPLIT, **{"f:1": "test"}))


class TestArguments(unittest.TestCase):
    def test_default_is_the_user_disjoint_protocol(self):
        self.assertEqual(v6.parse_args([]).split, "users")

    def test_campaign_mode_needs_its_inputs(self):
        with self.assertRaises(SystemExit):
            v6.main(["--split", "campaign_family"])

    def test_campaign_outputs_never_overwrite_the_committed_model(self):
        self.assertNotEqual(v6.CAMPAIGN_OUT_DIR, v6.OUT_DIR)
        self.assertNotEqual(v6.CAMPAIGN_OUT_DIR / "event_name_vocab.json", v6.VOCAB_PATH)


if __name__ == "__main__":
    unittest.main()
