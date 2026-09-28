"""
test_hgt_attention_dropout.py
=============================
Guards model_hgt's attention dropout on PyG versions whose HGTConv no longer
accepts `dropout=` / `group=`.

Before: both kwargs were dropped with a warning, so every HGT trained with
attn_dropout=0.1 trained without it. Now attention dropout is re-implemented by
dropping (edge, head) messages -- exactly equivalent to dropping the attention
coefficient -- and a group other than the fixed "sum" is an error.
"""

import unittest

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from model_hgt import HGTAnomalyDetector, _HGTConvAttnDropout, _make_hgt_conv

TRIPLE = ("User", "READ", "Resource")
METADATA = (["User", "Resource"], [TRIPLE])


def toy_data():
    torch.manual_seed(0)
    data = HeteroData()
    data["User"].x = torch.randn(6, 4)
    data["Resource"].x = torch.randn(5, 6)
    data[TRIPLE].edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 0]])
    data[TRIPLE].edge_attr = torch.randn(6, 7)
    data[TRIPLE].y = torch.zeros(6)
    return data


def model(attn_dropout, seed=0):
    torch.manual_seed(seed)
    return HGTAnomalyDetector(node_feat_dims={"User": 4, "Resource": 6}, edge_types=[TRIPLE],
                              edge_feat_dim=7, hidden_dim=16, heads=2, num_hgt_layers=1,
                              dropout=0.0, attn_dropout=attn_dropout)


def pyg_accepts_dropout():
    try:
        HGTConv(8, 8, METADATA, heads=2, dropout=0.1)
        return True
    except TypeError:
        return False


class TestConstruction(unittest.TestCase):
    def test_attention_dropout_is_never_silently_dropped(self):
        conv = _make_hgt_conv(16, METADATA, heads=2, group="sum", attn_dropout=0.1)
        if pyg_accepts_dropout():
            self.assertIsInstance(conv, HGTConv)       # PyG applies it natively
        else:
            self.assertIsInstance(conv, _HGTConvAttnDropout)
            self.assertEqual(conv.attn_dropout, 0.1)

    def test_unsupported_group_is_an_error_not_a_warning(self):
        try:
            HGTConv(8, 8, METADATA, heads=2, group="mean")
            self.skipTest("installed HGTConv still accepts group=")
        except TypeError:
            pass
        with self.assertRaises(ValueError):
            _make_hgt_conv(16, METADATA, heads=2, group="mean", attn_dropout=0.1)

    def test_sum_group_is_accepted(self):
        _make_hgt_conv(16, METADATA, heads=2, group="sum", attn_dropout=0.0)

    def test_no_extra_parameters_so_checkpoints_still_load(self):
        torch.manual_seed(0)
        plain = HGTConv(16, 16, METADATA, heads=2)
        torch.manual_seed(0)
        wrapped = _HGTConvAttnDropout(16, 16, METADATA, heads=2, attn_dropout=0.3)
        self.assertEqual(plain.state_dict().keys(), wrapped.state_dict().keys())
        wrapped.load_state_dict(plain.state_dict())


class TestBehaviour(unittest.TestCase):
    def setUp(self):
        if pyg_accepts_dropout():
            self.skipTest("installed HGTConv applies attention dropout natively")
        self.data = toy_data()

    def test_eval_mode_is_unaffected(self):
        a, b = model(0.0).eval(), model(0.5).eval()
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.allclose(a(self.data), b(self.data)))

    def test_zero_dropout_matches_the_plain_conv_in_training(self):
        a, b = model(0.0).train(), model(0.0).train()
        b.load_state_dict(a.state_dict())
        self.assertTrue(torch.allclose(a(self.data), b(self.data)))

    def test_training_mode_applies_dropout(self):
        a, b = model(0.0).train(), model(0.5).train()
        b.load_state_dict(a.state_dict())
        torch.manual_seed(1)
        self.assertFalse(torch.allclose(a(self.data), b(self.data)))

    def test_whole_edge_head_messages_are_dropped(self):
        """Each (edge, head) message block is either zeroed or scaled by
        1/(1-p) as a whole -- i.e. its attention coefficient is dropped."""
        conv = _HGTConvAttnDropout(8, 8, METADATA, heads=2, attn_dropout=0.5)
        messages = torch.ones(200, 8)                   # 200 edges x (2 heads x 4 dims)
        original = HGTConv.message
        HGTConv.message = lambda self, *a, **k: messages
        try:
            torch.manual_seed(0)
            unused = (None,) * 7   # the stub ignores message()'s inputs
            trained = conv.train().message(*unused)
            evaluated = conv.eval().message(*unused)
        finally:
            HGTConv.message = original
        blocks = trained.reshape(200, 2, 4)
        per_block = blocks[..., 0]
        self.assertTrue(torch.equal(blocks, per_block.unsqueeze(-1).expand_as(blocks)))
        self.assertTrue(set(per_block.unique().tolist()) <= {0.0, 2.0})
        self.assertTrue((per_block == 0).any() and (per_block == 2).any())
        self.assertTrue(torch.equal(evaluated, messages))

if __name__ == "__main__":
    unittest.main()
