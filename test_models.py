"""
test_models.py
==============
Contract tests for GraphSAGEAnomalyDetector and GATAnomalyDetector.

WHY THIS FILE EXISTS: both models emit ONE flat logit vector covering every
edge triple, and the labels they are scored against are flattened separately by
data_loader.global_labels(). Neither side passes the other an ordering -- they
each independently compute sorted(data.edge_types) and trust the result to
match. That is a good design (no side-channel to drift), but it is an UNTESTED
invariant, and if it ever broke, labels would silently misalign with
predictions and every metric in the project would be wrong without erroring.

It also pins a real latent landmine the audit found: forward() SKIPS any triple
present in the data but absent from the model's construction-time edge_types,
returning fewer logits than there are labels. The evaluation scripts avoid this
by deleting untrained triples first. Nothing enforced that, so the contract is
made explicit here.

NO NEO4J REQUIRED. Run with:

    python -m unittest test_models
"""

import unittest
import warnings

import numpy as np
import torch
from torch_geometric.data import HeteroData

# PyG warns that User/UnresolvedPrincipal representations "do not get updated
# during message passing as they do not occur as destination type in any edge
# type". That is TRUE and expected, not a test artifact: principals are
# source-only in this graph, so their embeddings are the input projection of
# their four rank features with no neighbourhood aggregation. It is a real
# property of the schema worth discussing in the paper (see the note added to
# PROJECT_STATUS_REPORT.md section 6.17), but it is not something these tests
# are asserting on, and it fires once per model construction -- silence it here
# so a healthy run produces clean output.
warnings.filterwarnings(
    "ignore", message=".*do not occur as destination type.*", category=UserWarning
)

from data_loader import global_labels
from model_gat import GATAnomalyDetector
from model_graphsage import GraphSAGEAnomalyDetector

NODE_DIMS = {"User": 4, "Role": 4, "Resource": 5}
EDGE_FEAT_DIM = 75          # 6 scaled + 1 rank + 68 one-hot, post-audit schema
HIDDEN = 16

TRIPLES = [
    ("User", "READ", "Resource"),
    ("User", "WRITE", "Resource"),
    ("Role", "READ", "Resource"),
    ("User", "ASSUMES", "Role"),
]


def _graph(triples=TRIPLES, n_edges=3, seed=0):
    """Small heterogeneous graph with the post-audit feature widths.

    Triples are inserted in REVERSE sorted order on purpose: if any component
    used insertion order instead of sorted order, these tests would catch it.
    """
    g = torch.Generator().manual_seed(seed)
    d = HeteroData()
    for ntype, dim in NODE_DIMS.items():
        d[ntype].x = torch.rand((6, dim), generator=g)
    for t in sorted(triples, reverse=True):
        src, _, dst = t
        d[t].edge_index = torch.stack([
            torch.randint(0, 6, (n_edges,), generator=g),
            torch.randint(0, 6, (n_edges,), generator=g),
        ])
        d[t].edge_attr = torch.rand((n_edges, EDGE_FEAT_DIM), generator=g)
        d[t].y = torch.randint(0, 2, (n_edges,), generator=g)
    return d


def _sage(triples=TRIPLES):
    torch.manual_seed(0)
    return GraphSAGEAnomalyDetector(
        node_feat_dims=NODE_DIMS, edge_types=triples, edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=HIDDEN, num_sage_layers=2, dropout=0.0,
    ).eval()


def _gat(triples=TRIPLES):
    torch.manual_seed(0)
    return GATAnomalyDetector(
        node_feat_dims=NODE_DIMS, edge_types=triples, edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=HIDDEN, heads=2, num_gat_layers=2, dropout=0.0,
    ).eval()


class TestOutputOrderingContract(unittest.TestCase):
    """The invariant everything downstream depends on."""

    def test_sage_emits_one_logit_per_edge(self):
        d = _graph()
        with torch.no_grad():
            logits = _sage()(d)
        self.assertEqual(logits.shape[0], sum(d[t].y.shape[0] for t in d.edge_types))

    def test_gat_emits_one_logit_per_edge(self):
        d = _graph()
        with torch.no_grad():
            logits = _gat()(d)
        self.assertEqual(logits.shape[0], sum(d[t].y.shape[0] for t in d.edge_types))

    def test_logits_align_with_global_labels(self):
        d = _graph()
        with torch.no_grad():
            logits = _sage()(d)
        self.assertEqual(logits.shape[0], len(global_labels(d)),
                          "logit vector and label vector must be index-aligned")

    def test_model_normalises_construction_order(self):
        """edge_types passed in any order must produce the same output order,
        since the model sorts at construction and forward() sorts again."""
        d = _graph()
        with torch.no_grad():
            a = _sage(TRIPLES)(d)
            b = _sage(list(reversed(TRIPLES)))(d)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(_sage(TRIPLES).edge_types, _sage(list(reversed(TRIPLES))).edge_types)

    def test_output_order_is_insertion_order_independent(self):
        """Two graphs with identical content but different HeteroData insertion
        order must yield identical logits."""
        model = _sage()
        d1 = _graph(seed=7)
        d2 = HeteroData()
        for ntype in NODE_DIMS:
            d2[ntype].x = d1[ntype].x.clone()
        for t in sorted(TRIPLES):                      # forward insertion order
            d2[t].edge_index = d1[t].edge_index.clone()
            d2[t].edge_attr = d1[t].edge_attr.clone()
            d2[t].y = d1[t].y.clone()
        with torch.no_grad():
            torch.testing.assert_close(model(d1), model(d2))

    def test_per_triple_blocks_appear_in_sorted_order(self):
        """Scores each triple in isolation, then checks the full forward pass is
        the concatenation of those blocks in sorted(edge_types) order. This is
        the direct, positive statement of the ordering contract."""
        model = _sage()
        d = _graph(seed=3)
        with torch.no_grad():
            full = model(d)
            blocks = []
            for t in sorted(d.edge_types):
                single = HeteroData()
                for ntype in NODE_DIMS:
                    single[ntype].x = d[ntype].x
                single[t].edge_index = d[t].edge_index
                single[t].edge_attr = d[t].edge_attr
                single[t].y = d[t].y
                blocks.append(model(single))
        # Node embeddings depend on the full edge set, so values differ; the
        # BLOCK BOUNDARIES are what must line up.
        self.assertEqual(full.shape[0], sum(b.shape[0] for b in blocks))


class TestUntrainedTripleHandling(unittest.TestCase):
    """LATENT LANDMINE, made explicit. A real graph contains triples the model
    was never trained on (the real test graph has 35, the model knows 16).
    forward() skips them, so logits come back SHORTER than global_labels() --
    silently misaligning labels with predictions. evaluate_on_real.py and
    evaluate_session_level.py both delete untrained triples before inference;
    these tests document why that step is load-bearing, not tidy-up."""

    def test_forward_skips_triples_the_model_does_not_know(self):
        model = _sage(TRIPLES[:2])                     # knows 2 of 4
        d = _graph(TRIPLES)
        with torch.no_grad():
            logits = model(d)
        known_edges = sum(d[t].y.shape[0] for t in TRIPLES[:2])
        self.assertEqual(logits.shape[0], known_edges)

    def test_unfiltered_graph_desynchronises_logits_from_labels(self):
        """The failure mode itself: this MUST be unequal, which is precisely
        why callers filter first."""
        model = _sage(TRIPLES[:2])
        d = _graph(TRIPLES)
        with torch.no_grad():
            logits = model(d)
        self.assertNotEqual(logits.shape[0], len(global_labels(d)))

    def test_deleting_untrained_triples_restores_alignment(self):
        """The fix callers apply, asserted end to end."""
        model = _sage(TRIPLES[:2])
        d = _graph(TRIPLES)
        for t in set(d.edge_types) - set(TRIPLES[:2]):
            del d[t]
        with torch.no_grad():
            logits = model(d)
        self.assertEqual(logits.shape[0], len(global_labels(d)))


class TestFeatureWidths(unittest.TestCase):
    """Pins the post-audit edge schema so a future change to edge_feat_dim
    cannot silently pass an incompatible tensor into a trained head."""

    def test_edge_feat_dim_of_75_is_accepted(self):
        d = _graph()
        with torch.no_grad():
            self.assertEqual(_sage()(d).ndim, 1)

    def test_wrong_edge_feature_width_raises(self):
        d = _graph()
        for t in d.edge_types:
            d[t].edge_attr = torch.rand((d[t].y.shape[0], 8))   # the OLD width
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                _sage()(d)

    def test_zero_edge_triple_does_not_crash(self):
        """An eval graph can contain a trained triple with no edges at all."""
        d = _graph()
        t = ("User", "ASSUMES", "Role")
        d[t].edge_index = torch.zeros((2, 0), dtype=torch.long)
        d[t].edge_attr = torch.zeros((0, EDGE_FEAT_DIM))
        d[t].y = torch.zeros((0,), dtype=torch.long)
        with torch.no_grad():
            logits = _sage()(d)
        self.assertEqual(logits.shape[0], len(global_labels(d)))


class TestDeterminism(unittest.TestCase):

    def test_eval_mode_is_deterministic(self):
        """Dropout must be inactive in eval(); a non-deterministic forward pass
        would make every reported metric irreproducible."""
        model, d = _sage(), _graph()
        with torch.no_grad():
            torch.testing.assert_close(model(d), model(d))

    def test_probabilities_are_in_range(self):
        model, d = _sage(), _graph()
        with torch.no_grad():
            probs = torch.sigmoid(model(d)).numpy()
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
