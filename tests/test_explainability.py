"""
test_explainability.py
======================
Guards graph_construction/explainability.py, which had no tests.

Two real bugs it pins:

  * Feature NAMES vs COLUMNS. EDGE_FEATURE_NAMES was a hand-written list in a
    different order than data_loader._edge_features() builds edge_attr, so from
    column 3 on every importance was reported under the wrong name, and the
    one-hot edge_type block was cut to its first column. Names now derive from
    data_loader.EDGE_ATTR_NUMERIC_COLS; the contract test builds real edge
    features and checks each named column holds that feature's value.
  * Scoring ORDER. The models score scored_edge_types(data) (reverse edges
    excluded) restricted to their own triples. The explainer walked
    sorted(data.edge_types), so with reverse edges or an untrained triple it
    explained the wrong edge (or raised on a reverse triple's missing labels).
"""

import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import HeteroData

from data_loader import (EDGE_ATTR_NUMERIC_COLS, EDGE_NUM_COLS, PrivilegePropagationGraphLoader,
                         UNK_CATEGORY, scored_edge_types)
from explainability import (EDGE_FEATURE_NAMES, EdgeExplainer, FeatureAblation, TargetEdge,
                            _feature_groups)

READ = ("User", "READ", "Resource")
LIST = ("User", "LIST", "Resource")          # labelled, but the toy model has no weights for it
WRITE = ("Role", "WRITE", "Resource")
REV = ("Resource", "REV_READ", "User")       # reverse mirror: no labels, never scored
N_NUM = len(EDGE_ATTR_NUMERIC_COLS)
N_COLS = N_NUM + 4                            # + a 4-class one-hot edge_type block


class LinearEdgeModel(nn.Module):
    """logit = edge_attr . w for every edge it scores, in the models' order."""

    def __init__(self, edge_types, n_cols=N_COLS, weights=None):
        super().__init__()
        self.edge_types = sorted(edge_types)
        self.w = nn.Parameter(torch.ones(n_cols) if weights is None else weights)

    def forward(self, data):
        return torch.cat([data[t].edge_attr @ self.w
                          for t in scored_edge_types(data) if t in self.edge_types])


def toy_graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    for ntype, n in (("User", 3), ("Role", 2), ("Resource", 3)):
        data[ntype].x = torch.zeros(n, 1)
    for triple, n in ((READ, 3), (LIST, 2), (WRITE, 2)):
        data[triple].edge_index = torch.zeros((2, n), dtype=torch.long)
        data[triple].edge_attr = torch.rand((n, N_COLS), generator=g) + 0.1
        data[triple].y = torch.zeros(n)
    data[REV].edge_index = torch.zeros((2, 3), dtype=torch.long)
    data[REV].edge_attr = torch.rand((3, N_COLS), generator=g)
    return data


class TestColumnContract(unittest.TestCase):
    def test_names_follow_the_loaders_numeric_columns_then_edge_type(self):
        self.assertEqual(EDGE_FEATURE_NAMES[:-1], EDGE_ATTR_NUMERIC_COLS)
        self.assertEqual(EDGE_FEATURE_NAMES[-1], "edge_type")
        self.assertEqual(EDGE_ATTR_NUMERIC_COLS[:len(EDGE_NUM_COLS)], EDGE_NUM_COLS)

    def test_each_named_column_holds_that_feature(self):
        """Build edge features with the real _edge_features and an identity
        scaler, and check every named position carries its own value."""
        loader = PrivilegePropagationGraphLoader.__new__(PrivilegePropagationGraphLoader)
        expected = {"hop_count": 11.0, "privilege_gain": 22.0, "privilege_gain_defined": 1.0,
                    "action_global_frequency_log": 3.0, "is_privilege_escalation_technique": 0.0,
                    "is_read_only": 1.0, "abnormal_path_frequency_rank": 0.77}
        df = pd.DataFrame([{
            "hop_count": 11, "privilege_gain": 22.0, "privilege_gain_defined": True,
            "action_global_frequency": float(np.expm1(3.0)), "is_privilege_escalation_technique": False,
            "is_read_only": True, "abnormal_path_frequency_rank": 0.77, "edge_type": "GetObject",
        }])
        loader.edge_scaler = StandardScaler(with_mean=False, with_std=False).fit(np.zeros((2, len(EDGE_NUM_COLS))))
        enc = LabelEncoder().fit(["GetObject", "PutObject", UNK_CATEGORY])
        attr = loader._edge_features(df, enc)[0]
        self.assertEqual(attr.shape[0], N_NUM + len(enc.classes_))
        for name, value in expected.items():
            self.assertAlmostEqual(float(attr[EDGE_ATTR_NUMERIC_COLS.index(name)]), value, places=5, msg=name)
        self.assertEqual(float(attr[N_NUM:].sum()), 1.0)                  # one-hot block after them

    def test_groups_cover_every_column_once(self):
        groups = _feature_groups(N_COLS, EDGE_FEATURE_NAMES)
        cols = [c for _, cs in groups for c in cs]
        self.assertEqual(sorted(cols), list(range(N_COLS)))
        self.assertEqual(dict(groups)["edge_type"], list(range(N_NUM, N_COLS)))


class TestScoringOrder(unittest.TestCase):
    def setUp(self):
        self.data = toy_graph()
        self.model = LinearEdgeModel([READ, WRITE])      # knows neither LIST nor REV
        self.explainer = EdgeExplainer(self.model)

    def test_scored_triples_match_the_model_output(self):
        triples = self.explainer._scored_triples(self.data)
        self.assertEqual(triples, [WRITE, READ])
        with torch.no_grad():
            n_out = self.model(self.data).shape[0]
        self.assertEqual(n_out, sum(self.data[t].y.shape[0] for t in triples))

    def test_explains_the_requested_edge(self):
        """With w = 1 the gradient of edge i's logit is 1 on edge i's own row
        and 0 elsewhere, so importance = |x_i| normalised. The wrong offset
        would read a zero-gradient row and report ~0 everywhere."""
        for local in range(3):
            got = self.explainer.explain(self.data, TargetEdge(READ, local))
            row = self.data[READ].edge_attr[local].detach().abs().numpy()
            want = {n: float(row[c].sum() / row.sum()) for n, c in _feature_groups(N_COLS, EDGE_FEATURE_NAMES)}
            for name, value in want.items():
                self.assertAlmostEqual(got[name], value, places=5, msg=(local, name))

    def test_probabilities_line_up_with_their_targets(self):
        probs, targets = self.explainer._flat_probs_and_targets(self.data)
        self.assertEqual(len(probs), len(targets))
        for p, t in zip(probs, targets):
            x = self.data[t.triple].edge_attr[t.local_index]
            self.assertAlmostEqual(float(p), float(torch.sigmoid(x.sum())), places=5)

    def test_top_k_respects_the_mask(self):
        data = self.data
        masks = {t: torch.zeros(data[t].y.shape[0], dtype=torch.bool) for t in scored_edge_types(data)}
        masks[READ][1] = True
        top = self.explainer.explain_top_k(data, masks, k=1)
        self.assertEqual(list(top), [TargetEdge(READ, 1)])

    def test_importances_are_normalised_with_one_hot_grouped(self):
        got = self.explainer.explain(self.data, TargetEdge(WRITE, 0))
        self.assertEqual(set(got), set(EDGE_FEATURE_NAMES))
        self.assertAlmostEqual(sum(got.values()), 1.0, places=5)


class MixingEdgeModel(nn.Module):
    """logit_i = x_i . w + (sum of every edge's row) . v -- each edge's score also
    depends on its neighbours, as it does through message passing."""

    def __init__(self, edge_types):
        super().__init__()
        self.edge_types = sorted(edge_types)
        g = torch.Generator().manual_seed(3)
        self.w = nn.Parameter(torch.rand(N_COLS, generator=g))
        self.v = nn.Parameter(torch.rand(N_COLS, generator=g))

    def forward(self, data):
        outs = []
        for t in scored_edge_types(data):
            if t in self.edge_types:
                x = data[t].edge_attr
                outs.append(x @ self.w + (x.sum(0) @ self.v))
        return torch.cat(outs)


class TestRepeatedExplanations(unittest.TestCase):
    def test_explaining_one_edge_does_not_leak_into_the_next(self):
        """Regression: edge_attr.grad accumulated across calls, so the second
        explanation on the same data included the first edge's gradient."""
        model = LinearEdgeModel([READ, WRITE])
        mixing = MixingEdgeModel([READ, WRITE])
        for m in (model, mixing):
            fresh = EdgeExplainer(m).explain(toy_graph(), TargetEdge(READ, 1))
            data = toy_graph()
            explainer = EdgeExplainer(m)
            explainer.explain(data, TargetEdge(READ, 0))
            again = explainer.explain(data, TargetEdge(READ, 1))
            for name in fresh:
                self.assertAlmostEqual(fresh[name], again[name], places=5, msg=(type(m).__name__, name))


class TestFeatureAblation(unittest.TestCase):
    def test_edge_type_ablation_zeroes_the_whole_one_hot_block(self):
        w = torch.zeros(N_COLS)
        w[N_NUM:] = 1.0                                   # model reads ONLY the one-hot block
        model = LinearEdgeModel([READ, WRITE], weights=w)
        data = toy_graph()
        before = {t: data[t].edge_attr.clone() for t in data.edge_types}

        def evaluate_fn(m, d, _masks):
            with torch.no_grad():
                return {"f1": float(m(d).sum())}

        drops = FeatureAblation(model).run(data, {}, evaluate_fn)
        baseline = evaluate_fn(model, data, {})["f1"]
        self.assertAlmostEqual(drops["edge_type"], baseline, places=4)
        for name in EDGE_ATTR_NUMERIC_COLS:
            self.assertAlmostEqual(drops[name], 0.0, places=5, msg=name)
        for t in data.edge_types:                          # originals restored
            self.assertTrue(torch.equal(data[t].edge_attr, before[t]))


if __name__ == "__main__":
    unittest.main()
