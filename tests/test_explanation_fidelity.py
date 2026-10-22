"""
test_explanation_fidelity.py
============================
Explanation FIDELITY, not just execution (execution/format is covered by
test_ensemble_explain, test_lstm_explain, test_explainability):

  1. Known cause, toy model -- a sequence model whose logit is driven by one action
     by construction: lstm_explain must rank that event first and the deletion test
     must agree with the ranking.
  2. Known cause, real LSTM -- a privilege-escalation write injected into a benign
     session, verified counterfactually to drive the later event's score: is it the
     top-ranked earlier event? (expected failure, see docstring)
  3. Deletion tests on the real models -- zeroing the top-attributed feature moves the
     logit more than zeroing a random one (LSTM IG; HGT gradient x input).
  4. Ensemble consistency -- the GNN / LSTM shares in each alert explanation add up to
     the risk the pipeline actually scored, with weight_graph from pipeline_config.json,
     and the graph explanation's logit is the explained model's own score.
  5. No invented actions -- for TP, FP, FN and low-confidence events, every action an
     explanation names occurs in the scored input, and no rule / technique is claimed
     that the input does not support.
  6. Real dev data -- top-1 / top-3 hit rate of the LSTM's event ranking against
     attack-labelled events, vs. the random-ranking baseline (sanity floor only).
     NOTE: the deployed LSTM (v6.3-ft) was fine-tuned on dev labels, so this is an
     in-sample number for that branch.

Uses the real checkpoints from pipeline_config.json (CPU). No test data is read.
"""

import contextlib
import io
import logging
import math
import os
import re
import tempfile
import unittest
from math import comb
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import ensemble_explain as ee
import feature_engine9 as fe9
import lstm_explain as lx
import pipeline as P
import train_lstm_transformer as tlt
from explainability import EDGE_FEATURE_NAMES, _feature_groups
from data_loader import scored_edge_types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "privilege-escalation")
DEV = os.path.join(DATA, "real_dataset_dev.csv")
ATTACK = os.path.join(ROOT, "samples", "cloudtrail", "synthetic_attack_chain.json")
CPU = torch.device("cpu")

_PIPE = {}


def setUpModule():
    logging.disable(logging.CRITICAL)


def tearDownModule():
    logging.disable(logging.NOTSET)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def real_pipeline():
    """The configured pipeline (3-seed HGT + v6.3-ft LSTM), loaded once, fresh buffer."""
    if not _PIPE:
        tmp = tempfile.mkdtemp()
        cfg = P.PipelineConfig.load()
        cfg.state_dir, cfg.alert_dir = os.path.join(tmp, "state"), os.path.join(tmp, "alerts")
        _PIPE["p"] = quiet(P.Pipeline, cfg, write_outputs=False)
    p = _PIPE["p"]
    p.buffer = pd.DataFrame()
    return p


def frozen_engine():
    d = tempfile.mkdtemp()
    return fe9.FeatureEngineer(event_name_vocab_path=os.path.join(ROOT, fe9.EVENT_NAME_VOCAB_FILE),
                               state_tracker_path=os.path.join(d, "t.json"), graph_state_path=os.path.join(d, "g.json"),
                               identity_state_path=os.path.join(d, "i.json"),
                               action_prior_path=os.path.join(ROOT, fe9.ACTION_PRIOR_FILE),
                               principal_prior_path=os.path.join(ROOT, fe9.PRINCIPAL_PRIOR_FILE),
                               freeze_vocab=True, freeze_priors=True)


def logit(p):
    return math.log(p / (1 - p))


# ── 1. toy model: the cause is known by construction ─────────────────────────

ATTACK_ID, ATTACK_NAME = 3, "CreateAccessKey"


class KnownCauseModel(nn.Module):
    """logit = 4 * [CreateAccessKey anywhere in the window] + 0.1 * sum(f0) - 3."""

    def __init__(self, vocab_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 1, padding_idx=0)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[ATTACK_ID] = 4.0

    def forward(self, idx, feats, lengths):
        mask = (torch.arange(idx.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).float()
        return (self.embedding(idx)[..., 0] * mask).sum(1) + 0.1 * (feats[..., 0] * mask).sum(1) - 3.0


def toy_seq(names_idx, log_id="toy:0"):
    rng = np.random.default_rng(3)
    n = len(names_idx)
    feats = rng.random((n, 5)).astype(np.float32) * 0.5
    feats[:, -1] = np.log1p(np.full(n, 30.0)).astype(np.float32)
    feats[0, -1] = 0.0
    idx, feats, L = tlt.pad_seq(np.array(names_idx, dtype=np.int64), feats, tlt.SEQ_LEN)
    return tlt.EventSeq(username="u", timestamp="2024-01-01", log_id=log_id, event_idxs=idx, feats=feats,
                        length=L, label=1, last_idx=int(idx[L - 1]), label_orig=1)


TOY_CKPT = {"feature_cols": ["f0", "f1", "f2", "f3"]}
TOY_VOCAB = {"ListBuckets": 1, "GetUser": 2, ATTACK_NAME: ATTACK_ID, "ListRoles": 4, "DescribeVpcs": 5}


class TestKnownCauseToyModel(unittest.TestCase):
    def explain(self, idxs):
        m = KnownCauseModel().eval()
        return m, lx.explain_seq(m, toy_seq(idxs), lx.feature_names(TOY_CKPT), lx.id_to_name(TOY_VOCAB), CPU, top_k=10)

    def test_injected_event_is_ranked_first_with_its_true_effect(self):
        _, e = self.explain([1, 2, ATTACK_ID, 4, 1, 5])
        top = e["top_events"][0]
        self.assertEqual((top["event_name"], top["position"]), (ATTACK_NAME, 2))
        self.assertAlmostEqual(top["effect"], 4.0 + 0.1 * 0.0, delta=0.06)   # its own f0 is in the effect too
        self.assertLess(max(abs(t["effect"]) for t in e["top_events"][1:]), 0.06)

    def test_deletion_of_the_top_event_lowers_the_score_most(self):
        _, e = self.explain([1, 2, ATTACK_ID, 4, 1, 5])
        drop = {t["position"]: e["score"] - t["score_without"] for t in e["top_events"]}
        top = e["top_events"][0]["position"]
        self.assertGreater(drop[top], 10 * max(abs(v) for k, v in drop.items() if k != top))

    def test_scored_attack_event_is_explained_by_its_own_action(self):
        _, e = self.explain([1, 2, 4, 1, ATTACK_ID])
        self.assertEqual(e["top_features"][0]["feature"], f"event_name={ATTACK_NAME}")
        self.assertLess(e["ig_completeness_error"], 0.05)


# ── 2. real LSTM, injected cause ─────────────────────────────────────────────

BENIGN = [("ListBuckets", "s3.amazonaws.com"), ("DescribeInstances", "ec2.amazonaws.com"),
          ("GetUser", "iam.amazonaws.com"), ("ListRoles", "iam.amazonaws.com"),
          ("DescribeVpcs", "ec2.amazonaws.com"), ("ListPolicies", "iam.amazonaws.com")]
LAST = ("ListBuckets", "s3.amazonaws.com")
INJECTED = ("CreateAccessKey", "AttachUserPolicy", "PutRolePolicy", "StopLogging")   # PE writes, in the LSTM vocab


def score_session(lstm, events):
    """One principal's events, one per minute, through the frozen live feature engine."""
    eng, recs = frozen_engine(), []
    for i, (name, src) in enumerate(events):
        r = fe9.normalize_cloudtrail_row({
            "timestamp": f"2024-03-05T10:{i:02d}:00Z", "event_name": name, "event_source": src,
            "principal_type": "IAMUser", "principal_arn": "arn:aws:iam::111122223333:user/alice", "username": "alice",
            "source_ip": "10.0.0.5", "user_agent": "aws-cli/2.0", "read_only": "true", "mfa_authenticated": "false",
            "aws_region": "us-east-1", "label": "0"})
        temporal = eng.get_temporal_features(r)
        eng.get_structural_data(r)
        recs.append({"log_id": f"s:{i}", "username": "alice", "timestamp": r["timestamp"], "event_name": name,
                     "label": 0, **dict(zip(fe9.TEMPORAL_COLS, temporal))})
    feats = list(lstm.ckpt["feature_cols"])
    seqs = tlt.build_event_sequences(tlt.prepare_score_frame(pd.DataFrame(recs), lstm.vocab, feats), feats)
    return tlt.score_seqs(lstm.model, seqs, lstm.device)["P_event"].to_numpy(), seqs


class TestInjectedCauseRealLSTM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lstm = real_pipeline().lstm
        cls.names = lx.feature_names(cls.lstm.ckpt)
        cls.cases = {}
        p0, s0 = score_session(cls.lstm, BENIGN + [LAST])
        for inj in INJECTED:
            p1, s1 = score_session(cls.lstm, BENIGN[:3] + [(inj, "iam.amazonaws.com")] + BENIGN[3:] + [LAST])
            e = lx.explain_seq(cls.lstm.model, s1[-1], cls.names, lx.id_to_name(cls.lstm.vocab), CPU, top_k=ee.TOP_K)
            changed = {cls.names[k] for k in range(len(cls.names))
                       if abs(float(s0[-1].feats[s0[-1].length - 1, k] - s1[-1].feats[s1[-1].length - 1, k])) > 1e-6}
            cls.cases[inj] = dict(cf=logit(p1[-1]) - logit(p0[-1]), exp=e, changed=changed)

    def test_injection_counterfactually_drives_the_later_score(self):
        """Precondition: the injected write is a real cause (logit +1 or more)."""
        for inj, c in self.cases.items():
            self.assertGreater(c["cf"], 1.0, inj)

    def test_reported_features_surface_the_mechanism_the_injection_changed(self):
        """The PE-context features of the scored event (pe_write_recent / log_secs_since_pe)
        are what the injection changes; one of them must be among the reported features."""
        for inj, c in self.cases.items():
            self.assertTrue({"pe_write_recent", "log_secs_since_pe"} & c["changed"], inj)
            self.assertTrue({f["feature"] for f in c["exp"]["top_features"]} & c["changed"], inj)

    @unittest.expectedFailure
    def test_injected_event_is_the_top_ranked_earlier_event(self):
        """KNOWN DEFECT (explanation fidelity): lstm_explain.event_effects is leave-one-out
        on the already-featurised window -- drop_event removes the row but keeps every
        later event's features as computed WITH it (pe_write_recent, log_secs_since_pe,
        session/velocity, identity state). A privilege-escalation write acts on later
        events mainly through those features, so LOO misses it: injecting
        CreateAccessKey / AttachUserPolicy / PutRolePolicy / StopLogging before a benign
        ListBuckets raises its logit by ~+1.3 (p 0.06 -> 0.20), yet the injected event's
        reported effect is within +/-0.06 and it is usually not ranked first (a benign
        ListPolicies with a negative effect often is, since ranking is by |effect|)."""
        for inj, c in self.cases.items():
            top = c["exp"]["top_events"][0]
            self.assertEqual(top["event_name"], inj)
            self.assertGreater(top["effect"], 0.5 * c["cf"])


# ── 3. deletion tests on the real models ─────────────────────────────────────

def dev_attack_sessions(n_sessions, seed=0, max_rows=300):
    dev = pd.read_csv(DEV, low_memory=False)
    sizes = dev[dev["session_label"] == 1].groupby("session_id").size()
    pick = list(np.random.default_rng(seed).permutation(sizes[(sizes >= 10) & (sizes <= max_rows)].index))[:n_sessions]
    return dev[dev["session_id"].isin(pick)].sort_values("timestamp", kind="stable")


def featurize_dev(rows, lstm):
    """Live-pipeline features (frozen engine) -> LSTM frame + per-event window labels."""
    eng, recs = frozen_engine(), []
    for i, r in zip(rows.index, rows.to_dict("records")):
        raw = {k: (None if isinstance(v, float) and math.isnan(v) else str(v)) for k, v in r.items()}
        n = fe9.normalize_cloudtrail_row(raw)
        try:
            temporal = eng.get_temporal_features(n)
            eng.get_structural_data(n)
        except ValueError:
            continue
        recs.append({"log_id": f"dev:{i}", "username": n["username"], "timestamp": n["timestamp"],
                     "event_name": n["event_name"], "label": int(r["label"]), **dict(zip(fe9.TEMPORAL_COLS, temporal))})
    df = pd.DataFrame(recs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    feats = list(lstm.ckpt["feature_cols"])
    frame = tlt.prepare_score_frame(df, lstm.vocab, feats)
    seqs = tlt.build_event_sequences(frame, feats)
    window_labels, window_ns = {}, int(pd.Timedelta(minutes=tlt.WINDOW_MINUTES) / pd.Timedelta(nanoseconds=1))
    for _, g in frame.groupby("username", sort=False):          # same windows as build_event_sequences
        g = g.sort_values("timestamp", kind="stable").reset_index(drop=True)
        ts, lab = tlt._ns(g["timestamp"]), g["label"].to_numpy()
        for i in range(len(g)):
            left = max(int(np.searchsorted(ts, ts[i] - window_ns, side="left")), i + 1 - tlt.SEQ_LEN)
            window_labels[str(g["log_id"][i])] = lab[left:i + 1]
    return seqs, window_labels


class TestDevAttackSessions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lstm = real_pipeline().lstm
        cls.names = lx.feature_names(cls.lstm.ckpt)
        cls.id2name = lx.id_to_name(cls.lstm.vocab)
        cls.seqs, cls.labels = featurize_dev(dev_attack_sessions(30), cls.lstm)

    def test_lstm_ig_deletion_top_feature_beats_random_feature(self):
        rng = np.random.default_rng(1)
        seqs = [s for s in self.seqs if s.label == 1 and s.length >= 3]
        seqs = [seqs[i] for i in rng.permutation(len(seqs))[:40]]
        d_top, d_rand = [], []
        for s in seqs:
            e = lx.explain_seq(self.lstm.model, s, self.names, self.id2name, CPU, top_k=len(self.names) + 1)
            numeric = [f["feature"] for f in e["top_features"] if not f["feature"].startswith("event_name=")]
            base = float(lx._logit(self.lstm.model, *lx._tensors(s, CPU))[0])

            def zeroed(name):
                k, f = self.names.index(name), np.array(s.feats).copy()
                f[: s.length, k] = 0.0                                        # the IG baseline value
                return abs(base - float(lx._logit(self.lstm.model, torch.as_tensor(np.array(s.event_idxs)).unsqueeze(0),
                                                  torch.as_tensor(f).unsqueeze(0), torch.as_tensor([s.length]))[0]))
            d_top.append(zeroed(numeric[0]))
            d_rand.append(zeroed(numeric[int(rng.integers(1, len(numeric)))]))
        print(f"\n  LSTM IG deletion |dlogit|: top feature {np.mean(d_top):.3f} vs random feature "
              f"{np.mean(d_rand):.3f} (n={len(seqs)}, top wins {np.mean(np.array(d_top) > np.array(d_rand)):.0%})")
        self.assertGreater(np.mean(d_top), np.mean(d_rand))
        self.assertGreater(np.mean(np.array(d_top) > np.array(d_rand)), 0.5)

    def test_reported_event_effects_reproduce_on_rescoring(self):
        s = max((s for s in self.seqs if s.label == 1), key=lambda s: s.length)
        e = lx.explain_seq(self.lstm.model, s, self.names, self.id2name, CPU, top_k=50)
        dt = self.names.index(lx.DT_FEATURE)
        for t in e["top_events"]:
            idx, f = lx.drop_event(np.array(s.event_idxs), np.array(s.feats), s.length, t["position"], dt)
            p = 1 / (1 + math.exp(-float(lx._logit(self.lstm.model, torch.as_tensor(idx).unsqueeze(0),
                                                    torch.as_tensor(f).unsqueeze(0), torch.as_tensor([s.length - 1]))[0])))
            self.assertAlmostEqual(p, t["score_without"], places=3)

    def test_lstm_event_ranking_hits_attack_events_better_than_random(self):
        """Top-1/top-3 hit rate: is the top-ranked earlier event attack-labelled? Only
        windows with both attack and benign earlier events count (otherwise any ranking
        hits or misses). Baseline: a uniformly random ranking of the same window."""
        rng = np.random.default_rng(2)
        cands = [s for s in self.seqs if s.label == 1 and 0 < self.labels[s.log_id][:-1].sum() < s.length - 1]
        cands = [cands[i] for i in rng.permutation(len(cands))[:80]]
        hit1 = hit3 = hit1_pos = base1 = base3 = 0.0
        for s in cands:
            e = lx.explain_seq(self.lstm.model, s, self.names, self.id2name, CPU, top_k=50)
            lab = self.labels[s.log_id][:-1]
            k, a = len(lab), int(lab.sum())
            order = [t["position"] for t in e["top_events"]]
            positive = [t["position"] for t in e["top_events"] if t["effect"] > 0]
            hit1 += lab[order[0]]
            hit3 += float(any(lab[j] for j in order[:3]))
            hit1_pos += lab[positive[0]] if positive else 0
            base1 += a / k
            base3 += 1 - comb(k - a, min(3, k)) / comb(k, min(3, k))
        n = len(cands)
        self.assertGreaterEqual(n, 15, "too few mixed windows sampled")
        print(f"\n  LSTM event-ranking hit rate on dev attack windows (n={n}, in-sample for v6.3-ft): "
              f"top-1 {hit1 / n:.3f} (first positive-effect {hit1_pos / n:.3f}) vs random {base1 / n:.3f}; "
              f"top-3 {hit3 / n:.3f} vs random {base3 / n:.3f}")
        self.assertGreater(hit1 / n, base1 / n)
        self.assertGreater(hit3 / n, base3 / n)


class TestGraphDeletion(unittest.TestCase):
    def test_hgt_top_feature_deletion_beats_random_feature(self):
        gnn = real_pipeline().gnn
        scorer = getattr(gnn, "scorers", [gnn])[0]
        st = pd.read_csv(os.path.join(DATA, "real_dataset_dev_structural.csv"), nrows=200)
        data, order = scorer.build_graph(st[["log_id"] + P.STRUCT_COLS + ["label"]])
        targets = list(np.random.default_rng(0).choice(order, 8, replace=False))
        exps = ee.explain_graph_events(scorer.model, data, order, targets, top_k=len(EDGE_FEATURE_NAMES))
        pos = {lid: i for i, lid in enumerate(order)}
        where = {str(l): (t, i) for t in scored_edge_types(data) for i, l in enumerate(data[t].log_id)}
        with torch.no_grad():
            base = scorer.model(data).numpy()
        d_top, d_rand = [], []
        for lid, e in exps.items():
            triple, i = where[lid]
            groups = dict(_feature_groups(data[triple].edge_attr.shape[1], EDGE_FEATURE_NAMES))

            def delta(name):
                orig = data[triple].edge_attr
                data[triple].edge_attr = orig.clone()
                data[triple].edge_attr[i, groups[name]] = 0.0
                with torch.no_grad():
                    v = float(scorer.model(data)[pos[lid]])
                data[triple].edge_attr = orig
                return abs(v - float(base[pos[lid]]))
            top = e["top_features"][0]["feature"]
            self.assertAlmostEqual(e["logit"], float(base[pos[lid]]), places=3)
            d_top.append(delta(top))
            d_rand.append(np.mean([delta(g) for g in groups if g != top]))
        print(f"\n  HGT grad x input deletion |dlogit|: top feature {np.mean(d_top):.3f} vs mean other "
              f"{np.mean(d_rand):.3f} (n={len(d_top)})")
        self.assertGreater(np.mean(d_top), np.mean(d_rand))


# ── 4/5. ensemble consistency and no invented actions, end to end ────────────

WORD = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"


class TestPipelineExplanations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = real_pipeline()
        cls.cfg_json = P.PipelineConfig.load()
        cls.pipe.cfg.explain_top_events = 1000
        dev = pd.read_csv(DEV, low_memory=False)
        attack = dev[dev["session_label"] == 1]["session_id"].drop_duplicates().iloc[[0, 5, 10, 20, 30, 40]]
        rows = pd.concat([dev[dev["session_id"].isin(attack)].head(320), dev[dev["session_label"] == 0].head(120)])
        tmp = tempfile.mkdtemp()
        cls.csv = os.path.join(tmp, "mixed.csv")
        rows.sort_values("timestamp").to_csv(cls.csv, index=False)
        names = set(dev["event_name"].dropna()) | set(pd.read_csv(os.path.join(DATA, "synthetic_cloudtrail.csv"),
                                                                 usecols=["event_name"])["event_name"].dropna())
        cls.action_names = sorted(names | set(cls.pipe.lstm.vocab) | set(fe9.CRITICAL_ACTIONS), key=len, reverse=True)

    def scored(self, path, name):
        self.pipe.buffer = pd.DataFrame()
        new = self.pipe.featurize(enumerate(fe9.iter_input_rows(path)), name)
        return quiet(self.pipe.score, new)

    def test_model_shares_add_up_to_the_scored_risk(self):
        scored = self.scored(ATTACK, "attack.json")
        exps = quiet(self.pipe.explain, scored)
        self.assertEqual(len(exps), len(scored))
        graph_rows = self.pipe._scored_inputs["graph_rows"]
        seed0 = getattr(self.pipe.gnn, "scorers", [self.pipe.gnn])[0]
        p_seed0 = dict(zip(*seed0.score(graph_rows[["log_id"] + P.STRUCT_COLS + ["label"]]).T.values))
        w = self.cfg_json.weight_graph
        self.assertEqual(self.pipe.cfg.weight_graph, w)
        for r in scored.itertuples():
            e, m = exps[r.log_id], exps[r.log_id]["models"]
            self.assertAlmostEqual(m["graph"]["contribution"] + m["sequence"]["contribution"], r.risk, delta=2e-4)
            self.assertAlmostEqual(e["risk_score"], r.risk_score, delta=0.011)
            self.assertAlmostEqual(m["sequence"]["probability"], r.p_sequence, delta=1e-4)
            self.assertAlmostEqual(m["graph"]["share"] + m["sequence"]["share"], 1.0, delta=2e-3)
            if pd.isna(r.p_graph):
                self.assertEqual((m["graph"]["weight"], m["sequence"]["weight"]), (0.0, 1.0))
                continue
            self.assertEqual(m["graph"]["weight"], w)
            self.assertAlmostEqual(m["graph"]["probability"], r.p_graph, delta=1e-4)
            # the graph explanation explains the first seed's own score of this edge
            self.assertAlmostEqual(1 / (1 + math.exp(-e["graph"]["logit"])), float(p_seed0[r.log_id]), delta=1e-3)
            self.assertLessEqual(sum(f["share"] for f in e["graph"]["top_features"]), 1.0 + 1e-3)

    def mentioned_actions(self, text):
        found, rest = set(), text
        for name in self.action_names:                 # longest first: GetUserPolicy before GetUser
            if re.search(WORD.format(re.escape(name)), rest):
                found.add(name)
                rest = re.sub(WORD.format(re.escape(name)), " ", rest)
        return found

    def test_explanations_never_name_actions_absent_from_the_input(self):
        scored = self.scored(self.csv, "mixed.csv")
        thr = self.pipe.cfg.alert_threshold
        lab = scored["label"] == 1
        cases = {"TP": scored[scored["alert"] & lab].head(3), "FP": scored[scored["alert"] & ~lab].head(3),
                 "FN": scored[~scored["alert"] & lab].head(3),
                 "low-confidence": scored.loc[(scored["risk"] - thr).abs().sort_values().index[:3]]}
        print("\n  explained cases: " + ", ".join(f"{k}={len(v)}" for k, v in cases.items()))
        self.assertTrue(all(len(v) for v in cases.values()), "slice lacks a TP / FP / FN / low-confidence case")
        pick = pd.concat(cases.values()).drop_duplicates("log_id")
        exps = quiet(self.pipe.explain, pick)
        self.assertEqual(len(exps), len(pick))
        graph_rows, seqs = self.pipe._scored_inputs["graph_rows"], self.pipe._scored_inputs["seqs"]
        id2name = lx.id_to_name(self.pipe.lstm.vocab)
        window = {s.log_id: {id2name.get(int(i), "?") for i in s.event_idxs[: s.length]} for s in seqs}
        in_graph = set(graph_rows["edge_type"].dropna())
        graph_ids = set(graph_rows["log_id"].astype(str))
        for r in pick.itertuples():
            e = exps[r.log_id]
            principal_actions = set(self.pipe.buffer.loc[self.pipe.buffer["username"] == r.username, "event_name"].dropna())
            allowed = principal_actions | in_graph
            self.assertEqual(self.mentioned_actions(e["summary"]) - allowed, set(), e["summary"])
            for t in (e["sequence"] or {}).get("top_events", []):
                self.assertIn(t["event_name"], window[r.log_id])
            for rel in (e["graph"] or {}).get("related_events", []):
                self.assertIn(rel["log_id"], graph_ids)
                self.assertEqual(rel["edge_type"], graph_rows.set_index("log_id").loc[rel["log_id"], "edge_type"])
            self.assertEqual("FAST-LANE" in e["summary"], r.event_name in fe9.CRITICAL_ACTIONS)
            self.assertIsNone(re.search(r"\bT1\d{3}\b", e["summary"]), "claims an ATT&CK technique")


if __name__ == "__main__":
    unittest.main()
