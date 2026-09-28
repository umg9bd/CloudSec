"""
test_lineage_verification.py
============================
Checks the feature engine's INFERRED identity lineage against the generator's
GROUND TRUTH (review point 6: campaign/hop annotations exist for "evaluation
and lineage verification", never as features).

The engine infers, from observable fields only, how many identity handoffs
separate the acting identity from an origin (causal_depth_normalized * 3) and
whether a handoff happened at all (principal_handoff). The generator records
the true value as hop_id in synthetic_campaign_annotations.csv.

Two different questions are kept apart:

  * Is the ENGINE right? Run it over the committed synthetic data with the two
    dataset defects below repaired IN MEMORY (nothing on disk changes): its
    inferred depth must equal hop_id on every campaign event.
  * Does the DATASET carry the links a real CloudTrail log carries? Two
    defects currently break them, so on the file as committed the engine
    recovers only 14 of 252 handoff events:
      1. AssumeRole requests roleSessionName X, but the role's later events act
         as assumed-role/ROLE/Y with Y != X (0 of 126 match). Real CloudTrail
         always uses the requested name; it is the only observable link.
      2. 72 of 640 campaign events precede their own parent in file order
         (same-second ties sorted without regard to causality), e.g. AssumeRole
         listed before the CreateRole that makes it possible.
    Those checks are marked expectedFailure. When the generator is fixed they
    will report "unexpected success": delete the decorator then.
"""

import json
import os
import unittest

import pandas as pd

import feature_engine9 as fe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "privilege-escalation")
EVENTS = os.path.join(DATA, "synthetic_cloudtrail.csv")
ANNOTATIONS = os.path.join(DATA, "synthetic_campaign_annotations.csv")


def load():
    if not (os.path.exists(EVENTS) and os.path.exists(ANNOTATIONS)):
        raise unittest.SkipTest("synthetic dataset / annotations not present")
    events = pd.read_csv(EVENTS, dtype=str, keep_default_na=False, low_memory=False)
    ann = pd.read_csv(ANNOTATIONS, dtype=str, keep_default_na=False)
    if len(events) != len(ann):
        raise AssertionError(f"annotations ({len(ann)}) do not align with events ({len(events)})")
    events["_uid"] = ann["event_uid"].values
    events["_parent"] = ann["parent_event_id"].values
    events["_hop"] = ann["hop_id"].astype(int).values
    return events


def repaired(events):
    """In-memory copy with the two dataset defects corrected (see module docstring)."""
    df = events.copy()
    rename = {}
    for _, r in df[df.event_name == "AssumeRole"].iterrows():
        if not r.request_params_raw:
            continue
        params = json.loads(r.request_params_raw)
        if "roleSessionName" not in params:
            continue
        for j in df.index[df._parent == r._uid]:
            old = df.at[j, "principal_arn"]
            if ":assumed-role/" in old:
                rename[old] = old.rsplit("/", 1)[0] + "/" + params["roleSessionName"]
    df["principal_arn"] = df.principal_arn.map(lambda a: rename.get(a, a))

    row_of = {u: i for i, u in zip(df.index, df._uid) if u}

    def depth(uid, guard=0):
        parent = df.at[row_of[uid], "_parent"] if uid in row_of else ""
        return 0 if not parent or parent not in row_of or guard > 20 else 1 + depth(parent, guard + 1)

    df["_depth"] = [depth(u) if u else 0 for u in df._uid]
    df["_ts"] = pd.to_datetime(df.timestamp, utc=True, format="mixed")
    return df.sort_values(["_ts", "_depth"], kind="mergesort")


def infer(df):
    engine = fe.FeatureEngineer()
    depth, handoff = [], []
    for _, r in df.iterrows():
        ctx = engine._context(fe.normalize_cloudtrail_row(r.to_dict()))
        depth.append(round(ctx["causal_depth_normalized"] * 3))
        handoff.append(ctx["principal_handoff"])
    out = df.copy()
    out["_inferred_depth"], out["_inferred_handoff"] = depth, handoff
    return out


class TestEngineRecoversGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = load()
        cls.fixed = infer(repaired(cls.events))
        cls.campaign = cls.fixed[cls.fixed._hop >= 0]

    def test_campaigns_contain_multi_hop_events(self):
        self.assertGreater((self.campaign._hop >= 1).sum(), 0)
        self.assertGreater((self.campaign._hop >= 2).sum(), 0)

    def test_inferred_depth_equals_hop_id_on_every_campaign_event(self):
        wrong = self.campaign[self.campaign._inferred_depth != self.campaign._hop]
        self.assertEqual(len(wrong), 0, wrong[["event_name", "_hop", "_inferred_depth"]].head().to_string())

    def test_handoff_flag_matches_hop_id(self):
        expected = (self.campaign._hop >= 1).astype(int)
        self.assertTrue((self.campaign._inferred_handoff == expected).all())


class TestNoInventedHandoffs(unittest.TestCase):
    """On the file as committed: whatever the engine misses, it must not
    invent a handoff for a campaign event whose true depth is 0."""

    @classmethod
    def setUpClass(cls):
        cls.as_is = infer(load())

    def test_no_handoff_on_depth_zero_campaign_events(self):
        root = self.as_is[self.as_is._hop == 0]
        self.assertGreater(len(root), 0)
        self.assertEqual(int(root._inferred_handoff.sum()), 0)

    def test_inferred_depth_never_exceeds_the_truth(self):
        c = self.as_is[self.as_is._hop >= 0]
        self.assertTrue((c._inferred_depth <= c._hop).all())


class TestDatasetLineagePreconditions(unittest.TestCase):
    """What a real CloudTrail log guarantees and the synthetic data must too.
    Known failures -- see the module docstring."""

    @classmethod
    def setUpClass(cls):
        cls.events = load()

    @unittest.expectedFailure
    def test_assumed_sessions_use_the_requested_session_name(self):
        acting = set(self.events.principal_arn)
        ar = self.events[(self.events.event_name == "AssumeRole") & (self.events.request_params_raw != "")]
        missing = 0
        for raw in ar.request_params_raw:
            p = json.loads(raw)
            if "roleSessionName" not in p:
                continue
            acct, role = p["roleArn"].split(":")[4], p["roleArn"].rsplit("/", 1)[-1]
            used_by_children = any(a.startswith(f"arn:aws:sts::{acct}:assumed-role/{role}/") for a in acting)
            if used_by_children and f"arn:aws:sts::{acct}:assumed-role/{role}/{p['roleSessionName']}" not in acting:
                missing += 1
        self.assertEqual(missing, 0, f"{missing} AssumeRole calls whose session acts under another name")

    @unittest.expectedFailure
    def test_parents_precede_children_in_file_order(self):
        position = {u: i for i, u in enumerate(self.events._uid) if u}
        late = sum(1 for i, p in enumerate(self.events._parent) if p and position.get(p, -1) > i)
        self.assertEqual(late, 0, f"{late} events appear before their own parent")

    def test_every_parent_id_exists(self):
        uids = set(self.events._uid)
        dangling = [p for p in self.events._parent if p and p not in uids]
        self.assertEqual(dangling, [])


if __name__ == "__main__":
    unittest.main()
