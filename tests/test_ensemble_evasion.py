"""
test_ensemble_evasion.py
========================
Can an attacker keep a detected session under the alert threshold by reshaping its CloudTrail?
Each attack session is scored by the real pipeline (3-seed HGT + LSTM ensemble, deployed
pipeline_config.json) from a fresh state, then again after one perturbation:

  pad          3 benign read-only calls by the same principal between every attack call
  pad_heavy    10 per attack call (attack events become ~9% of the session)
  user_agent   every user agent renamed to a stock aws-cli string
  spread_3h    one attack call every 3 hours (beyond the LSTM's 1 h history)
  spread_30h   one attack call every 30 hours (beyond the 24 h graph window: no two in one graph)
  region       every call moved to another region
  combined     spread_3h + user_agent + region + pad

Session risk = the highest ensemble risk over the session's attack events (the fast-lane rule is
separate and not counted). An evasion succeeds when a session detected at baseline drops below the
alert threshold. Sessions: the committed synthetic attack-chain sample and the first labelled attack
sessions of real_dataset_dev.csv (first 40 events each); a session already missed at baseline is
reported, not counted as an evasion.
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from test_full_pipeline_integration import ROOT, sandbox_pipeline  # noqa: E402

import feature_engine9 as fe9                                     # noqa: E402

DEV = os.path.join(ROOT, "datasets", "privilege-escalation", "real_dataset_dev.csv")
ATTACK = os.path.join(ROOT, "samples", "cloudtrail", "synthetic_attack_chain.json")
N_DEV_ROWS, SESSION_LEN, N_REAL_SESSIONS = 4000, 40, 3
BENIGN_CALLS = [("ListBuckets", "s3.amazonaws.com"), ("DescribeInstances", "ec2.amazonaws.com"),
                ("GetCallerIdentity", "sts.amazonaws.com"), ("ListUsers", "iam.amazonaws.com")]
STOCK_UA = "aws-cli/2.15.0 Python/3.11.6 Linux/5.15.0 exe/x86_64.ubuntu.22 prompt/off command/s3.ls"


def _ts(t):
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def pad(rows, k=3):
    """k benign calls by the same identity between consecutive attack calls; attack times unchanged."""
    out = []
    times = [pd.Timestamp(r["timestamp"]) for r in rows]
    for i, r in enumerate(rows):
        out.append(dict(r, label=1))
        gap = (times[i + 1] - times[i]) if i + 1 < len(rows) else pd.Timedelta(seconds=k + 1)
        for j in range(k):
            name, source = BENIGN_CALLS[(i + j) % len(BENIGN_CALLS)]
            out.append(dict(r, event_name=name, event_source=source, read_only="true", error_code="",
                            request_params_raw="{}", target_resource="aws_service", label=0,
                            timestamp=_ts(times[i] + gap * (j + 1) / (k + 1))))
    return out


def user_agent(rows):
    return [dict(r, user_agent=STOCK_UA) for r in rows]


def spread(rows, hours):
    t0 = pd.Timestamp(rows[0]["timestamp"])
    return [dict(r, timestamp=_ts(t0 + pd.Timedelta(hours=hours * i))) for i, r in enumerate(rows)]


def region(rows):
    return [dict(r, aws_region="eu-west-1" if r.get("aws_region") != "eu-west-1" else "ap-southeast-2") for r in rows]


PERTURBATIONS = {
    "pad": pad,
    "pad_heavy": lambda r: pad(r, k=10),
    "user_agent": user_agent,
    "spread_3h": lambda r: spread(r, 3),
    "spread_30h": lambda r: spread(r, 30),
    "region": region,
    "combined": lambda r: pad(region(user_agent(spread(r, 3)))),
}


class EnsembleEvasionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        os.chdir(ROOT)
        cls.tmp = tempfile.mkdtemp()
        cls.pipe = sandbox_pipeline(cls.tmp, write_outputs=False, explain_top_events=0)
        cls.thr = cls.pipe.cfg.alert_threshold
        with open(ATTACK, encoding="utf-8") as f:
            records = json.load(f)["Records"]
        sessions = {"synthetic_attack_chain": [dict(fe9.normalize_cloudtrail_row(r), label=1) for r in records]}
        by_user = {}
        for _, r in zip(range(N_DEV_ROWS), fe9.iter_input_rows(DEV)):
            if str(r.get("label")) == "1":
                by_user.setdefault(r["username"], []).append(dict(r, label=1))
        for user, rows in list(by_user.items())[:N_REAL_SESSIONS]:
            sessions[f"dev:{user}"] = rows[:SESSION_LEN]
        cls.results = {}
        for name, rows in sessions.items():
            base = cls.session_risk(rows, f"{name}|baseline")
            entry = {"n_attack_events": len(rows), "baseline": base, "detected_at_baseline": base["max_risk"] >= cls.thr}
            for pname, f in PERTURBATIONS.items():
                entry[pname] = cls.session_risk(f(copy.deepcopy(rows)), f"{name}|{pname}")
            cls.results[name] = entry
        cls.report()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def session_risk(cls, rows, name):
        cls.pipe.reload_state()                                    # fresh state: no history from earlier runs
        new = cls.pipe.featurize(list(enumerate(rows)), name.replace(":", "_").replace("|", "_"))
        scored = cls.pipe.score(new)
        att = scored[scored["label"] == 1]
        return {"max_risk": round(float(att["risk"].max()), 4), "n_alert_events": int(att["alert"].sum()),
                "graph_scored_share": round(float(att["p_graph"].notna().mean()), 3),
                "fast_lane_events": int(att["fast_lane"].sum())}

    @classmethod
    def report(cls):
        lines = [f"\n[EVASION] session max ensemble risk (alert threshold {cls.thr})"]
        lines.append("  " + f"{'session':52s}{'baseline':>9}" + "".join(f"{p:>11}" for p in PERTURBATIONS))
        for name, e in cls.results.items():
            lines.append("  " + f"{name[:51]:52s}{e['baseline']['max_risk']:>9.3f}"
                         + "".join(f"{e[p]['max_risk']:>11.3f}" for p in PERTURBATIONS)
                         + ("" if e["detected_at_baseline"] else "   (missed at baseline)"))
        print("\n".join(lines), file=sys.stderr)

    def evaded(self, perturbation):
        return {name: (e["baseline"]["max_risk"], e[perturbation]["max_risk"])
                for name, e in self.results.items()
                if e["detected_at_baseline"] and e[perturbation]["max_risk"] < self.thr}

    def test_sessions_under_test(self):
        detected = [n for n, e in self.results.items() if e["detected_at_baseline"]]
        self.assertIn("synthetic_attack_chain", detected)
        self.assertGreaterEqual(len(detected), 3)

    def test_benign_padding_does_not_evade(self):
        self.assertEqual(self.evaded("pad"), {})

    def test_heavy_benign_padding_does_not_evade(self):
        self.assertEqual(self.evaded("pad_heavy"), {})

    def test_renamed_user_agent_does_not_evade(self):
        self.assertEqual(self.evaded("user_agent"), {})

    def test_spreading_over_hours_does_not_evade(self):
        self.assertEqual(self.evaded("spread_3h"), {})

    def test_spreading_beyond_the_graph_window_does_not_evade(self):
        self.assertEqual(self.evaded("spread_30h"), {})

    def test_region_change_does_not_evade(self):
        self.assertEqual(self.evaded("region"), {})

    def test_combined_perturbation_does_not_evade(self):
        self.assertEqual(self.evaded("combined"), {})


if __name__ == "__main__":
    unittest.main()
