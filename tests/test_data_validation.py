"""
test_data_validation.py
=======================
Data-quality guards that sit upstream of both models.

  1. validate_synthetic_vs_real.py -- its chi-square / KS comparison functions on
     small synthetic frames whose answer is known, and the documented expectations
     (docs/PROJECT_STATUS_REPORT.md: "5 of 8 fields pass ... event_source/event_name
     remain intentionally unmatched ... hour_of_day is borderline (KS=0.11)")
     re-checked on the committed DEV slice only. The report's own numbers were
     computed on real_dataset_combined.csv, which contains the test rows, so they are
     not recomputed here (see the LIMITATION skip).
  2. Data prep (feature_engine9): missing values, duplicate events, invalid labels,
     malformed events and schema drift either fail loudly or are handled in a stated,
     tested way.
  3. Derived datasets (cloudtrail_structural/temporal, real_dataset_dev_*) line up
     row-for-row with the source CSV they were built from.

real_dataset_test.csv is only touched for a row count (shape check), never scored.
"""

import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import feature_engine9 as fe9

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "privilege-escalation")
SYNTHETIC = os.path.join(DATA, "synthetic_cloudtrail.csv")
DEV = os.path.join(DATA, "real_dataset_dev.csv")
TEST = os.path.join(DATA, "real_dataset_test.csv")

_spec = importlib.util.spec_from_file_location("validate_synthetic_vs_real",
                                               os.path.join(DATA, "validate_synthetic_vs_real.py"))
vsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vsr)

# What the status report says about the 8 tested fields.
INTENTIONALLY_UNMATCHED = {"event_source", "event_name"}   # flagged by design, not defects
BORDERLINE = {"hour_of_day"}                               # KS~0.11: just over the 0.10 cutoff


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def eight_fields(real, synthetic):
    """The exact field list and options validate_synthetic_vs_real.main() uses."""
    n = vsr.TOP_N_HIGH_CARDINALITY
    out = [quiet(vsr.chi_square_field, real[c], synthetic[c], c)
           for c in ("read_only", "principal_type", "mfa_authenticated")]
    out += [quiet(vsr.chi_square_field, real[c], synthetic[c], c, as_null_flag=True)
            for c in ("target_resource", "error_code")]
    out += [quiet(vsr.chi_square_field, real[c], synthetic[c], c, top_n=n) for c in ("event_source", "event_name")]
    out.append(quiet(vsr.ks_field, real["timestamp"].dt.hour, synthetic["timestamp"].dt.hour, "hour_of_day"))
    return {r["field"]: r for r in out}


# ── 1a. the comparison functions on frames with a known answer ──────────────────

class TestComparisonFunctions(unittest.TestCase):
    def test_identical_categorical_distributions_pass(self):
        s = pd.Series(["a"] * 50 + ["b"] * 30 + [None] * 20)
        r = quiet(vsr.chi_square_field, s, s.copy(), "x")
        self.assertEqual(r["verdict"], "PASS")
        self.assertAlmostEqual(r["effect_size"], 0.0, places=9)

    def test_disjoint_categories_are_flagged(self):
        r = quiet(vsr.chi_square_field, pd.Series(["a"] * 100), pd.Series(["b"] * 100), "x")
        self.assertEqual(r["verdict"], "FLAG")
        self.assertGreater(r["effect_size"], 0.9)

    def test_verdict_follows_effect_size_not_p_value(self):
        """The module's stated design: at this sample size a negligible gap is
        'significant'. A 1-point shift on 100k rows must PASS despite p << 0.05."""
        real = pd.Series(["a"] * 50_000 + ["b"] * 50_000)
        syn = pd.Series(["a"] * 51_000 + ["b"] * 49_000)
        r = quiet(vsr.chi_square_field, real, syn, "x")
        self.assertLess(r["p_value"], 1e-3)
        self.assertEqual(r["verdict"], "PASS")

    def test_null_flag_mode_compares_presence_only(self):
        real = pd.Series(["arn:x"] * 70 + [None] * 30)
        syn = pd.Series([f"arn:{i}" for i in range(70)] + [None] * 30)   # different values, same null rate
        self.assertEqual(quiet(vsr.chi_square_field, real, syn, "x", as_null_flag=True)["verdict"], "PASS")
        self.assertEqual(quiet(vsr.chi_square_field, real, syn, "x")["verdict"], "FLAG")

    def test_top_n_buckets_unseen_rare_categories_into_other(self):
        """event_name/event_source are compared top-N + 'other': synthetic actions that
        never occur in real data are not individually counted against it."""
        real = pd.Series(["A"] * 400 + ["B"] * 300 + [f"r{i}" for i in range(300)])
        syn = pd.Series(["A"] * 400 + ["B"] * 300 + [f"s{i}" for i in range(300)])
        self.assertEqual(quiet(vsr.chi_square_field, real, syn, "x", top_n=2)["verdict"], "PASS")
        self.assertEqual(quiet(vsr.chi_square_field, real, syn, "x")["verdict"], "FLAG")

    def test_ks_threshold_is_strict_so_0_11_is_flagged(self):
        """'Borderline (KS=0.11)' in the report is still a FLAG under the module's
        rule (PASS only when the statistic is < 0.10), and exactly 0.10 flags too."""
        real = np.zeros(100)
        for n_ones, want in ((11, "FLAG"), (10, "FLAG"), (9, "PASS")):
            syn = np.array([0] * (100 - n_ones) + [1] * n_ones, dtype=float)
            r = quiet(vsr.ks_field, pd.Series(real), pd.Series(syn), "hour_of_day")
            self.assertAlmostEqual(r["effect_size"], n_ones / 100, places=9)
            self.assertEqual(r["verdict"], want, n_ones)

    def test_single_category_field_does_not_crash(self):
        r = quiet(vsr.chi_square_field, pd.Series(["x"] * 5), pd.Series(["x"] * 7), "x")
        self.assertEqual((r["effect_size"], r["verdict"]), (0.0, "PASS"))


# ── 1b. the documented expectations, on committed dev + synthetic ───────────────

class TestDocumentedExpectationsOnDev(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        real = pd.read_csv(DEV, low_memory=False)
        syn = pd.read_csv(SYNTHETIC, low_memory=False)
        for d in (real, syn):
            d["timestamp"] = pd.to_datetime(d["timestamp"], format="ISO8601", utc=True)
        cls.res = eight_fields(real, syn)
        print("\n  dev vs synthetic: " + ", ".join(
            f"{k.split(' ')[0]}={v['effect_size']:.3f}:{v['verdict']}" for k, v in cls.res.items()))

    def field(self, name):
        return next(v for k, v in self.res.items() if k.split(" ")[0] == name)

    def test_eight_fields_are_tested(self):
        self.assertEqual(len(self.res), 8)

    def test_event_source_and_event_name_are_intentionally_unmatched(self):
        """Flagged, and flagged by design: they are excluded from the defect count."""
        for name in INTENTIONALLY_UNMATCHED:
            self.assertEqual(self.field(name)["verdict"], "FLAG", name)
        defects = {k.split(" ")[0] for k, v in self.res.items() if v["verdict"] == "FLAG"}
        self.assertTrue(INTENTIONALLY_UNMATCHED <= defects)

    def test_hour_of_day_verdict_follows_the_module_threshold(self):
        r = self.field("hour_of_day")
        self.assertEqual(r["verdict"], "PASS" if r["effect_size"] < vsr.EFFECT_SIZE_THRESHOLD else "FLAG")

    def test_dev_slice_verdicts_are_pinned(self):
        """On dev alone 4 of 8 pass: the four presence/flag fields. principal_type
        (V~0.115) and hour_of_day (KS~0.24) exceed 0.10 on this slice, so the report's
        '5 of 8' and 'KS=0.11' (computed on dev+test combined) are NOT reproducible
        from dev alone. Pinned so a regenerated synthetic set shows up here."""
        want = {"read_only": "PASS", "mfa_authenticated": "PASS", "target_resource": "PASS",
                "error_code": "PASS", "principal_type": "FLAG", "event_source": "FLAG",
                "event_name": "FLAG", "hour_of_day": "FLAG"}
        got = {k.split(" ")[0]: v["verdict"] for k, v in self.res.items()}
        self.assertEqual(got, want)

    @unittest.skip("LIMITATION: the documented '5 of 8 fields pass' / 'hour_of_day KS=0.11' was computed "
                   "by validate_synthetic_vs_real.main() on real_dataset_combined.csv, which contains every "
                   "real_dataset_test.csv row. Recomputing it would read the held-out test split, so it is "
                   "not re-verified; the dev-only verdicts are pinned above instead.")
    def test_five_of_eight_on_combined_real_data(self):
        pass


# ── 2. data prep: bad input either fails loudly or is handled as stated ─────────

PATH_GLOBALS = ("DATA_DIR", "STRUCT_OUT", "TEMPORAL_OUT", "STATE_FILE", "EVENT_NAME_VOCAB_FILE",
                "STATE_TRACKER_FILE", "GRAPH_NODE_STATE_FILE", "IDENTITY_STATE_FILE",
                "ACTION_PRIOR_FILE", "PRINCIPAL_PRIOR_FILE")
FIELDS = ["timestamp", "event_name", "event_source", "principal_type", "principal_arn", "username", "label"]


def base_row(i=0, **kw):
    r = {"timestamp": f"2024-01-01T10:{i:02d}:00Z", "event_name": "ListRoles", "event_source": "iam.amazonaws.com",
         "principal_type": "IAMUser", "principal_arn": "arn:aws:iam::111122223333:user/u1",
         "username": "u1", "label": "0"}
    r.update(kw)
    return r


class PrepSandbox(unittest.TestCase):
    """feature_engine9's path globals redirected to a temp dir; nothing in the repo is written."""

    def setUp(self):
        self.saved = {n: getattr(fe9, n) for n in PATH_GLOBALS}
        self.tmp = tempfile.mkdtemp()
        fe9.DATA_DIR = self.tmp
        for n in PATH_GLOBALS[1:]:
            setattr(fe9, n, os.path.join(self.tmp, os.path.basename(self.saved[n])))

    def tearDown(self):
        for n, v in self.saved.items():
            setattr(fe9, n, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def engine(self):
        return fe9.FeatureEngineer(event_name_vocab_path=fe9.EVENT_NAME_VOCAB_FILE,
                                   state_tracker_path=fe9.STATE_TRACKER_FILE,
                                   graph_state_path=fe9.GRAPH_NODE_STATE_FILE,
                                   identity_state_path=fe9.IDENTITY_STATE_FILE,
                                   action_prior_path=fe9.ACTION_PRIOR_FILE,
                                   principal_prior_path=fe9.PRINCIPAL_PRIOR_FILE)

    def write_csv(self, name, rows, fields=FIELDS):
        path = os.path.join(self.tmp, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return path

    def run_batch_file(self, path):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            n = fe9.process_batch_file(self.engine(), path)
        struct = pd.read_csv(fe9.STRUCT_OUT, dtype=str, keep_default_na=False) if os.path.exists(fe9.STRUCT_OUT) \
            else pd.DataFrame()
        return n, struct, out.getvalue()


class TestMissingAndMalformedValues(PrepSandbox):
    def test_missing_timestamp_is_skipped_loudly_and_row_indices_are_kept(self):
        path = self.write_csv("in.csv", [base_row(0), base_row(1, timestamp=""), base_row(2)])
        n, struct, log = self.run_batch_file(path)
        self.assertEqual(n, 2)
        self.assertEqual(list(struct["log_id"]), ["in.csv:0", "in.csv:2"])   # no shift after the skip
        self.assertIn("[SKIP]", log)
        self.assertIn("row 1", log)

    def test_unparseable_timestamp_is_skipped_loudly(self):
        path = self.write_csv("in.csv", [base_row(0), base_row(1, timestamp="yesterday"),
                                         base_row(2, timestamp="2024-13-45T99:00:00Z")])
        n, struct, log = self.run_batch_file(path)
        self.assertEqual(list(struct["log_id"]), ["in.csv:0"])
        self.assertEqual(log.count("[SKIP]"), 2)

    def test_missing_optional_fields_become_explicit_absent_flags(self):
        """Unknown MFA / read-only status is encoded as 'absent', never as a claimed value."""
        row = fe9.normalize_cloudtrail_row({k: v for k, v in base_row().items()})
        f = dict(zip(fe9.TEMPORAL_COLS, self.engine().get_temporal_features(row)))
        self.assertEqual((f["no_mfa"], f["mfa_absent"]), (0, 1))
        self.assertEqual((f["is_write_action"], f["read_only_absent"]), (0, 1))
        self.assertEqual((f["has_error"], f["has_access_key"]), (0, 0))

    def test_malformed_json_line_fails_loudly(self):
        path = os.path.join(self.tmp, "in.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"eventTime": "2024-01-01T10:00:00Z", "eventName": "ListRoles"}) + "\n{broken\n")
        it = fe9.iter_input_rows(path)
        self.assertEqual(next(it)["event_name"], "ListRoles")
        with self.assertRaises(json.JSONDecodeError):
            next(it)


class TestDuplicates(PrepSandbox):
    def test_duplicate_rows_without_an_event_id_are_kept_each_with_its_own_log_id(self):
        """Documented-by-test handling: identical CSV rows (real dev has 1,294) are not
        silently dropped; each keeps its row-index log_id, so session joins stay exact."""
        path = self.write_csv("in.csv", [base_row(0), base_row(0), base_row(1)])
        n, struct, _ = self.run_batch_file(path)
        self.assertEqual(n, 3)
        self.assertEqual(struct["log_id"].nunique(), 3)

    @unittest.expectedFailure
    def test_duplicate_cloudtrail_records_with_the_same_event_id_are_featurized_once(self):
        """KNOWN DEFECT: CloudTrail can deliver the same record more than once (same
        eventID). Neither iter_input_rows/normalize_cloudtrail_row nor process_batch_file
        (or pipeline.featurize) de-duplicates on eventID -- normalize_cloudtrail_row drops
        the eventID entirely -- so a re-delivered record is featurized and scored twice
        and inflates the principal's action_velocity / events_per_minute state."""
        rec = {"eventID": "11111111-2222-3333-4444-555555555555", "eventTime": "2024-01-01T10:00:00Z",
               "eventName": "CreateAccessKey", "eventSource": "iam.amazonaws.com",
               "userIdentity": {"type": "IAMUser", "arn": "arn:aws:iam::111122223333:user/u1", "userName": "u1"}}
        path = os.path.join(self.tmp, "in.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"Records": [rec, dict(rec)]}, f)
        n, _, _ = self.run_batch_file(path)
        self.assertEqual(n, 1)


class TestInvalidLabels(unittest.TestCase):
    def test_canonical_labels_are_counted(self):
        p = fe9.AdaptiveRiskPrior({}, path=None, frozen=False)
        for y in ("1", 1, "0", 0):
            p.update("X", y)
        self.assertEqual(tuple(p.counts["X"]), (2, 4))

    @unittest.expectedFailure
    def test_non_canonical_labels_are_not_silently_treated_as_benign(self):
        """KNOWN DEFECT: AdaptiveRiskPrior.update counts an attack only when
        str(label) == '1'. A float-formatted label ('1.0', what pandas writes for a label
        column that ever held a NaN) or any garbage value ('yes', 'abc', '-1') is silently
        folded into the label-derived prior as BENIGN, with no error. (pipeline.featurize
        separately coerces unparseable labels to 0 with pd.to_numeric(errors='coerce').)"""
        p = fe9.AdaptiveRiskPrior({}, path=None, frozen=False)
        p.update("X", "1.0")
        self.assertEqual(tuple(p.counts["X"]), (1, 1))
        with self.assertRaises(ValueError):
            p.update("X", "abc")


class TestSchemaDrift(PrepSandbox):
    def test_extra_input_column_is_ignored(self):
        rows = [base_row(i, event_name=n) for i, n in enumerate(["ListRoles", "CreateUser", "GetUser"])]
        plain = self.write_csv("a.csv", rows)
        extra = self.write_csv("b.csv", [dict(r, surprise="x") for r in rows], FIELDS + ["surprise"])
        feats = []
        for path in (plain, extra):
            e = self.engine()
            feats.append([(e.get_structural_data(r), e.get_temporal_features(r)) for r in fe9.iter_input_rows(path)])
        self.assertEqual(feats[0], feats[1])

    @unittest.expectedFailure
    def test_missing_event_name_column_gets_the_placeholder_or_fails(self):
        """KNOWN DEFECT: get_structural_data intends 'unknown_action' for a row without an
        action (log.get('event_name', 'unknown_action')), but normalize_cloudtrail_row
        always sets the key (to None when the column is absent), so the default never
        applies: edge_type is None, written as an empty field in the structural CSV and
        read back as NaN by the graph loader -- neither the placeholder nor a loud error."""
        path = self.write_csv("in.csv", [base_row(0)], [c for c in FIELDS if c != "event_name"])
        row = next(fe9.iter_input_rows(path))
        try:
            edge_type = self.engine().get_structural_data(row)["edge_type"]
        except ValueError:
            return                                   # a loud failure is also acceptable
        self.assertEqual(edge_type, "unknown_action")

    def test_output_with_an_older_column_layout_is_refused(self):
        with open(fe9.STRUCT_OUT, "w", encoding="utf-8") as f:
            f.write("log_id,source_node,target_node,edge_type,label\n")     # pre-fe9 layout
        path = self.write_csv("in.csv", [base_row(0)])
        with self.assertRaises(SystemExit) as ctx:
            self.run_batch_file(path)
        self.assertIn("different column layout", str(ctx.exception))


class TestSessionConsistencyOnDev(unittest.TestCase):
    """Session-level evaluation groups events by session_id; it relies on these."""

    @classmethod
    def setUpClass(cls):
        cls.dev = pd.read_csv(DEV, usecols=["session_id", "data_source", "label", "session_label"])

    def test_every_event_has_a_session_and_binary_labels(self):
        self.assertFalse(self.dev["session_id"].isna().any())
        self.assertTrue(set(self.dev["label"].unique()) <= {0, 1})
        self.assertTrue(set(self.dev["session_label"].unique()) <= {0, 1})

    def test_a_session_has_one_source_and_one_label(self):
        g = self.dev.groupby("session_id")
        self.assertEqual(int(g["data_source"].nunique().max()), 1)
        self.assertEqual(int(g["session_label"].nunique().max()), 1)

    def test_session_label_is_the_max_of_its_event_labels(self):
        g = self.dev.groupby("session_id")
        self.assertTrue((g["session_label"].first() == g["label"].max()).all())


# ── 3. derived datasets line up with their source rows ─────────────────────────

def check_derived(tc, source_csv, structural_csv, temporal_csv):
    name = os.path.basename(source_csv)
    src = list(fe9.iter_input_rows(source_csv))
    st = pd.read_csv(structural_csv, dtype=str, keep_default_na=False)
    tp = pd.read_csv(temporal_csv, dtype=str, keep_default_na=False)
    ids = [f"{name}:{i}" for i in range(len(src))]
    tc.assertEqual(list(st["log_id"]), ids, "structural rows must be the source rows, in order, none skipped")
    tc.assertEqual(sorted(tp["log_id"]), sorted(ids), "temporal must hold exactly the same events")
    tc.assertEqual(list(st["edge_type"]), [r["event_name"] or "" for r in src])
    tc.assertEqual(list(st["source_node"]), [r["principal_arn"] for r in src])
    tc.assertEqual(list(st["target_node"]), [r["target_resource"] for r in src])
    tc.assertEqual([int(float(x)) for x in st["label"]], [int(float(r["label"])) for r in src])
    tp = tp.set_index("log_id").loc[ids]
    tc.assertEqual(list(tp["username"]), [r["username"] for r in src])
    tc.assertEqual(list(tp["timestamp"]), [r["timestamp"] for r in src])
    tc.assertEqual([int(float(x)) for x in tp["label"]], [int(float(r["label"])) for r in src])


class TestDerivedDatasetsAlign(unittest.TestCase):
    def test_synthetic_structural_and_temporal_match_the_source(self):
        check_derived(self, SYNTHETIC, os.path.join(DATA, "cloudtrail_structural.csv"),
                      os.path.join(DATA, "cloudtrail_temporal.csv"))

    def test_dev_structural_and_temporal_match_the_source(self):
        check_derived(self, DEV, os.path.join(DATA, "real_dataset_dev_structural.csv"),
                      os.path.join(DATA, "real_dataset_dev_temporal.csv"))

    def test_campaign_annotations_align_with_synthetic_rows(self):
        ann = pd.read_csv(os.path.join(DATA, "synthetic_campaign_annotations.csv"), dtype=str, keep_default_na=False)
        syn = pd.read_csv(SYNTHETIC, usecols=["label"], dtype=str, keep_default_na=False)
        self.assertEqual(list(ann["log_id"]), [f"synthetic_cloudtrail.csv:{i}" for i in range(len(syn))])
        self.assertEqual(list(ann["label"]), list(syn["label"]))

    def test_test_split_derived_files_have_the_source_shape(self):
        """Shape only -- no test value is read beyond counting rows and ids."""
        n = len(pd.read_csv(TEST, usecols=["timestamp"]))
        for suffix in ("structural", "temporal"):
            ids = pd.read_csv(os.path.join(DATA, f"real_dataset_test_{suffix}.csv"), usecols=["log_id"])["log_id"]
            self.assertEqual(sorted(ids), sorted(f"real_dataset_test.csv:{i}" for i in range(n)), suffix)


if __name__ == "__main__":
    unittest.main()
