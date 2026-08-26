"""
test_evaluation_integrity.py
============================
Unit tests for the evaluation path: the edge->session join, the session
aggregation rule, the paired baseline comparison, and the two guards that stop
a silently-wrong number from being produced.

WHY THIS FILE EXISTS: the audit's single most serious finding (V-1) was not a
crash or a bad model -- it was a REPORTED NUMBER that looked fine and was
meaningless, because the rule baseline was measured on 397 dev+test sessions
while the model was measured on 238 test sessions. Nothing failed. Nothing
warned. The comparison simply was not valid.

Every test here pins one way this pipeline can produce a plausible wrong
answer:

  - baseline measured on a different population than the model -> TestPairedBaselineComparison
  - scoring a graph against the wrong CSV                       -> TestGraphProvenanceGuard
  - joining edges to sessions by an index from another file     -> TestRowIndexRecovery
  - sessions with no in-schema edges silently vanishing         -> TestSessionAggregation

NO NEO4J AND NO CHECKPOINT REQUIRED. Run with:

    python -m unittest test_evaluation_integrity
"""

import unittest

import numpy as np
import pandas as pd

from evaluate_session_level import (
    GUARDDUTY,
    check_graph_provenance,
    parse_row_indices,
    report_baseline_comparison,
    session_max_scores,
)


def _raw_df(rows):
    """rows: list of (session_id, session_label, event_name)."""
    return pd.DataFrame(
        [{"session_id": s, "session_label": y, "event_name": e} for s, y, e in rows]
    )


class TestGraphProvenanceGuard(unittest.TestCase):
    """Neo4j holds exactly ONE graph at a time. Before provenance stamping,
    nothing stopped this script from scoring the test graph against the dev
    CSV: row indices exist in both files, so the join succeeded and produced a
    plausible, entirely meaningless F1."""

    def test_matching_graph_and_csv_passes(self):
        check_graph_provenance("real_dataset_test_structural.csv", "real_dataset_test.csv")

    def test_mismatched_graph_and_csv_exits(self):
        with self.assertRaises(SystemExit) as ctx:
            check_graph_provenance("real_dataset_dev_structural.csv", "real_dataset_test.csv")
        msg = str(ctx.exception)
        self.assertIn("GRAPH MISMATCH", msg)
        self.assertIn("real_dataset_dev_structural.csv", msg)

    def test_error_names_the_command_that_fixes_it(self):
        with self.assertRaises(SystemExit) as ctx:
            check_graph_provenance("cloudtrail_structural.csv", "real_dataset_test.csv")
        self.assertIn("build_graph.py", str(ctx.exception))

    def test_unstamped_graph_warns_rather_than_failing(self):
        """Graphs built before provenance stamping must still load, or every
        previously-built graph becomes unusable. stdout is captured so the
        warning does not leak into the test runner's output -- we assert it was
        emitted rather than letting it print."""
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            check_graph_provenance(None, "real_dataset_test.csv")
        self.assertIn("predates provenance stamping", buf.getvalue())

    def test_synthetic_graph_against_real_csv_is_caught(self):
        with self.assertRaises(SystemExit):
            check_graph_provenance("cloudtrail_structural.csv", "real_dataset_dev.csv")


class TestRowIndexRecovery(unittest.TestCase):
    """log_id is "<raw_csv_filename>:<row_index>". Recovering the row index is
    how each edge finds its session. Indexing a raw CSV with a row number that
    originated in a DIFFERENT file is silent corruption -- every edge gets
    attributed to some real-looking but wrong session."""

    def test_recovers_indices_in_order(self):
        ids = ["real_dataset_test.csv:0", "real_dataset_test.csv:7", "real_dataset_test.csv:3"]
        np.testing.assert_array_equal(
            parse_row_indices(ids, "real_dataset_test.csv", 10), np.array([0, 7, 3])
        )

    def test_log_id_from_another_file_exits(self):
        ids = ["real_dataset_test.csv:0", "real_dataset_dev.csv:1"]
        with self.assertRaises(SystemExit) as ctx:
            parse_row_indices(ids, "real_dataset_test.csv", 10)
        self.assertIn("LOG_ID MISMATCH", str(ctx.exception))

    def test_index_past_the_end_of_the_csv_exits(self):
        """Catches a structural CSV that has drifted out of sync with its raw
        CSV -- which is exactly what would happen if feature engineering ever
        skipped a row while numbering log_ids by a filtered counter."""
        with self.assertRaises(SystemExit) as ctx:
            parse_row_indices(["real_dataset_test.csv:99"], "real_dataset_test.csv", 10)
        self.assertIn("OUT OF RANGE", str(ctx.exception))

    def test_last_valid_index_is_accepted(self):
        out = parse_row_indices(["real_dataset_test.csv:9"], "real_dataset_test.csv", 10)
        self.assertEqual(out[0], 9)

    def test_unparseable_log_id_raises(self):
        with self.assertRaises(ValueError):
            parse_row_indices(["no_row_index_here"], "real_dataset_test.csv", 10)

    def test_filenames_containing_colons_still_parse(self):
        """The regex is greedy on the prefix and anchors the index at the end,
        so only the FINAL colon separates the row number."""
        out = parse_row_indices(["a:b.csv:4"], "a:b.csv", 10)
        self.assertEqual(out[0], 4)

    def test_empty_input_does_not_crash(self):
        self.assertEqual(len(parse_row_indices([], "real_dataset_test.csv", 10)), 0)


class TestSessionAggregation(unittest.TestCase):
    """Session score = MAX edge probability, mirroring the rule baseline's
    "flag the session if ANY event trips a rule". The two must be the same unit
    or the comparison is meaningless -- which was the original reason
    evaluate_on_real.py's edge-level F1 could not be compared to the baseline."""

    def setUp(self):
        self.raw = _raw_df([
            ("s1", 1, "CreateUser"), ("s1", 1, "AttachUserPolicy"), ("s1", 1, "GetObject"),
            ("s2", 0, "ListBuckets"), ("s2", 0, "GetObject"),
            ("s3", 1, "StopLogging"),
        ])
        self.index = self.raw.drop_duplicates("session_id").set_index("session_id").index

    def test_session_score_is_the_max_of_its_edges(self):
        row_idx = np.array([0, 1, 2, 3, 4, 5])
        probs = np.array([0.1, 0.9, 0.2, 0.3, 0.05, 0.6])
        scores, _ = session_max_scores(row_idx, probs, self.raw, self.index)
        np.testing.assert_allclose(scores, [0.9, 0.3, 0.6])

    def test_a_single_high_edge_carries_the_whole_session(self):
        """This IS the detection mechanism, and the paper must describe it
        precisely: one correctly-flagged action per attack session, not
        accurate per-action classification."""
        row_idx = np.array([0, 1, 2])
        probs = np.array([0.01, 0.99, 0.01])
        scores, _ = session_max_scores(row_idx, probs, self.raw, self.index)
        self.assertAlmostEqual(scores[0], 0.99)

    def test_sessions_with_no_in_schema_edges_score_zero_not_dropped(self):
        """Out-of-schema edges are excluded from scoring, so some sessions have
        no scorable edge at all. They must default to benign and REMAIN in the
        denominator -- dropping them would quietly shrink the evaluation set
        and inflate every metric."""
        row_idx = np.array([0])          # only s1 has a scored edge
        probs = np.array([0.8])
        scores, _ = session_max_scores(row_idx, probs, self.raw, self.index)
        self.assertEqual(len(scores), 3, "all sessions must remain in the evaluation")
        np.testing.assert_allclose(scores, [0.8, 0.0, 0.0])

    def test_scores_align_with_the_label_index_order(self):
        row_idx = np.array([5, 0, 3])
        probs = np.array([0.7, 0.2, 0.4])
        scores, _ = session_max_scores(row_idx, probs, self.raw, self.index)
        by_session = dict(zip(self.index, scores))
        self.assertAlmostEqual(by_session["s1"], 0.2)
        self.assertAlmostEqual(by_session["s2"], 0.4)
        self.assertAlmostEqual(by_session["s3"], 0.7)


class TestPairedBaselineComparison(unittest.TestCase):
    """REGRESSION for the audit's V-1, the finding that would have sunk the
    paper. The script used to PRINT a hardcoded "F1=0.732", a figure computed
    on real_dataset_combined.csv (397 dev+test sessions), directly beneath a
    model score computed on 238 test sessions. On test alone the same rule set
    scores 0.747. The fix is structural: the baseline is now computed on
    exactly the sessions just scored, so the populations cannot diverge."""

    def setUp(self):
        # 4 attack sessions, 4 benign. GuardDuty rules fire on AttachUserPolicy
        # and StopLogging; a1/a2 are catchable, a3/a4 are credential-access
        # flavoured and invisible to the rule set -- the real recall gap.
        self.raw = _raw_df([
            ("a1", 1, "AttachUserPolicy"),
            ("a2", 1, "StopLogging"),
            ("a3", 1, "GetSecretValue"),
            ("a4", 1, "GetParameters"),
            ("b1", 0, "ListBuckets"),
            ("b2", 0, "GetObject"),
            ("b3", 0, "DescribeInstances"),
            ("b4", 0, "ListRoles"),
        ])
        self.sessions_true = self.raw.drop_duplicates("session_id") \
                                      .set_index("session_id")["session_label"]
        self.y_true = self.sessions_true.to_numpy()

    def _capture(self, y_model):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report_baseline_comparison(self.raw, self.sessions_true, self.y_true, y_model)
        return buf.getvalue()

    def test_baseline_is_computed_from_the_scored_sessions(self):
        """The rule set catches a1 and a2 only: P=1.000, R=0.500, F1=0.667 on
        THESE eight sessions. If this ever prints 0.732 or 0.747 instead, the
        hardcoded literal is back."""
        out = self._capture(np.array([1, 1, 1, 1, 0, 0, 0, 0]))
        self.assertIn(GUARDDUTY, out)
        self.assertIn("0.667", out)

    def test_no_hardcoded_baseline_figure_is_printed(self):
        out = self._capture(np.array([1, 1, 1, 1, 0, 0, 0, 0]))
        for stale in ("0.732", "0.672, 0.790"):
            self.assertNotIn(stale, out, f"hardcoded baseline {stale!r} reappeared")

    def test_reports_a_paired_difference_with_an_interval(self):
        out = self._capture(np.array([1, 1, 1, 1, 0, 0, 0, 0]))
        self.assertIn("PAIRED bootstrap", out)
        self.assertIn("95% CI", out)
        self.assertIn("p =", out)

    def test_a_perfect_model_beats_the_rule_set(self):
        out = self._capture(np.array([1, 1, 1, 1, 0, 0, 0, 0]))
        self.assertIn("+0.3333", out)

    def test_an_identical_model_shows_no_difference_and_is_not_significant(self):
        """Guards against a comparison that always flatters the model: feeding
        it the rule set's own predictions must yield delta 0 and a
        not-significant verdict."""
        rules = self.raw.groupby("session_id")["event_name"].apply(set) \
                        .reindex(self.sessions_true.index)
        from evaluate_baselines import RULES
        y_rule = rules.apply(lambda s: int(bool(s & RULES[GUARDDUTY]))).to_numpy()
        out = self._capture(y_rule)
        self.assertIn("+0.0000", out)
        self.assertIn("NOT significant", out)

    def test_a_worse_model_reports_a_negative_difference(self):
        out = self._capture(np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        self.assertIn("PAIRED bootstrap on (GNN - rule) F1: -", out)
        self.assertIn("NOT significant", out)

    def test_verdict_language_matches_the_interval(self):
        """A CI that includes zero must never be described as beating the
        baseline -- the exact overstatement the audit flagged."""
        out = self._capture(np.array([1, 1, 1, 0, 0, 0, 0, 0]))
        if "NOT significant" in out:
            self.assertNotIn("is significant at the 5% level.\n", out)

    def test_is_deterministic_across_runs(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(self._capture(y), self._capture(y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
