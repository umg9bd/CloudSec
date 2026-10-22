"""
test_dashboard.py
=================
Guards cloudsec_dashboard.py, the Streamlit console over the live pipeline's output, against
showing something other than what the pipeline wrote:

  - Scores: every HGT / LSTM / Risk value in the processed-log table is the pipeline's own
    risk_scores.csv value for that log_id (pipeline.OUTPUT_COLS), in arrival order; duplicates
    keep their first row (the table selection is a row position); alert/fast_lane parse to bools;
    files from before newer columns existed still load. Filters show exactly the flagged rows.
  - Alerts: alerts/*.json (pipeline.Pipeline.emit schema) are indexed by log_id and by
    (principal, source_file); a half-written file is skipped, not a crash; a new alert or new
    scored rows show up on the next refresh -- also under real st.cache_data (AppTest).
  - Explanations: the "why" text is derived only from the alert's explanation (model shares,
    feature labels, earlier events) -- nothing invented. One known defect is kept as an
    expectedFailure.

Fast: no torch, no Neo4j. Unit/console logic runs against a recording stub of `streamlit`
(module-level st.* calls run on import); one end-to-end run uses streamlit.testing AppTest on a
temporary copy of the script, so the repo's own output/ and alerts/ are never read or written.
"""

import ast
import contextlib
import glob
import html
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD = os.path.join(ROOT, "cloudsec_dashboard.py")


def _literal_from_source(path, func=None, name=None):
    """Reads a schema from source without importing it (pipeline / ensemble_explain pull in
    torch, ~30 s here): a module-level list assignment, or the keys of a function's returned
    dict literal."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if name and isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
        if func and isinstance(node, ast.FunctionDef) and node.name == func:
            ret = [n for n in ast.walk(node) if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)][-1]
            return [k.value for k in ret.value.keys]
    raise LookupError(name or func)


OUTPUT_COLS = _literal_from_source(os.path.join(ROOT, "pipeline.py"), name="OUTPUT_COLS")
EXPLANATION_KEYS = _literal_from_source(os.path.join(ROOT, "ensemble_explain.py"), func="combine")


# ── a recording stand-in for streamlit ──────────────────────────────────────
class _Ctx:
    def __init__(self, st):
        self.st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def markdown(self, body, **k):
        self.st.log.append(body)

    def radio(self, *a, **k):
        return self.st.radio_value


class FakeStreamlit:
    def __init__(self):
        self.fragments, self.column_config = {}, mock.MagicMock()
        self.reset()

    def reset(self):
        self.log, self.tables, self.selected_rows, self.radio_value = [], [], [], "All events"

    def markdown(self, body, **k):
        self.log.append(body)

    info = write = caption = success = error = warning = subheader = markdown

    def cache_data(self, fn=None, **k):
        return fn if fn is not None else (lambda f: f)

    def fragment(self, run_every=None):
        def deco(f):                  # registered, not run: the import-time call is a no-op
            self.fragments[f.__name__] = f
            return lambda *a, **k: None
        return deco

    def columns(self, spec, **k):
        return [_Ctx(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def dataframe(self, table, **k):
        self.tables.append(table)
        return SimpleNamespace(selection=SimpleNamespace(rows=list(self.selected_rows)))

    def set_page_config(self, **k):
        pass

    sidebar = property(lambda self: _Ctx(self))
    form = expander = lambda self, *a, **k: _Ctx(self)
    text_input = lambda self, *a, value="", **k: value
    form_submit_button = button = lambda self, *a, **k: False
    toggle = lambda self, *a, value=True, **k: value


FAKE = FakeStreamlit()
_spec = importlib.util.spec_from_file_location("cloudsec_dashboard_under_test", DASHBOARD)
dash = importlib.util.module_from_spec(_spec)
with mock.patch.dict(sys.modules, {"streamlit": FAKE}):
    _spec.loader.exec_module(dash)
console = FAKE.fragments["console"]


def text(bodies):
    """Rendered markdown as plain text (tags dropped, entities unescaped)."""
    return html.unescape(re.sub(r"<[^>]+>", "", "\n".join(bodies)))


# ── fixtures in the pipeline's own schemas ──────────────────────────────────
W, THRESHOLD = 0.5, 0.5


def scored_row(log_id, user, event, p_graph, p_seq, fast_lane=False, reason=None, minute=0):
    risk = p_seq if p_graph is None else W * p_graph + (1 - W) * p_seq
    return {"log_id": log_id, "timestamp": f"2026-01-01 00:{minute:02d}:00+00:00", "username": user,
            "principal_arn": f"arn:aws:iam::111:user/{user}", "source_ip": "10.0.0.1", "event_name": event,
            "source_node": f"arn:aws:iam::111:user/{user}", "target_node": "arn:aws:s3:::bucket/secret-key",
            "edge_type": event, "p_graph": np.nan if p_graph is None else p_graph, "p_sequence": p_seq,
            "risk_score": round(risk * 10, 2), "alert": risk >= THRESHOLD, "fast_lane": fast_lane,
            "fast_lane_reason": reason}


def explanation(row, seq_feats, seq_events, graph_feats, related=()):
    """ensemble_explain.combine's output shape for one scored row (shares computed the same way)."""
    pg, ps = (None if pd.isna(row["p_graph"]) else row["p_graph"]), row["p_sequence"]
    g = W * pg if pg is not None else 0.0
    s = (1 - W) * ps if pg is not None else ps
    risk = g + s
    exp = {
        "log_id": row["log_id"], "event_name": row["event_name"], "principal": row["username"],
        "timestamp": row["timestamp"], "risk_score": round(risk * 10, 2), "threshold": THRESHOLD,
        "driven_by": "graph" if g >= s else "sequence",
        "models": {"graph": {"probability": pg, "weight": W if pg is not None else 0.0,
                             "contribution": round(g, 4), "share": round(g / risk, 4) if risk else 0.0},
                   "sequence": {"probability": ps, "weight": (1 - W) if pg is not None else 1.0,
                                "contribution": round(s, 4), "share": round(s / risk, 4) if risk else 0.0}},
        "graph": None if pg is None else {"top_features": graph_feats, "related_events": list(related)},
        "sequence": {"score": ps, "top_features": seq_feats, "top_events": seq_events},
        "fast_lane": row["fast_lane_reason"], "summary": f"risk {risk * 10:.2f}/10 (alert at {THRESHOLD * 10:.2f})",
    }
    assert list(exp) == EXPLANATION_KEYS, "fixture drifted from ensemble_explain.combine"
    return exp


def alert_json(source_file, principal, rows, explanations):
    """pipeline.Pipeline.emit's alert schema."""
    rows = sorted(rows, key=lambda r: -r["risk_score"])
    return {"alert_id": f"{principal}-{source_file}", "source_file": source_file, "principal": principal,
            "principal_arn": rows[0]["principal_arn"], "first_seen": rows[0]["timestamp"],
            "last_seen": rows[-1]["timestamp"], "max_risk_score": rows[0]["risk_score"],
            "n_flagged_events": len(rows), "fast_lane_reasons": sorted({r["fast_lane_reason"] for r in rows
                                                                         if r["fast_lane_reason"]}),
            "threshold": THRESHOLD,
            "events": [{"log_id": r["log_id"], "timestamp": r["timestamp"], "event_name": r["event_name"],
                        "target": r["target_node"], "p_graph": None if pd.isna(r["p_graph"]) else r["p_graph"],
                        "p_sequence": r["p_sequence"], "risk_score": r["risk_score"]} for r in rows],
            "explanations": explanations}


SEQ_FEATS = [{"feature": "burst", "label": "Burst of activity", "contribution_window": 0.30},
             {"feature": "calm", "label": "Usual hours", "contribution_window": -0.20},
             {"feature": "night", "label": "Off-hours activity", "contribution_window": 0.10}]
SEQ_EVENTS = [{"event_name": "ListUsers", "minutes_before": 3.0, "effect": 0.12, "score_without": 0.58},
              {"event_name": "<UNK>", "minutes_before": 0.5, "effect": -0.05, "score_without": 0.75}]
GRAPH_FEATS = [{"feature": "privilege_gain", "label": "Privilege gain", "share": 0.6},
               {"feature": "hop_count", "label": "Hops from the principal", "share": 0.3}]

ROWS = [scored_row("f1.json:0", "alice", "ListBuckets", 0.10, 0.10, minute=0),
        scored_row("f1.json:1", "alice", "AttachUserPolicy", 0.90, 0.70, minute=1),
        scored_row("f1.json:2", "alice", "PutUserPolicy", 0.80, 0.60, minute=2),
        scored_row("f1.json:3", "bob", "StopLogging", None, 0.30, fast_lane=True,
                   reason="CloudTrail logging stopped", minute=3)]
EXP_TOP = explanation(ROWS[1], SEQ_FEATS, SEQ_EVENTS, GRAPH_FEATS)
EXP_FL = explanation(ROWS[3], SEQ_FEATS[:1], [], [])


class DashboardSandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.scores = os.path.join(self.tmp, "output", "risk_scores.csv")
        self.alerts = os.path.join(self.tmp, "alerts")
        self.config = os.path.join(self.tmp, "pipeline_config.json")
        os.makedirs(os.path.dirname(self.scores))
        os.makedirs(self.alerts)
        with open(self.config, "w", encoding="utf-8") as f:
            json.dump({"alert_threshold": THRESHOLD, "weight_graph": W}, f)
        self.patches = [mock.patch.object(dash, k, v) for k, v in
                        (("SCORES", self.scores), ("ALERTS", self.alerts), ("CONFIG", self.config))]
        for p in self.patches:
            p.start()
        FAKE.reset()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_scores(self, rows, append=False):
        """Exactly what pipeline.append_csv writes: OUTPUT_COLS, header once, appended."""
        pd.DataFrame(rows)[OUTPUT_COLS].to_csv(self.scores, mode="a" if append else "w", index=False,
                                               header=not append)

    def write_alert(self, alert):
        with open(os.path.join(self.alerts, f"alert_{alert['alert_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(alert, f, indent=2)

    def write_fixture(self):
        self.write_scores(ROWS)
        self.write_alert(alert_json("f1.json", "alice", ROWS[1:3], [EXP_TOP]))
        self.write_alert(alert_json("f1.json", "bob", ROWS[3:], [EXP_FL]))


# ═══════════════════════════════════════════════════════════════════════════
# 1. loading what the pipeline wrote
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadScores(DashboardSandbox):
    def test_values_equal_pipeline_csv(self):
        self.write_scores(ROWS)
        df = dash.load_scores(0)
        self.assertEqual(list(df["log_id"]), [r["log_id"] for r in ROWS])
        for col in ("p_sequence", "risk_score"):
            np.testing.assert_array_equal(df[col].to_numpy(), [r[col] for r in ROWS])
        np.testing.assert_array_equal(df["p_graph"].to_numpy(), [r["p_graph"] for r in ROWS])  # NaN kept
        self.assertEqual(list(df["alert"]), [r["alert"] for r in ROWS])
        self.assertEqual(list(df["fast_lane"]), [r["fast_lane"] for r in ROWS])
        self.assertEqual(df["alert"].dtype, bool)

    def test_duplicate_log_id_keeps_first_so_row_positions_stay_stable(self):
        self.write_scores(ROWS)
        self.write_scores([dict(ROWS[1], risk_score=0.01)], append=True)
        df = dash.load_scores(0)
        self.assertEqual(len(df), len(ROWS))
        self.assertEqual(float(df.loc[df["log_id"] == "f1.json:1", "risk_score"].iloc[0]), ROWS[1]["risk_score"])

    def test_file_from_before_newer_columns_still_loads(self):
        old_cols = [c for c in OUTPUT_COLS if c not in ("fast_lane", "fast_lane_reason", "principal_arn",
                                                         "source_ip")]
        pd.DataFrame(ROWS)[old_cols].to_csv(self.scores, index=False)
        df = dash.load_scores(0)
        self.assertFalse(df["fast_lane"].any())
        self.assertTrue(df["fast_lane_reason"].isna().all())

    def test_alert_threshold_is_pipeline_config_on_the_0_10_scale(self):
        self.assertEqual(dash.alert_threshold(), THRESHOLD * 10)
        os.remove(self.config)
        self.assertEqual(dash.alert_threshold(), 5.0)


class TestLoadAlerts(DashboardSandbox):
    def test_indexes_explanations_by_log_id_and_alerts_by_principal_and_file(self):
        self.write_fixture()
        alerts, exps, by_group = dash.load_alerts()
        self.assertEqual(len(alerts), 2)
        self.assertEqual(exps, {"f1.json:1": EXP_TOP, "f1.json:3": EXP_FL})
        self.assertEqual(set(by_group), {("alice", "f1.json"), ("bob", "f1.json")})
        # console looks a row's group up by log_id minus its ":<row>" suffix -- must match source_file
        self.assertIn(("alice", ROWS[2]["log_id"].rsplit(":", 1)[0]), by_group)

    def test_half_written_alert_is_skipped_and_new_alert_appears_next_call(self):
        self.write_fixture()
        with open(os.path.join(self.alerts, "alert_partial.json"), "w", encoding="utf-8") as f:
            f.write('{"alert_id": "x", "princ')
        self.assertEqual(len(dash.load_alerts()[0]), 2)
        row = scored_row("f2.json:0", "carol", "CreateAccessKey", 0.9, 0.9)
        exp = explanation(row, SEQ_FEATS, [], GRAPH_FEATS)
        self.write_alert(alert_json("f2.json", "carol", [row], [exp]))
        alerts, exps, _ = dash.load_alerts()
        self.assertEqual(len(alerts), 3)
        self.assertEqual(exps["f2.json:0"], exp)


# ═══════════════════════════════════════════════════════════════════════════
# 2. explanation text: derived from the alert, nothing invented
# ═══════════════════════════════════════════════════════════════════════════

class TestExplanationText(DashboardSandbox):
    def test_reasons_are_the_explanations_own_drivers(self):
        m = EXP_TOP["models"]
        self.assertEqual(text(dash.plain_reasons(EXP_TOP)).splitlines(), [
            f"Activity pattern ({m['sequence']['share']:.0%} of the risk): Burst of activity and Off-hours activity.",
            "What came before: ListUsers, 3 min earlier, made it look more like an attack.",
            f"Access graph ({m['graph']['share']:.0%} of the risk): Privilege gain and Hops from the principal.",
        ])  # "Usual hours" (negative) and the <UNK> event (lowered the risk) are not reasons

    def test_graph_unscored_says_so_and_claims_no_graph_share(self):
        out = text(dash.plain_reasons(EXP_FL))
        self.assertIn("Access graph: no score for this kind of access", out)
        self.assertNotIn("% of the risk): Privilege", out)
        self.assertIn(f"Activity pattern ({EXP_FL['models']['sequence']['share']:.0%} of the risk)", out)

    def test_share_bar_and_technical_numbers_are_the_explanations(self):
        dash.render_explanation(EXP_TOP)
        out = text(FAKE.log)
        m = EXP_TOP["models"]
        self.assertIn(f"Access graph {m['graph']['share']:.0%}", out)
        self.assertIn(f"Activity pattern {m['sequence']['share']:.0%}", out)
        self.assertIn(f"HGT score {m['graph']['probability']:.0%} · LSTM score {m['sequence']['probability']:.0%} · "
                      f"risk {EXP_TOP['risk_score']:.2f}/10", out)
        self.assertIn(f"log_id {EXP_TOP['log_id']}", out)
        # earlier-event points = (LSTM score - score without it) * 100
        self.assertIn(f"{(0.70 - 0.58) * 100:+.1f} pts", out)

    def test_technical_hgt_na_when_graph_unscored(self):
        dash.render_technical(EXP_FL)
        self.assertIn("HGT score n/a", text(FAKE.log))
        self.assertIn("No graph score for this event", text(FAKE.log))

    def test_explanation_strings_are_html_escaped(self):
        exp = json.loads(json.dumps(EXP_TOP))
        exp["sequence"]["top_features"][0]["label"] = "<script>x</script>"
        self.assertNotIn("<script>", "".join(dash.plain_reasons(exp)))

    @unittest.expectedFailure
    def test_no_factor_fallback_does_not_invent_a_claim(self):
        """KNOWN DEFECT: when plain_reasons() finds no driver, render_explanation shows the fixed
        text "No single factor stands out; both models scored it moderately high." regardless of
        the scores. pipeline.explain explains every FAST-LANE event whatever its risk, so a
        rule-flagged event with HGT p=0.05 and LSTM p=0.40 (risk 2.25/10, alert at 5.00; graph
        share 11% < MIN_DRIVER_SHARE, LSTM's only feature lowered the risk) is shown as
        'both models scored it moderately high' -- a claim the explanation does not support."""
        row = scored_row("f3.json:0", "dave", "DeleteTrail", 0.05, 0.40, fast_lane=True, reason="trail deleted")
        exp = explanation(row, [SEQ_FEATS[1]], [], GRAPH_FEATS)
        dash.render_explanation(exp)
        self.assertNotIn("scored it moderately high", text(FAKE.log))

    def test_event_header_shows_csv_risk_and_config_threshold(self):
        self.write_scores(ROWS)
        df = dash.load_scores(0)
        for i, banner in ((1, "ALERT -- risk {:.2f} out of 10 (alerts start at 5.00)"),
                          (0, "Not flagged -- risk {:.2f} out of 10 (alerts start at 5.00)")):
            FAKE.reset()
            dash.render_event_header(df.iloc[i], dash.alert_threshold())
            self.assertIn(banner.format(ROWS[i]["risk_score"]), text(FAKE.log))
        FAKE.reset()
        dash.render_event_header(df.iloc[3], 5.0)
        self.assertIn("CloudTrail logging stopped -- flagged by rule", text(FAKE.log))


# ═══════════════════════════════════════════════════════════════════════════
# 3. the console page (stubbed streamlit)
# ═══════════════════════════════════════════════════════════════════════════

class TestConsole(DashboardSandbox):
    def run_console(self, selected=(), radio="All events"):
        FAKE.reset()
        FAKE.selected_rows, FAKE.radio_value = list(selected), radio
        console()
        return FAKE.tables[-1] if FAKE.tables else None, text(FAKE.log)

    def test_waits_for_first_scores(self):
        table, out = self.run_console()
        self.assertIsNone(table)
        self.assertIn("Waiting for the pipeline's first scores", out)

    def test_table_values_equal_pipeline_csv_in_arrival_order(self):
        self.write_fixture()
        table, out = self.run_console()
        self.assertEqual(list(table["Action"]), [r["event_name"] for r in ROWS])
        np.testing.assert_array_equal(table["HGT"].to_numpy(), [r["p_graph"] for r in ROWS])
        np.testing.assert_array_equal(table["LSTM"].to_numpy(), [r["p_sequence"] for r in ROWS])
        np.testing.assert_array_equal(table["Risk"].to_numpy(), [r["risk_score"] for r in ROWS])
        self.assertEqual(list(table["Flag"]), ["", "▲ ALERT", "▲ ALERT", "⚡ FAST-LANE"])
        self.assertEqual(list(table["Why"]), ["", "explained", "", "explained"])
        self.assertEqual(list(table["Target"]), ["secret-key"] * 4)
        avg = pd.Series([r["risk_score"] for r in ROWS]).mean()
        for stat in ("4Events processed", "2Alert events", "1Fast-lane events", "2Alerts raised",
                     f"{avg:.2f}Avg risk /10", "2 principals alerted"):
            self.assertIn(stat, out.replace("\n", ""))

    def test_filters_show_exactly_the_flagged_rows(self):
        self.write_fixture()
        alerts_view, _ = self.run_console(radio="Alerts")
        self.assertEqual(list(alerts_view["Action"]), ["AttachUserPolicy", "PutUserPolicy"])
        fl_view, _ = self.run_console(radio="Fast-lane")
        self.assertEqual(list(fl_view["Action"]), ["StopLogging"])

    def test_default_panel_is_latest_explained_alert_with_its_own_reasons(self):
        self.write_fixture()
        _, out = self.run_console()
        self.assertIn("the latest explained alert", out)
        for reason in dash.plain_reasons(EXP_FL):           # ROWS[3] is the last explained row
            self.assertIn(text([reason]), out)
        self.assertIn("CloudTrail logging stopped -- flagged by rule", out)   # ROWS[3] is fast-lane

    def test_selected_row_shows_that_rows_explanation(self):
        self.write_fixture()
        _, out = self.run_console(selected=[1])
        self.assertIn("Why was this event flagged?", out)
        self.assertIn(f"risk {ROWS[1]['risk_score']:.2f} out of 10", out)
        for reason in dash.plain_reasons(EXP_TOP):
            self.assertIn(text([reason]), out)

    def test_unexplained_alert_row_points_to_its_groups_top_explanation(self):
        self.write_fixture()
        _, out = self.run_console(selected=[2])               # PutUserPolicy: flagged, not explained
        self.assertIn("Only the riskiest events of each alert get their own explanation.", out)
        self.assertIn(f"part of an alert on alice (2 flagged events). Its top event, AttachUserPolicy "
                      f"({EXP_TOP['risk_score']:.2f}/10), was flagged because:", out)
        for reason in dash.plain_reasons(EXP_TOP):
            self.assertIn(text([reason]), out)

    def test_new_rows_and_alerts_appear_after_an_update(self):
        self.write_fixture()
        before, _ = self.run_console()
        new = [scored_row("f2.json:0", "carol", "CreateAccessKey", 0.95, 0.85, minute=9)]
        exp = explanation(new[0], SEQ_FEATS, [], GRAPH_FEATS)
        self.write_scores(new, append=True)
        self.write_alert(alert_json("f2.json", "carol", new, [exp]))
        after, out = self.run_console()
        self.assertEqual(len(after), len(before) + 1)
        self.assertEqual(after.iloc[-1]["Action"], "CreateAccessKey")      # appended at the bottom
        self.assertEqual(float(after.iloc[-1]["Risk"]), new[0]["risk_score"])
        self.assertIn("3Alerts raised", out.replace("\n", ""))
        self.assertIn("ALERT -- risk 9.00 out of 10", out)                 # newest explained alert
        for reason in dash.plain_reasons(exp):
            self.assertIn(text([reason]), out)


# ═══════════════════════════════════════════════════════════════════════════
# 4. end to end under real Streamlit (caching included)
# ═══════════════════════════════════════════════════════════════════════════

try:
    import streamlit
    import streamlit.elements.arrow as _st_arrow
    from streamlit.testing.v1 import AppTest
    import inspect
    # cloudsec_dashboard calls st.dataframe(width="stretch"); older Streamlit only takes an int.
    STRETCH_OK = '"stretch"' in inspect.getsource(_st_arrow)
except ImportError:  # pragma: no cover
    AppTest, STRETCH_OK = None, False


def _known_defect_unless(ok):
    return (lambda f: f) if ok else unittest.expectedFailure


@unittest.skipUnless(AppTest is not None, "streamlit.testing not available")
class TestRealStreamlitRun(DashboardSandbox):
    """The real script, on a copy placed beside a sandbox output/ + alerts/ (it resolves both
    from its own directory), so real st.cache_data / st.fragment run."""

    def setUp(self):
        super().setUp()
        self.script = os.path.join(self.tmp, "cloudsec_dashboard.py")
        shutil.copy(DASHBOARD, self.script)

    def run_app(self, at=None, shim_width=True):
        """shim_width: test-side only -- drops width="stretch" on Streamlit builds that reject it,
        so the rest of the page (and its caching) can still be checked there. The unshimmed run
        is test_runs_on_installed_streamlit."""
        at = at or AppTest.from_file(self.script, default_timeout=60)
        orig = streamlit.dataframe

        def dataframe(*a, width=None, **k):
            return orig(*a, **k) if width == "stretch" else orig(*a, width=width, **k)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {"CLOUDSEC_SMTP_USER": "", "CLOUDSEC_SMTP_PASSWORD": ""}))
            if shim_width and not STRETCH_OK:
                stack.enter_context(mock.patch.object(streamlit, "dataframe", dataframe))
            at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at, text([m.value for m in at.markdown])

    @_known_defect_unless(STRETCH_OK)
    def test_runs_on_installed_streamlit(self):
        """KNOWN DEFECT (on Streamlit builds without width="stretch", e.g. the native 1.42 here):
        cloudsec_dashboard.py line ~411 calls st.dataframe(..., width="stretch"), which Streamlit
        1.42 rejects inside console() with "TypeError: 'str' object cannot be interpreted as an
        integer" (streamlit/elements/arrow.py: proto.width = width), so the processed-log table
        and the "why" panel never render. Only the Dockerfile's pinned streamlit==1.64.0 accepts
        it; temporal-analysis/requirements-ui.txt still allows streamlit>=1.32.0. Passes (no
        expectedFailure) where the installed Streamlit supports width="stretch"."""
        self.write_fixture()
        at, _ = self.run_app(shim_width=False)
        self.assertEqual(len(at.dataframe[0].value), len(ROWS))

    def test_scores_shown_then_update_picked_up(self):
        self.write_fixture()
        at, out = self.run_app()
        table = at.dataframe[0].value
        np.testing.assert_array_equal(table["Risk"].to_numpy(), [r["risk_score"] for r in ROWS])
        np.testing.assert_array_equal(table["LSTM"].to_numpy(), [r["p_sequence"] for r in ROWS])
        for reason in dash.plain_reasons(EXP_FL):
            self.assertIn(text([reason]), out)

        new = [scored_row("f2.json:0", "carol", "CreateAccessKey", 0.95, 0.85, minute=9)]
        self.write_scores(new, append=True)
        self.write_alert(alert_json("f2.json", "carol", new, [explanation(new[0], SEQ_FEATS, [], GRAPH_FEATS)]))
        future = time.time() + 5                     # load_scores is cached on the file's mtime
        os.utime(self.scores, (future, future))
        at, out = self.run_app(at)
        table = at.dataframe[0].value
        self.assertEqual(len(table), len(ROWS) + 1)
        self.assertEqual(float(table["Risk"].iloc[-1]), new[0]["risk_score"])
        self.assertIn("3Alerts raised", out.replace("\n", ""))


if __name__ == "__main__":
    unittest.main()
