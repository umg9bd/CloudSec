"""
test_build_graph.py
===================
graph_construction/build_graph.py is a thin CLI over
neo4j_graph_builder.build_graph(), which WIPES Neo4j (`MATCH (n) DETACH
DELETE n`) and reloads it from one structural CSV. Neo4j holds exactly one
graph at a time, so two things matter more than anything else here:

  - the graph is stamped with the CSV it came from (_GraphProvenance), in a
    form evaluate_session_level.check_graph_provenance accepts / rejects
    correctly                                                 -> TestProvenanceStamp
  - bad input (missing columns, missing file, empty or malformed rows) fails
    BEFORE the database is touched, so a typo cannot wipe a good graph
                                                              -> TestInvalidInputDoesNotWipeGraph
  - the CLI hands its argument to the builder                 -> TestCli

NO NEO4J REQUIRED: a fake `neo4j` module records every Cypher statement.
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import build_graph as cli
import neo4j_graph_builder as nb
from evaluate_session_level import check_graph_provenance
from offline_graph import load_offline

HEADER = "log_id,source_node,target_node,edge_type,label\n"
ROWS = (
    "f.csv:0,arn:aws:iam::123456789012:user/alice,arn:aws:iam::123456789012:role/dev,AssumeRole,0\n"
    "f.csv:1,arn:aws:sts::123456789012:assumed-role/dev/s,arn:aws:s3:::bucket1,GetObject,1\n"
    "f.csv:2,arn:aws:iam::123456789012:user/alice,iam.amazonaws.com,ListRoles,0\n"
)


class FakeSession:
    def __init__(self, log, fail_on=None):
        self.log, self.fail_on = log, fail_on

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        q = " ".join(query.split())
        if self.fail_on and self.fail_on in q and any(self.fail_on in c for c, _ in self.log):
            raise RuntimeError("simulated Neo4j failure mid-ingest")
        self.log.append((q, params))


class FakeNeo4j:
    """Stands in for the neo4j package; build_graph imports it lazily."""

    def __init__(self, fail_on=None):
        self.log, self.drivers = [], 0
        module = types.ModuleType("neo4j")
        module.GraphDatabase = types.SimpleNamespace(driver=self._driver)
        self.module, self.fail_on = module, fail_on

    def _driver(self, *a, **k):
        self.drivers += 1
        return types.SimpleNamespace(session=lambda: FakeSession(self.log, self.fail_on), close=lambda: None)

    def queries(self, prefix):
        return [(q, p) for q, p in self.log if q.startswith(prefix)]


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._csv_path = nb.CSV_PATH  # build_graph.py mutates this module global

    def tearDown(self):
        nb.CSV_PATH = self._csv_path
        shutil.rmtree(self.tmp, ignore_errors=True)

    def csv(self, text, name="cloudtrail_structural.csv", subdir="datasets"):
        d = os.path.join(self.tmp, subdir)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def build(self, path, fake=None):
        fake = fake or FakeNeo4j()
        nb.CSV_PATH = path
        with mock.patch.dict(sys.modules, {"neo4j": fake.module}), contextlib.redirect_stdout(io.StringIO()):
            nb.build_graph()
        return fake


class TestProvenanceStamp(Sandbox):
    def test_stamp_is_csv_basename_written_once(self):
        fake = self.build(self.csv(HEADER + ROWS, "real_dataset_test_structural.csv"))
        stamps = fake.queries("CREATE (:_GraphProvenance")
        self.assertEqual(len(stamps), 1)
        self.assertEqual(stamps[0][1], {"csv": "real_dataset_test_structural.csv"})

    def test_stamp_follows_the_wipe_and_precedes_every_node(self):
        fake = self.build(self.csv(HEADER + ROWS))
        order = [q for q, _ in fake.log]
        wipe = order.index("MATCH (n) DETACH DELETE n")
        stamp = next(i for i, q in enumerate(order) if q.startswith("CREATE (:_GraphProvenance"))
        first_node = next(i for i, q in enumerate(order) if q.startswith("MERGE"))
        self.assertTrue(all(q.startswith("CREATE CONSTRAINT") for q in order[:wipe]))
        self.assertLess(wipe, stamp)
        self.assertLess(stamp, first_node)

    def test_stamp_satisfies_the_evaluation_guard(self):
        fake = self.build(self.csv(HEADER + ROWS, "real_dataset_test_structural.csv"))
        stamped = fake.queries("CREATE (:_GraphProvenance")[0][1]["csv"]
        check_graph_provenance(stamped, "real_dataset_test.csv")
        with self.assertRaises(SystemExit):
            check_graph_provenance(stamped, "real_dataset_dev.csv")

    def test_every_row_becomes_one_typed_edge(self):
        fake = self.build(self.csv(HEADER + ROWS))
        edges = fake.queries("MATCH (src:")
        self.assertEqual(len(edges), 3)
        self.assertEqual([p["log_id"] for _, p in edges], ["f.csv:0", "f.csv:1", "f.csv:2"])
        self.assertEqual([p["relation"] for _, p in edges], ["ASSUMES", "READ", "LIST"])
        self.assertEqual([p["is_attack"] for _, p in edges], [0, 1, 0])
        # alice, dev (canonicalised: role ARN and assumed-role session are one node), bucket1, s3 service
        self.assertEqual(len(fake.queries("MERGE")), 4)

    def test_empty_source_becomes_unresolved_principal(self):
        fake = self.build(self.csv(HEADER + "f.csv:0,,arn:aws:s3:::b,GetObject,0\n"))
        self.assertTrue(fake.queries("MERGE (n:UnresolvedPrincipal"))

    def test_offline_graph_carries_its_source_name(self):
        import pandas as pd
        df = pd.read_csv(self.csv(HEADER + ROWS))
        with contextlib.redirect_stdout(io.StringIO()):
            _, meta = load_offline(df, source_name="real_dataset_test_structural.csv")
            _, unstamped = load_offline(df)
        self.assertEqual(meta["source_csv"], "real_dataset_test_structural.csv")
        self.assertIsNone(unstamped["source_csv"])

    @unittest.expectedFailure
    def test_failed_build_leaves_no_provenance_stamp(self):
        """KNOWN DEFECT: build_graph() writes the _GraphProvenance stamp right
        after the wipe, BEFORE ingesting nodes/edges, and there is no
        completion marker. If ingestion dies part-way (Neo4j drop, bad row),
        Neo4j holds a PARTIAL graph stamped as built from the CSV, and
        check_graph_provenance (the guard the stamp exists for) accepts it."""
        fake = FakeNeo4j(fail_on="MATCH (src:")  # second edge create fails
        with self.assertRaises(RuntimeError):
            self.build(self.csv(HEADER + ROWS), fake)
        self.assertEqual(fake.queries("CREATE (:_GraphProvenance"), [])


class TestInvalidInputDoesNotWipeGraph(Sandbox):
    def assert_rejected_before_neo4j(self, path, exc=Exception):
        fake = FakeNeo4j()
        with self.assertRaises(exc) as ctx:
            self.build(path, fake)
        self.assertEqual(fake.drivers, 0, "Neo4j was contacted (and would have been wiped)")
        self.assertEqual(fake.log, [])
        return ctx.exception

    def test_missing_required_column(self):
        text = "log_id,source_node,target_node,label\nf:0,u,t,0\n"
        err = self.assert_rejected_before_neo4j(self.csv(text), ValueError)
        self.assertIn("edge_type", str(err))

    def test_missing_file(self):
        self.assert_rejected_before_neo4j(os.path.join(self.tmp, "nope.csv"), FileNotFoundError)

    def test_header_only_csv(self):
        self.assert_rejected_before_neo4j(self.csv(HEADER))

    def test_missing_label_value(self):
        self.assert_rejected_before_neo4j(self.csv(HEADER + "f:0,arn:aws:iam::1:user/u,arn:aws:s3:::b,GetObject,\n"))

    def test_duplicate_log_ids(self):
        dup = HEADER + ("f:0,arn:aws:iam::1:user/u,arn:aws:s3:::b,GetObject,0\n"
                        "f:0,arn:aws:iam::1:user/u,arn:aws:s3:::c,PutObject,1\n")
        self.assert_rejected_before_neo4j(self.csv(dup))

    @unittest.expectedFailure
    def test_header_only_csv_reports_empty_input(self):
        """KNOWN DEFECT: a structural CSV with the right header but no rows is
        rejected (safely, before Neo4j), but with pandas'
        KeyError "None of ['log_id'] are in the columns" from
        compute_graph's set_index -- telling the user a column that IS present
        is missing. build_graph() validates columns with a ValueError but never
        checks for zero rows."""
        self.assert_rejected_before_neo4j(self.csv(HEADER), ValueError)


class TestCli(Sandbox):
    def test_argument_is_passed_to_the_builder(self):
        path = self.csv(HEADER + ROWS)
        seen = []
        with mock.patch.object(sys, "argv", ["build_graph.py", path]), \
                mock.patch.object(nb, "build_graph", lambda: seen.append(nb.CSV_PATH)):
            cli.main()
        self.assertEqual(seen, [path])

    def test_cli_end_to_end_stamps_provenance(self):
        path = self.csv(HEADER + ROWS, "real_dataset_dev_structural.csv")
        fake = FakeNeo4j()
        with mock.patch.object(sys, "argv", ["build_graph.py", path]), \
                mock.patch.dict(sys.modules, {"neo4j": fake.module}), contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        self.assertEqual(fake.queries("CREATE (:_GraphProvenance")[0][1]["csv"], "real_dataset_dev_structural.csv")

    def test_missing_argument_is_a_usage_error(self):
        with mock.patch.object(sys, "argv", ["build_graph.py"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
