"""
conftest.py
=============
Repo-root pytest configuration.

BUG FOUND DURING AUDIT (see FINAL_REPORT.md "Bugs found and fixed"):
neo4j_graph_builder.py lives in graph_construction/, but infer.py,
incremental_updater.py, and test_incremental_updater.py all import it as a
bare top-level module (`import neo4j_graph_builder as nb`). That only
resolves if graph_construction/ happens to already be on sys.path.
incremental_updater.py and infer.py now each carry their own explicit
sys.path.insert fixing this for normal runtime use (see the comments at
their imports) — this conftest.py additionally guarantees it for pytest,
regardless of which test module pytest happens to collect first or in
what order, without requiring every test file to carry its own path-fixing
boilerplate.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_construction"))
