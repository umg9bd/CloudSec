"""
Thin CLI wrapper around neo4j_graph_builder.py's build_graph(), which
otherwise only reads a hardcoded CSV_PATH module constant. Lets any
structural CSV be loaded into Neo4j with one clean command instead of an
inline multi-line python -c snippet.

Usage (from the repo root):
    python graph_construction/build_graph.py datasets/privilege-escalation/cloudtrail_structural.csv
    python graph_construction/build_graph.py datasets/privilege-escalation/real_dataset_test_structural.csv
    python graph_construction/build_graph.py datasets/privilege-escalation/real_dataset_dev_structural.csv

WARNING: build_graph() does `MATCH (n) DETACH DELETE n` before loading --
this replaces whatever graph is currently in Neo4j, it does not merge.
"""
import argparse

import neo4j_graph_builder as nb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", help="Path to a structural CSV (log_id, source_node, target_node, edge_type, label)")
    args = p.parse_args()
    nb.CSV_PATH = args.csv_path
    nb.build_graph()


if __name__ == "__main__":
    main()
