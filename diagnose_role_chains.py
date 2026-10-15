"""
Measures whether the privilege-propagation chain

    User -> ASSUMES -> Role -> ACTION -> Resource

actually materialises in each dataset, and how hop_count / privilege_gain
can therefore behave.

Run it before accepting any claim that the chain "returns 0 results". The
answer depends entirely on WHICH LEVEL you measure at:

  * raw CSV strings   -- an AssumeRole event names the IAM role ARN
                         (arn:aws:iam::A:role/R) while the same role, when it
                         later acts, appears as an STS session ARN
                         (arn:aws:sts::A:assumed-role/R/session). Compared as
                         strings these NEVER match, so a naive check reports 0.
  * graph node keys   -- what the builder actually constructs. Both forms
                         normalise to Role(R), so the chain does exist.

Measuring at the wrong level is the single easiest way to "discover" a bug
that isn't there, which is why this script reports both.

    python diagnose_role_chains.py
"""

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, "graph_construction")
from neo4j_graph_builder import parse_principal, parse_target  # noqa: E402

import privilege_features as pf  # noqa: E402


def analyse(path: str) -> None:
    df = pd.read_csv(path, low_memory=False)
    is_assume = df.edge_type.astype(str).str.upper().str.contains("ASSUME", na=False).values

    # --- level 1: raw strings, the naive check -------------------------------
    raw_assumed = set(df.target_node[is_assume].astype(str))
    raw_actors = set(df.source_node[~is_assume].astype(str))

    # --- level 2: graph node keys, replicating neo4j_graph_builder exactly ----
    pinfos = df["source_node"].apply(parse_principal)
    tinfos = df["target_node"].apply(parse_target)
    src = [pf.node_key_for_principal(a, i.principal_type, i.name)
           for a, i in zip(df["source_node"], pinfos)]
    known_roles = {i.name for i in pinfos
                   if i.principal_type in ("AssumedRole", "AWSServiceLinkedRole")}
    known_users = {i.name for i in pinfos if i.principal_type == "IAMUser"}
    dst = [pf.node_key_for_target(t.value, t.resource_type, t.service,
                                  known_roles, known_users) for t in tinfos]

    assumed_nodes = [d for d, m in zip(dst, is_assume) if m]
    assumed_roles = {d for d in assumed_nodes if d.label == "Role"}
    # roles that got demoted to Resource because nothing ever saw them ACT
    orphaned = {d for d in assumed_nodes if d.label != "Role"}
    actors = {s for s, m in zip(src, is_assume) if not m}

    hubs = assumed_roles & actors
    hop2_edges = sum(1 for s, m in zip(src, is_assume) if not m and s in hubs)

    print(f"\n=== {os.path.basename(path)}  ({len(df):,} edges) ===")
    print(f"  AssumeRole-type events                  : {int(is_assume.sum()):,}")
    print(f"  raw-string overlap (naive, MISLEADING)  : {len(raw_assumed & raw_actors)}")
    print(f"  assumed targets resolved to Role nodes  : {len(assumed_roles):,}")
    print(f"  assumed targets left as non-Role        : {len(orphaned):,}  <- never act, so never linked")
    print(f"  TRUE 2-hop hub roles                    : {len(hubs):,}")
    print(f"  action edges at hop_count == 2          : {hop2_edges:,} "
          f"({hop2_edges / len(df):.1%} of the dataset)")
    if hubs:
        print(f"  hub roles: {sorted(k.key for k in hubs)[:6]}")


def main() -> None:
    paths = sorted(glob.glob("datasets/privilege-escalation/*structural*.csv"))
    if not paths:
        sys.exit("No structural CSVs found — run from the repo root.")
    for p in paths:
        analyse(p)
    print("\nhop_count == 2 requires the source node to be a Role that was itself")
    print("the target of an ASSUMES edge (privilege_features.hop_count).")
    print("privilege_gain is defined only on those hop-2 edges.")


if __name__ == "__main__":
    main()
