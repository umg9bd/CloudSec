"""
Reports campaign-diversity statistics for the paper's dataset section
(review point 5). Answers "how many INDEPENDENT campaigns produced the malicious
events, across how many families/topologies, with what multi-principal/multi-hop
structure" -- so "11,520 events" becomes a scientifically meaningful claim about
variation, not just a row count.

Run:
    python campaign_stats.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main():
    ann = pd.read_csv(HERE / "synthetic_campaign_annotations.csv", low_memory=False)
    raw = pd.read_csv(HERE / "synthetic_cloudtrail.csv", low_memory=False)

    ann["campaign_id"] = ann["campaign_id"].astype("string").fillna("")
    camps = ann[ann.campaign_id != ""]

    raw["campaign_id"] = raw["campaign_id"].astype("string").fillna("")
    chain = raw[(raw.campaign_id != "") & (raw.stage_index >= 0)]

    n_campaigns = camps.campaign_id.nunique()
    n_families = camps.chain_name.nunique()
    per_family = camps.groupby("chain_name").campaign_id.nunique().sort_values(ascending=False)

    hops = chain.groupby("campaign_id").hop_id.max() + 1          # 0-based -> count
    stages = chain.groupby("campaign_id").stage_index.max() + 1
    principals = chain.groupby("campaign_id").principal_arn.nunique()
    accounts = chain.assign(acct=chain.principal_arn.str.extract(r"::(\d+):")[0]) \
        .groupby("campaign_id").acct.nunique()

    print("=" * 64)
    print("CAMPAIGN DIVERSITY  (synthetic attack corpus)")
    print("=" * 64)
    print(f"independent campaigns          : {n_campaigns}")
    print(f"campaign families (topologies) : {n_families}")
    print(f"labelled chain-stage events    : {len(chain):,}")
    print()
    print("campaigns per family:")
    for fam, n in per_family.items():
        print(f"    {n:4d}  {fam}")
    print()
    print(f"hops per campaign        : min {hops.min()}  max {hops.max()}  mean {hops.mean():.2f}")
    print(f"stages per campaign      : min {stages.min()}  max {stages.max()}  mean {stages.mean():.2f}")
    print(f"principals per campaign  : min {principals.min()}  max {principals.max()}  mean {principals.mean():.2f}")
    print(f"multi-principal campaigns (>1 principal) : {int((principals > 1).sum())} / {len(principals)} "
          f"({(principals > 1).mean():.0%})")
    print(f"multi-hop campaigns (>=2 hops)           : {int((hops >= 2).sum())} / {len(hops)} "
          f"({(hops >= 2).mean():.0%})")
    print(f"cross-account campaigns (>1 account)     : {int((accounts > 1).sum())} / {len(accounts)}")
    print()
    print("For the paper: \"We generated {} independent attack campaigns across {} "
          "campaign families/topologies, {:.0%} of them multi-principal and {:.0%} "
          "multi-hop, varying principals, privilege transitions, resources and "
          "temporal spacing.\"".format(n_campaigns, n_families,
                                       (principals > 1).mean(), (hops >= 2).mean()))


if __name__ == "__main__":
    main()
