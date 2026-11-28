"""
Large evaluation on REAL held-out prompts drawn from the project dataset
(team cloud slice + licensed public benchmarks: deepset, jackhhao, lakera).

Only the held-out splits (test / public_test) are used -- prompts the gate never
trained on. Reports detection / leak / false-positive rates overall, per source
(so 'real public' vs 'team' is explicit), and per attack category.

Run from repo root:
    myenv\\Scripts\\python.exe manual_prompt_eval\\run_eval_dataset.py [N_per_label]
"""
import os
import sys
import json
import csv
import random
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from prompt_injection_gate.screening_service import ScreeningService, screen_prompt  # noqa: E402

N_PER_LABEL = int(sys.argv[1]) if len(sys.argv) > 1 else 18
random.seed(42)

BENIGN = {"benign"}
SKIP_AS_ATTACK = {"ambiguous"}  # reported separately, not counted as attack or benign

def load_heldout():
    rows = []
    for name, splits in [("prompt_injection_merged.jsonl", {"test", "public_test"}),
                         ("prompt_injection_cloud.jsonl", {"test"})]:
        p = os.path.join(ROOT, "datasets", "prompt-injection", name)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("split") in splits and (r.get("text") or "").strip():
                rows.append(r)
    return rows

rows = load_heldout()
# stratified sample: up to N per primary_label
by_label = collections.defaultdict(list)
for r in rows:
    by_label[r.get("primary_label", "?")].append(r)
sample = []
for lab, rs in by_label.items():
    random.shuffle(rs)
    sample.extend(rs[:N_PER_LABEL])
random.shuffle(sample)

svc = ScreeningService()
out = []
for r in sample:
    lab = r.get("primary_label", "?")
    is_benign = lab in BENIGN
    is_ambiguous = lab in SKIP_AS_ATTACK
    hist = r.get("conversation_history") or None
    if isinstance(hist, str):
        hist = [hist]
    contract = screen_prompt(r["text"], conversation_history=hist, service=svc)
    safe = bool(contract["handoff_allowed"])
    out.append({
        "primary_label": lab, "source": r.get("source", "?"),
        "is_benign": is_benign, "is_ambiguous": is_ambiguous,
        "decision": contract["decision"], "handoff_allowed": safe,
        "text": r["text"][:100].replace("\n", " "),
    })

# ---- aggregate ----
attacks = [o for o in out if not o["is_benign"] and not o["is_ambiguous"]]
benign = [o for o in out if o["is_benign"]]
ambig = [o for o in out if o["is_ambiguous"]]

def rate(sub, cond):
    n = sum(1 for o in sub if cond(o))
    return n, len(sub), (f"{100.0*n/len(sub):4.0f}%" if sub else " n/a")

att_c = rate(attacks, lambda o: not o["handoff_allowed"])   # caught
att_l = rate(attacks, lambda o: o["handoff_allowed"])        # leaked SAFE
ben_fp = rate(benign, lambda o: not o["handoff_allowed"])    # benign blocked/review

L = []
L.append("# Stage-1 Gate -- Evaluation on REAL Held-out Prompts\n")
L.append(f"Sampled {len(out)} held-out prompts (never trained on): "
         f"{len(attacks)} attacks, {len(benign)} benign, {len(ambig)} ambiguous.\n")
L.append("## Headline")
L.append(f"- Attacks caught as NOT-SAFE : {att_c[0]}/{att_c[1]}  ({att_c[2]})")
L.append(f"- Attacks leaked as SAFE     : {att_l[0]}/{att_l[1]}  ({att_l[2]})   <- safety-critical")
L.append(f"- Benign sent to review/block: {ben_fp[0]}/{ben_fp[1]}  ({ben_fp[2]})\n")

# per source (real public vs team)
L.append("## Detection by source  (shows it works on REAL public prompts, not just team data)")
L.append("| source | caught | attacks | detection |")
L.append("|---|---:|---:|---|")
bysrc = collections.defaultdict(lambda: [0, 0])
for o in attacks:
    bysrc[o["source"]][1] += 1
    if not o["handoff_allowed"]:
        bysrc[o["source"]][0] += 1
for s in sorted(bysrc):
    c, t = bysrc[s]
    L.append(f"| {s} | {c} | {t} | {100.0*c/t:4.0f}% |")

# per label
L.append("\n## Detection by attack category")
L.append("| category | caught | total |")
L.append("|---|---:|---:|")
bylab = collections.defaultdict(lambda: [0, 0])
for o in attacks:
    bylab[o["primary_label"]][1] += 1
    if not o["handoff_allowed"]:
        bylab[o["primary_label"]][0] += 1
for lab in sorted(bylab):
    c, t = bylab[lab]
    L.append(f"| {lab} | {c} | {t} |")

summary = "\n".join(L) + "\n"
open(os.path.join(HERE, "dataset_eval_summary.md"), "w", encoding="utf-8").write(summary)
with open(os.path.join(HERE, "dataset_eval_results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print(summary)
print("[wrote] dataset_eval_summary.md, dataset_eval_results.csv")
