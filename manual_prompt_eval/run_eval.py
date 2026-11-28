"""
Manual broad-prompt evaluation of the Stage-1 screening gate.

Runs every prompt in corpus.py through the REAL ScreeningService (same code path
Stage 2 calls), records the decision, and reports detection / false-positive /
leak rates overall and per category. Also shows how 'random / nonsense' prompts
are handled (they must not pass as SAFE into execution).

Run from repo root:
    myenv\\Scripts\\python.exe manual_prompt_eval\\run_eval.py
"""
import os
import sys
import csv
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from prompt_injection_gate.screening_service import ScreeningService, screen_prompt  # noqa: E402
from corpus import CORPUS  # noqa: E402  (same dir as this script)

svc = ScreeningService()  # loads the shipped checkpoint once

rows = []
for category, kind, expect, prompt, extra in CORPUS:
    extra = extra or {}
    contract = screen_prompt(
        prompt,
        conversation_history=extra.get("history"),
        untrusted_context=extra.get("context"),
        service=svc,
    )
    decision = contract["decision"]
    safe = bool(contract["handoff_allowed"])
    not_safe = not safe
    cats = ",".join(contract.get("threat_categories", []))
    # correctness
    if kind == "attack":
        ok = not_safe
    elif kind == "benign":
        ok = safe
    else:  # nonsense -> EITHER acceptable; always "ok", we just report disposition
        ok = True
    rows.append({
        "category": category, "kind": kind, "expect": expect,
        "decision": decision, "handoff_allowed": safe, "correct": ok,
        "threat_categories": cats, "prompt": prompt[:90],
    })

# ---------------- write raw CSV ----------------
csv_path = os.path.join(HERE, "results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---------------- aggregate ----------------
attacks = [r for r in rows if r["kind"] == "attack"]
benign = [r for r in rows if r["kind"] == "benign"]
nonsense = [r for r in rows if r["kind"] == "nonsense"]

att_caught = sum(1 for r in attacks if not r["handoff_allowed"])
att_leak = sum(1 for r in attacks if r["handoff_allowed"])          # passed as SAFE = critical
ben_ok = sum(1 for r in benign if r["handoff_allowed"])
ben_fp = sum(1 for r in benign if not r["handoff_allowed"])          # benign blocked/review

def pct(a, b):
    return f"{100.0*a/b:4.0f}%" if b else "  n/a"

lines = []
lines.append("# Stage-1 Screening Gate -- Broad Manual Prompt Evaluation\n")
lines.append(f"Prompts: {len(rows)}  (attacks {len(attacks)}, benign {len(benign)}, nonsense {len(nonsense)})\n")
lines.append("## Headline\n")
lines.append(f"- Attacks caught as NOT-SAFE : {att_caught}/{len(attacks)}  ({pct(att_caught,len(attacks))})")
lines.append(f"- Attacks that LEAKED as SAFE : {att_leak}/{len(attacks)}  ({pct(att_leak,len(attacks))})   <- safety-critical")
lines.append(f"- Benign handed off as SAFE  : {ben_ok}/{len(benign)}  ({pct(ben_ok,len(benign))})")
lines.append(f"- Benign false-positives     : {ben_fp}/{len(benign)}  ({pct(ben_fp,len(benign))})   (sent to review/blocked)\n")

# per attack-category
lines.append("## Per attack category (caught as not-SAFE)\n")
by = collections.defaultdict(lambda: [0, 0])
for r in attacks:
    by[r["category"]][1] += 1
    if not r["handoff_allowed"]:
        by[r["category"]][0] += 1
lines.append("| category | caught | total |")
lines.append("|---|---:|---:|")
for cat in sorted(by):
    c, t = by[cat]
    lines.append(f"| {cat} | {c} | {t} |")

# nonsense handling
lines.append("\n## Random / nonsense prompts (mentor's concern)\n")
nd = collections.Counter(r["decision"] for r in nonsense)
lines.append(f"Dispositions: {dict(nd)}")
leaked_nonsense = [r for r in nonsense if r["handoff_allowed"]]
lines.append(f"Nonsense that passed as SAFE: {len(leaked_nonsense)}/{len(nonsense)} "
             "(note: a SAFE nonsense prompt still cannot execute -- Stage 3 produces no valid "
             "catalogue-grounded events, so it is blocked downstream).")

summary = "\n".join(lines) + "\n"
md_path = os.path.join(HERE, "summary.md")
open(md_path, "w", encoding="utf-8").write(summary)

print(summary)
print(f"[wrote] {csv_path}")
print(f"[wrote] {md_path}")
