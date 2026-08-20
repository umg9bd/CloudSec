# Demo Guide

A runnable script for showing this project to your mentor, with exact expected
output at every step so nothing on screen is a surprise. Commands are
PowerShell (your primary shell) run from the repo root, `C:\CloudSec\CloudSec`.

**Read this first:** the headline result is now a real, verified win —
session-level F1=0.823 on real held-out test data, beating the rule-based
baseline's F1=0.732, checked with proper dev/test discipline, threshold
stability, and a bootstrap confidence interval. Getting there required
diagnosing and fixing a genuine synthetic-to-real generalization gap first —
that diagnostic journey is *part of the pitch*, not something to hide. One
honest caveat to state alongside the win: it's a session-level effect (the
model correctly flags at least one edge per attack session); edge-level
accuracy on individual actions is still weak (AUC≈0.54). See "What to say"
at the bottom for exactly how to frame this.

---

## 0. Before the meeting — do this ahead of time, not live

The two slowest, most failure-prone steps are starting Neo4j and rebuilding a
graph from a 30,000-row CSV (several minutes, and depends on Docker actually
being up). Do these **before** your mentor sits down, so a live demo is never
blocked on a multi-minute wait or a Docker hiccup.

### 0.1 Start Docker Desktop and the Neo4j container

```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
# wait ~30-60s for Docker's daemon to actually come up, then:
docker start neo4j-local
```

**Expect:** `docker start neo4j-local` prints `neo4j-local` on success. If
`docker ps` errors with *"failed to connect to the docker API"*, Docker
Desktop itself isn't running yet — this happened twice during development
because the container had been left stopped since a previous session. Just
wait longer and retry; there's no way to speed up Docker's own startup.

### 0.2 Confirm Neo4j is actually accepting connections

The container reporting "Up" does **not** mean Neo4j's bolt listener is ready
yet — there's a real gap between the two. Activate your venv first:

```powershell
C:\CloudSec\myenv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
python -c "from neo4j import GraphDatabase; d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','test1234')); d.verify_connectivity(); d.close(); print('ready')"
```

**Expect:** prints `ready`. If it raises `ServiceUnavailable`, wait ~10-20s
and retry — this is normal right after `docker start`.

### 0.3 Set environment variables for every command below

Every command in this guide assumes these are set in your current shell
session (PowerShell doesn't persist them between windows):

```powershell
$env:PYTHONPATH = "C:\CloudSec\CloudSec;C:\CloudSec\CloudSec\graph_construction"
$env:PYTHONIOENCODING = "utf-8"
cd C:\CloudSec\CloudSec
```

`PYTHONIOENCODING` matters specifically on Windows — without it, print
statements containing ✅ or other Unicode characters crash the script with a
`cp1252` codec error on an otherwise-successful run.

### 0.4 Rebuild whichever graph Part 4 will use

Pick **one** — Neo4j holds one graph at a time; loading a new one replaces
the last. For the honest-result demo (Part 4), load the real test graph:

```powershell
python build_graph.py datasets/privilege-escalation/real_dataset_test_structural.csv
```

**Expect** (takes ~5-8 minutes for this file's ~30,000 rows):
```
Loading datasets/privilege-escalation/real_dataset_test_structural.csv ...
Action access-level resolver: policy_sentry (445 AWS services, offline)
  29,948 rows | 4263 labelled attack | 1196 nodes | 29948 edges
Creating constraints ...
Clearing existing graph ...
Ingesting nodes ...
Ingesting 29,948 typed edges ...
  ... 500 rows processed
  [... many more progress lines ...]
✅ Privilege Propagation Graph built: 1196 nodes, 29948 typed edges.
```

If instead you want to lead with the **live graph visualization** (Part 2)
or **live training** (Part 3), load the synthetic graph instead:

```powershell
python build_graph.py datasets/privilege-escalation/cloudtrail_structural.csv
```

**Expect** (faster, ~2-3 minutes, ~9,900 rows):
```
  9,860 rows | 520 labelled attack | 7480 nodes | 9860 edges
✅ Privilege Propagation Graph built: 7480 nodes, 9860 typed edges.
```

---

## 1. Recommended live flow (~10-15 minutes)

| # | What | Live or pre-run? | Time |
|---|---|---|---|
| 2 | Graph visualization in Neo4j Browser | Live | 1 min |
| 3 | Live GNN training on synthetic data | Live | ~2 min |
| 4 | Real-data evaluation (the honest result) | Live (graph pre-built in 0.4) | ~1 min |
| 5 | Session-level result + the length-confound finding | Live | ~1 min |

---

## 2. Graph visualization (needs the **synthetic** graph loaded — see 0.4)

Open **http://localhost:7474** in a browser, log in with `neo4j` / `test1234`.
Run this query:

```cypher
MATCH (u)-[a:ASSUMES]->(r:Role)-[x]->(res)
WHERE x.is_attack = 1
RETURN u, a, r, x, res LIMIT 10
```

**Expect:** a small graph showing a principal assuming a role, then that role
acting on a resource, with the edges flagged as attack. This is a good
opening visual, and it's directly the concrete evidence for one of this
session's two verified bug fixes — before the fix, this exact pattern did not
exist anywhere in the training data (zero `Role`-sourced attack edges); after
it, there are 40 `READ`, 20 `WRITE`, 20 `PERMISSIONS_MANAGEMENT`, and 40
`ASSUMES→Role` attack edges. Worth saying out loud: "this is the canonical
AWS privilege-escalation pattern — assume a role, then abuse it — and until
this fix, the model had never once seen a labeled example of it."

---

## 3. Live training (needs the **synthetic** graph loaded — see 0.4)

```powershell
python train.py --model sage
```

**Expect** (~2 minutes total, early stopping around epoch 55-70):
```
INFO | Epoch   5/100 | loss=... | val_F1=... | val_AUC=... | ...s
[... F1 climbs steadily ...]
INFO | Early stopping at epoch ~55-70 (patience=15).
Final TEST evaluation -- GRAPHSAGE
INFO | Acc=0.99  P=0.9-0.97  R=0.88-0.96  F1=0.92-0.94  AUC=0.999  (n=1480)
```
Exact numbers vary slightly run to run (random init/split), but should land
in this range every time — that's expected and fine to say out loud if asked.

**Say when this finishes:** "This F1=0.93 is on synthetic held-out data —
data generated by the same process as training data. It proves the pipeline
works end to end. It is *not* the real-world number — that's next."

If you ran this live, you now need to re-wrap the checkpoint before Part 4,
since training overwrote it:

```powershell
python infer.py --wrap-checkpoint checkpoints/best_GraphSAGE.pt --wrapped-output checkpoints/best_GraphSAGE_wrapped.pt
```

**Expect:** ends with `Wrapped checkpoint saved to checkpoints/best_GraphSAGE_wrapped.pt`.
Then reload the real test graph (0.4's second command) before Part 4 — Neo4j
now holds the synthetic graph from this step, not the real one.

**If you don't need to demo live training**, skip this whole section and go
straight to Part 4 using the already-committed checkpoints — no retraining
needed.

---

## 4. Real-data evaluation — edge-level (needs the **real test** graph loaded — see 0.4)

```powershell
python evaluate_on_real.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage
```

**Expect** (~1 minute):
```
Model trained on 16 edge-type triples; real graph has 35 triples.
EXCLUDING 4205/30189 real edges (13.9%) outside the trained schema:
    [... list of excluded triples ...]

SUMMARY: P=0.207  R=0.037  F1=0.062  AUC=0.537
Compare against GuardDuty-style rule baseline: F1=0.732 [95% CI: 0.672, 0.790]
```

**Say:** "Edge-level, the model is still weak — AUC=0.54, barely above
random. It is not accurately classifying most individual actions. That's
not the whole story, though — session level is next, and it's where this
actually works. Don't skip past this number to get there; it's an honest
part of the result."

---

## 5. Session-level result — the verified win

```powershell
python evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage --raw-csv datasets/privilege-escalation/real_dataset_test.csv --threshold 0.35
```

**Expect:**
```
Sessions: 238 total | 236 have >=1 in-schema edge | 2 have ZERO in-schema edges (predicted benign by default)

SESSION-LEVEL @ threshold=0.35: P=0.859  R=0.790  F1=0.823
Compare directly against GuardDuty-style rule baseline: F1=0.732 [95% CI: 0.672, 0.790]
```

**Say:** "Aggregated to session level — the same unit the rule baseline uses
— F1=0.823, beating the baseline's 0.732. This threshold (0.35) was selected
entirely on a separate dev set, then checked exactly once here, so this isn't
picking the best-looking number after the fact. We also bootstrapped a
confidence interval — [0.766, 0.875] — and its lower bound is still above the
baseline's point estimate. And we checked it isn't just re-deriving session
length: the correlation between the model's score and session length is
0.55, not the ~1.0 you'd see from a disguised length-counter, and the
score separates attack from benign far more sharply than length alone does."

**The honest framing that ties 4 and 5 together, worth saying explicitly:**
"The model isn't accurately scoring every individual action — it's correctly
flagging at least one action per attack session while staying quiet on
benign sessions. That's a real, legitimate detection mechanism — it's how
most SOC alerting actually works, you don't need every log line right, you
need the aggregate alert to fire correctly — but it's a different, more
precise claim than 'the model understands each action,' and stating it
precisely is what makes this credible under questioning."

---

## 6. Optional backups, if there's time or your mentor asks

**Rule-based baselines** (already-computed numbers, safe to just describe
rather than re-run live — several seconds either way):
```powershell
python datasets/privilege-escalation/evaluate_baselines.py
```
Reports: Minimal SIEM F1=0.504, GuardDuty-style F1=0.732, Post-incident
(unfair upper bound) F1=0.913.

**GAT instead of GraphSAGE** — same commands, substitute the checkpoint and
`--model gat`. Worth knowing before you're asked: the current GAT checkpoint
predates the fix behind the F1=0.823 result and has not yet been re-trained
and re-checked with it — if asked, say GAT is still being re-verified rather
than citing its old (worse) numbers as current.

**Threshold sweep on dev data, showing how 0.35 was selected** (needs the dev
graph loaded via
`python build_graph.py datasets/privilege-escalation/real_dataset_dev_structural.csv`):
```powershell
python evaluate_session_level.py --checkpoint checkpoints/best_GraphSAGE_wrapped.pt --model sage --raw-csv datasets/privilege-escalation/real_dataset_dev.csv --sweep
```
Shows a broad F1=0.86-0.90 plateau across thresholds 0.25-0.45 (best
F1=0.901 at 0.35) — good material if asked "how did you pick the threshold,"
since it demonstrates the choice wasn't fit to the test set.

---

## 7. If something breaks live

- **Docker/Neo4j connection errors** (`ServiceUnavailable`, `ConnectionRefusedError`):
  Docker Desktop or the container isn't up. Run 0.1-0.2 again. This is the
  single most likely failure mode — it happened twice during development,
  always because Docker had been closed since the last session.
- **`AttributeError: 'NodeStorage' object has no attribute 'x'`**: the loaded
  graph is missing an entire node type the model was trained on (e.g. no
  `Policy` nodes in a small split). Already fixed in `data_loader.py` — if
  you see this, the fix didn't make it into your checkout; check
  `git status`.
- **Unicode/`cp1252` crash on an otherwise-finished run**: `$env:PYTHONIOENCODING`
  wasn't set in this shell session — see 0.3.
- **Import errors (`ModuleNotFoundError: neo4j_graph_builder` etc.)**:
  `$env:PYTHONPATH` wasn't set in this shell session — see 0.3.
- **A rebuilt graph doesn't match what a command expects**: remember Neo4j
  holds exactly one graph at a time. If Part 4 gives strange numbers, check
  you last ran `build_graph.py` on `real_dataset_test_structural.csv`, not
  something else.

---

## What to say, overall

Lead with the result, then the journey that earned it: **"We built a
complete, real, end-to-end pipeline — real red-team attack data collected
across 4 independent AWS accounts, a validated synthetic data generator, a
heterogeneous GNN trained on privilege-propagation graphs. Evaluated
honestly against real attack data, session-level F1=0.823, beating an
11-rule GuardDuty-style baseline's F1=0.732 — checked with a dev-only
selected threshold, a bootstrap confidence interval, and a control for the
one confound we found along the way. Getting there required real diagnostic
work: two structural bugs found and fixed, one plausible fix tested and
correctly rejected when it made things worse, before finding the one that
actually worked. The honest caveat: this is a session-level result — the
model flags at least one action per attack session correctly, individual
action-level accuracy is still weak, and that's stated precisely rather than
glossed over."**

That framing is stronger than either "it works" alone or "here's an honest
failure" alone — it's both, in the right order, and it's true.
