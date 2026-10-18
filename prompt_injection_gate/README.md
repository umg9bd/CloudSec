# Stage-1 — Prompt-injection and malicious-prompt screening gate

An **isolated, additive** module. It screens a user prompt (and any untrusted
context) *before* the downstream cloud-security workflow, and forwards the prompt
to Stage 2 **only** when the decision is `SAFE`. It reads nothing from the
existing pipeline and modifies none of it.

> Scope: Stage 1 only — decide whether a prompt is safe to forward. It does **not**
> harden prompts, generate CloudTrail events, build the AWS graph, judge IAM
> authorisation, or execute anything. Those are later stages.

## Architecture (defence in depth)

```
request ─▶ input validation ─▶ bounded normalisation ─▶ rules engine ─┐
                                                                       ├─▶ decision combiner ─▶ SAFE-only handoff
                           fine-tuned classifier ──────────────────────┘        │
                                                                        explanation (blocked/review)
```

| Component | File | Role |
|---|---|---|
| Schemas / contract | `schemas.py` | request/response dataclasses; enforces the SAFE-only invariants in `__post_init__` |
| Config | `config.py` | all tunables via env vars; no secrets; thresholds come from the checkpoint |
| Input validation | `input_validation.py` | type/size/field limits → controlled non-SAFE on failure |
| Normalisation | `normalization.py` | NFKC, confusable folding, zero-width/bidi stripping, **data-only** base64/hex/rot13 decode (never executed) |
| Rules engine | `rule_engine.py` + `rules_catalogue.yaml` | deterministic structural signals, source-grounded, versioned |
| Classifier | `classifier_interface.py` | fine-tuned checkpoint behind a validated interface; unavailable ⇒ UNCERTAIN |
| Decision combiner | `decision_combiner.py` | fuses rules + model; **fails closed** |
| Explanation | `explanation.py` | deterministic, category-based, no secrets, no thresholds |
| Service | `screening_service.py` | orchestrates all of the above; builds the handoff |

### The two signals are independent
The **rules engine** applies explicit structural checks (encodings, hidden
characters, fake verdict fields, credential formats, word-class co-occurrence).
The **classifier** judges intent from learned patterns. The **combiner** is the
only place they meet, and neither alone can authorise a prompt:

- a confident malicious classification is never flipped to SAFE by empty rules;
- a high-severity / block rule is never ignored because the model says SAFE;
- a serious rule-vs-model disagreement goes to `REVIEW_REQUIRED`, never SAFE;
- any failure (validation error, classifier unavailable/error, bad catalogue)
  yields `REVIEW_REQUIRED` or `MALICIOUS` — **never** SAFE by default;
- a prompt is SAFE only when every check ran and all agree it is clean.

## Decisions and the handoff contract

`decision ∈ {SAFE, MALICIOUS, REVIEW_REQUIRED}`. The downstream-facing dict:

```json
{
  "request_id": "...",
  "decision": "SAFE",
  "handoff_allowed": true,
  "safe_prompt_for_next_stage": "Create an S3 bucket for the public website assets.",
  "threat_categories": [],
  "explanation": "No malicious prompt indicators were identified by the required screening checks.",
  "rules_version": "0.1.0",
  "classifier_version": "distilbert-mbert-pi-v1@<sha>"
}
```

Enforced in code (`schemas.ScreeningResponse.__post_init__`, and tested):
`handoff_allowed` is true **iff** `decision == SAFE`; `safe_prompt_for_next_stage`
is populated only for SAFE and is the **unaltered original** prompt; a caller
cannot manufacture a SAFE decision by putting a verdict field in the prompt (the
verdict is always computed, never parsed from input).

## Using it from Stage 2

```python
from prompt_injection_gate import screen_prompt

result = screen_prompt("Create an S3 bucket for the public website assets.")
if result["handoff_allowed"]:
    hand_to_stage2(result["safe_prompt_for_next_stage"])
else:
    show_user(result["explanation"])   # MALICIOUS or REVIEW_REQUIRED
```

Or hold a configured service and pass untrusted context / history:

```python
from prompt_injection_gate import ScreeningService
from prompt_injection_gate.schemas import ScreeningRequest, ContextItem

svc = ScreeningService()
resp = svc.screen(ScreeningRequest(
    prompt="Summarise the attached ticket.",
    untrusted_context=[ContextItem(ticket_body, source="ticket-482", trust="tool_output")],
))
audit_record = svc.audit_record(req, resp)   # no raw prompt unless PIGATE_LOG_RAW_INPUT=1
```

No existing file is changed to integrate: Stage 2 imports this package and
branches on `handoff_allowed`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PIGATE_CLASSIFIER_DIR` | `artifacts/distilbert-mbert-pi-v1` | checkpoint dir; missing ⇒ fail closed |
| `PIGATE_RULES_CATALOGUE` | `rules_catalogue.yaml` | rules file |
| `PIGATE_MAX_PROMPT_CHARS` | 20000 | prompt size limit |
| `PIGATE_MAX_CONTEXT_ITEMS` | 32 | untrusted-context count limit |
| `PIGATE_RULE_BLOCK_SEVERITY` | `high` | rule severity that blocks SAFE |
| `PIGATE_LOG_RAW_INPUT` | off | opt-in raw-prompt logging (default: sha256 only) |

## The classifier

- **Base model:** `distilbert/distilbert-base-multilingual-cased` (Apache-2.0),
  pinned by revision. Multilingual, so mixed-language prompts tokenise properly;
  runs on CPU. (`microsoft/deberta-v3-small` was tried first but its CPU attention
  made a training epoch impractically slow on this machine — documented, not hidden.)
- **Head:** `1 + K` sigmoid outputs — attack vs benign, plus one score per threat
  category (multi-label).
- **Thresholds** are **not** hard-coded. The training run derives them from the
  held-out dev slice and writes them to `artifacts/<run>/gate_model.json`
  (`malicious_threshold`, `safe_threshold`, per-category thresholds). Scores are
  **not** treated as calibrated probabilities — only these dev cut-offs are used.
- **Model status:** a checkpoint is only "fine-tuned" once `train_eval_transformer.py`
  has produced it. Until then the classifier reports `unavailable` and the gate
  fails closed. See `results/<run>.json` for the trained model's metrics.

## Train / evaluate

```bash
# 1. build the dataset (the data author's scripts; not part of this module)
python datasets/prompt-injection/merge_public.py

# 2. fine-tune + evaluate with the dev-grounded protocol
prompt_injection_gate/.venv/Scripts/python.exe -u \
    prompt_injection_gate/training/train_eval_transformer.py
```

`train_eval_transformer.py` **imports** the dataset harness
(`datasets/prompt-injection/train_eval_baseline.py`) so the split, threshold rule
and reported numbers match the baseline exactly — only the model differs. It
reports **separately** on `cloud_test` and each `public_test:*` (the
out-of-distribution / "does it generalise to human-written prompts" check), plus
per-category recall, the ambiguous routing rate, and single-prompt CPU latency,
and writes a full manifest to `results/<run>.json`.

The training environment is a module-local venv (`.venv/`, gitignored) with
`transformers`; the base repo's dependencies are untouched.

## Tests

```bash
cd prompt_injection_gate/tests && python -m pytest -q
```

Structural and fail-closed tests run with no checkpoint; model-dependent
adversarial tests are skipped until a checkpoint exists (then auto-run). Attack
examples in the adversarial suite come from the dataset's **held-out** split, so
the detector is never evaluated only on what its rules were built from. Each test
documents its intended classification.

## Deviations from the suggested layout

- `prepare_dataset.py` / `train_classifier.py` / `evaluate_classifier.py` are
  **not** duplicated here: the dataset already has a reviewed generator
  (`generate_prompt_dataset.py`), a public-merge step (`merge_public.py`) and a
  shared eval harness. `training/train_eval_transformer.py` is the train+evaluate
  path and reuses that harness rather than forking its logic. `dataset_schema.md`
  documents the schema those scripts produce.

## Source grounding

| Source | Used for |
|---|---|
| OWASP LLM01:2025 Prompt Injection | direct/indirect injection, jailbreak, obfuscation taxonomy and rule references |
| OWASP LLM06:2025 Excessive Agency | credential/exfiltration/privilege/destruction categories and defences |
| MITRE ATLAS (release 2026.09) | technique ids on rules and dataset labels (verified against source data) |
| Microsoft AI Red Teaming Agent | obfuscation strategies, XPIA/indirect and agentic test design |
| Microsoft PyRIT | adversarial-test concepts only — **not** a runtime dependency |

## Limitations

- Not a guarantee: screening reduces, not eliminates, the chance a malicious
  prompt reaches Stage 2.
- The dataset's cloud slice is synthetic and unreviewed; thin on multi-turn,
  obfuscated and indirect templates. Recall on those is weaker — see
  `results/<run>.json` per-category numbers.
- Rules are structural; a novel paraphrase with no structural tell leans entirely
  on the classifier.
- Indirect screening is offered as an interface; it is **not** wired into the
  existing pipeline's tool-execution path (that would require changing it).
- Thresholds are set for a small dev slice; re-tune as the dataset grows.
