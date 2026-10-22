# Stage-1 screening dataset — schema and provenance

The Stage-1 classifier trains on the project's prompt-injection dataset, which
lives **outside this module** at `datasets/prompt-injection/` (owned by the data
author; this module does not modify it). This file documents the schema the
training/eval code in `prompt_injection_gate/training/` expects, and how the set
is assembled.

## How the dataset is built

1. `datasets/prompt-injection/generate_prompt_dataset.py` — the **cloud-specific
   slice**: team-written AWS attacks (credential theft, privilege escalation,
   disabling logging, exfiltration, data destruction, system-prompt extraction,
   indirect injection) and their **hard-safe** counterparts (legitimate key
   rotation, IAM/S3 admin). Deterministic, grouped by `template_id`.
2. `datasets/prompt-injection/merge_public.py` — downloads and maps the licensed
   public sets into the same schema, keeping their own splits separate:
   - `deepset/prompt-injections` (Apache-2.0),
   - `jackhhao/jailbreak-classification` (Apache-2.0),
   - `Lakera/gandalf_ignore_instructions` (MIT).
   Output: `prompt_injection_merged.jsonl` (gitignored; regenerable).

The merged file is what training reads. If only the cloud slice is present,
training falls back to it and says so.

## Row schema (one JSON object per line)

| field | type | meaning |
|---|---|---|
| `id` | str | stable hash of `text` |
| `template_id` | str | generator template; the split is grouped on this so no template crosses train/test |
| `text` | str | the prompt or content |
| `channel` | str | `user_prompt` \| `retrieved_content` \| `tool_output` — the untrusted-channel signal |
| `labels` | list[str] | multi-label threat categories (may be several) |
| `primary_label` | str | the headline label; `benign` / `ambiguous` are the non-attack classes |
| `category` | str | fine-grained category or `hardsafe_*` |
| `atlas_ids` | list[str] | MITRE ATLAS / ATT&CK technique ids |
| `atlas_tactic` | str | ATLAS tactic id where applicable |
| `is_obfuscated` | bool | encoded/obfuscated surface form |
| `is_indirect` | bool | arrives via a non-user channel |
| `multi_turn` | bool | `conversation_history` is populated |
| `conversation_history` | list[str] | prior turns, oldest first (multi-turn rows) |
| `rationale` | str | why this label (for human review) |
| `cloud_specific` | bool | from the cloud slice |
| `source` | str | `team_synthetic` or the public dataset id |
| `license` | str | per-source license |
| `dataset_version` | str | e.g. `0.2.0` |
| `reviewed_by` | str | human reviewer initials — **empty until reviewed** |
| `split` | str | `train` / `test` (cloud, grouped) or `public_train` / `public_test` |

## Label → model output mapping

The model has `1 + K` sigmoid outputs: output 0 is **attack vs benign** (the
gate's main signal); outputs 1..K are the `K` categories in
`train_eval_transformer.CATEGORIES`. `ambiguous` rows are **not trained on** —
they are the REVIEW_REQUIRED outcome and are reported separately.

Category outputs are grounded in OWASP LLM01/LLM06 and MITRE ATLAS:

| model category | OWASP | ATLAS id |
|---|---|---|
| prompt_injection_direct | LLM01 (direct) | AML.T0051.000 |
| prompt_injection_indirect | LLM01 (indirect) | AML.T0051.001 |
| jailbreak | LLM01 (jailbreak) | AML.T0054 |
| system_prompt_extraction | LLM01 / disclosure | AML.T0056 |
| credential_harvesting | LLM06 | AML.T0098 / AML.T0055 |
| exfiltration | LLM06 | AML.T0086 |
| privilege_escalation | LLM06 (excessive agency) | AML.T0053 |
| defense_evasion_logging | LLM06 | ATT&CK T1562.008 (tactic AML.TA0007) |
| data_destruction | LLM06 | AML.T0101 |
| obfuscation | LLM01 (evasion) | AML.T0068 |

## Honest limitations (carry into any write-up)

- The cloud slice is **synthetic and unreviewed** (`reviewed_by` empty). A human
  review pass is required before the labels are trusted.
- Diversity is limited by template count; count templates, not rows.
- Thin categories: multi-turn, obfuscation, indirect. Grow by **templates**.
- Do **not** train on the cloud slice alone — it does not generalise; the public
  merge is what provides human-written signal (measured separately as
  `public_test:*`).
