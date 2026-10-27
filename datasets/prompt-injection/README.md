# Stage-1 Prompt-Injection Dataset (cloud-specific slice) — v0.4.0

The team-generated, **cloud-specific** portion of the Stage-1 prompt-injection
dataset — the gap public sets don't cover (AWS credential theft, privilege
escalation, disabling logging, exfiltration, data destruction, system-prompt
extraction, indirect injection in tool/retrieved content), plus the **hard-safe**
counterparts that keep a security gate from blocking legitimate cloud work.

It is **not the whole Stage-1 dataset** and **must not be trained on alone**
(see "Honest numbers"). Merge it with the public sets via `merge_public.py`.

## v0.4.0 — terminal-punctuation robustness
Adds 350 benign rows (`dataset_version: "0.4.0"`): each benign cloud base in both its
no-period and with-period form, because v2 flipped SAFE/REVIEW on a trailing period.
Same base → same `template_id` → same split; every earlier row is unchanged.
Total 2,283 rows.

## v0.3.0 — benign cloud-ops negatives
Adds 675 benign rows / 154 templates (all `dataset_version: "0.3.0"`): ~117 hand-written
cloud-ops and security-hardening request templates (S3, IAM, KMS, logging, VPC, DBs,
secrets, IaC, non-English, security-worded rewrites) plus the Stage-2 catalogue's
`required_prompt_clause` / safer-alternative texts, so the gate stops flagging the
safer rewrites Stage 2 produces. The v0.2.0 rows are byte-identical; the new templates
have their own seeded 80/20 template split. Total 1,933 rows / 418 templates. These
are synthetic (one author) -- real team-written benign prompts are still the gap.

## v0.2.0 — Stage-1 review fixes applied
This version addresses the Stage-1 review in full:
- **Template-grouped, reproducible split** — `template_id` per row; no template
  crosses train/test (enforced by an assert); fixed per-label seed (no `hash()`).
  Output is **byte-identical across runs**.
- **Multi-label** (`labels` is a list) with **corrected ATLAS mapping**:
  privilege escalation → `AML.T0053` (and "ignore above + grant admin" also
  carries `AML.T0051.000`); disabling logging → **ATT&CK `T1562.008`** (ATLAS
  tactic `AML.TA0007`); `credential_harvesting` / `exfiltration` /
  `system_prompt_extraction` are separate; **obfuscation is a flag**
  (`is_obfuscated`) on top of the decoded intent.
- **More templates** for the thin categories (264 templates, was ~168), varied
  obfuscation openings, **Unicode / mixed-language** attacks, and **general
  (non-AWS) benign** prompts so "safe" isn't only short AWS admin.
- New fields: `template_id, conversation_history, rationale, dataset_version,
  license, is_obfuscated, is_indirect`.

## Files
| File | What |
|---|---|
| `generate_prompt_dataset.py` | generator (v0.4.0). Deterministic; re-run to regenerate. |
| `prompt_injection_cloud.jsonl` / `.csv` | the cloud slice (2,283 rows / 418 templates) |
| `merge_public.py` | maps + merges the licensed public sets, keeping their splits apart |

Regenerate: `python datasets/prompt-injection/generate_prompt_dataset.py`
Merge public: `pip install datasets && python datasets/prompt-injection/merge_public.py`

## Schema
`id, template_id, text, channel, labels, primary_label, category, atlas_ids,
atlas_tactic, is_obfuscated, is_indirect, multi_turn, conversation_history,
rationale, cloud_specific, source, license, dataset_version, reviewed_by, split`
- **labels**: multi-label list (CSV: pipe-joined). e.g. `["prompt_injection_direct","privilege_escalation"]`.
- **channel**: `user_prompt | retrieved_content | tool_output` (indirect-injection signal).
- **template_id**: groups all renders of one base template — the split holds out whole templates.
- **conversation_history**: list of turns for multi-turn rows (CSV: ` ||| `-joined); `text` is the joined view.
- **reviewed_by**: empty on cloud rows — **needs a human reviewer before training**.
- **split**: `train`/`test` (cloud, grouped by template). Public rows (after merge) use `public_train`/`public_test`.

## Composition (cloud slice)
1,258 rows / 264 templates. By primary label (rows / templates):
benign 352/49, jailbreak 159/21, prompt_injection_indirect 135/64, ambiguous 119/14,
credential_harvesting 95/27, exfiltration 81/24, privilege_escalation 76/18,
defense_evasion_logging 75/17, data_destruction 63/15, prompt_injection_direct 59/10,
system_prompt_extraction 44/5.
Channels: user_prompt 1,102 · retrieved_content 119 · tool_output 37.
Flags: obfuscated 85 · indirect 135 · multi_turn 26 · non-ASCII 13.

## Honest numbers (why you must merge the public sets)
- A TF-IDF baseline scores ~**0.40** macro-F1 on the **grouped** split (it scored a
  meaningless **1.000** on the old leaky split — that's the bug this version fixes).
  ~0.40 is the real difficulty; report numbers only on the grouped/held-out split.
- Trained on this slice alone, a model flags a large share of **human-written**
  benign prompts as attacks. The cloud slice gives coverage, **not** realism —
  the human-written public sets supply that. Use `merge_public.py`, train on
  `train + public_train`, and report **separately** on `cloud test` and each
  `public_test` (out-of-distribution).

## Known-thin spots (grow these next)
- `system_prompt_extraction` (5 templates) and non-ASCII (13 rows) are the
  thinnest; multi-turn is 5 skeletons. Add templates (not more renders).

## Honest limitations (disclose in the paper)
- Rows are **synthetic/team-written** and **unreviewed** (`reviewed_by=""`); a
  classifier can learn writing style, so the realism must come from the public +
  agent-benchmark data. **Do not train on this slice alone.**
- Count **templates**, not rows, when judging diversity.
- Attacks are standard, publicly-documented (OWASP/ATLAS) patterns, authored to
  train a **defensive** classifier that blocks them.

## Novelty note
A cloud-security prompt-injection + hard-safe benchmark does not exist publicly,
so this slice (once reviewed + expanded + merged) is a legitimate **dataset
contribution**, even though prompt-injection *detection* as a technique is prior
art. The value is the cloud-specific + hard-safe coverage, not the classifier.
