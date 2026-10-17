# Stage-1 Prompt-Injection Dataset (cloud-specific slice)

This directory holds the **cloud-specific, team-generated** portion of the Stage-1
dataset — the gap that public prompt-injection sets and agent benchmarks do **not**
cover (AWS credential theft, privilege escalation, defense evasion, exfiltration,
indirect injection in tool output, and the **hard-safe** counterparts that keep a
security gate from blocking legitimate cloud work).

It is **not the whole Stage-1 dataset.** It is the differentiator. Combine it with
the public + benchmark data below for realism and volume.

## Files
| File | What |
|---|---|
| `generate_prompt_dataset.py` | the generator (templates + paraphrase augmentation). Re-run to regenerate. |
| `prompt_injection_cloud.jsonl` / `.csv` | the generated rows (2,526), same content both formats |

Regenerate: `python datasets/prompt-injection/generate_prompt_dataset.py` (seeded, deterministic).

## Schema
`id, text, channel, label, category, atlas_id, cloud_specific, multi_turn, source, reviewed_by, split`
- **channel**: `user_prompt` | `retrieved_content` | `tool_output` — the untrusted-context signal (instruction-like text in data the agent *reads* is indirect injection).
- **label**: `benign | injection_direct | injection_indirect | jailbreak | secret_extraction | defense_evasion | ambiguous`
- **atlas_id**: MITRE ATLAS technique (e.g. `AML.T0051.000` direct injection, `.001` indirect, `T0054` jailbreak, `T0056` system-prompt extraction, `T0068` obfuscation, `T0098` credential harvesting).
- **reviewed_by**: **empty on purpose** — every row needs a human reviewer before training (the spec requires reviewed labels).
- **split**: `train`/`test`, stratified 80/20 by label (seeded).

## Current composition (this slice only)
~2,526 rows: **attack 1,181 · benign 1,142 · ambiguous 203**.
Labels: benign 1142, secret_extraction 411, defense_evasion 278, injection_direct 242, ambiguous 203, jailbreak 137, injection_indirect 113.
Channels: user_prompt 2,388 · retrieved_content 113 · tool_output 25.

## How to build the full Stage-1 set (TODO for the team)
1. **Add the licensed public sets** (human-written, generic signal):
   - `deepset/prompt-injections` (Apache-2.0), `jackhhao/jailbreak-classification` (Apache-2.0), `Lakera/gandalf_ignore_instructions` (MIT).
   - Map their binary labels into this schema (injection→`injection_direct`/`secret_extraction`, jailbreak→`jailbreak`); set `source` accordingly, `cloud_specific=false`.
   - **Exclude** `xTRam1/safe-guard-prompt-injection` until its license is confirmed.
2. **Check + add agent/indirect benchmarks** (closest to the AWS-agent gaps): **InjecAgent**, **AgentDojo**, **BIPIA** (indirect injection in documents). Verify licenses first.
3. **Human-review** a sample of every label (fill `reviewed_by`); report inter-annotator agreement.
4. **Grow thin categories** (jailbreak, indirect) with more hand-written diversity.

## Honest limitations (disclose in the paper)
- These rows are **synthetic/team-written**. A classifier can learn the team's *writing style* rather than real attacks, so the realistic signal must come from the human-written public + benchmark data; keep these paraphrase-diverse and **do not train on this slice alone**.
- Paraphrase augmentation multiplies surface forms of the same intent — good for robustness, but count *templates*, not rows, when judging true diversity.
- Labels here are **unreviewed** until `reviewed_by` is filled.
- The cloud attacks are **standard, publicly-documented (OWASP/ATLAS) patterns**, authored to train a *defensive* classifier that blocks them.

## Note on novelty
A cloud-security prompt-injection + hard-safe benchmark does **not** exist publicly — so this slice (once reviewed and expanded) is a legitimate **dataset contribution**, even though prompt-injection *detection* as a technique is prior art (Llama Guard, Bedrock, US 12689653). Put the value here, not in the classifier architecture.
