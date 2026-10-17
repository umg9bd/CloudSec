"""
merge_public.py
===============
Merges the licensed PUBLIC prompt-injection sets into the Stage-1 schema and
combines them with the cloud-specific slice (prompt_injection_cloud.jsonl).

Why: the cloud slice alone doesn't generalize to human-written prompts (a model
trained on it flags 45-85% of real benign prompts as attacks -- review #4). The
public sets provide the human-written realism and general-benign coverage.

The public sets are NOT redistributed here (licenses + size). This script loads
them from Hugging Face (needs `pip install datasets` and network) and maps them:

  deepset/prompt-injections        Apache-2.0  label 1 -> prompt_injection_direct, 0 -> benign
  jackhhao/jailbreak-classification Apache-2.0  "jailbreak" -> jailbreak, "benign" -> benign
  Lakera/gandalf_ignore_instructions MIT        all attack -> system_prompt_extraction (secret-via-override)

IMPORTANT for the out-of-distribution check (review #6): each public set keeps
its OWN native train/test split (prefixed `public_`), so the public TEST rows
are never mixed into the cloud TRAIN. Train a model on cloud+public train, then
report separately on (a) cloud test and (b) each public test.

Run:  python datasets/prompt-injection/merge_public.py
Output: prompt_injection_merged.jsonl  (cloud slice + mapped public rows)
"""
import json
import os
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD = os.path.join(HERE, "prompt_injection_cloud.jsonl")
OUT = os.path.join(HERE, "prompt_injection_merged.jsonl")

# dataset name -> (hf_id, license, split_map, row_mapper)
SOURCES = {
    "deepset_prompt_injections": dict(
        hf="deepset/prompt-injections", license="Apache-2.0",
        text_col="text", label_col="label",
        map=lambda v: (["prompt_injection_direct"] if int(v) == 1 else ["benign"])),
    "jackhhao_jailbreak": dict(
        hf="jackhhao/jailbreak-classification", license="Apache-2.0",
        text_col="prompt", label_col="type",
        map=lambda v: (["jailbreak"] if str(v).lower().startswith("jail") else ["benign"])),
    "lakera_gandalf": dict(
        hf="Lakera/gandalf_ignore_instructions", license="MIT",
        text_col="text", label_col=None,
        map=lambda v: ["system_prompt_extraction"]),   # all attacks (secret via override)
}
LABEL_ATLAS = {
    "prompt_injection_direct": "AML.T0051.000", "jailbreak": "AML.T0054",
    "system_prompt_extraction": "AML.T0056", "benign": "",
}


def to_row(text, labels, source, license_, split):
    atlas = [LABEL_ATLAS.get(l, "") for l in labels if LABEL_ATLAS.get(l)]
    primary = "benign" if "benign" in labels else labels[0]
    return dict(
        id=hashlib.sha1(text.encode("utf-8")).hexdigest(),
        template_id="public:" + source,          # one group per public source
        text=text, channel="user_prompt", labels=labels, primary_label=primary,
        category="public", atlas_ids=atlas, atlas_tactic="",
        is_obfuscated=False, is_indirect=False, multi_turn=False,
        conversation_history=[], rationale="public dataset label",
        cloud_specific=False, source=source, license=license_,
        dataset_version="public", reviewed_by="original_dataset",
        split="public_" + split,                  # keep public splits SEPARATE
    )


def main():
    rows = [json.loads(l) for l in open(CLOUD, encoding="utf-8")]
    print(f"cloud slice: {len(rows)} rows")
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n`datasets` not installed. Run: pip install datasets")
        print("Then re-run this script. Writing cloud-only merged file for now.")
        _write(rows); return

    for name, cfg in SOURCES.items():
        try:
            ds = load_dataset(cfg["hf"])
        except Exception as e:
            print(f"  SKIP {name} ({cfg['hf']}): {type(e).__name__}: {str(e)[:80]}")
            continue
        n = 0
        for split in ds:                              # native splits: train/test/validation
            native = "test" if "test" in split or "val" in split else "train"
            for ex in ds[split]:
                text = ex.get(cfg["text_col"])
                if not text:
                    continue
                v = ex.get(cfg["label_col"]) if cfg["label_col"] else None
                rows.append(to_row(text, cfg["map"](v), name, cfg["license"], native))
                n += 1
        print(f"  + {name}: {n} rows ({cfg['license']})")
    _write(rows)


def _write(rows):
    # dedupe by id across cloud + public
    seen, uniq = set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"]); uniq.append(r)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"\nwrote {len(uniq)} rows -> {os.path.basename(OUT)}")
    print("splits:", dict(Counter(r["split"] for r in uniq)))
    print("sources:", dict(Counter(r["source"] for r in uniq)))
    print("\nEvaluate: train on (cloud) train + public_train; report SEPARATELY on"
          " cloud test, public_test (OOD). Never tune on any *_test split.")


if __name__ == "__main__":
    main()
