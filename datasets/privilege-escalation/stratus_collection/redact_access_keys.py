"""
Redact AWS Access Key IDs from the dataset before publication.

GitHub secret-scanning flags AWS Access Key IDs (AKIA.../ASIA...) that appear
in CloudTrail-derived data. These IDs are the PUBLIC half of a credential --
recorded by CloudTrail on every API call, unusable without the secret key
(which is NOT in this repo). So this is not a leaked-credential emergency; it is
a data-hygiene step so a PUBLISHED dataset carries no real principal key IDs and
GitHub stops flagging it.

The key ID value is not a feature or a graph node in this pipeline -- feature
engine 9 uses only `has_access_key` (present vs empty, feature_engine9.py). So
every real ID is replaced with a deterministic, non-empty, non-AWS-format token:

    AKIAXZ5NGD3MQUGZJYPU  ->  REDACTEDAK<10-hex-of-hash>

Deterministic and 1:1, so the SAME real ID maps to the SAME token in every file
(dev/test/combined and the .graph_node_state composites like `root||AKIA...`),
which keeps principal identity and graph structure exactly as they were. The
token does not start with an AWS key prefix, so it will not be re-flagged.

Usage:
    # dry run (default): report what WOULD change, write nothing
    python redact_access_keys.py

    # apply in place
    python redact_access_keys.py --apply

    # restrict to specific files (default: every tracked file that contains a key ID)
    python redact_access_keys.py --apply --files ../real_dataset_combined.csv

IMPORTANT for a shared repo: run this on the mainline/integration branch, commit
once, THEN do the history rewrite (see REDACTION_PLAN.md). Running it on a
feature branch first makes every redacted data file conflict at merge time.
Redacting forward is non-destructive; the history rewrite + force-push is the
destructive, coordinate-with-the-team step and is NOT done by this script.
"""

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# GitHub's AWS Access Key ID pattern: a fixed set of 4-char prefixes + 16 base32
# chars. Matching the same set means we redact exactly what the scanner flags.
KEY_ID_RE = re.compile(r"\b(A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b")

# Salt keeps tokens stable across runs but not trivially reversible via a plain
# hash of the ID. It is not a secret (the IDs are not secret either) -- it only
# needs to be constant, so it is committed with the script.
_SALT = "cloudsec-privilege-escalation-dataset-redaction-v1"


def token_for(key_id: str) -> str:
    h = hashlib.sha256((_SALT + key_id).encode()).hexdigest()[:10]
    # Preserve the AK/AS type letters (long-term vs temporary) for anyone
    # eyeballing the data; the prefix REDACTEDAK/REDACTEDAS is not an AWS format.
    return f"REDACTED{key_id[:2]}{h}"


def tracked_files_with_keys() -> list:
    """Every tracked file whose current content contains an AWS key ID."""
    out = subprocess.run(
        ["git", "grep", "-lE", KEY_ID_RE.pattern, "HEAD", "--", "."],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout
    files = []
    for line in out.splitlines():
        # `git grep HEAD` prefixes each path with "HEAD:"
        path = line.split(":", 1)[1] if line.startswith("HEAD:") else line
        files.append(REPO_ROOT / path)
    return files


def redact_file(path: Path, apply: bool, mapping: dict) -> tuple:
    """Returns (replacements, distinct_ids_in_file). Streams the file so large
    CSVs don't have to fit in memory twice."""
    replacements = 0
    ids_here = set()

    def sub(m):
        nonlocal replacements
        kid = m.group(0)
        ids_here.add(kid)
        replacements += 1
        tok = mapping.get(kid)
        if tok is None:
            tok = token_for(kid)
            mapping[kid] = tok
        return tok

    if not path.exists():
        return 0, ids_here
    tmp = path.with_suffix(path.suffix + ".redacted.tmp")
    with open(path, "r", encoding="utf-8", errors="surrogatepass") as fin, \
         open(tmp, "w", encoding="utf-8", errors="surrogatepass", newline="") as fout:
        for line in fin:
            fout.write(KEY_ID_RE.sub(sub, line))
    if apply and replacements:
        tmp.replace(path)
    else:
        tmp.unlink()
    return replacements, ids_here


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    p.add_argument("--files", nargs="*", help="Specific files (default: all tracked files with key IDs)")
    args = p.parse_args()

    if args.files:
        files = [Path(f).resolve() for f in args.files]
    else:
        files = tracked_files_with_keys()

    print(f"{'APPLYING' if args.apply else 'DRY RUN'} over {len(files)} file(s)\n")
    mapping = {}
    total = 0
    all_ids = set()
    per_file = []
    for f in files:
        n, ids = redact_file(f, args.apply, mapping)
        if n:
            per_file.append((n, f))
        total += n
        all_ids |= ids

    for n, f in sorted(per_file, reverse=True)[:15]:
        try:
            rel = f.relative_to(REPO_ROOT)
        except ValueError:
            rel = f
        print(f"  {n:>8,}  {rel}")
    if len(per_file) > 15:
        print(f"  ... and {len(per_file) - 15} more file(s)")

    print(f"\nDistinct real key IDs: {len(all_ids)}   Total occurrences: {total:,}")
    if all_ids:
        ex = sorted(all_ids)[0]
        print(f"Example mapping: {ex} -> {token_for(ex)}")
    if not args.apply:
        print("\nDry run only -- nothing written. Re-run with --apply to redact.")
    else:
        print("\nDone. Verify: git grep -nE '(AKIA|ASIA)[A-Z0-9]{16}' -- . should return nothing.")
        print("Then commit, and see REDACTION_PLAN.md for the history rewrite.")


if __name__ == "__main__":
    main()
