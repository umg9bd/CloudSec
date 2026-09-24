# Access Key ID redaction — remediation plan

## What happened
GitHub secret-scanning flagged AWS **Access Key IDs** (`AKIA…`/`ASIA…`) in the
CloudTrail dataset. Verified across the full git history:

- **No** secret access keys, session tokens, private keys, or plaintext passwords.
- Only Access Key **IDs** — the *public* half of a credential, recorded by
  CloudTrail on every API call. **Unusable without the secret key**, which is not
  in this repo and cannot be derived from the ID.

So this is **not a leaked-credential emergency.** It is a data-hygiene issue for
a repo that will be published/submitted.

## Scope
- **10,591** distinct key IDs, **127,520** occurrences, across **3,677** files
  (3,667 raw CloudTrail JSONs under `stratus_own_runs/CloudTrail/` + 10 derived
  dataset/state files).
- The key ID value is **not** a model feature or a graph node — `feature_engine9`
  uses only `has_access_key` (present vs empty). Redacting the values changes no
  results.

## Fix — in order of what actually reduces risk

### 1. Deactivate the exposed keys (do this first, it's what matters)
In each of **your own** sandbox accounts (536697249497, 268769775703,
482745810921, 261390480702, …), IAM → Users → Security credentials →
**Deactivate/Delete** the old access keys. Once the keys are inactive, the
exposed IDs are inert trivia. The Invictus-dataset keys (account 123837392027,
user `benjamin`) are **not yours** — leave them; they are public-dataset data.

### 2. Redact the dataset (makes the published repo clean)
Run the deterministic redactor **on the mainline/integration branch**, not a
feature branch (otherwise all 3,677 files conflict at merge):

```
git checkout <integration-branch>
python datasets/privilege-escalation/stratus_collection/redact_access_keys.py            # dry run, review
python datasets/privilege-escalation/stratus_collection/redact_access_keys.py --apply
git grep -nE '(AKIA|ASIA)[A-Z0-9]{16}' -- .    # must return nothing
git commit -am "security: redact AWS access key IDs from CloudTrail dataset"
```

Same real ID → same token everywhere, so principal identity and graph structure
are preserved. Every other branch should then merge/rebase this commit (the
redaction is deterministic, so branches converge).

### 3. (Optional, for a pristine public history) Rewrite history
Redaction in step 2 cleans the current tree, but the old key IDs remain in past
commits. If you need them gone from history too — **destructive, coordinate with
the whole team first, everyone must re-clone afterward:**

```
pip install git-filter-repo
# build a replacements file mapping each real ID to its token, then:
git filter-repo --replace-text replacements.txt
git push --force --all
git push --force --tags
```
Every teammate then re-clones (their old clones still contain the IDs).
Because the keys are deactivated (step 1), this is polish, not a security need —
decide as a team whether it's worth the disruption.

### 4. Prevent recurrence
- Consider `.gitignore`-ing raw captures (`stratus_own_runs/CloudTrail/*.json`)
  and committing only the **redacted** derived datasets the pipeline needs.
- A pre-commit hook that greps for `(AKIA|ASIA)[A-Z0-9]{16}` blocks future
  accidental commits.

### 5. The GitHub alert
Once keys are deactivated (and ideally step 2 done), dismiss the alert as
*“used in tests / not a live credential — CloudTrail dataset identifiers.”* That
is accurate. Don't dismiss before deactivating.
