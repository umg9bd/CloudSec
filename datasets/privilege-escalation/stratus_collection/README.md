# Real attack data collection — teammate setup

This produces the real, held-out test data for the privilege-escalation detection project, using
[Stratus Red Team](https://stratus-red-team.cloud/) to actually run attack techniques in a sandbox
AWS account and capturing the resulting CloudTrail logs. Each teammate runs this independently, on
their own AWS account, and everyone's results get merged via git. You can run at the same time as
teammates — separate accounts mean there's nothing to conflict over.

**Do this on your own AWS account, not a shared one.** Nobody needs anyone else's credentials.

## 1. AWS account

If you don't already have one, create an [AWS Free Tier account](https://aws.amazon.com/free/).
Prefer a fresh account with nothing else running in it — the IAM user below gets broad permissions,
which is fine for a dedicated sandbox account but not something to mix with anything else you use AWS for.

## 2. Create an IAM user for this project

1. AWS Console → **IAM** → **Users** → **Create user** → name it e.g. `stratus-redteam`
2. Click into the new user → **Permissions** tab → **Add permissions** → **Attach policies directly**
3. Search `AdministratorAccess`, check it, → **Next** → **Add permissions**
   (Stratus needs broad permissions to create/attach IAM roles, policies, Lambda functions, etc. This
   is safe because it's a dedicated sandbox account.)
4. Same user → **Security credentials** tab → **Create access key** → choose **Command Line Interface (CLI)**
5. Save the Access Key ID and Secret Access Key somewhere safe. **Do not paste these into Slack, a
   commit, or anywhere outside your own machine's AWS config.**

## 3. Install the AWS CLI

Download and run: https://awscli.amazonaws.com/AWSCLIV2.msi (Windows) — or the equivalent for your OS
from the [AWS CLI install docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Verify: `aws --version`

## 4. Configure your credentials

```
aws configure
```
Paste in your Access Key ID and Secret Access Key when prompted. **For "Default region name", type
`us-east-1`** (don't leave it blank — a blank region causes Stratus to fail with an authentication
error even though your credentials are fine). Default output format: `json`.

Verify it worked:
```
aws sts get-caller-identity
```
This should print your Account ID, User ID, and ARN. Keep your Account ID handy — you'll need it below.

## 5. Set up CloudTrail (this is what captures the attack logs)

Easiest path — use the AWS Console wizard, which creates the S3 bucket and its access policy for you:

1. AWS Console → **CloudTrail** → **Trails** → **Create trail**
2. Trail name: `stratus-redteam-trail` (or anything memorable)
3. Storage location: choose **Create new S3 bucket**, let it name itself or name it something like
   `stratus-redteam-cloudtrail-<your-account-id>`
4. Leave "Management events" enabled (this is what we need — API calls, not data events)
5. Under trail settings, make it **multi-region** (so you don't miss events if a technique runs outside `us-east-1`)
6. Create the trail, and confirm on the trail's page that logging is **ON**

**Write down your bucket name and account ID** — both are required arguments later.

## 6. Set a budget alert (cheap insurance, not strictly required)

Billing Console → **Budgets** → **Create budget** → Cost budget → $5/month → add an email alert.

## 7. Install Stratus Red Team

Download the release for your OS from the
[Stratus Red Team releases page](https://github.com/DataDog/stratus-red-team/releases/latest)
(e.g. `stratus-red-team_Windows_x86_64.tar.gz` for Windows, `_Darwin_` for Mac, `_Linux_` for Linux).
Extract it and put the `stratus` (or `stratus.exe`) binary somewhere on your PATH.

Verify:
```
stratus list
```
You should see a table of attack techniques.

## 8. Get the code

Clone the project repo and go to this folder:
```
cd datasets/privilege-escalation/stratus_collection
```
You'll need Python 3 with no extra packages required (everything here uses the standard library).

## 9. (Recommended) Run one smoke-test technique manually first

Before running a full batch, confirm your setup works end to end with one cheap, safe technique:
```
stratus warmup aws.credential-access.ssm-retrieve-securestring-parameters
stratus detonate aws.credential-access.ssm-retrieve-securestring-parameters
stratus revert aws.credential-access.ssm-retrieve-securestring-parameters
stratus cleanup aws.credential-access.ssm-retrieve-securestring-parameters
```
If all four commands complete without errors, you're set up correctly.

## 10. Run a real session

Use **your own name/handle** as `--collector` — this keeps your data file separate from teammates' so
they merge cleanly in git, with no conflicts.

```
python run_detonations.py --collector <your_name> --reps 3
```
This runs all 11 techniques, 3 reps each (33 runs), logging every run to `manifest_<your_name>.csv`.
Takes roughly 45-90 minutes. Safe to let it run in the background while you do other things.

Then pull down and verify the actual CloudTrail logs. You'll need your own bucket name and account ID —
if you don't have them written down from step 5, look them up:
```
aws sts get-caller-identity --query Account --output text
aws cloudtrail describe-trails --query "trailList[0].S3BucketName" --output text
```
Then:
```
python collect_real_logs.py --collector <your_name> --bucket <your-bucket-name> --account <your-account-id>
```
Run this again ~20-30 minutes later to catch anything CloudTrail delivered late — it only pulls new
files, so it's always safe to re-run.

Check `manifest_<your_name>_verified.csv` afterward: each row should eventually show
`logs_confirmed = yes`. If a technique's `cleanup_status` ever shows anything other than `ok`, check
the AWS console for that technique before running more — it means something may not have been
cleaned up.

## 11. How many sessions, and when

Team target: **~18 reps per technique in total, pooled across everyone.** Coordinate loosely (a quick
message in the group chat is enough) so the 6ish total sessions land on different days/times rather
than all at once — e.g. 2 sessions each for two people, 1 each for the other two, each person picking
times that don't all overlap with each other. Running literally at the same time as a teammate is not
a technical problem (separate accounts, nothing shared) — the reason to spread sessions out is so the
combined real dataset has genuine variety in when attacks happened, not because of any conflict risk.

## 12. What to commit and push

**Do commit:**
- Any changes to the `.py` scripts (if you improve something, share it)
- Your `manifest_<your_name>.csv` and `manifest_<your_name>_verified.csv`
- The organized logs under `../stratus_own_runs/CloudTrail/` (plain `.json` files — these are the actual dataset)

**Never commit:**
- Your AWS credentials, `.aws/` folder, or access keys — anywhere, ever
- The `raw_s3/` folder (compressed originals, redundant with the organized JSON, already gitignored)

The `.gitignore` in this repo already blocks the credential and `raw_s3/` paths as a safety net, but
double-check `git status` before your first push regardless.
