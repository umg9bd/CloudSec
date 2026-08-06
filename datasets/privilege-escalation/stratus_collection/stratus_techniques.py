"""
Shared technique config for the Phase 1 real-data collection scripts
(run_detonations.py, collect_real_logs.py).

expected_events: the CloudTrail eventName(s) actually emitted during
DETONATION (not warmup), taken verbatim from `stratus show <technique>`.
Used to verify a run's logs actually arrived, instead of trusting that
"a log file exists" means the right events are in it.
"""

TECHNIQUES = [
    {
        "id": "aws.persistence.iam-create-admin-user",
        "tactic": "persistence",
        "expected_events": ["CreateUser", "AttachUserPolicy"],
        "cost_note": None,
    },
    {
        "id": "aws.privilege-escalation.iam-update-user-login-profile",
        "tactic": "privilege-escalation",
        "expected_events": ["UpdateLoginProfile"],
        "cost_note": None,
    },
    {
        "id": "aws.persistence.iam-backdoor-user",
        "tactic": "persistence",
        "expected_events": ["CreateAccessKey"],
        "cost_note": None,
    },
    {
        "id": "aws.persistence.iam-create-backdoor-role",
        "tactic": "persistence",
        "expected_events": ["CreateRole", "AttachRolePolicy"],
        "cost_note": None,
    },
    {
        "id": "aws.persistence.iam-create-user-login-profile",
        "tactic": "persistence",
        "expected_events": ["CreateLoginProfile"],
        "cost_note": None,
    },
    {
        "id": "aws.persistence.lambda-backdoor-function",
        "tactic": "persistence",
        "expected_events": ["AddPermission20150331v2"],
        "cost_note": "Creates a real Lambda function during warmup; well within free tier for a single function.",
    },
    {
        "id": "aws.credential-access.ec2-get-password-data",
        "tactic": "credential-access",
        "expected_events": ["AssumeRole", "GetPasswordData"],
        "cost_note": None,
    },
    {
        "id": "aws.credential-access.secretsmanager-retrieve-secrets",
        "tactic": "credential-access",
        "expected_events": ["ListSecrets", "GetSecretValue"],
        "cost_note": "Creates real Secrets Manager secrets (~$0.40/secret/month, prorated daily). "
                      "Negligible if cleanup runs promptly, but don't leave a failed run uncleaned.",
    },
    {
        "id": "aws.credential-access.ssm-retrieve-securestring-parameters",
        "tactic": "credential-access",
        "expected_events": ["DescribeParameters", "GetParameters"],
        "cost_note": None,
    },
    {
        "id": "aws.defense-evasion.cloudtrail-stop",
        "tactic": "defense-evasion",
        "expected_events": ["StopLogging"],
        "cost_note": "Creates its own throwaway CloudTrail trail during warmup and stops that trail - "
                      "does not touch stratus-redteam-trail.",
    },
    {
        "id": "aws.defense-evasion.cloudtrail-event-selectors",
        "tactic": "defense-evasion",
        "expected_events": ["PutEventSelectors"],
        "cost_note": "Creates its own throwaway CloudTrail trail during warmup - "
                      "does not touch stratus-redteam-trail.",
    },
]

TECHNIQUE_IDS = [t["id"] for t in TECHNIQUES]
