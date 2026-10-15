"""
attack_taxonomy.py
==================
The single source of truth for MITRE ATT&CK labels in this project.

WHY THIS EXISTS
---------------
Tactic labels used to live in three unrelated hand-written dictionaries
(`TECHNIQUES` in stratus_techniques.py, `ATTACK_EVENTS` in explore.ipynb,
`ATTACK_CHAINS` in generate_synthetic_data.py), each keyed on eventName alone.
They disagreed: `CreateAccessKey` was "privilege-escalation" in one and
"persistence" in another, `AttachRolePolicy` and `CreateLoginProfile` likewise.
And the column holding them was called `attack_technique` although every value
was a TACTIC ("persistence"), never a technique ID ("T1098.001").

This module fixes both:

  * TACTIC and TECHNIQUE are separate fields (`attack_tactic`,
    `attack_technique_id`), and every (tactic, technique) pair is checked
    against ATT&CK's own tactic membership for that technique.
  * An eventName does NOT determine its tactic. The same `AttachRolePolicy` is
    persistence when it arms a backdoor role and privilege escalation when it
    grants the caller's own path admin. The tactic comes from the CONTEXT that
    produced the event -- the Stratus technique that was detonated (real data)
    or the campaign stage that emitted it (synthetic data) -- and this module
    only validates it.

Technique IDs are only assigned where a defensible ATT&CK mapping exists. Where
none does, the event is recorded as UNMAPPED and the reason is written down
here, rather than an ID being invented to fill the column.

ATT&CK version: Enterprise v15 (tactic memberships below are taken from the
technique pages; re-check them if you move to a newer release).
"""

from __future__ import annotations

# ── Tactics (slug -> ATT&CK tactic) ──────────────────────────────────────────
# Slugs match the values already used by the real dataset, so existing counts
# and reports stay comparable.
TACTICS = {
    "initial-access":       {"id": "TA0001", "name": "Initial Access"},
    "persistence":          {"id": "TA0003", "name": "Persistence"},
    "privilege-escalation": {"id": "TA0004", "name": "Privilege Escalation"},
    "defense-evasion":      {"id": "TA0005", "name": "Defense Evasion"},
    "credential-access":    {"id": "TA0006", "name": "Credential Access"},
    "discovery":            {"id": "TA0007", "name": "Discovery"},
    "lateral-movement":     {"id": "TA0008", "name": "Lateral Movement"},
    "exfiltration":         {"id": "TA0010", "name": "Exfiltration"},
}

UNMAPPED = "unmapped"

# ── Techniques used by this project ──────────────────────────────────────────
# `tactics` is ATT&CK's own membership list for the technique. `basis` records
# why the AWS API calls in this project are mapped to it.
TECHNIQUES = {
    "T1078.004": {
        "name": "Valid Accounts: Cloud Accounts",
        "tactics": {"initial-access", "persistence", "privilege-escalation", "defense-evasion"},
        "basis": "sts:AssumeRole into a role the actor was not meant to use: acting "
                 "through a legitimate cloud identity's temporary credentials.",
    },
    "T1098": {
        "name": "Account Manipulation",
        "tactics": {"persistence", "privilege-escalation"},
        "basis": "Changing who can use an identity: role trust policies "
                 "(CreateRole/UpdateAssumeRolePolicy trusting the attacker) and console "
                 "passwords (Create/UpdateLoginProfile). ATT&CK has no role-trust or "
                 "login-profile sub-technique, so the parent is used rather than guessing one.",
    },
    "T1098.001": {
        "name": "Account Manipulation: Additional Cloud Credentials",
        "tactics": {"persistence", "privilege-escalation"},
        "basis": "iam:CreateAccessKey on an existing or attacker-created user.",
    },
    "T1098.003": {
        "name": "Account Manipulation: Additional Cloud Roles",
        "tactics": {"persistence", "privilege-escalation"},
        "basis": "Granting permissions to an identity: Attach*/Put*Policy, "
                 "AddUserToGroup, Create/SetDefaultPolicyVersion.",
    },
    "T1136.003": {
        "name": "Create Account: Cloud Account",
        "tactics": {"persistence"},
        "basis": "iam:CreateUser.",
    },
    "T1555.006": {
        "name": "Credentials from Password Stores: Cloud Secrets Management Stores",
        "tactics": {"credential-access"},
        "basis": "Enumerating and reading Secrets Manager secrets and SSM SecureString "
                 "parameters (List/Describe then Get), including the KMS Decrypt the read "
                 "triggers. The enumeration calls are kept under this technique because "
                 "they are steps of the same harvesting procedure, not free-standing discovery.",
    },
    "T1552": {
        "name": "Unsecured Credentials",
        "tactics": {"credential-access"},
        "basis": "ec2:GetPasswordData (retrieving a Windows instance's administrator "
                 "password). No sub-technique covers it, so the parent is used.",
    },
    "T1562.008": {
        "name": "Impair Defenses: Disable or Modify Cloud Logs",
        "tactics": {"defense-evasion"},
        "basis": "cloudtrail:StopLogging, DeleteTrail, PutEventSelectors.",
    },
    "T1537": {
        "name": "Transfer Data to Cloud Account",
        "tactics": {"exfiltration"},
        "basis": "s3:PutBucketPolicy granting an external account read access to a bucket.",
    },
    "T1580": {
        "name": "Cloud Infrastructure Discovery",
        "tactics": {"discovery"},
        "basis": "Describe/List calls against compute and storage (DescribeInstances, "
                 "DescribeTrails, GetBucketPolicy) made as a scoped step inside a campaign.",
    },
}

# Events with no defensible technique mapping, and why. Recorded so a reader
# can see the gap was a decision, not an omission.
UNMAPPED_REASONS = {
    "AddPermission20150331v2": "Lambda resource policy opened to principal '*'. Stratus files "
                                "it under persistence, but no ATT&CK technique describes "
                                "exposing a function through its resource policy.",
    "DeleteBucketPolicy": "Removing a bucket policy can be exfiltration prep or cleanup; the "
                          "data does not say which, so no technique is claimed.",
    "ConsoleLogin": "A console sign-in with credentials obtained earlier in the campaign. "
                    "It is the USE of the manipulated account (tactic comes from the "
                    "campaign stage); the manipulation itself carries the technique.",
}


def validate(tactic: str | None, technique_id: str | None) -> None:
    """Raises ValueError unless (tactic, technique_id) is a consistent pair.

    `technique_id` may be UNMAPPED (any tactic allowed) or None (only when
    tactic is also None, i.e. a benign row)."""
    if tactic is None and technique_id is None:
        return
    if tactic not in TACTICS:
        raise ValueError(f"unknown ATT&CK tactic slug {tactic!r}; expected one of {sorted(TACTICS)}")
    if technique_id == UNMAPPED:
        return
    tech = TECHNIQUES.get(technique_id)
    if tech is None:
        raise ValueError(f"technique {technique_id!r} is not defined in attack_taxonomy.TECHNIQUES")
    if tactic not in tech["tactics"]:
        raise ValueError(
            f"{technique_id} ({tech['name']}) is not an ATT&CK {tactic!r} technique; "
            f"its tactics are {sorted(tech['tactics'])}")


# ── Legacy: the invictus capture's event-name dictionary ─────────────────────
# The 2023 invictus capture was labeled by eventName alone (explore.ipynb,
# section 1) and its raw CloudTrail is not in this repo, so its labels cannot
# be re-derived from context. This dictionary is kept ONLY to reproduce and
# annotate those historical labels; rows labeled through it carry
# attack_mapping_basis = "event_name_dictionary" so they can be told apart
# from context-labeled rows. Do not use it to label anything new.
INVICTUS_EVENT_LABELS = {
    # eventName: (tactic, technique_id)
    "CreateAccessKey":         ("privilege-escalation", "T1098.001"),
    "CreateLoginProfile":      ("privilege-escalation", "T1098"),
    "UpdateLoginProfile":      ("privilege-escalation", "T1098"),
    "AttachUserPolicy":        ("privilege-escalation", "T1098.003"),
    "AttachRolePolicy":        ("privilege-escalation", "T1098.003"),
    "AttachGroupPolicy":       ("privilege-escalation", "T1098.003"),
    "PutUserPolicy":           ("privilege-escalation", "T1098.003"),
    "PutRolePolicy":           ("privilege-escalation", "T1098.003"),
    "PutGroupPolicy":          ("privilege-escalation", "T1098.003"),
    "CreatePolicyVersion":     ("privilege-escalation", "T1098.003"),
    "SetDefaultPolicyVersion": ("privilege-escalation", "T1098.003"),
    "AddUserToGroup":          ("privilege-escalation", "T1098.003"),
    "CreateUser":              ("persistence",          "T1136.003"),
    "CreateRole":              ("persistence",          "T1098"),
    "UpdateAssumeRolePolicy":  ("persistence",          "T1098"),
    "StopLogging":             ("defense-evasion",      "T1562.008"),
    "DeleteTrail":             ("defense-evasion",      "T1562.008"),
    "UpdateTrail":             ("defense-evasion",      "T1562.008"),
    "PutEventSelectors":       ("defense-evasion",      "T1562.008"),
    "GetSecretValue":          ("credential-access",    "T1555.006"),
    "GetPasswordData":         ("credential-access",    "T1552"),
    "PutBucketPolicy":         ("exfiltration",         "T1537"),
    "DeleteBucketPolicy":      ("exfiltration",         UNMAPPED),
}

for _tactic, _tech in INVICTUS_EVENT_LABELS.values():
    validate(_tactic, _tech)
