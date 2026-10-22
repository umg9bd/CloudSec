"""
provenance.py
=============
Source provenance and catalogue-version tracking.

WHY THIS EXISTS
---------------
Every control that reaches a prompt must be traceable to a named source, and the
module must be honest about HOW FAR that source was checked. Three levels exist:

    unverified         cited from the author's knowledge; the source was not read
    section_checked    the cited section was read in the fetched source on `retrieved_at`
    snapshot_verified  a stored snapshot exists and `content_hash` is its SHA-256

This file defines what each level requires, whether a control may be applied under the
current policy, and the freshness summary attached to every response, so a downstream
reader can see exactly which catalogue version produced a hardened prompt and how much
of it has been verified. It never touches the network: fetching and snapshotting is
source_sync.py's job and never happens on the request path.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .models import Control, _plain

PROVENANCE_LEVELS = ("unverified", "section_checked", "snapshot_verified")


def sha256_bytes(data: bytes) -> str:
    """'sha256:<hex>' for a bytes snapshot (the format stored in `content_hash`)."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def provenance_problems(control: Control) -> List[str]:
    """Rules that make a control traceable. Checked for every control at load time."""
    cid = control.control_id
    problems: List[str] = []
    for name in ("source_name", "source_url", "source_version_or_section"):
        if not str(getattr(control, name) or "").strip():
            problems.append(f"{cid}: {name} must not be empty")
    status = control.provenance_status
    if status == "unverified":
        if control.content_hash is not None or control.retrieved_at is not None:
            problems.append(f"{cid}: an unverified control must not carry retrieved_at or content_hash")
    elif status == "section_checked":
        if not control.retrieved_at:
            problems.append(f"{cid}: section_checked requires retrieved_at")
    elif status == "snapshot_verified":
        if not control.retrieved_at or not control.content_hash:
            problems.append(f"{cid}: snapshot_verified requires retrieved_at and content_hash")
    else:
        problems.append(f"{cid}: unknown provenance_status {status!r}")
    if control.derived and not control.supporting_references:
        problems.append(f"{cid}: a derived recommendation must list its supporting_references")
    return problems


def control_usability(control: Control, policy: Any) -> Tuple[bool, List[str]]:
    """May this control be applied under the current policy? Returns (usable, warnings)."""
    if control.active_status != "active":
        return False, []
    if policy.require_snapshot_verified_provenance and control.provenance_status != "snapshot_verified":
        return False, [f"{control.control_id}: skipped because policy requires snapshot-verified provenance "
                       f"(this control is {control.provenance_status})"]
    if control.source_status == "unavailable":
        if policy.on_unavailable_source == "skip":
            return False, [f"{control.control_id}: skipped because its source is currently unavailable"]
        return True, [f"{control.control_id}: its source is currently unavailable; the last known-good text was "
                      f"applied and could not be re-checked"]
    return True, []


def source_reference_dict(control: Control) -> Dict[str, Any]:
    return _plain(control.source_reference())


def freshness_report(catalogue: Any) -> Dict[str, Any]:
    """Deterministic summary of how much of the catalogue is verified (attached to every response)."""
    active = [c for c in catalogue.controls if c.active_status == "active"]
    counts = {level: sum(1 for c in active if c.provenance_status == level) for level in PROVENANCE_LEVELS}
    fresh = catalogue.freshness
    report: Dict[str, Any] = {
        "catalogue_version": catalogue.catalogue_version,
        "last_sync_status": fresh.get("last_sync_status", "never_synced"),
        "last_sync_attempt": fresh.get("last_sync_attempt"),
        "last_known_good_version": fresh.get("last_known_good_version"),
        "active_controls": len(active),
        "provenance_counts": counts,
        "unavailable_sources": sorted(c.control_id for c in active if c.source_status == "unavailable"),
    }
    if fresh.get("last_sync_status", "never_synced") == "never_synced":
        report["note"] = ("No source synchronisation has run for this catalogue; "
                          "see provenance_counts for what has been verified.")
    elif fresh.get("last_sync_status") == "failed":
        report["note"] = "The last source synchronisation failed; the last known-good catalogue is in use."
    return report
