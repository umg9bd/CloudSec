"""Explicit, review-gated synchronisation for the security-control catalogue.

Network access is confined to this module and only happens when a user explicitly runs
``python -m security_hardening.source_sync sync``. The request-hardening path never imports
or calls this module.

The synchroniser is deliberately conservative:
* HTTPS only, with a fixed allow-list of official source hosts and safe redirects;
* bounded downloads and timeouts;
* content hashes cover the exact bytes written to the local snapshot;
* source updates create a candidate catalogue; they do not change the active catalogue
  unless ``--activate`` is explicitly supplied;
* candidates are schema/semantic validated before activation;
* unsupported or unverified source sections are not silently activated;
* activation writes a backup, and rollback restores the most recent backup.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from .catalogue import CATALOGUE_PATH, CatalogueError, load_catalogue, parse_catalogue
from .config import load_scope
from .provenance import sha256_bytes

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT_DIR = PACKAGE_DIR / "snapshots" / "live"
DEFAULT_CANDIDATE_DIR = PACKAGE_DIR / "catalogue" / "candidates"
DEFAULT_BACKUP_DIR = PACKAGE_DIR / "catalogue" / "backups"
DEFAULT_STATE_PATH = PACKAGE_DIR / "data" / "source_sync_state.json"
ALLOWED_HOSTS = frozenset({
    "docs.aws.amazon.com", "csrc.nist.gov", "nvlpubs.nist.gov",
    "cheatsheetseries.owasp.org", "api-security.owasp.org", "top10.owasp.org",
    "owasp.org", "community.owasp.org",
})
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 15


class SourceSyncError(RuntimeError):
    """A synchronisation operation could not be completed safely."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._skip_depth:
            return
        self.parts.append(text)
        if self._in_title:
            self.title_parts.append(text)

    @property
    def text(self) -> str:
        return " ".join(self.parts)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts)


def _check_url(url: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise SourceSyncError(f"refusing non-HTTPS source URL: {url}")
    if not parsed.hostname or parsed.hostname.lower() not in ALLOWED_HOSTS:
        raise SourceSyncError(f"source host is not allow-listed: {parsed.hostname or '<missing>'}")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise SourceSyncError("source URL may not include credentials or a non-standard port")
    return parsed


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_source(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS,
                 max_bytes: int = MAX_DOWNLOAD_BYTES) -> Dict[str, Any]:
    """Fetch a source from allow-listed HTTPS domains and return raw bytes plus metadata."""
    _check_url(url)
    request = urllib.request.Request(url, headers={
        "User-Agent": "CloudSec-ControlCatalogue-Sync/1.0 (explicit operator initiated)",
        "Accept": "text/html,application/pdf,application/xhtml+xml;q=0.9,*/*;q=0.1",
    }, method="GET")
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            _check_url(final_url)
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise SourceSyncError(f"source exceeded maximum download size ({max_bytes} bytes)")
            if not body or len(body) < 128:
                raise SourceSyncError("source response is empty or implausibly small")
            return {
                "url": url, "final_url": final_url,
                "status_code": getattr(response, "status", 200),
                "content_type": response.headers.get_content_type() if response.headers else "application/octet-stream",
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "body": body,
            }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceSyncError(f"could not retrieve {url}: {type(exc).__name__}: {exc}") from exc


def _normalise_content(raw: bytes, content_type: str) -> Tuple[str, str]:
    # The configured baseline sources are HTML landing/document pages. PDFs are preserved byte-for-byte
    # as snapshots but not marked section-verified by this simple extractor.
    if "pdf" in content_type.lower() or raw[:4] == b"%PDF":
        return "", "PDF document (text extraction not performed)"
    text = raw.decode("utf-8", errors="replace")
    extractor = _TextExtractor()
    try:
        extractor.feed(text)
        extractor.close()
        return extractor.text, extractor.title
    except Exception:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)), ""


def _required_terms(control: Mapping[str, Any]) -> List[str]:
    """Extract modest, explicit source-section hints for a conservative relevance check."""
    section = str(control.get("source_version_or_section", "")).lower()
    cid = str(control.get("control_id", "")).lower()
    groups: List[List[str]] = [
        ["least privilege", "least-privilege", "permissions"],
        ["wildcard", "least privilege", "permissions"],
        ["access analyzer", "trust policy", "external"],
        ["permissions boundaries", "permissions boundary", "delegate"],
        ["temporary credentials", "roles", "access keys"],
        ["block public access", "public access", "bucket policies"],
        ["encryption", "tls", "secure transport"],
        ["versioning", "object lock", "delete"],
        ["security groups", "traffic", "ingress"],
        ["automate", "security best practices", "infrastructure as code"],
        ["prepared statements", "parameterized queries", "sql injection"],
        ["command injection", "parameterization", "operating system commands"],
        ["allowlist", "allow-list", "input validation"],
        ["server side request forgery", "ssrf", "remote resource"],
        ["path traversal", "cwe-22", "broken access control"],
        ["broken object level authorization", "object-level authorization", "authentication"],
        ["secrets management", "secrets", "credentials"],
        ["sensitive information", "passwords", "access tokens", "logs"],
        ["resource consumption", "rate limit", "request size"],
        ["least privilege", "database permissions", "parameterized queries"],
    ]
    by_id = {
        "iam-leastpriv-001": groups[0], "iam-wildcard-002": groups[1],
        "iam-trust-003": groups[2], "iam-escalate-004": groups[3], "iam-creds-005": groups[4],
        "s3-public-001": groups[5], "s3-encrypt-002": groups[6], "s3-delete-003": groups[7],
        "vpc-sgroup-001": groups[8], "dev-change-002": groups[9], "app-sqli-001": groups[10],
        "app-errors-010": ["generic response", "stack trace", "internal details", "server-side"],
        "app-output-011": ["output encoding", "HTML Contexts", "JavaScript Contexts", "URL Contexts"],
        "app-deps-012": ["software supply chain", "dependencies", "vulnerabilities", "software bill of materials"],
        "dev-verify-001": ["PW.7", "PW.8", "secure software development framework"],
        "app-cmdi-002": groups[11], "app-input-003": groups[12], "app-ssrf-004": groups[13],
        "app-path-005": groups[14], "app-authz-006": groups[15], "app-secrets-007": groups[16],
        "app-logging-008": groups[17], "app-resource-009": groups[18], "db-access-001": groups[19],
    }
    return by_id.get(cid, [w for w in re.findall(r"[a-z]{4,}", section) if w not in {"section", "source", "owasp", "amazon", "best", "practices"}][:6])


def _section_matches(control: Mapping[str, Any], text: str, title: str, content_type: str) -> bool:
    """True when enough of a control's known section cues appear in fetched page text."""
    if not text or "pdf" in content_type.lower():
        return False
    haystack = (title + " " + text).lower()
    terms = _required_terms(control)
    hits = sum(1 for term in terms if term.lower() in haystack)
    # A single cue is only enough for a one- or two-term page-specific section check;
    # broader sources require two independent cues. Control text is not modified by this check.
    threshold = 1 if len(terms) <= 2 else 2
    return hits >= threshold


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SourceSyncError(f"cannot read catalogue {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("controls"), list):
        raise SourceSyncError(f"catalogue {path} is not a controls mapping")
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _bump_patch(version: str) -> str:
    parts = version.split(".")
    if len(parts) != 3 or not all(x.isdigit() for x in parts):
        raise SourceSyncError(f"invalid semantic catalogue version: {version}")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def sync_catalogue(*, catalogue_path: Path = CATALOGUE_PATH,
                   snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
                   candidate_dir: Path = DEFAULT_CANDIDATE_DIR,
                   state_path: Path = DEFAULT_STATE_PATH,
                   backup_dir: Path = DEFAULT_BACKUP_DIR,
                   activate: bool = False,
                   approved_controls: Optional[Iterable[str]] = None,
                   fetcher: Callable[[str], Dict[str, Any]] = fetch_source) -> Dict[str, Any]:
    """Retrieve sources and create a validated candidate; activate only when explicitly requested."""
    catalogue_path = Path(catalogue_path)
    snapshot_dir, candidate_dir, state_path, backup_dir = map(Path, (snapshot_dir, candidate_dir, state_path, backup_dir))
    approved = set(approved_controls or ())
    data = _read_yaml(catalogue_path)
    original_version = str(data.get("catalogue_version", "0.0.0"))
    old_controls = {c.get("control_id"): copy.deepcopy(c) for c in data["controls"]}
    state: Dict[str, Any] = {"schema_version": 1, "sources": {}, "last_attempt": None}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
        except (OSError, json.JSONDecodeError):
            # A corrupt state file is not trusted; preserving the catalogue is safer than guessing.
            state["state_recovered_from_corruption"] = True
    source_map: Dict[str, List[Dict[str, Any]]] = {}
    for control in data["controls"]:
        url = str(control.get("source_url", ""))
        if url.startswith("https://"):
            source_map.setdefault(url, []).append(control)
    report: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "catalogue": str(catalogue_path), "from_version": original_version,
        "source_count": len(source_map), "available": [], "unchanged": [], "changed": [],
        "unavailable": [], "section_not_verified": [], "demoted_controls": [],
        "activated_controls": [], "candidate": None, "activated": False,
    }
    nowstamp = _stamp()
    changed_data = False
    for url, controls in sorted(source_map.items()):
        source_before = state.get("sources", {}).get(url, {})
        try:
            result = fetcher(url)
            body = result.get("body")
            if not isinstance(body, bytes) or not body:
                raise SourceSyncError("fetcher returned no byte body")
            final_url = str(result.get("final_url") or url)
            _check_url(final_url)
            content_type = str(result.get("content_type") or "text/html")
            title_text, title = _normalise_content(body, content_type)
            if not title_text and "pdf" not in content_type.lower() and body[:4] != b"%PDF":
                raise SourceSyncError("could not extract text from source page")
            raw_hash = sha256_bytes(body)
            snapshot_name = f"{urllib.parse.urlsplit(url).hostname}_{hashlib.sha256(url.encode()).hexdigest()[:10]}_{raw_hash[7:19]}{'.pdf' if 'pdf' in content_type.lower() or body[:4] == b'%PDF' else '.html'}"
            snapshot_path = snapshot_dir / snapshot_name
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(body)
            old_hash = source_before.get("raw_sha256")
            state_source_status = "new_baseline" if not old_hash else ("unchanged" if old_hash == raw_hash else "changed")
            report["available"].append({"url": url, "title": title, "sha256": raw_hash,
                                        "status": state_source_status, "snapshot": str(snapshot_path)})
            if state_source_status == "unchanged":
                report["unchanged"].append(url)
            elif state_source_status == "changed":
                report["changed"].append(url)
            passed, failed = [], []
            for control in controls:
                if _section_matches(control, title_text, title, content_type):
                    passed.append(control)
                else:
                    failed.append(control)
                    report["section_not_verified"].append({"control_id": control["control_id"], "source_url": url,
                                                            "section": control.get("source_version_or_section"),
                                                            "reason": "expected source-section cues were not found; requires manual review"})
            for control in passed:
                control["retrieved_at"] = str(result.get("retrieved_at") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
                control["content_hash"] = raw_hash
                control["provenance_status"] = "snapshot_verified"
                control["source_status"] = "available"
                control["source_name"] = str(control.get("source_name") or title or urllib.parse.urlsplit(url).hostname)
                control["_sync_snapshot"] = str(snapshot_path.relative_to(PACKAGE_DIR)) if snapshot_path.is_relative_to(PACKAGE_DIR) else str(snapshot_path)
                if control.get("active_status") == "draft" and control.get("control_id") in approved:
                    control["active_status"] = "active"
                    report["activated_controls"].append(control["control_id"])
                changed_data = True
            for control in failed:
                # Candidate-only demotion: never silently delete last known-good provenance, but keep a
                # source-status signal so policy can warn/skip until an operator reviews it.
                if control.get("active_status") == "active":
                    control["active_status"] = "draft"
                    control["provenance_status"] = "unverified"
                    control["retrieved_at"] = None
                    control["content_hash"] = None
                    report["demoted_controls"].append(control["control_id"])
                    changed_data = True
            state.setdefault("sources", {})[url] = {
                "raw_sha256": raw_hash, "last_retrieved_at": result.get("retrieved_at"),
                "final_url": final_url, "content_type": content_type, "snapshot": str(snapshot_path),
                "status": state_source_status,
            }
        except Exception as exc:  # network/content failures are recorded; the known-good catalogue stays usable
            report["unavailable"].append({"url": url, "reason": f"{type(exc).__name__}: {exc}"})
            state.setdefault("sources", {}).setdefault(url, {}).update({"status": "unavailable", "last_error": f"{type(exc).__name__}: {exc}"})
            for control in controls:
                # Keep content hash/provenance: it identifies the last-known-good snapshot. A network
                # outage does not invalidate previously verified evidence; it does mark the source down.
                control["source_status"] = "unavailable"
                changed_data = True
    # Drop private synchroniser field from catalogue, retaining only its snapshot file in the tree.
    for control in data["controls"]:
        control.pop("_sync_snapshot", None)
    data["catalogue_version"] = _bump_patch(original_version)
    data["updated_at"] = datetime.now(timezone.utc).date().isoformat()
    sync_incomplete = bool(report["unavailable"] or report["section_not_verified"])
    data["freshness"] = {
        **(data.get("freshness") or {}),
        "last_sync_attempt": report["started_at"],
        "last_sync_status": "failed" if sync_incomplete else "ok",
        "last_known_good_version": original_version,
        "note": "Sync creates a candidate only. Activate after source review; hashes identify exact local raw snapshots. A partial sync cannot be activated.",
    }
    data.setdefault("changelog", []).insert(0, {
        "version": data["catalogue_version"], "date": data["updated_at"],
        "change": "Source synchronisation metadata refreshed. Candidate requires operator review; control wording is not auto-rewritten.",
    })
    # Candidate validation before any activation. Unknown draft source documents remain drafts.
    try:
        parse_catalogue(data, scope=load_scope(), path=catalogue_path)
    except CatalogueError as exc:
        report["validation_errors"] = exc.problems
        report["candidate_rejected"] = True
        state["last_attempt"] = {"at": report["started_at"], "status": "failed_validation", "report": report}
        _write_json(state_path, state)
        report_path = state_path.parent / f"source_sync_report_{nowstamp}.json"
        _write_json(report_path, report)
        raise SourceSyncError("candidate catalogue failed validation: " + "; ".join(exc.problems[:8])) from exc
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"security_controls_candidate_{nowstamp}.yaml"
    candidate_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")
    report["candidate"] = str(candidate_path)
    if activate and not sync_incomplete:
        current = _read_yaml(catalogue_path)
        parse_catalogue(current, scope=load_scope(), path=catalogue_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"security_controls_{original_version}_{nowstamp}.yaml"
        shutil.copy2(catalogue_path, backup_path)
        shutil.copy2(candidate_path, catalogue_path)
        report["activated"] = True
        report["backup"] = str(backup_path)
        report["to_version"] = data["catalogue_version"]
    else:
        report["to_version"] = original_version if activate else data["catalogue_version"]
        if activate and sync_incomplete:
            report["activation_blocked_reason"] = (
                "Sync is incomplete: one or more source URLs were unavailable or expected section cues "
                "were not found. Candidate was retained for review; active catalogue and last-known-good "
                "version were not changed."
            )
    report_path = state_path.parent / f"source_sync_report_{nowstamp}.json"
    report["report_path"] = str(report_path)
    attempt_status = "partial" if sync_incomplete else ("activated" if report["activated"] else "candidate_created")
    state["last_attempt"] = {"at": report["started_at"], "status": attempt_status, "report": report}
    _write_json(state_path, state)
    _write_json(report_path, report)
    return report


def rollback_catalogue(*, catalogue_path: Path = CATALOGUE_PATH, backup_dir: Path = DEFAULT_BACKUP_DIR) -> Path:
    """Restore the latest synchroniser backup after validating it."""
    catalogue_path, backup_dir = Path(catalogue_path), Path(backup_dir)
    backups = sorted(backup_dir.glob("security_controls_*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        raise SourceSyncError(f"no rollback backup found in {backup_dir}")
    backup = backups[0]
    data = _read_yaml(backup)
    parse_catalogue(data, scope=load_scope(), path=backup)
    catalogue_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, catalogue_path)
    return backup


def current_status(*, catalogue_path: Path = CATALOGUE_PATH, state_path: Path = DEFAULT_STATE_PATH) -> Dict[str, Any]:
    data = _read_yaml(Path(catalogue_path))
    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8")) if Path(state_path).exists() else {}
    except (OSError, json.JSONDecodeError):
        state = {"state_file_warning": "missing or invalid state file"}
    return {
        "catalogue_version": data.get("catalogue_version"),
        "freshness": data.get("freshness", {}),
        "active_controls": sum(c.get("active_status") == "active" for c in data["controls"]),
        "verified_active_controls": sum(c.get("active_status") == "active" and c.get("provenance_status") in {"section_checked", "snapshot_verified"} for c in data["controls"]),
        "active_unverified_controls": [c.get("control_id") for c in data["controls"] if c.get("active_status") == "active" and c.get("provenance_status") == "unverified"],
        "source_states": {url: source.get("status") for url, source in state.get("sources", {}).items()},
        "network_access_on_status": False,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m security_hardening.source_sync",
        description="Explicit source synchronisation for the security-control catalogue; no automatic activation.")
    subs = parser.add_subparsers(dest="command", required=True)
    status_parser = subs.add_parser("status", help="show local provenance/sync state without network access")
    status_parser.add_argument("--catalogue", type=Path, default=CATALOGUE_PATH)
    status_parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    sync_parser = subs.add_parser("sync", help="fetch allow-listed sources and create a validated candidate")
    sync_parser.add_argument("--catalogue", type=Path, default=CATALOGUE_PATH)
    sync_parser.add_argument("--activate", action="store_true", help="activate the validated candidate after backup; use only after review")
    sync_parser.add_argument("--approve", action="append", default=[], metavar="CONTROL_ID", help="explicitly permit a draft control to activate if its source section verifies")
    sync_parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="per-source request timeout in seconds")
    roll_parser = subs.add_parser("rollback", help="restore the most recent synchroniser backup")
    roll_parser.add_argument("--catalogue", type=Path, default=CATALOGUE_PATH)
    roll_parser.add_argument("--backups", type=Path, default=DEFAULT_BACKUP_DIR)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = current_status(catalogue_path=args.catalogue, state_path=args.state)
        elif args.command == "rollback":
            restored = rollback_catalogue(catalogue_path=args.catalogue, backup_dir=args.backups)
            result = {"restored_from": str(restored), "catalogue": str(args.catalogue)}
        else:
            def timed_fetch(url: str) -> Dict[str, Any]:
                return fetch_source(url, timeout=args.timeout)
            result = sync_catalogue(catalogue_path=args.catalogue, activate=args.activate,
                                    approved_controls=args.approve, fetcher=timed_fetch)
    except (SourceSyncError, CatalogueError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
