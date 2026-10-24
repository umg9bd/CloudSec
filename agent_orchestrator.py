"""
agent_orchestrator.py
=====================
The agent security gate, end to end: one prompt in, an approved (or refused) AWS action out.

    Stage 1  prompt_injection_gate      screen the prompt; MALICIOUS / REVIEW_REQUIRED stops here,
                                        with the reason
    Stage 2  prompt_hardening_realtime  a SAFE prompt is dropped into its incoming/ folder; its
                                        watcher writes the hardened prompt template to output/
    Stage 3  log-generation             predict the AWS calls and the CloudTrail events they emit
    Stage 4  pipeline.py models         score those predicted events with HGT + LSTM against the
                                        history the live pipeline last saved -- a dry run, never saved
    Execute  (simulated)                write the approved events, freshly stamped, into incoming/:
                                        the real-time pipeline scores them for real, adds them to the
                                        agent's history, and the dashboard shows them

A revision goes round the whole loop again: the edited request is re-screened (Stage 1), re-hardened
(Stage 2), re-predicted (Stage 3) and re-scored (Stage 4). The security section Stage 2 appends is
never taken from the user's edit -- it is regenerated from the request every round, so a clause
cannot be removed by deleting it from the template.

Execution is simulated: no AWS API is called. What runs is exactly what was approved -- execute()
refuses events whose content (calls, targets, parameters, principal) differs from the plan Stage 4
approved; only the timestamps and IDs a real execution would create are new.

agent_console.py is the interactive UI over this module. Headless:

    python agent_orchestrator.py "Create a private S3 bucket named audit-logs-prod with versioning."
    python agent_orchestrator.py --execute "..."      # also execute it if every stage approves
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "graph_construction", ROOT / "temporal-analysis",
           ROOT / "prompt_hardening_realtime", ROOT / "log-generation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

STAGE2_ROOT = ROOT / "prompt_hardening_realtime"
STAGE2_INCOMING = STAGE2_ROOT / "incoming"
STAGE2_OUTPUT = STAGE2_ROOT / "output"
PIPELINE_INCOMING = ROOT / "incoming"
PIPELINE_STAGING = ROOT / ".incoming_staging"     # same staging folder feed_incoming.py uses
AUDIT_LOG = ROOT / "output" / "agent_audit.jsonl"

DEFAULT_PRINCIPAL = "arn:aws:iam::111122223333:user/agentic-operator"
AGENT_SOURCE_IP = "10.20.30.40"
AGENT_USER_AGENT = "cloudsec-agent/1.0 (simulated execution)"
STAGE2_PROCEED = {"HARDENED", "ALREADY_ADEQUATE"}
SECURITY_BLOCK_PREFIX = "--- SECURITY HARDENING"   # prompt_hardener.BLOCK_BEGIN, without its version
STAGE2_WAIT_SECONDS = 20.0


# ── small helpers (no model stack needed; unit-tested) ────────────────────────

def split_template(text: str) -> tuple[str, str]:
    """(request, security_section) of a Stage 2 prompt template. The request is the part a user
    may edit; the security section is CloudSec's and is regenerated on every revision."""
    i = text.find(SECURITY_BLOCK_PREFIX)
    if i < 0:
        return text.strip(), ""
    return text[:i].strip(), text[i:].strip()


def _status(value: Any) -> str:
    """Enum or string -> its plain string (Stage 3 dumps pydantic enums as Enum members)."""
    return str(getattr(value, "value", value))


def materialize_events(records: list[dict], principal_arn: str,
                       start: Optional[dt.datetime] = None, spacing_seconds: int = 2) -> list[dict]:
    """Stage 3's predicted records, as the CloudTrail an execution starting at `start` would log:
    fresh eventTime/eventID/requestID, the agent's identity, source IP and user agent. The content
    that was assessed -- event name/source, region, request parameters -- is left untouched."""
    start = start or dt.datetime.now(dt.timezone.utc)
    account = principal_arn.split(":")[4] if principal_arn.count(":") >= 5 else "111122223333"
    user = principal_arn.rsplit("/", 1)[-1]
    out = []
    for i, rec in enumerate(records):
        r = copy.deepcopy(rec)
        r["eventTime"] = (start + dt.timedelta(seconds=spacing_seconds * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r["eventID"] = str(uuid.uuid4())
        r["requestID"] = str(uuid.uuid4())
        r["userIdentity"] = {"type": "IAMUser", "principalId": "AIDA" + hashlib.sha1(principal_arn.encode()).hexdigest()[:16].upper(),
                             "arn": principal_arn, "accountId": account, "userName": user}
        r["recipientAccountId"] = account
        r["sourceIPAddress"] = AGENT_SOURCE_IP
        r["userAgent"] = AGENT_USER_AGENT
        out.append(r)
    return out


def plan_fingerprint(records: list[dict]) -> str:
    """Hash of what an execution DOES: the principal, each call and its target and parameters, in
    order. Timestamps and IDs are excluded, so the approved plan and its execution match."""
    keep = [{"principal": (r.get("userIdentity") or {}).get("arn"), "source": r.get("eventSource"),
             "name": r.get("eventName"), "region": r.get("awsRegion"), "params": r.get("requestParameters")}
            for r in records]
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def plan_blockers(stage2: Optional[dict], stage3: Optional[dict], assessment: Optional[dict]) -> list[str]:
    """Why this round may NOT execute; [] means every stage approved it."""
    reasons = []
    if not stage2:
        return ["Stage 2 has not produced a hardened template."]
    s2 = stage2.get("status")
    if s2 not in STAGE2_PROCEED:
        reasons.append(f"Stage 2 returned {s2}: the request must be revised before it can run.")
    for q in stage2.get("clarification_questions", []):
        if q.get("blocking"):
            reasons.append(f"Stage 2 needs an answer: {q.get('question')}")
    if not stage3:
        return reasons + ["Stage 3 has not predicted the AWS activity."]
    s3 = _status((stage3.get("stage3_contract") or {}).get("status"))
    if s3 != "GENERATED":
        reasons.append(f"Stage 3 returned {s3}: the predicted activity is incomplete, so it cannot be "
                       "approved (see its open questions).")
    if not (stage3.get("downstream_lstm_records") or {}).get("Records"):
        reasons.append("Stage 3 predicted no CloudTrail events to assess.")
    if not assessment:
        return reasons + ["Stage 4 has not scored the predicted events."]
    if assessment.get("error"):
        reasons.append(f"Stage 4 could not score the events ({assessment['error']}).")
    for e in assessment.get("events", []):
        if e["fast_lane"]:
            reasons.append(f"{e['event_name']}: {e['fast_lane_reason']} (fast-lane rule).")
        elif e["alert"]:
            reasons.append(f"{e['event_name']} scored risk {e['risk_score']:.2f}/10, at or above the alert "
                           f"threshold {assessment['threshold'] * 10:.2f}/10.")
    return reasons


def audit(event: str, **fields) -> None:
    """One JSON line per pipeline step. The prompt itself is never written, only its hash."""
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "event": event, **fields}
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass   # the audit trail must never stop the gate


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _atomic_json(dest: Path, payload: Any, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=staging)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)       # one rename: the watcher never sees a half-written file
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Stage 4: dry-run scorer over the live pipeline's models ───────────────────

class DryRunScorer:
    """HGT + LSTM scores for hypothetical events, against the state the live pipeline last saved.

    Wraps a pipeline.Pipeline built with write_outputs=False: it never writes scores, alerts or
    state, and never emails. Before each assessment it re-reads the stored state, so the score
    reflects the agent's real history -- and anything it scored is forgotten with the next reload."""

    def __init__(self):
        import pipeline as pl                      # torch + checkpoints: only when Stage 4 is used
        self.pl = pl
        self.p = pl.Pipeline(pl.PipelineConfig.load(), write_outputs=False)
        self.cfg = self.p.cfg
        self._lock = threading.Lock()

    def _reload(self):
        for attempt in range(5):                   # the watcher may be mid-save
            try:
                self.p.reload_state()
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.3)

    def assess(self, records: list[dict], name: str) -> dict:
        import feature_engine9 as fe9
        with self._lock:
            self._reload()
            rows = [(i, fe9.normalize_cloudtrail_row(r)) for i, r in enumerate(records)]
            new = self.p.featurize(rows, name)
            if new.empty:
                return {"events": [], "threshold": self.cfg.alert_threshold, "max_risk": 0.0,
                        "error": "no event could be featurised"}
            scored = self.p.score(new)
            flagged = scored[scored["alert"] | scored["fast_lane"]]
            explanations = self.p.explain(flagged) if len(flagged) else {}
            # Preserve the exact structural input window used by Pipeline.score(). The proposed
            # dry-run rows are in memory only; this copy is read-only enrichment data and is not
            # written to the live history or any output file.
            scored_inputs = getattr(self.p, "_scored_inputs", {}) or {}
            graph_rows = scored_inputs.get("graph_rows")
            graph_window = graph_rows.copy(deep=True) if hasattr(graph_rows, "copy") else None
        events = []
        for r in scored.to_dict("records"):
            exp = explanations.get(str(r["log_id"])) or {}
            is_flagged = bool(r["alert"] or r["fast_lane"])
            blast_radius = None
            if is_flagged:
                # Blast-radius enrichment must never affect the model score or fail closed/open
                # behavior. Failures are explicitly returned as unavailable metadata.
                try:
                    from blast_radius_ui import calculate_event_blast_radius
                    blast_radius = calculate_event_blast_radius(
                        graph_window if graph_window is not None else [],
                        source_node=r.get("source_node"),
                        principal_arn=r.get("principal_arn"),
                        username=r.get("username"),
                    )
                except Exception as exc:
                    blast_radius = {
                        "status": "calculation_failed",
                        "message": f"Blast-radius calculation failed: {type(exc).__name__}: {exc}",
                        "limitation": (
                            "Observed reachability from CloudTrail-derived graph data only; this is not proof "
                            "of all permissions granted by IAM policies."
                        ),
                        "window_rows": 0, "origin": None, "score": None,
                        "reachable_count": None, "critical_count": None,
                        "asset_categories": {}, "reachable_resources": [], "paths": [],
                        "snapshot": {"nodes": [], "edges": []},
                    }
            events.append({
                "log_id": str(r["log_id"]),
                "event_name": r["event_name"], "target": r["target_node"], "edge_type": r["edge_type"],
                "source_node": r.get("source_node"), "principal_arn": r.get("principal_arn"),
                "username": r.get("username"), "timestamp": str(r.get("timestamp")),
                "p_graph": None if r["p_graph"] != r["p_graph"] else round(float(r["p_graph"]), 4),
                "p_sequence": round(float(r["p_sequence"]), 4),
                "risk": round(float(r["risk"]), 4), "risk_score": float(r["risk_score"]),
                "alert": bool(r["alert"]), "fast_lane": bool(r["fast_lane"]),
                "fast_lane_reason": fe9.CRITICAL_ACTIONS.get(r["event_name"]) if r["fast_lane"] else None,
                "why": exp.get("summary"), "reasons": exp.get("reasons", []),
                # Keep the original structured graph/sequence explanation alongside the legacy
                # summary and reasons so callers can display real feature/event attribution.
                "explanation": copy.deepcopy(exp) if exp else None,
                "blast_radius": blast_radius,
            })
        return {"events": events, "threshold": self.cfg.alert_threshold,
                "weight_graph": self.cfg.weight_graph,
                "max_risk": max((e["risk"] for e in events), default=0.0)}


# ── the gate ──────────────────────────────────────────────────────────────────

@dataclass
class Round:
    """One pass of a request through the gate. A revision is a new Round."""
    request: str
    request_id: Optional[str] = None
    stage1: Optional[dict] = None
    stage2: Optional[dict] = None
    stage2_via: Optional[str] = None
    stage3: Optional[dict] = None
    assessment: Optional[dict] = None
    approved_records: list = field(default_factory=list)
    fingerprint: Optional[str] = None
    blockers: list = field(default_factory=list)
    stopped_at: Optional[str] = None
    execution: Optional[dict] = None

    @property
    def approved(self) -> bool:
        return self.stopped_at is None and not self.blockers and bool(self.approved_records)


class AgentGate:
    def __init__(self, principal_arn: str = DEFAULT_PRINCIPAL, stage2_wait: float = STAGE2_WAIT_SECONDS):
        from prompt_injection_gate.config import load_config
        from prompt_injection_gate.pipeline_bridge import PipelineBridge
        cfg = load_config()
        cfg.stage2_incoming_dir = str(STAGE2_INCOMING)
        cfg.stage2_output_dir = str(STAGE2_OUTPUT)
        cfg.stage2_root = str(STAGE2_ROOT)
        cfg.stage3_dir = str(ROOT / "log-generation")
        self.bridge = PipelineBridge(cfg)
        self.principal_arn = principal_arn
        self.stage2_wait = stage2_wait
        self._scorer: Optional[DryRunScorer] = None
        self._scorer_lock = threading.Lock()

    @property
    def classifier_ready(self) -> bool:
        return bool(self.bridge.service.classifier.available)

    def scorer(self) -> DryRunScorer:
        with self._scorer_lock:
            if self._scorer is None:
                self._scorer = DryRunScorer()
            return self._scorer

    # Stage 1
    def screen(self, prompt: str):
        from prompt_injection_gate.schemas import ScreeningRequest
        resp = self.bridge.service.screen(ScreeningRequest(prompt=prompt))
        return resp, {
            "decision": resp.decision.value, "request_id": resp.request_id,
            "explanation": resp.explanation, "threat_categories": list(resp.threat_categories),
            "triggered_rule_ids": list(resp.triggered_rule_ids), "reason_codes": list(resp.reason_codes),
            "attack_score": resp.classifier_scores.get("attack"),
            "classifier_version": resp.classifier_version, "latency_ms": resp.latency_ms,
        }

    # Stage 2: through the watcher's inbox, as deployed; in process if no watcher answers
    def harden(self, resp) -> tuple[Optional[dict], str]:
        delivered = self.bridge.forward(resp)          # writes only on SAFE
        if delivered is None:
            return None, "not delivered"
        out = STAGE2_OUTPUT / f"{delivered.stem}.response.json"
        deadline = time.time() + self.stage2_wait
        while time.time() < deadline:
            if out.exists():
                try:
                    return json.loads(out.read_text(encoding="utf-8")), "watcher"
                except (json.JSONDecodeError, OSError):
                    pass                               # being replaced; read again
            time.sleep(0.2)
        request = {"request_id": resp.request_id, "screening_status": self.bridge.cfg.handoff_screening_status,
                   "original_prompt": resp.safe_prompt_for_next_stage}
        return self.bridge.run_stage2_inline(request), "in process (no Stage 2 watcher answered)"

    # Stage 3
    def predict(self, original: str, hardened: str, request_id: str) -> Optional[dict]:
        adapter = self.bridge._stage3_adapter()
        if adapter is None:
            return None
        return adapter.process_stage2_output(original_prompt=original, hardened_prompt=hardened,
                                             principal_arn=self.principal_arn, request_id=request_id)

    def run(self, request: str, on_stage=None) -> Round:
        """Stages 1-4 for one request. Stops at the first stage that refuses it."""
        step = on_stage or (lambda name: None)
        rnd = Round(request=request.strip())
        step("Stage 1 - screening the prompt")
        resp, rnd.stage1 = self.screen(rnd.request)
        rnd.request_id = resp.request_id
        audit("stage1", request_id=rnd.request_id, prompt_sha=_sha(rnd.request), decision=rnd.stage1["decision"],
              threats=rnd.stage1["threat_categories"], rules=rnd.stage1["triggered_rule_ids"])
        if not resp.handoff_allowed:
            rnd.stopped_at = "stage1"
            return rnd

        step("Stage 2 - hardening the prompt (incoming/ -> watcher)")
        rnd.stage2, rnd.stage2_via = self.harden(resp)
        audit("stage2", request_id=rnd.request_id, via=rnd.stage2_via,
              status=(rnd.stage2 or {}).get("status"),
              added=[c.get("control_id") for c in (rnd.stage2 or {}).get("added_controls", [])])
        if not rnd.stage2 or rnd.stage2.get("status") not in STAGE2_PROCEED or not rnd.stage2.get("revised_prompt"):
            rnd.stopped_at = "stage2"
            rnd.blockers = plan_blockers(rnd.stage2, None, None)[:1]
            return rnd

        step("Stage 3 - predicting the AWS calls and CloudTrail events")
        rnd.stage3 = self.predict(rnd.request, rnd.stage2["revised_prompt"], rnd.request_id)
        contract = (rnd.stage3 or {}).get("stage3_contract") or {}
        audit("stage3", request_id=rnd.request_id, status=_status(contract.get("status")),
              events=(rnd.stage3 or {}).get("downstream_event_sequence"))
        records = ((rnd.stage3 or {}).get("downstream_lstm_records") or {}).get("Records") or []
        if not rnd.stage3 or _status(contract.get("status")) != "GENERATED" or not records:
            rnd.stopped_at = "stage3"
            rnd.blockers = plan_blockers(rnd.stage2, rnd.stage3, None)
            return rnd

        step("Stage 4 - scoring the predicted events with HGT + LSTM")
        planned = materialize_events(records, self.principal_arn)
        try:
            rnd.assessment = self.scorer().assess(planned, f"dryrun_{rnd.request_id}")
        except Exception as e:                         # a scorer failure must refuse, not approve
            rnd.assessment = {"events": [], "threshold": None, "max_risk": None,
                              "error": f"{type(e).__name__}: {e}"}
        rnd.blockers = plan_blockers(rnd.stage2, rnd.stage3, rnd.assessment)
        if rnd.blockers:
            rnd.stopped_at = "stage4"
        else:
            rnd.approved_records = planned
            rnd.fingerprint = plan_fingerprint(planned)
        audit("stage4", request_id=rnd.request_id, max_risk=rnd.assessment.get("max_risk"),
              approved=rnd.approved, blockers=rnd.blockers)
        return rnd

    # Execute (simulated)
    def execute(self, rnd: Round) -> dict:
        """Writes the approved plan's CloudTrail, as executed now, into the pipeline's incoming/."""
        if not rnd.approved:
            raise PermissionError("this request was not approved by every stage; it cannot execute")
        executed = materialize_events(rnd.approved_records, self.principal_arn)
        if plan_fingerprint(executed) != rnd.fingerprint:
            raise PermissionError("the execution differs from the approved plan; refusing to run it")
        name = f"agent_{rnd.request_id}.json"
        _atomic_json(PIPELINE_INCOMING / name, {"Records": executed}, PIPELINE_STAGING)
        rnd.execution = {"file": name, "events": len(executed),
                         "executed_at": executed[0]["eventTime"] if executed else None}
        audit("execute", request_id=rnd.request_id, file=name, events=len(executed), fingerprint=rnd.fingerprint)
        return rnd.execution


def execution_status(file_name: str) -> dict:
    """Where the real-time pipeline is with an executed file, and its live scores once it has them."""
    import csv
    state = ("waiting" if (PIPELINE_INCOMING / file_name).exists()
             or (PIPELINE_INCOMING / ".processing" / file_name).exists()
             else "scored" if (PIPELINE_INCOMING / "processed" / file_name).exists()
             else "failed" if (PIPELINE_INCOMING / "failed" / file_name).exists() else "unknown")
    rows = []
    scores = ROOT / "output" / "risk_scores.csv"
    if state == "scored" and scores.exists():
        try:
            with open(scores, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r.get("log_id", "").startswith(file_name + ":")]
        except OSError:
            pass
    return {"state": state, "rows": rows}


# ── headless CLI ──────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one request through the CloudSec agent gate.")
    ap.add_argument("prompt")
    ap.add_argument("--execute", action="store_true", help="execute (simulated) if every stage approves")
    ap.add_argument("--principal", default=DEFAULT_PRINCIPAL)
    args = ap.parse_args(argv)

    gate = AgentGate(principal_arn=args.principal)
    rnd = gate.run(args.prompt, on_stage=lambda s: print(f"... {s}", flush=True))
    s1 = rnd.stage1
    print(f"\nStage 1  {s1['decision']}: {s1['explanation']}")
    if rnd.stopped_at == "stage1":
        return 1
    print(f"Stage 2  {rnd.stage2.get('status') if rnd.stage2 else None} (via {rnd.stage2_via}); "
          f"controls added: {[c['control_id'] for c in (rnd.stage2 or {}).get('added_controls', [])]}")
    if rnd.stage3:
        c = rnd.stage3["stage3_contract"]
        print(f"Stage 3  {_status(c['status'])}: {' -> '.join(rnd.stage3['downstream_event_sequence']) or '(no events)'}")
    if rnd.assessment:
        for e in rnd.assessment["events"]:
            hgt = "n/a" if e["p_graph"] is None else f"{e['p_graph']:.3f}"
            print(f"Stage 4  {e['event_name']:<28} HGT {hgt}  LSTM {e['p_sequence']:.3f}  "
                  f"risk {e['risk_score']:.2f}/10{'  ALERT' if e['alert'] else ''}")
    if rnd.blockers:
        print("\nNOT APPROVED:\n  - " + "\n  - ".join(rnd.blockers))
        return 1
    print("\nAPPROVED: every stage passed.")
    if args.execute:
        ex = gate.execute(rnd)
        print(f"Executed (simulated): {ex['events']} event(s) -> incoming/{ex['file']}")
        for _ in range(60):
            st = execution_status(ex["file"])
            if st["state"] in ("scored", "failed"):
                print(f"Real-time pipeline: {st['state']}; {len(st['rows'])} event(s) in output/risk_scores.csv")
                break
            time.sleep(1)
        else:
            print("Real-time pipeline has not picked it up yet (is pipeline.py --watch incoming running?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
