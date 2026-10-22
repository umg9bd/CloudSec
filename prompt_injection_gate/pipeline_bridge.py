"""
pipeline_bridge.py
==================
Connects the Stage-1 screening gate to the stages that follow it, without
modifying any file in those stages.

    Stage 1 (this module)        screen a prompt
         |  SAFE only
         v
    prompt_hardening_realtime/incoming/<request_id>.json      <-- the integration seam
         |
         v
    Stage 2 (security_hardening) harden the prompt -> output/<id>.response.json
         |  HARDENED / ALREADY_ADEQUATE
         v
    Stage 3 (log-generation)     predict the CloudTrail events the action would emit

THE DELIVERY RULE
-----------------
A prompt is handed on *only* when Stage 1 returns SAFE. That is not a second
check here -- `ScreeningResponse` already guarantees `handoff_allowed` is true
iff the decision is SAFE and that `safe_prompt_for_next_stage` is the original
prompt byte-for-byte. `forward()` simply obeys that invariant: SAFE is written,
MALICIOUS and REVIEW_REQUIRED are not. Stage 2's own policy independently only
admits an approved status, so a delivered file is always an approved, SAFE one.

The request file is written with the same atomic temp-then-rename pattern Stage 2
documents, so a watcher never sees a half-written file. Screening itself stays a
pure, side-effect-free computation; delivery is this separate, explicit step.

The Stage 2 / Stage 3 calls are imported lazily and fail soft: if a neighbour
module or its dependency is absent, the SAFE prompt is still delivered to
`incoming/` (the real deployment seam) and the in-process demo of that stage is
reported as unavailable rather than raising.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from .config import GateConfig, load_config
from .schemas import ContextItem, Decision, ScreeningRequest, ScreeningResponse
from .screening_service import ScreeningService

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def _safe_stem(request_id: str) -> str:
    """A filesystem-safe request filename stem (Stage 2 keys output on it)."""
    stem = _SAFE_NAME.sub("-", request_id).strip("-") or f"pig-{uuid.uuid4().hex[:12]}"
    return stem[:200]


class PipelineBridge:
    """Deliver SAFE prompts to Stage 2 and (optionally) run the chain end to end."""

    def __init__(self, cfg: Optional[GateConfig] = None, service: Optional[ScreeningService] = None):
        self.cfg = cfg or load_config()
        self.service = service or ScreeningService(self.cfg)
        self._stage2 = None          # lazily built HardeningService (None => tried and unavailable)
        self._stage2_tried = False
        self._stage3 = None          # lazily built Stage3PipelineAdapter
        self._stage3_tried = False

    # -- the integration seam: write a SAFE prompt into Stage 2's inbox ---------

    def _atomic_write_request(self, request_id: str, prompt: str) -> Path:
        incoming = Path(self.cfg.stage2_incoming_dir)
        incoming.mkdir(parents=True, exist_ok=True)
        dest = incoming / f"{_safe_stem(request_id)}.json"
        payload = {
            "request_id": request_id,
            "screening_status": self.cfg.handoff_screening_status,   # "SAFE" (only reached on SAFE)
            "original_prompt": prompt,
        }
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=incoming)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dest)     # atomic on the same filesystem; watcher never sees a partial file
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return dest

    def forward(self, resp: ScreeningResponse) -> Optional[Path]:
        """Deliver to Stage 2 iff the decision is SAFE. Returns the written path, else None."""
        if not self.cfg.deliver_on_safe:
            return None
        if not resp.handoff_allowed:              # True iff SAFE (enforced in ScreeningResponse)
            return None
        rid = resp.request_id or f"pig-{uuid.uuid4().hex[:12]}"
        # safe_prompt_for_next_stage is the ORIGINAL prompt, unaltered, and is
        # non-None exactly when handoff is allowed.
        return self._atomic_write_request(rid, resp.safe_prompt_for_next_stage or "")

    def screen_and_forward(self, prompt: str, untrusted_context: Optional[list] = None,
                           session_id: Optional[str] = None) -> dict[str, Any]:
        """Screen one prompt; deliver it to Stage 2 if (and only if) it is SAFE."""
        ctx = []
        for c in untrusted_context or []:
            ctx.append(c if isinstance(c, ContextItem)
                       else ContextItem(c.get("content", ""), c.get("source", "other"), c.get("trust", "other")))
        resp = self.service.screen(ScreeningRequest(prompt=prompt, untrusted_context=ctx, session_id=session_id))
        path = self.forward(resp)
        return {
            "decision": resp.decision.value,
            "forwarded": path is not None,
            "request_id": resp.request_id,
            "incoming_path": str(path) if path else None,
            "response": resp,
        }

    # -- in-process demo of the downstream stages (deployment uses the watcher) -

    def _stage2_service(self):
        if not self._stage2_tried:
            self._stage2_tried = True
            try:
                if self.cfg.stage2_root not in sys.path:
                    sys.path.insert(0, self.cfg.stage2_root)
                from security_hardening.integration import HardeningService  # type: ignore
                self._stage2 = HardeningService()
            except Exception as exc:          # missing module / catalogue / dep -> unavailable, not fatal
                self._stage2 = None
                self._stage2_error = f"{type(exc).__name__}: {exc}"
        return self._stage2

    def run_stage2_inline(self, request: dict) -> Optional[dict]:
        """Compute Stage 2's response for a request dict, in process, with no queue side effects.

        This is the same HardeningService the directory watcher runs; calling it
        here lets the demo show the hardened prompt without starting the watcher
        or disturbing other files already in `incoming/`.
        """
        svc = self._stage2_service()
        if svc is None:
            return None
        try:
            return svc.handle(request)
        except Exception:                      # the service is contracted never to raise, but stay safe
            return None

    def _stage3_adapter(self):
        if not self._stage3_tried:
            self._stage3_tried = True
            try:
                import importlib
                if self.cfg.stage3_dir not in sys.path:
                    sys.path.insert(0, self.cfg.stage3_dir)   # resolves the module's own `from engine import ...`
                adapter_mod = importlib.import_module("adapter")
                self._stage3 = adapter_mod.Stage3PipelineAdapter()
            except Exception as exc:
                self._stage3 = None
                self._stage3_error = f"{type(exc).__name__}: {exc}"
        return self._stage3

    def run_stage3(self, original_prompt: str, hardened_prompt: str,
                   request_id: Optional[str] = None) -> Optional[dict]:
        """Predict the CloudTrail events Stage 2's hardened prompt would emit (Stage 3)."""
        adapter = self._stage3_adapter()
        if adapter is None:
            return None
        try:
            return adapter.process_stage2_output(
                original_prompt=original_prompt, hardened_prompt=hardened_prompt, request_id=request_id)
        except Exception:
            return None

    def run_end_to_end(self, prompt: str, untrusted_context: Optional[list] = None,
                       session_id: Optional[str] = None, run_downstream: bool = True) -> dict[str, Any]:
        """Screen -> deliver (SAFE only) -> harden -> predict events. Stops at the first gate that blocks."""
        res = self.screen_and_forward(prompt, untrusted_context, session_id)
        out: dict[str, Any] = {
            "stage1": {
                "decision": res["decision"],
                "forwarded": res["forwarded"],
                "request_id": res["request_id"],
                "incoming_path": res["incoming_path"],
                "explanation": res["response"].explanation,
                "threat_categories": list(res["response"].threat_categories),
            },
            "stage2": None,
            "stage3": None,
        }
        if not res["forwarded"] or not run_downstream:
            return out

        request = {
            "request_id": res["request_id"],
            "screening_status": self.cfg.handoff_screening_status,
            "original_prompt": res["response"].safe_prompt_for_next_stage,
        }
        s2 = self.run_stage2_inline(request)
        out["stage2"] = s2
        if s2 and s2.get("status") in ("HARDENED", "ALREADY_ADEQUATE") and s2.get("revised_prompt"):
            s3 = self.run_stage3(prompt, s2["revised_prompt"], res["request_id"])
            if s3 is not None:
                out["stage3"] = {
                    "status": s3.get("stage3_contract", {}).get("status"),
                    "event_sequence": s3.get("downstream_event_sequence", []),
                    "gnn_structural_rows": s3.get("downstream_gnn_structural_rows", []),
                    "cloudtrail_records": s3.get("downstream_lstm_records", {}),
                }
        return out


_DEFAULT: Optional[PipelineBridge] = None


def get_bridge(cfg: Optional[GateConfig] = None) -> PipelineBridge:
    """Process-wide default bridge (builds the screening service once)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PipelineBridge(cfg)
    return _DEFAULT
