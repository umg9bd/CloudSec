"""
bridge_cli.py
=============
Screen a prompt and, if it is SAFE, deliver it to Stage 2's `incoming/` folder.
Optionally run the whole chain (Stage 1 -> Stage 2 -> Stage 3) and print each stage.

    python -m prompt_injection_gate.bridge_cli "Create a private S3 bucket for audit logs."
    python -m prompt_injection_gate.bridge_cli --no-downstream "..."   # just screen + deliver
    python -m prompt_injection_gate.bridge_cli --json "..."            # machine-readable
    echo "..." | python -m prompt_injection_gate.bridge_cli -          # read prompt from stdin

A MALICIOUS or REVIEW_REQUIRED prompt is held: nothing is written to incoming/ and
the chain stops. Without a trained classifier the gate fails closed (REVIEW_REQUIRED),
so nothing is delivered until a model is present.
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .pipeline_bridge import PipelineBridge


def _read_prompt(arg: str | None) -> str:
    if arg is None or arg == "-":
        return sys.stdin.read().strip()
    return arg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Screen a prompt and deliver it to Stage 2 if SAFE.")
    ap.add_argument("prompt", nargs="?", help="the prompt to screen (or '-' / omit to read stdin)")
    ap.add_argument("--no-downstream", action="store_true",
                    help="screen and deliver only; do not run Stage 2/3 in process")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)

    prompt = _read_prompt(args.prompt)
    if not prompt:
        ap.error("no prompt given (pass it as an argument or on stdin)")

    bridge = PipelineBridge(load_config())
    result = bridge.run_end_to_end(prompt, run_downstream=not args.no_downstream)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    s1 = result["stage1"]
    icon = {"SAFE": "[SAFE]", "MALICIOUS": "[BLOCKED]", "REVIEW_REQUIRED": "[REVIEW]"}.get(s1["decision"], s1["decision"])
    print(f"\nStage 1  {icon}  {s1['explanation']}")
    if s1["threat_categories"]:
        print("         threat categories: " + ", ".join(s1["threat_categories"]))
    if not s1["forwarded"]:
        print("         held -- not delivered to Stage 2.\n")
        return 0
    print(f"         delivered to Stage 2 incoming/ as {s1['incoming_path']}")

    s2 = result["stage2"]
    if s2 is None:
        print("\nStage 2  (not run in process; a running watcher will pick the file up)\n")
        return 0
    print(f"\nStage 2  [{s2.get('status')}]  controls added: "
          f"{len(s2.get('added_controls', []))}, satisfied: {len(s2.get('satisfied_controls', []))}")

    s3 = result["stage3"]
    if s3 is None:
        print("\nStage 3  (not run)\n")
        return 0
    seq = s3.get("event_sequence", [])
    print(f"\nStage 3  [{s3.get('status')}]  predicted CloudTrail events: "
          + (" -> ".join(seq) if seq else "(none)"))
    print(f"         {len(s3.get('gnn_structural_rows', []))} structural row(s) for the GNN/LSTM scorer.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
