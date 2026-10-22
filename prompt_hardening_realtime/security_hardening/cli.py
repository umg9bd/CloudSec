"""
cli.py
======
    python -m security_hardening harden --input request.json [--output response.json]
    python -m security_hardening harden --prompt "..." --screening-status SAFE [--json]
    python -m security_hardening validate-catalogue [--catalogue PATH]
    python -m security_hardening explain --prompt "..."

`harden --input` accepts one JSON object, a JSON array, or JSON Lines (one request per line) and writes
the same shape back. Exit status: 0 = every request returned HARDENED or ALREADY_ADEQUATE,
2 = at least one request needs attention (conflict, clarification, scope, screening, error),
1 = usage / unreadable input. The CLI never calls the network or AWS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

from .catalogue import CatalogueError, load_catalogue
from .integration import HardeningService
from .models import PROCEED_STATUSES


def _read_requests(path: str) -> List[Any]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _summary(resp: dict) -> str:
    lines = [f"[{resp['status']}] request_id={resp['request_id']} catalogue={resp.get('source_catalogue_version')} review={resp['requires_user_review']}"]
    for key, label in (("added_controls", "added"), ("satisfied_controls", "already satisfied"), ("deferred_controls", "deferred")):
        if resp.get(key):
            lines.append(f"  {label}: " + ", ".join(c["control_id"] for c in resp[key]))
    for c in resp.get("conflicts", []):
        lines.append(f"  CONFLICT {c['control_id']}: {c['why']}\n    alternative: {c['proposed_alternative']}")
    for q in resp.get("clarification_questions", []):
        lines.append(f"  {'BLOCKING ' if q['blocking'] else ''}question: {q['question']}")
    for w in resp.get("warnings", []):
        lines.append(f"  warning: {w}")
    for e in resp.get("errors", []):
        lines.append(f"  error {e['code']}: {e['message']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m security_hardening", description="Stage 2 security prompt hardening")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harden", help="harden one or more approved prompts")
    h.add_argument("--input", help="request JSON / JSON array / JSONL file, or - for stdin")
    h.add_argument("--output", help="write responses here (default: print)")
    h.add_argument("--prompt", help="harden a single prompt given on the command line")
    h.add_argument("--screening-status", default=None, help="upstream screening status for --prompt (no default: missing is an error)")
    h.add_argument("--request-id", default="cli-request")
    h.add_argument("--json", action="store_true", help="print full JSON responses instead of a summary")
    v = sub.add_parser("validate-catalogue", help="load and validate a catalogue file")
    v.add_argument("--catalogue")
    e = sub.add_parser("explain", help="show the facts the analyser extracts from a prompt")
    e.add_argument("--prompt", required=True)
    from .directory_cli import add_watch_parser, run_watch
    add_watch_parser(sub)
    args = ap.parse_args(argv)

    if args.cmd == "watch":
        return run_watch(args)

    if args.cmd == "validate-catalogue":
        try:
            cat = load_catalogue(Path(args.catalogue) if args.catalogue else None)
        except CatalogueError as exc:
            print("INVALID:", *exc.problems, sep="\n  - ")
            return 2
        print(f"valid: {cat.catalogue_id} {cat.version} ({len(cat.active())} active of {len(cat.controls)} controls, sha256 {cat.sha256[:12]})")
        return 0

    if args.cmd == "explain":
        from .config import load_scope
        from .task_analyzer import TaskAnalyzer
        cat = load_catalogue()
        profile = TaskAnalyzer(cat, load_scope()).analyze(args.prompt)
        for fact, values in profile.facts.items():
            if values:
                print(f"{fact:26s} {sorted(values)}")
        sf = profile.services
        print(f"{'scope':26s} approved={sf.approved} excluded={sf.excluded} out_of_scope={sf.out_of_scope}")
        if profile.intent_unclear:
            print("intent unclear: permission/access wording but nothing to secure")
        return 0

    # harden
    if args.prompt is not None:
        requests: List[Any] = [{"request_id": args.request_id, "original_prompt": args.prompt,
                                **({"screening_status": args.screening_status} if args.screening_status else {})}]
    elif args.input:
        try:
            requests = _read_requests(args.input)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read requests: {exc}", file=sys.stderr)
            return 1
    else:
        ap.error("harden needs --input or --prompt")
    service = HardeningService()
    responses = [service.handle(r) for r in requests]
    if args.output:
        Path(args.output).write_text(json.dumps(responses[0] if len(responses) == 1 else responses, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(responses)} response(s) to {args.output}")
    for resp in responses:
        if args.json and not args.output:
            print(json.dumps(resp, indent=2))
        else:
            print(_summary(resp))
            if resp["status"] in {s.value for s in PROCEED_STATUSES} and not args.output and not args.json:
                print("\n" + resp["revised_prompt"])
    return 0 if all(r["status"] in {s.value for s in PROCEED_STATUSES} for r in responses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
