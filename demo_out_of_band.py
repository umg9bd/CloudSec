"""
demo_out_of_band.py -- simulate the agent's identity being used OUTSIDE the gate.

The agent gate only sees what is asked of it. If the agent's credentials are then used directly --
a compromised agent, a leaked key, a second tool -- none of that passes Stages 1-4. This drops such
activity, as the CloudTrail it would leave, into incoming/: a multi-step privilege-escalation chain
under the agent's own principal. The real-time pipeline scores it like any other log, and the
alert appears on the Live Dashboard under the agent's name -- the post-execution loop doing its job.

    python demo_out_of_band.py                         # IAM persistence chain as agentic-operator
    python demo_out_of_band.py --chain recon_then_role
    python demo_out_of_band.py --principal-name my-agent

Chains (log-generation/session_generator.py): iam_user_persist (CreateUser -> login profile ->
access key -> AdministratorAccess), recon_then_role (recon -> admin role -> AssumeRole -> secrets ->
StopLogging), secrets_loot (secret theft and staging). Demo data only: no AWS API is called.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import agent_orchestrator as ao

sys.path.insert(0, str(ao.ROOT / "log-generation"))
import session_generator as sg  # noqa: E402

CHAINS = ("iam_user_persist", "recon_then_role", "secrets_loot")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", choices=CHAINS, default="iam_user_persist")
    ap.add_argument("--principal-name", default=ao.DEFAULT_PRINCIPAL.rsplit("/", 1)[-1],
                    help="the agent's IAM user name (default: the agent gate's principal)")
    ap.add_argument("--incoming", default=str(ao.PIPELINE_INCOMING), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    now = dt.datetime.now(dt.timezone.utc)
    records = sg.generate_privilege_escalation_chain(args.chain, attacker_name=args.principal_name, base_time=now)
    name = f"out_of_band_{args.chain}_{now.strftime('%H%M%S')}.json"
    incoming = Path(args.incoming)
    ao._atomic_json(incoming / name, {"Records": records}, incoming.parent / ".incoming_staging")
    ao.audit("out_of_band_demo", file=name, chain=args.chain, principal=args.principal_name, events=len(records))
    print(f"Dropped {len(records)} out-of-band events ({args.chain}) for user/{args.principal_name} into "
          f"incoming/{name}.\nWatch the Live Dashboard: the real-time pipeline scores them within a few seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
