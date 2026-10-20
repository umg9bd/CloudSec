"""
CLI / Generator script to produce and save predicted CloudTrail logs to disk.

Usage:
    python log-generation/generate.py
    python log-generation/generate.py --prompt "Create an S3 bucket named dev-assets" --output-dir log-generation/generated_logs
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure current directory is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from adapter import Stage3PipelineAdapter
    from engine import Stage3Engine
except ImportError:
    from stage3_intent_event_predictor.adapter import Stage3PipelineAdapter
    from stage3_intent_event_predictor.engine import Stage3Engine


DEFAULT_WORKFLOWS = [
    {
        "name": "s3_secure_data_lake",
        "original_prompt": "Create an S3 bucket named analytics-prod-data-lake for company financial reports",
        "hardened_prompt": "TASK: Create S3 bucket named analytics-prod-data-lake. SECURITY REQUIREMENTS: Require server-side encryption with AES256. Block all public access."
    },
    {
        "name": "kms_key_with_rotation",
        "original_prompt": "Create a KMS customer managed key for encrypting production RDS database tables",
        "hardened_prompt": "TASK: Create KMS key for RDS. SECURITY REQUIREMENTS: Enable annual key rotation."
    },
    {
        "name": "ec2_web_tier_security_group",
        "original_prompt": "Create an EC2 security group named web-tier-secgroup in vpc-0123456789abcdef0",
        "hardened_prompt": "TASK: Create security group web-tier-secgroup in vpc-0123456789abcdef0."
    },
    {
        "name": "lambda_image_processor",
        "original_prompt": "Create a Lambda function named image-processor-worker with execution role arn:aws:iam::111122223333:role/lambda-s3-execution-role",
        "hardened_prompt": "TASK: Deploy Lambda function image-processor-worker."
    }
]


def generate_and_save_logs(
    prompt: str,
    hardened_prompt: str,
    output_dir: str,
    filename_prefix: str = "generated",
    principal_arn: str = "arn:aws:iam::111122223333:user/agentic-operator"
):
    os.makedirs(output_dir, exist_ok=True)
    adapter = Stage3PipelineAdapter()

    result = adapter.process_stage2_output(
        original_prompt=prompt,
        hardened_prompt=hardened_prompt,
        principal_arn=principal_arn,
        request_id=f"gen-{filename_prefix}"
    )

    # 1. Save standard CloudTrail JSON Records file
    cloudtrail_records = result["downstream_lstm_records"]
    cloudtrail_path = os.path.join(output_dir, f"{filename_prefix}_cloudtrail.json")
    with open(cloudtrail_path, "w", encoding="utf-8") as f:
        json.dump(cloudtrail_records, f, indent=2)

    # 2. Save full Stage 3 analysis contract (includes graph, provenance, dependencies)
    contract_data = result["stage3_contract"]
    contract_path = os.path.join(output_dir, f"{filename_prefix}_stage3_contract.json")
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract_data, f, indent=2)

    # 3. Save GNN structural rows
    gnn_rows = result["downstream_gnn_structural_rows"]
    gnn_path = os.path.join(output_dir, f"{filename_prefix}_gnn_structural.json")
    with open(gnn_path, "w", encoding="utf-8") as f:
        json.dump(gnn_rows, f, indent=2)

    event_count = len(cloudtrail_records.get("Records", []))
    print(f"Generated {event_count} CloudTrail event(s):")
    print(f"   CloudTrail JSON: {cloudtrail_path}")
    print(f"   Stage 3 Report:  {contract_path}")
    print(f"   GNN Graph Rows:  {gnn_path}")
    return cloudtrail_path


def main():
    parser = argparse.ArgumentParser(description="Generate and save predicted CloudTrail logs to disk")
    parser.add_argument("--prompt", help="User original prompt", default=None)
    parser.add_argument("--hardened-prompt", help="Security-hardened prompt", default=None)
    parser.add_argument("--output-dir", help="Directory to save logs", default=os.path.join(_current_dir, "generated_logs"))
    parser.add_argument("--name", help="Prefix name for output files", default="custom_run")
    args = parser.parse_args()

    if args.prompt:
        hardened = args.hardened_prompt or args.prompt
        generate_and_save_logs(args.prompt, hardened, args.output_dir, args.name)
    else:
        print(f"=== Generating default benchmark workflows into '{args.output_dir}' ===")
        for wf in DEFAULT_WORKFLOWS:
            print(f"\nProcessing workflow: {wf['name']}...")
            generate_and_save_logs(
                prompt=wf["original_prompt"],
                hardened_prompt=wf["hardened_prompt"],
                output_dir=args.output_dir,
                filename_prefix=wf["name"]
            )
        print("\nAll default workflows generated successfully!")


if __name__ == "__main__":
    main()
