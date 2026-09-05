"""
Module 6 — Standalone CLI Drift Simulation Runner (drift/run_drift_simulation.py)

Usage:
    python drift/run_drift_simulation.py --train-version ce3_2023 --live-version ce3_2026_04 --n-samples 50000
"""

import sys
import os
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure UTF-8 console output for Windows CLI
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from drift.evaluation import run_rule_drift_experiment, format_drift_markdown_report


def main():
    parser = argparse.ArgumentParser(description="Module 6: Rule-Version Concept Drift Simulator")
    parser.add_argument("--train-version", type=str, default="ce3_2023", help="Training-time baseline rule version")
    parser.add_argument("--live-version", type=str, default="ce3_2026_04", help="Live target environment rule version")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of benchmark transactions to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-json", type=str, default="drift/drift_results.json", help="Output JSON path")
    parser.add_argument("--output-md", type=str, default="drift/DRIFT_REPORT.md", help="Output Markdown report path")
    args = parser.parse_args()

    print("=" * 70)
    print("Module 6: Rule-Version Concept Drift Simulator")
    print(f"Comparing Training Baseline [{args.train_version}] vs Live Environment [{args.live_version}]")
    print("=" * 70)

    result = run_rule_drift_experiment(
        training_version=args.train_version,
        live_version=args.live_version,
        n_transactions=args.n_samples,
        seed=args.seed,
    )

    md_report = format_drift_markdown_report(result)
    result_dict = result.to_dict()

    # Save outputs
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n" + md_report)
    print("\n" + "=" * 70)
    print(f"Results successfully saved to {args.output_json} and {args.output_md}")
    print("=" * 70)


if __name__ == "__main__":
    main()
