"""
Module 6 — Rule Drift Experiment Runner & Formatter (drift/evaluation.py)

Runs the rule-version drift experiment and renders formatted markdown comparison tables.
"""

from typing import Dict, List, Any, Optional
import json
import pandas as pd

from drift.simulator import RuleDriftSimulator, RuleDriftSimulationResult
from drift.detector import RuleDriftDetector, RuleDriftReport


def run_rule_drift_experiment(
    training_version: str = "ce3_2023",
    live_version: str = "ce3_2026_04",
    n_transactions: int = 50000,
    seed: int = 42,
) -> RuleDriftSimulationResult:
    """Convenience helper running the rule drift experiment."""
    sim = RuleDriftSimulator(
        training_version=training_version,
        live_version=live_version,
        seed=seed,
    )
    return sim.run_simulation(n_transactions=n_transactions)


def format_drift_markdown_report(result: RuleDriftSimulationResult) -> str:
    """Renders a comprehensive GitHub Markdown report comparing matched vs mismatched regimes."""
    b = result.matched_baseline
    m = result.mismatched_silent_failure
    a = result.matched_adapted
    d = result.drift_detector_report

    md_lines = [
        "# Module 6 — Rule-Version Concept Drift Simulation Report",
        "**Silent Evaluation Failure Analysis & Drift Detection (Visa CE3.0 Evolution)**",
        "",
        "> **APPROXIMATION NOTICE:** All rule-version files (`ce3_2023.yaml`, `ce3_2025_10.yaml`, `ce3_2026_04.yaml`) are **approximated for demonstration** based on publicly available Visa CE3.0 documentation and industry research.",
        "",
        "---",
        "",
        "## 1. Executive Summary & Drift Detector Verdict",
        f"- **Baseline Training Version:** `{result.training_version}`",
        f"- **Live Production Environment Version:** `{result.live_version}`",
        f"- **Drift Detection Status:** **`{d.severity}`** (TVD = `{d.total_variation_distance:.4f}` | Cosine Div = `{d.cosine_divergence:.4f}`)",
        f"- **Recommended Remediation:** `{d.recommended_action}`",
        f"- **Diagnostic Narrative:** {d.summary_narrative}",
        "",
        "### Key Divergent Criteria Breakdown",
        "| Criterion | Baseline Weight (2023) | Live Weight (2026) | Absolute Delta | Relative Shift (%) | Direction |",
        "|---|---|---|---|---|---|",
    ]

    for crit in d.primary_divergent_criteria:
        md_lines.append(
            f"| `{crit.criterion}` | {crit.baseline_weight:.2f} | {crit.live_weight:.2f} | {crit.absolute_delta:+.3f} | {crit.percentage_change:+.1f}% | **{crit.direction}** |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Quantitative Performance Drop (Silent Failure Demonstration)",
        "Comparing model performance across the three operating regimes on the same dispute records:",
        "",
        "| Metric / Business Dimension | Regime 1: Matched Baseline (`2023` in `2023`) | Regime 2: Mismatched Silent Failure (`2023` in `2026`) | Regime 3: Adapted Modern (`2026` in `2026`) | Degradation Impact (Regime 2 vs 3) |",
        "|---|---|---|---|---|",
        f"| **PR-AUC** | `{b.pr_auc:.4f}` | `{m.pr_auc:.4f}` | `{a.pr_auc:.4f}` | **-{result.pr_auc_drop:.4f}** ({(-result.pr_auc_drop/a.pr_auc*100):.1f}%) |",
        f"| **ROC-AUC** | `{b.roc_auc:.4f}` | `{m.roc_auc:.4f}` | `{a.roc_auc:.4f}` | **-{(a.roc_auc - m.roc_auc):.4f}** |",
        f"| **F1 Score** | `{b.f1_score:.4f}` | `{m.f1_score:.4f}` | `{a.f1_score:.4f}` | **-{result.f1_drop:.4f}** |",
        f"| **Contested Win Rate (Count)** | `{b.count_win_rate_contested:.2f}%` | `{m.count_win_rate_contested:.2f}%` | `{a.count_win_rate_contested:.2f}%` | **-{result.count_win_rate_drop_pct:.2f}%** |",
        f"| **Contested Win Rate (Amount)**| `{b.amount_win_rate_contested:.2f}%` | `{m.amount_win_rate_contested:.2f}%` | `{a.amount_win_rate_contested:.2f}%` | **-{(a.amount_win_rate_contested - m.amount_win_rate_contested):.2f}%** |",
        f"| **False Positives (Lost Contests)** | {b.fp_count} cases | {m.fp_count} cases | {a.fp_count} cases | **+{m.fp_count - a.fp_count} excess FP cases** |",
        f"| **False Negatives (Missed Wins)** | {b.fn_count} cases | {m.fn_count} cases | {a.fn_count} cases | **+{m.fn_count - a.fn_count} excess FN cases** |",
        f"| **False Positive Cost (₹ INR)** | ₹{b.fp_cost_inr:,.2f} | ₹{m.fp_cost_inr:,.2f} | ₹{a.fp_cost_inr:,.2f} | **+₹{result.fp_cost_increase_inr:,.2f}** |",
        f"| **False Negative Cost (₹ INR)** | ₹{b.fn_cost_inr:,.2f} | ₹{m.fn_cost_inr:,.2f} | ₹{a.fn_cost_inr:,.2f} | **+₹{result.fn_cost_increase_inr:,.2f}** |",
        f"| **Total Monetary Error Cost (₹)** | ₹{b.total_error_cost_inr:,.2f} | ₹{m.total_error_cost_inr:,.2f} | ₹{a.total_error_cost_inr:,.2f} | **+₹{result.total_error_cost_increase_inr:,.2f}** |",
        f"| **Net Realized Portfolio PnL (₹)**| **₹{b.net_realized_pnl_inr:,.2f}** | **₹{m.net_realized_pnl_inr:,.2f}** | **₹{a.net_realized_pnl_inr:,.2f}** | **-₹{result.net_pnl_loss_inr:,.2f} Net Loss** |",
        "",
        "---",
        "",
        "## 3. Why the Silent Failure Occurs (§2.6 Literature Review Insight)",
        "1. **IP Match Deprecation:** The 2023 model relies on static IP matches (+0.10 weight). In 2026, mobile proxies and CGNAT rendered IP matching unreliable, causing issuers to downweight IP (+0.05). The 2023 model overconfidently contests fraudulent transactions that merely share an IP subnet.",
        "2. **3DS OTP Step-Up Expansion:** In 2026, 3DS OTP step-up authentication became a decisive criterion (+0.30 weight, +1400% surge vs 2023). The 2023 model discounts 3DS (+0.02) and declines to contest winnable disputes where shipping was ambiguous but strong OTP authentication was present.",
        "3. **Drift Detector Intervention:** The `RuleDriftDetector` tracks Total Variation Distance and flags the exact criteria responsible, prompting automatic recalibration before merchant revenue is forfeited.",
        "",
        "---",
    ])

    return "\n".join(md_lines)
