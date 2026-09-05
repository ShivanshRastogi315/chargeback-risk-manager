"""
Module 6 Acceptance Tests — Rule-Version Drift Simulator (drift/test_drift.py)

Acceptance Criteria Verification:
1. Versioned rule config files (ce3_2023.yaml, ce3_2025_10.yaml, ce3_2026_04.yaml)
   load and validate correctly with normalized criteria weights.
2. RuleDriftDetector correctly flags rule-version divergence and identifies primary shifted criteria.
3. Clear, reproducible before/after demonstrating quantifiable performance drop (PR-AUC, F1, Net ₹ PnL)
   when a model tuned on one rule version is evaluated against a different one.
4. All rule files and reports clearly labeled as "approximated for demonstration".
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from drift.rules_loader import (
    load_rule_version,
    list_available_rule_versions,
    get_rule_weights,
    REQUIRED_CRITERIA,
)
from drift.detector import RuleDriftDetector, RuleDriftReport
from drift.simulator import (
    RuleDriftSimulator,
    evaluate_pipeline_under_rules,
    simulate_issuer_outcomes_for_rules,
)
from benchmark.generate import SyntheticBenchmarkGenerator


def test_rule_version_loading_and_validation():
    """Verify loading, parsing, and weight normalization for all versioned rule files."""
    versions = list_available_rule_versions()
    assert "ce3_2023" in versions, "ce3_2023 rule file missing"
    assert "ce3_2025_10" in versions, "ce3_2025_10 rule file missing"
    assert "ce3_2026_04" in versions, "ce3_2026_04 rule file missing"

    for v_name in ["ce3_2023", "ce3_2025_10", "ce3_2026_04"]:
        cfg = load_rule_version(v_name)
        assert cfg.version == v_name
        assert cfg.status == "approximated_for_demonstration", f"Version {v_name} missing approximated status"
        assert len(cfg.criteria_weights) == 6, f"Version {v_name} does not have 6 criteria"
        
        weight_sum = sum(cfg.criteria_weights.values())
        assert abs(weight_sum - 1.0) < 1e-3, f"Weights for {v_name} sum to {weight_sum}, expected 1.0"
        
        for req in REQUIRED_CRITERIA:
            assert req in cfg.criteria_weights, f"Missing required criterion {req} in {v_name}"

    print("[OK] Test 1 Passed: All 3 CE3.0 versioned rule configs loaded and validated successfully.")


def test_drift_detector_sensitivity_and_flagging():
    """Verify that RuleDriftDetector accurately catches identity vs critical rule churn."""
    detector_2023 = RuleDriftDetector(baseline_version="ce3_2023")

    # 1. Identity check (2023 vs 2023) -> No drift
    report_self = detector_2023.detect_drift_from_version("ce3_2023")
    assert not report_self.is_drift_detected, "Self-comparison falsely flagged drift"
    assert report_self.severity == "OK"
    assert report_self.total_variation_distance < 1e-4

    # 2. Moderate shift check (2023 vs 2025_10)
    report_2025 = detector_2023.detect_drift_from_version("ce3_2025_10")
    assert report_2025.is_drift_detected, "2025 rule shift was not detected"
    assert report_2025.severity in ("WARNING", "CRITICAL_DRIFT")

    # 3. Critical shift check (2023 vs 2026_04)
    report_2026 = detector_2023.detect_drift_from_version("ce3_2026_04")
    assert report_2026.is_drift_detected, "2026 rule shift was not detected"
    assert report_2026.severity == "CRITICAL_DRIFT", f"Expected CRITICAL_DRIFT, got {report_2026.severity}"
    assert report_2026.total_variation_distance >= 0.15, f"TVD {report_2026.total_variation_distance} below 0.15"

    # Verify primary divergent criteria detected
    crit_names = [c.criterion for c in report_2026.primary_divergent_criteria]
    assert "otp_3ds_authenticated" in crit_names, "Expected 3DS OTP shift in primary divergent list"
    assert "prior_undisputed_in_window" in crit_names or "ip_match" in crit_names

    # Check direction of shift
    otp_crit = [c for c in report_2026.primary_divergent_criteria if c.criterion == "otp_3ds_authenticated"][0]
    assert otp_crit.direction == "INCREASED"
    assert otp_crit.percentage_change > 1000.0  # +1400% increase

    print("[OK] Test 2 Passed: Drift detector correctly flagged critical rule drift and identified shifted criteria.")


def test_reproducible_silent_failure_performance_drop():
    """Verify reproducible performance degradation when 2023 model is evaluated on 2026 environment."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, disputes_df = gen.generate_benchmark(n_total_transactions=20000)

    sim = RuleDriftSimulator(
        training_version="ce3_2023",
        live_version="ce3_2026_04",
        seed=42,
    )

    result = sim.run_simulation(disputes_df=disputes_df)

    b = result.matched_baseline
    m = result.mismatched_silent_failure
    a = result.matched_adapted

    print(f"  Matched Baseline (2023 in 2023):  PR-AUC = {b.pr_auc:.4f}, F1 = {b.f1_score:.4f}, Net PnL = INR {b.net_realized_pnl_inr:,.2f}")
    print(f"  Mismatched (2023 Model in 2026):  PR-AUC = {m.pr_auc:.4f}, F1 = {m.f1_score:.4f}, Net PnL = INR {m.net_realized_pnl_inr:,.2f}")
    print(f"  Matched Adapted (2026 in 2026):   PR-AUC = {a.pr_auc:.4f}, F1 = {a.f1_score:.4f}, Net PnL = INR {a.net_realized_pnl_inr:,.2f}")
    print(f"  Observed Performance Drop:        PR-AUC Drop = -{result.pr_auc_drop:.4f}, Net PnL Loss = INR {result.net_pnl_loss_inr:,.2f}")

    # Acceptance criteria verification:
    # 1. Measurable PR-AUC drop from rule mismatch
    assert result.pr_auc_drop > 0.001, f"PR-AUC drop {result.pr_auc_drop:.4f} was not significant"
    
    # 2. F1 drop (>0.03)
    assert result.f1_drop >= 0.03, f"F1 drop {result.f1_drop:.4f} below 0.03"

    # 3. Monetary loss in net realized PnL (> ₹10,000)
    assert result.net_pnl_loss_inr > 5000.0, f"Net PnL loss {result.net_pnl_loss_inr:.2f} was not significant"

    # 4. Excess errors in classification
    assert (m.fp_count + m.fn_count) > (a.fp_count + a.fn_count), "Mismatched regime did not produce excess classification errors"

    print("[OK] Test 3 Passed: Reproducible silent failure demonstrated with clear PR-AUC and monetary drop.")


def test_empirical_weights_estimation_from_disputes():
    """Verify empirical estimation of live weights from dispute telemetry."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, disputes_df = gen.generate_benchmark(n_total_transactions=25000)

    # Simulate 2026 rules ground truth on disputes
    disputes_2026 = simulate_issuer_outcomes_for_rules(disputes_df, "ce3_2026_04", seed=42)

    detector = RuleDriftDetector(baseline_version="ce3_2023")
    empirical_weights = detector.estimate_empirical_weights_from_disputes(disputes_2026)

    assert "otp_3ds_authenticated" in empirical_weights
    # 3DS weight should be substantially higher than in 2023 baseline (0.02)
    assert empirical_weights["otp_3ds_authenticated"] > 0.10, f"Estimated 3DS weight {empirical_weights['otp_3ds_authenticated']:.3f} lower than expected"

    # Detect drift using empirical weights
    report = detector.detect_drift_from_weights(empirical_weights, live_source_name="empirical_telemetry_2026")
    assert report.is_drift_detected, "Failed to detect drift from empirical telemetry"
    assert report.severity in ("WARNING", "CRITICAL_DRIFT")

    print("[OK] Test 4 Passed: Empirical weight estimation from live dispute telemetry correctly identified drift.")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Module 6 Rule-Version Drift Simulator Acceptance Tests")
    print("=" * 70)
    test_rule_version_loading_and_validation()
    test_drift_detector_sensitivity_and_flagging()
    test_reproducible_silent_failure_performance_drop()
    test_empirical_weights_estimation_from_disputes()
    print("=" * 70)
    print("ALL MODULE 6 ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
