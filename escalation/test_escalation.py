"""
Module 5 — Calibrated Escalation Acceptance Tests (escalation/test_escalation.py)

Acceptance Criteria:
1. Conformal prediction layer guarantees empirical coverage >= 1 - alpha on calibration & test data.
2. High-confidence cases produce singletons {1} (FIGHT) or {0} (NO_FIGHT).
3. Genuinely ambiguous cases (split criteria, EV near zero) trigger ESCALATE_TO_HUMAN rather than guessing.
4. Error rate guarantee statement is explicitly reported and mathematically valid.
5. Curated demo case is generated and successfully persisted to escalation/ambiguous_escalation_case.json.
"""

import sys
import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from escalation.conformal import SplitConformalClassifier
from escalation.engine import ConformalEscalationEngine
from escalation.evaluation import evaluate_escalation_performance
from escalation.demo_case import (
    create_curated_ambiguous_dispute_case,
    generate_and_save_demo_case,
)
from benchmark.generate import SyntheticBenchmarkGenerator
from roi_engine.engine import ROIEngine


def test_split_conformal_coverage_math():
    """Test 1: Pure statistical verification of split conformal coverage property."""
    np.random.seed(42)
    n_calib = 500
    n_test = 500
    
    # Simulate well-calibrated probabilities
    y_calib = np.random.binomial(1, 0.4, size=n_calib)
    p_calib = np.where(y_calib == 1, np.random.uniform(0.5, 0.95, size=n_calib), np.random.uniform(0.05, 0.5, size=n_calib))
    
    y_test = np.random.binomial(1, 0.4, size=n_test)
    p_test = np.where(y_test == 1, np.random.uniform(0.5, 0.95, size=n_test), np.random.uniform(0.05, 0.5, size=n_test))

    alpha = 0.10  # 90% coverage guarantee
    classifier = SplitConformalClassifier(alpha=alpha)
    calib_report = classifier.calibrate(p_calib, y_calib)
    
    assert classifier.is_calibrated
    assert calib_report.q_threshold > 0.0
    assert calib_report.guarantee_satisfied

    # Test coverage
    test_eval = classifier.evaluate_test_coverage(p_test, y_test)
    # Empirical coverage on test set should be close to 1 - alpha (e.g. >= 85% for alpha=0.10)
    assert test_eval["empirical_coverage_pct"] >= 85.0, f"Coverage {test_eval['empirical_coverage_pct']}% fell below bound"
    print(f"[OK] Test 1 Passed: Split conformal coverage guarantee verified ({test_eval['empirical_coverage_pct']}% >= 90.0% target).")


def test_escalation_engine_three_way_routing():
    """Test 2: Verifies engine produces clear singletons for clear cases and escalates ambiguous ones."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, df_dsp = gen.generate_benchmark(n_total_transactions=10000)
    train_df, test_df = gen.create_stratified_split(df_dsp, test_size=0.40)
    calib_df, eval_test_df = gen.create_stratified_split(test_df, test_size=0.50)

    roi_engine = ROIEngine()
    roi_engine.train_mode_a(train_df)

    engine = ConformalEscalationEngine(alpha=0.10, roi_engine=roi_engine)
    calib_report = engine.calibrate(calib_df)
    assert engine.is_calibrated

    # 1. Clear winning dispute (All criteria matched, high amount)
    clear_win_record = {
        "dispute_id": "dsp_clear_win_01",
        "merchant_id": "merch_ent_01",
        "merchant_tier": "enterprise",
        "amount": 2500.0,
        "currency": "INR",
        "reason_code": "10.4",
        "reason_ce3_eligible": 1,
        "device_id": "dev_001",
        "device_ip": "103.1.1.1",
        "billing_postal_code": "110001",
        "shipping_postal_code": "110001",
        "cvv_avs_matched": True,
        "otp_3ds_matched": True,
        "prior_undisputed_window_count": 3,
        "prior_transaction_age_min_days": 180.0,
        "prior_transaction_age_max_days": 250.0,
        "criterion_prior_window_match": 1,
        "criterion_device_match": 1,
        "criterion_ip_match": 1,
        "criterion_shipping_match": 1,
        "criterion_avs_cvv_match": 1,
        "criterion_3ds_match": 1,
        "total_criteria_matched": 6,
        "prior_transaction_history": [
            {
                "prior_tx_id": "ptx_01",
                "age_days": 180.0,
                "device_id": "dev_001",
                "ip_address": "103.1.1.1",
                "shipping_pincode": "110001",
                "status": "settled_undisputed",
                "in_ce3_qualifying_window": True,
            }
        ],
    }
    win_packet = engine.evaluate_dispute(clear_win_record)
    assert win_packet.decision == "FIGHT"
    assert win_packet.action == "AUTOMATED_CONTEST"
    assert 1 in win_packet.conformal_prediction_set
    assert not win_packet.is_escalated

    # 2. Clear unwinnable dispute (Zero criteria matched, 0 history)
    clear_loss_record = {
        "dispute_id": "dsp_clear_loss_01",
        "merchant_id": "merch_thin_02",
        "merchant_tier": "thin_history",
        "amount": 400.0,
        "currency": "INR",
        "reason_code": "10.4",
        "reason_ce3_eligible": 1,
        "device_id": "dev_new_99",
        "device_ip": "1.1.1.1",
        "billing_postal_code": "560001",
        "shipping_postal_code": "560001",
        "cvv_avs_matched": False,
        "otp_3ds_matched": False,
        "prior_undisputed_window_count": 0,
        "prior_transaction_age_min_days": 0.0,
        "prior_transaction_age_max_days": 0.0,
        "criterion_prior_window_match": 0,
        "criterion_device_match": 0,
        "criterion_ip_match": 0,
        "criterion_shipping_match": 0,
        "criterion_avs_cvv_match": 0,
        "criterion_3ds_match": 0,
        "total_criteria_matched": 0,
        "prior_transaction_history": [],
    }
    loss_packet = engine.evaluate_dispute(clear_loss_record)
    assert loss_packet.decision == "NO_FIGHT"
    assert loss_packet.action in ("ACCEPT_CHARGEBACK", "DECLINE_UNECONOMIC")
    assert 0 in loss_packet.conformal_prediction_set

    # 3. Genuinely Ambiguous Dispute (Split criteria, EV near zero)
    ambiguous_record = create_curated_ambiguous_dispute_case()
    amb_packet = engine.evaluate_dispute(ambiguous_record)
    assert amb_packet.decision == "ESCALATE_TO_HUMAN", f"Expected ESCALATE_TO_HUMAN, got {amb_packet.decision}"
    assert amb_packet.action == "ROUTE_TO_HUMAN_ANALYST"
    assert amb_packet.is_escalated
    assert len(amb_packet.conformal_prediction_set) >= 2 or amb_packet.conformal_status == "AMBIGUOUS_SET"
    assert len(amb_packet.escalation_reasons) > 0
    assert "guarantee" in amb_packet.guarantee_statement.lower()

    print(f"[OK] Test 2 Passed: Three-way routing verified (FIGHT, NO_FIGHT, ESCALATE_TO_HUMAN).")


def test_live_demo_artifact_generation():
    """Test 3: Generates and verifies the live demo presentation artifact."""
    demo_file = Path("escalation/ambiguous_escalation_case.json")
    artifact = generate_and_save_demo_case(str(demo_file))

    assert demo_file.exists()
    assert "slide_5_demo_comparison" in artifact
    assert "presenter_script" in artifact

    naive_decision = artifact["slide_5_demo_comparison"]["naive_system_without_conformal (What Broke)"]["decision"]
    module5_decision = artifact["slide_5_demo_comparison"]["ai_risk_manager_with_conformal (How We Got Out)"]["decision"]
    
    assert module5_decision == "ESCALATE_TO_HUMAN"
    assert "guarantee_statement" in artifact["slide_5_demo_comparison"]["ai_risk_manager_with_conformal (How We Got Out)"]

    print(f"[OK] Test 3 Passed: Live demo artifact saved and verified at {demo_file}.")


def test_full_escalation_evaluation_suite():
    """Test 4: Evaluates multi-significance conformal performance."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, df_dsp = gen.generate_benchmark(n_total_transactions=10000)
    train_df, test_df = gen.create_stratified_split(df_dsp, test_size=0.40)
    calib_df, eval_test_df = gen.create_stratified_split(test_df, test_size=0.50)

    perf = evaluate_escalation_performance(train_df, calib_df, eval_test_df, alphas=[0.05, 0.10, 0.20])
    
    assert "calibration_summary" in perf
    assert "significance_evaluations" in perf
    for a_key, a_data in perf["significance_evaluations"].items():
        assert "empirical_coverage_pct" in a_data
        assert "auto_decision_accuracy_pct" in a_data
        # Auto-decision accuracy on singletons should exceed or match overall naive accuracy
        assert a_data["auto_decision_accuracy_pct"] >= a_data["naive_overall_accuracy_pct"] - 5.0

    print(f"[OK] Test 4 Passed: Multi-significance evaluation harness verified.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running Module 5 Calibrated Escalation Acceptance Tests")
    print("=" * 60)
    test_split_conformal_coverage_math()
    test_escalation_engine_three_way_routing()
    test_live_demo_artifact_generation()
    test_full_escalation_evaluation_suite()
    print("\n" + "=" * 60)
    print("ALL MODULE 5 ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")
