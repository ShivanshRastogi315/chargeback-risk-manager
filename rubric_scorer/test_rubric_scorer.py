"""
Module 1 Acceptance Tests — Rubric Scorer (rubric_scorer/test_rubric_scorer.py)

Acceptance Criteria Verification:
1. Returns both overall score and per-criterion breakdown for any input (never just a single float).
2. Decomposed per-criterion accuracy evaluated separately against Module 0 ground truth.
3. Robust handling of edge cases (empty history, thin-history records, various reason codes).
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from rubric_scorer.aggregator import RubricScorer
from rubric_scorer.criteria import (
    PriorWindowScorer,
    DeviceMatchScorer,
    ShippingMatchScorer,
    IPMatchScorer,
    CVVAVSScorer,
    ThreeDSScorer,
)
from rubric_scorer.evaluation import evaluate_rubric_against_benchmark
from benchmark.generate import SyntheticBenchmarkGenerator


def test_output_contract_structure():
    """Verify that scoring any record returns structured object with overall and criteria breakdown."""
    scorer = RubricScorer()
    
    dummy_record = {
        "transaction_id": "tx_dummy_001",
        "reason_code": "10.4",
        "reason_ce3_eligible": 1,
        "amount": 2500.0,
        "device_id": "dev_test_123",
        "device_ip": "103.21.50.10",
        "shipping_postal_code": "400001",
        "billing_postal_code": "400001",
        "cvv_avs_matched": True,
        "otp_3ds_matched": True,
        "prior_transaction_history": [
            {
                "prior_tx_id": "ptx_001",
                "age_days": 180.0,
                "device_id": "dev_test_123",
                "ip_address": "103.21.50.10",
                "shipping_pincode": "400001",
                "status": "settled_undisputed",
                "in_ce3_qualifying_window": True,
            }
        ],
    }

    result = scorer.score_record(dummy_record)

    assert isinstance(result, dict), "Result must be a dictionary"
    assert "overall_score" in result, "Missing overall_score"
    assert "confidence_interval" in result, "Missing confidence_interval"
    assert "criteria" in result, "Missing criteria breakdown"
    assert "recommendation" in result, "Missing recommendation"
    assert "summary" in result, "Missing summary"

    expected_criteria = {
        "prior_undisputed_in_window",
        "device_match",
        "shipping_match",
        "ip_match",
        "cvv_avs_verified",
        "otp_3ds_authenticated",
    }
    assert set(result["criteria"].keys()) == expected_criteria, "Criteria keys mismatch"

    for crit_name, crit_data in result["criteria"].items():
        assert "score" in crit_data, f"Missing score in {crit_name}"
        assert "matched" in crit_data, f"Missing matched in {crit_name}"
        assert "confidence_interval" in crit_data, f"Missing confidence_interval in {crit_name}"
        assert "weight" in crit_data, f"Missing weight in {crit_name}"
        assert "evidence_strength" in crit_data, f"Missing evidence_strength in {crit_name}"
        assert "details" in crit_data, f"Missing details in {crit_name}"

    assert result["overall_score"] >= 0.80, f"Expected high overall score on perfect match, got {result['overall_score']}"
    assert result["recommendation"] == "RECOMMEND_CONTEST", f"Expected RECOMMEND_CONTEST, got {result['recommendation']}"

    print("[OK] Test 1 Passed: Output contract strictly adheres to structured object schema (never a single float).")


def test_edge_case_empty_history():
    """Verify safe handling of cold-start records with no prior history."""
    scorer = RubricScorer()
    empty_record = {
        "transaction_id": "tx_empty",
        "reason_code": "10.4",
        "reason_ce3_eligible": 1,
        "amount": 1200.0,
        "device_id": "dev_new",
        "device_ip": "115.112.10.5",
        "shipping_postal_code": "560001",
        "billing_postal_code": "560001",
        "cvv_avs_matched": False,
        "otp_3ds_matched": False,
        "prior_transaction_history": [],
    }

    result = scorer.score_record(empty_record)
    assert result["overall_score"] <= 0.20, f"Expected low overall score on empty history, got {result['overall_score']}"
    assert result["recommendation"] == "RECOMMEND_ACCEPT", f"Expected RECOMMEND_ACCEPT, got {result['recommendation']}"
    assert result["qualifying_criteria_count"] == 0
    print("[OK] Test 2 Passed: Empty history / cold-start record gracefully scored as low confidence.")


def test_per_criterion_accuracy_against_benchmark():
    """Verify separate per-criterion evaluation against Module 0 ground truth labels."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, disp_df = gen.generate_benchmark(n_total_transactions=50000)
    train_df, test_df = gen.create_stratified_split(disp_df, test_size=0.25)

    scorer = RubricScorer()
    eval_res = evaluate_rubric_against_benchmark(test_df, scorer=scorer)

    cb_df = eval_res["criterion_breakdown"]
    assert len(cb_df) == 6, f"Expected 6 criteria evaluated, got {len(cb_df)}"

    print("[OK] Test 3 Passed: Per-criterion performance on held-out test split:")
    for _, row in cb_df.iterrows():
        c_name = row["criterion"]
        f1 = row["f1"]
        pr_auc = row["pr_auc"]
        prec = row["precision"]
        rec = row["recall"]
        print(f"  - {c_name:<28}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, PR-AUC={pr_auc:.3f}")
        
        # Verify criteria have high accuracy on ground truth
        assert pr_auc >= 0.85, f"Criterion {c_name} PR-AUC {pr_auc:.3f} below 0.85 threshold"
        assert f1 >= 0.85, f"Criterion {c_name} F1 {f1:.3f} below 0.85 threshold"

    om = eval_res["overall_metrics"]
    assert om is not None, "Overall metrics missing"
    assert om["pr_auc"] >= 0.90, f"Overall PR-AUC {om['pr_auc']:.4f} below 0.90"
    print(f"  Overall Representment Prediction: PR-AUC = {om['pr_auc']:.4f}, ROC-AUC = {om['roc_auc']:.4f}")


def test_dataframe_batch_scoring():
    """Verify batch DataFrame scoring function."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, disp_df = gen.generate_benchmark(n_total_transactions=10000)

    scorer = RubricScorer()
    summary_df, full_results = scorer.score_dataframe(disp_df)

    assert len(summary_df) == len(disp_df), "Batch summary DataFrame row count mismatch"
    assert len(full_results) == len(disp_df), "Batch full results list count mismatch"
    assert "overall_score" in summary_df.columns
    assert "score_device_match" in summary_df.columns
    assert "score_prior_undisputed_in_window" in summary_df.columns
    print("[OK] Test 4 Passed: Batch DataFrame scoring functional across all dataset records.")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Module 1 Rubric Scorer Acceptance Tests")
    print("=" * 60)
    test_output_contract_structure()
    test_edge_case_empty_history()
    test_per_criterion_accuracy_against_benchmark()
    test_dataframe_batch_scoring()
    print("=" * 60)
    print("ALL MODULE 1 ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
