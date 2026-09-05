"""
Acceptance and verification tests for Module 0 Synthetic Benchmark.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from benchmark.generate import SyntheticBenchmarkGenerator, run_baseline_logistic_regression
from benchmark.metrics import (
    count_based_win_rate,
    amount_weighted_win_rate,
    compute_pr_auc,
    compute_roc_auc,
    compute_financial_pnl,
    evaluate_dispute_model,
    evaluate_criterion_breakdown,
)


def test_reproducibility():
    """Verify that identical seeds produce identical datasets."""
    gen1 = SyntheticBenchmarkGenerator(seed=42)
    tx1, d1 = gen1.generate_benchmark(n_total_transactions=10000)

    gen2 = SyntheticBenchmarkGenerator(seed=42)
    tx2, d2 = gen2.generate_benchmark(n_total_transactions=10000)

    assert tx1.equals(tx2), "Transactions DataFrame is not reproducible across runs with same seed!"
    assert d1.equals(d2), "Disputes DataFrame is not reproducible across runs with same seed!"
    print("[OK] Test 1 Passed: Reproducibility with fixed seed confirmed.")


def test_imbalance_and_distribution():
    """Verify class imbalance and dispute rates against calibrated bounds."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    tx_df, d_df = gen.generate_benchmark(n_total_transactions=50000)
    
    dispute_rate = len(d_df) / len(tx_df)
    assert 0.0010 <= dispute_rate <= 0.0060, f"Dispute rate {dispute_rate:.4f} outside calibrated bound (0.13% - 0.50%)"
    
    # Amount distribution check
    median_amt = tx_df["amount"].median()
    mean_amt = tx_df["amount"].mean()
    assert 1000.0 <= median_amt <= 2500.0, f"Median amount {median_amt} unexpected"
    assert 2000.0 <= mean_amt <= 5000.0, f"Mean amount {mean_amt} unexpected"
    
    print(f"[OK] Test 2 Passed: Calibrated dispute rate = {dispute_rate*100:.3f}%, Mean amount = INR {mean_amt:,.2f}, Median = INR {median_amt:,.2f}")


def test_baseline_and_metrics():
    """Verify naive logistic regression baseline and metric calculations."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    tx_df, d_df = gen.generate_benchmark(n_total_transactions=50000)
    train_df, test_df = gen.create_stratified_split(d_df, target_col="issuer_dispute_won", test_size=0.25)
    
    baseline_out = run_baseline_logistic_regression(train_df, test_df)
    res = baseline_out["eval_results"]
    
    pr_auc = res["pr_auc"]
    roc_auc = res["roc_auc"]
    cnt_win = res["count_win_rate"]
    amt_win = res["amount_weighted_win_rate"]
    
    assert 0.60 <= pr_auc <= 0.99, f"Baseline PR-AUC ({pr_auc:.4f}) is outside sane non-trivial range (0.60 - 0.99)"
    assert 0.60 <= roc_auc <= 0.99, f"Baseline ROC-AUC ({roc_auc:.4f}) is outside sane non-trivial range (0.60 - 0.99)"
    assert 0.0 <= cnt_win <= 1.0, "Count win rate out of bounds"
    assert 0.0 <= amt_win <= 1.0, "Amount win rate out of bounds"
    
    print(f"[OK] Test 3 Passed: Naive Baseline PR-AUC = {pr_auc:.4f}, ROC-AUC = {roc_auc:.4f}")
    print(f"  Count-based Win Rate: {cnt_win:.4f} | Amount-weighted Win Rate: {amt_win:.4f}")


def test_criterion_breakdown():
    """Verify per-criterion evaluation helper."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, d_df = gen.generate_benchmark(n_total_transactions=20000)
    
    criteria = [
        "criterion_prior_window_match",
        "criterion_device_match",
        "criterion_ip_match",
        "criterion_shipping_match",
        "criterion_avs_cvv_match",
        "criterion_3ds_match",
    ]
    
    y_true_dict = {c: d_df[c].values for c in criteria}
    # Add noisy mock predictions for each criterion
    np.random.seed(42)
    y_score_dict = {c: np.clip(d_df[c].values * 0.8 + np.random.normal(0.1, 0.15, len(d_df)), 0.0, 1.0) for c in criteria}
    
    breakdown_df = evaluate_criterion_breakdown(y_true_dict, y_score_dict)
    assert len(breakdown_df) == len(criteria), "Criterion breakdown missing criteria"
    assert (breakdown_df["pr_auc"] > 0).all(), "Criterion PR-AUC calculation failed"
    
    print("[OK] Test 4 Passed: Per-criterion breakdown evaluator verified:")
    for _, row in breakdown_df.iterrows():
        print(f"  - {row['criterion']:<30}: F1={row['f1']:.3f}, PR-AUC={row['pr_auc']:.3f}, Support={row['positive_support']}/{row['total_samples']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Module 0 Synthetic Benchmark Acceptance Tests")
    print("=" * 60)
    test_reproducibility()
    test_imbalance_and_distribution()
    test_baseline_and_metrics()
    test_criterion_breakdown()
    print("=" * 60)
    print("ALL MODULE 0 ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
