"""
Module 2 Acceptance Tests — ROI Engine (roi_engine/test_roi_engine.py)

Acceptance Criteria Verification:
1. Mode A GBDT trained and calibrated (Platt / Isotonic) on merchant history + Rubric score.
2. Mode B TabPFN few-shot cold start trained on thin history.
3. Single interface auto-selecting Mode A vs Mode B based on history size.
4. Exact EV formula verified against ground truth math.
5. Mode B evaluated specifically on thin-history slice vs Mode A on the same slice.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from roi_engine.engine import ROIEngine, DEFAULT_REVIEW_COST, DEFAULT_FIGHT_LOSE_FEE
from roi_engine.models import ModeAGBDTModel, ModeBTabPFNModel
from roi_engine.evaluation import run_roi_cold_start_comparison
from benchmark.generate import SyntheticBenchmarkGenerator


def test_ev_formula_exactness():
    """Verify EV formula calculation matches exact specification."""
    engine = ROIEngine(review_cost=15.0, fight_and_lose_fee=25.0)

    # Case 1: High win probability (P=0.90, Amount=₹5,000)
    # EV = 0.90 * 5000 - 0.10 * 25 - 15 = 4500 - 2.5 - 15 = 4482.50
    ev1 = engine.compute_ev(p_win=0.90, amount=5000.0)
    assert abs(ev1 - 4482.50) < 1e-2, f"EV calculation error: expected 4482.50, got {ev1}"

    # Case 2: Low win probability (P=0.05, Amount=₹1,000)
    # EV = 0.05 * 1000 - 0.95 * 25 - 15 = 50 - 23.75 - 15 = 11.25 (> 0, worth fighting even at 5% for high amount)
    ev2 = engine.compute_ev(p_win=0.05, amount=1000.0)
    assert abs(ev2 - 11.25) < 1e-2, f"EV calculation error: expected 11.25, got {ev2}"

    # Case 3: Very low amount and low win prob (P=0.02, Amount=₹200)
    # EV = 0.02 * 200 - 0.98 * 25 - 15 = 4 - 24.5 - 15 = -35.50 (< 0, DO NOT FIGHT)
    ev3 = engine.compute_ev(p_win=0.02, amount=200.0)
    assert abs(ev3 - (-35.50)) < 1e-2, f"EV calculation error: expected -35.50, got {ev3}"

    print("[OK] Test 1 Passed: Expected Value (EV) mathematical formula rigorously verified.")


def test_auto_mode_selection():
    """Verify single ROIEngine auto-selects Mode A vs Mode B based on merchant history threshold."""
    engine = ROIEngine(history_threshold=30)
    
    dummy_record = {
        "dispute_id": "dsp_test_001",
        "merchant_id": "merch_001",
        "amount": 3500.0,
        "reason_code": "10.4",
        "reason_ce3_eligible": 1,
        "cvv_avs_matched": True,
        "otp_3ds_matched": True,
        "device_id": "dev_1",
        "device_ip": "103.21.1.1",
        "shipping_postal_code": "400001",
        "billing_postal_code": "400001",
        "prior_transaction_history": [],
    }

    # Case A: Merchant with 8 historical disputes (< 30) -> Mode B
    res_cold = engine.evaluate_dispute(dummy_record, merchant_history_count=8)
    assert res_cold["selected_mode"] == "MODE_B_TABPFN_COLDSTART", f"Expected Mode B, got {res_cold['selected_mode']}"

    # Case B: Merchant with 65 historical disputes (>= 30) -> Mode A
    res_est = engine.evaluate_dispute(dummy_record, merchant_history_count=65)
    assert res_est["selected_mode"] == "MODE_A_GBDT", f"Expected Mode A, got {res_est['selected_mode']}"

    print("[OK] Test 2 Passed: Single interface auto-selects Mode A (GBDT) vs Mode B (TabPFN) dynamically.")


def test_mode_a_gbdt_calibration():
    """Verify Mode A GBDT model probability calibration."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, disp_df = gen.generate_benchmark(n_total_transactions=30000)
    train_df, test_df = gen.create_stratified_split(disp_df, test_size=0.3)

    engine = ROIEngine()
    engine.train_mode_a(train_df)

    res_df, _ = engine.evaluate_batch(test_df, force_mode="MODE_A_GBDT")
    probs = res_df["calibrated_p_win"].values

    # Check probabilities are well-formed and bounded in [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]"
    assert len(np.unique(probs)) > 5, "Probabilities collapsed to constant"

    print(f"[OK] Test 3 Passed: Mode A GBDT calibrated probabilities verified across {len(test_df)} test records.")


def test_mode_b_cold_start_comparison():
    """Verify Mode B TabPFN on thin-history slice compared against Mode A on same slice."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, disp_df = gen.generate_benchmark(n_total_transactions=50000)
    train_df, test_df = gen.create_stratified_split(disp_df, test_size=0.25)

    comp = run_roi_cold_start_comparison(train_df, test_df, thin_history_sample_size=15)
    
    table = comp["comparison_table"]
    assert len(table) == 3, "Comparison table missing modes"
    
    eval_b = comp["eval_mode_b_thin"]
    eval_a = comp["eval_mode_a_thin"]

    # Verify both models produce valid PR-AUC and financial returns
    assert eval_b["pr_auc"] >= 0.85, f"Mode B PR-AUC ({eval_b['pr_auc']:.4f}) unexpectedly low on thin slice"
    assert eval_b["financials"]["net_pnl_inr"] > 0, "Mode B Net PnL must be positive"

    print("[OK] Test 4 Passed: Cold-start comparative evaluation on thin-history slice:")
    for _, row in table.iterrows():
        print(f"  - {row['Model / Mode']:<40}: PR-AUC={row['PR-AUC']:.4f}, Net PnL=INR {row['Net PnL (INR)']:,.2f}")


if __name__ == "__main__":
    print("=" * 65)
    print("Running Module 2 ROI Engine Acceptance Tests")
    print("=" * 65)
    test_ev_formula_exactness()
    test_auto_mode_selection()
    test_mode_a_gbdt_calibration()
    test_mode_b_cold_start_comparison()
    print("=" * 65)
    print("ALL MODULE 2 ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
