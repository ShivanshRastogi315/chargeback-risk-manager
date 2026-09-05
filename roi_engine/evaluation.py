"""
Module 2 — ROI Engine Evaluation & Thin-History Benchmark (roi_engine/evaluation.py)

Compares Mode A (GBDT) vs Mode B (TabPFN Few-Shot) on thin-history merchant slices
to demonstrate the cold-start gap being closed.
"""

from typing import Dict, Any, Optional, Tuple
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roi_engine.engine import ROIEngine
from roi_engine.models import ModeAGBDTModel, ModeBTabPFNModel
from benchmark.metrics import (
    evaluate_dispute_model,
    count_based_win_rate,
    amount_weighted_win_rate,
    compute_pr_auc,
    compute_roc_auc,
)
from benchmark.generate import SyntheticBenchmarkGenerator


def run_roi_cold_start_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    thin_history_sample_size: int = 15,
) -> Dict[str, Any]:
    """
    Evaluates Mode A vs Mode B on thin-history merchant cold-start slices.
    
    Procedure:
    1. Filter thin-history merchants ('merchant_tier' == 'thin_history') from test set.
    2. Simulate a thin merchant history pool (e.g. 15 cases) from the training set.
    3. Train Mode A only on this thin pool.
    4. Provide the thin pool as few-shot in-context context to Mode B (TabPFN).
    5. Evaluate both on the held-out thin-history test slice.
    """
    # Slice thin-history cases
    if "merchant_tier" in test_df.columns:
        test_thin = test_df[test_df["merchant_tier"] == "thin_history"].copy().reset_index(drop=True)
        test_enterprise = test_df[test_df["merchant_tier"] == "enterprise"].copy().reset_index(drop=True)
    else:
        test_thin = test_df.iloc[:len(test_df)//2].copy().reset_index(drop=True)
        test_enterprise = test_df.iloc[len(test_df)//2:].copy().reset_index(drop=True)

    if len(test_thin) == 0:
        test_thin = test_df.copy().reset_index(drop=True)

    # Simulate thin training pool (10-20 samples)
    if "merchant_tier" in train_df.columns:
        train_thin_candidates = train_df[train_df["merchant_tier"] == "thin_history"]
        if len(train_thin_candidates) >= thin_history_sample_size:
            thin_train_pool = train_thin_candidates.sample(n=thin_history_sample_size, random_state=42).reset_index(drop=True)
        else:
            thin_train_pool = train_df.sample(n=min(len(train_df), thin_history_sample_size), random_state=42).reset_index(drop=True)
    else:
        thin_train_pool = train_df.sample(n=min(len(train_df), thin_history_sample_size), random_state=42).reset_index(drop=True)

    # 1. Full Mode A trained on all enterprise history
    engine_full = ROIEngine()
    engine_full.train_mode_a(train_df)

    # 2. Mode A trained only on thin merchant history
    engine_thin_mode_a = ROIEngine()
    engine_thin_mode_a.train_mode_a(thin_train_pool)

    # 3. Mode B (TabPFN) trained on thin merchant history
    engine_thin_mode_b = ROIEngine()
    engine_thin_mode_b.train_mode_b(thin_train_pool)

    # Evaluate on held-out thin test slice
    y_true_thin = test_thin["issuer_dispute_won"].values
    amounts_thin = test_thin["amount"].values

    # Predictions
    res_mode_a_thin_df, _ = engine_thin_mode_a.evaluate_batch(test_thin, force_mode="MODE_A_GBDT")
    res_mode_b_thin_df, _ = engine_thin_mode_b.evaluate_batch(test_thin, force_mode="MODE_B_TABPFN_COLDSTART")
    res_mode_a_full_df, _ = engine_full.evaluate_batch(test_thin, force_mode="MODE_A_GBDT")

    eval_mode_a_thin = evaluate_dispute_model(
        y_true=y_true_thin,
        y_score=res_mode_a_thin_df["calibrated_p_win"].values,
        amounts=amounts_thin,
        threshold=0.50,
    )

    eval_mode_b_thin = evaluate_dispute_model(
        y_true=y_true_thin,
        y_score=res_mode_b_thin_df["calibrated_p_win"].values,
        amounts=amounts_thin,
        threshold=0.50,
    )

    eval_mode_a_full = evaluate_dispute_model(
        y_true=y_true_thin,
        y_score=res_mode_a_full_df["calibrated_p_win"].values,
        amounts=amounts_thin,
        threshold=0.50,
    )

    comparison_rows = [
        {
            "Model / Mode": "Mode A (GBDT - Thin History Only)",
            "History Available": f"{len(thin_train_pool)} disputes",
            "PR-AUC": eval_mode_a_thin["pr_auc"],
            "ROC-AUC": eval_mode_a_thin["roc_auc"],
            "Brier Score": eval_mode_a_thin["brier_score"],
            "Count Win Rate": eval_mode_a_thin["count_win_rate"],
            "Amount Win Rate": eval_mode_a_thin["amount_weighted_win_rate"],
            "Net PnL (INR)": eval_mode_a_thin["financials"]["net_pnl_inr"],
            "Recovery %": eval_mode_a_thin["financials"]["recovery_rate"] * 100,
        },
        {
            "Model / Mode": "Mode B (TabPFN - Few-Shot Cold Start)",
            "History Available": f"{len(thin_train_pool)} disputes",
            "PR-AUC": eval_mode_b_thin["pr_auc"],
            "ROC-AUC": eval_mode_b_thin["roc_auc"],
            "Brier Score": eval_mode_b_thin["brier_score"],
            "Count Win Rate": eval_mode_b_thin["count_win_rate"],
            "Amount Win Rate": eval_mode_b_thin["amount_weighted_win_rate"],
            "Net PnL (INR)": eval_mode_b_thin["financials"]["net_pnl_inr"],
            "Recovery %": eval_mode_b_thin["financials"]["recovery_rate"] * 100,
        },
        {
            "Model / Mode": "Mode A (GBDT - Full Cross-Merchant History)",
            "History Available": f"{len(train_df)} disputes",
            "PR-AUC": eval_mode_a_full["pr_auc"],
            "ROC-AUC": eval_mode_a_full["roc_auc"],
            "Brier Score": eval_mode_a_full["brier_score"],
            "Count Win Rate": eval_mode_a_full["count_win_rate"],
            "Amount Win Rate": eval_mode_a_full["amount_weighted_win_rate"],
            "Net PnL (INR)": eval_mode_a_full["financials"]["net_pnl_inr"],
            "Recovery %": eval_mode_a_full["financials"]["recovery_rate"] * 100,
        },
    ]

    return {
        "thin_test_count": len(test_thin),
        "thin_train_history_count": len(thin_train_pool),
        "comparison_table": pd.DataFrame(comparison_rows),
        "models": {
            "MODE_A_GBDT_THIN": {
                "model_name": "Mode A (GBDT - Thin History Only)",
                "history_count": len(thin_train_pool),
                "pr_auc": eval_mode_a_thin["pr_auc"],
                "roc_auc": eval_mode_a_thin["roc_auc"],
                "f1_score": eval_mode_a_thin["f1_score"],
                "brier_score": eval_mode_a_thin["brier_score"],
                "net_pnl_inr": eval_mode_a_thin["financials"]["net_pnl_inr"],
            },
            "MODE_B_TABPFN_FEWSHOT": {
                "model_name": "Mode B (TabPFN - Few-Shot Cold Start)",
                "history_count": len(thin_train_pool),
                "pr_auc": eval_mode_b_thin["pr_auc"],
                "roc_auc": eval_mode_b_thin["roc_auc"],
                "f1_score": eval_mode_b_thin["f1_score"],
                "brier_score": eval_mode_b_thin["brier_score"],
                "net_pnl_inr": eval_mode_b_thin["financials"]["net_pnl_inr"],
            },
            "MODE_A_GBDT_FULL": {
                "model_name": "Mode A (GBDT - Full Cross-Merchant History)",
                "history_count": len(train_df),
                "pr_auc": eval_mode_a_full["pr_auc"],
                "roc_auc": eval_mode_a_full["roc_auc"],
                "f1_score": eval_mode_a_full["f1_score"],
                "brier_score": eval_mode_a_full["brier_score"],
                "net_pnl_inr": eval_mode_a_full["financials"]["net_pnl_inr"],
            },
        },
        "eval_mode_a_thin": eval_mode_a_thin,
        "eval_mode_b_thin": eval_mode_b_thin,
        "eval_mode_a_full": eval_mode_a_full,
    }


def main():
    print("=" * 70)
    print("Evaluating Module 2 ROI Engine: Mode A vs Mode B Cold-Start Comparison")
    print("=" * 70)

    data_dir = os.path.join(REPO_ROOT, "benchmark", "data")
    train_path = os.path.join(data_dir, "train_disputes.parquet")
    test_path = os.path.join(data_dir, "test_disputes.parquet")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
    else:
        print("Generating fresh benchmark...")
        gen = SyntheticBenchmarkGenerator(seed=42)
        _, disp_df = gen.generate_benchmark(n_total_transactions=50000)
        train_df, test_df = gen.create_stratified_split(disp_df, test_size=0.25)

    comp_results = run_roi_cold_start_comparison(train_df, test_df, thin_history_sample_size=15)
    print(f"\nEvaluated on {comp_results['thin_test_count']} thin-history test disputes.")
    print(f"Merchant Cold-Start History Size: {comp_results['thin_train_history_count']} disputes.\n")

    table = comp_results["comparison_table"]
    for _, row in table.iterrows():
        print(f"[{row['Model / Mode']}] (History: {row['History Available']})")
        print(f"  PR-AUC:                   {row['PR-AUC']:.4f}")
        print(f"  ROC-AUC:                  {row['ROC-AUC']:.4f}")
        print(f"  Brier Score:              {row['Brier Score']:.4f}")
        print(f"  Count-based Win Rate:     {row['Count Win Rate']:.4f}")
        print(f"  Amount-weighted Win Rate: {row['Amount Win Rate']:.4f}")
        print(f"  Net Recovered PnL:        INR {row['Net PnL (INR)']:,.2f} ({row['Recovery %']:.2f}% recovery)")
        print("-" * 50)


if __name__ == "__main__":
    main()
