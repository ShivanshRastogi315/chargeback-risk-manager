"""
Module 1 — Rubric Scorer Evaluation & Validation (rubric_scorer/evaluation.py)

Evaluates rubric scorer outputs against Module 0 per-criterion ground truth
and overall issuer representment outcomes.
"""

from typing import Dict, Any, Optional, Tuple
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubric_scorer.aggregator import RubricScorer
from benchmark.metrics import (
    evaluate_dispute_model,
    evaluate_criterion_breakdown,
    count_based_win_rate,
    amount_weighted_win_rate,
    compute_pr_auc,
    compute_roc_auc,
)

# Mapping of rubric criterion names to Module 0 ground truth column names
CRITERIA_GROUND_TRUTH_MAP = {
    "prior_undisputed_in_window": "criterion_prior_window_match",
    "device_match": "criterion_device_match",
    "shipping_match": "criterion_shipping_match",
    "ip_match": "criterion_ip_match",
    "cvv_avs_verified": "criterion_avs_cvv_match",
    "otp_3ds_authenticated": "criterion_3ds_match",
}


def evaluate_rubric_against_benchmark(
    test_df: pd.DataFrame,
    scorer: Optional[RubricScorer] = None,
    threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Evaluates RubricScorer against benchmark dataset with ground-truth per-criterion labels.
    
    Returns:
    - per_criterion_df: DataFrame of precision, recall, f1, pr_auc per criterion
    - overall_metrics: Comprehensive metrics for overall win/loss dispute prediction
    - scored_df: DataFrame with predictions attached
    """
    if scorer is None:
        scorer = RubricScorer()

    summary_df, full_results = scorer.score_dataframe(test_df)
    
    # Prepare per-criterion evaluation
    y_true_dict = {}
    y_score_dict = {}
    for crit_name, gt_col in CRITERIA_GROUND_TRUTH_MAP.items():
        if gt_col in test_df.columns:
            y_true_dict[crit_name] = test_df[gt_col].values
            y_score_dict[crit_name] = summary_df[f"score_{crit_name}"].values

    criterion_breakdown_df = evaluate_criterion_breakdown(
        y_true_dict=y_true_dict,
        y_score_dict=y_score_dict,
        threshold=0.50,
    )

    # Evaluate overall representment outcome prediction if ground truth outcome exists
    overall_metrics = None
    if "issuer_dispute_won" in test_df.columns:
        y_true_overall = test_df["issuer_dispute_won"].values
        y_score_overall = summary_df["overall_score"].values
        amounts = test_df["amount"].values if "amount" in test_df.columns else np.ones(len(test_df))

        overall_metrics = evaluate_dispute_model(
            y_true=y_true_overall,
            y_score=y_score_overall,
            amounts=amounts,
            threshold=threshold,
        )

    # Attach ground truth to summary_df for convenience
    scored_df = pd.concat([test_df.reset_index(drop=True), summary_df.drop(columns=["dispute_id"], errors="ignore")], axis=1)

    return {
        "criterion_breakdown": criterion_breakdown_df,
        "overall_metrics": overall_metrics,
        "scored_df": scored_df,
        "full_results": full_results,
    }


def main():
    print("=" * 65)
    print("Evaluating Module 1 Rubric Scorer on Module 0 Benchmark Data")
    print("=" * 65)

    data_path = os.path.join(REPO_ROOT, "benchmark", "data", "test_disputes.parquet")
    csv_fallback = os.path.join(REPO_ROOT, "benchmark", "data", "test_disputes.csv")

    if os.path.exists(data_path):
        test_df = pd.read_parquet(data_path)
    elif os.path.exists(csv_fallback):
        test_df = pd.read_csv(csv_fallback)
    else:
        print("Test dataset not found. Generating fresh benchmark...")
        from benchmark.generate import SyntheticBenchmarkGenerator
        gen = SyntheticBenchmarkGenerator(seed=42)
        _, disp_df = gen.generate_benchmark(n_total_transactions=50000)
        _, test_df = gen.create_stratified_split(disp_df, test_size=0.25)

    scorer = RubricScorer()
    eval_results = evaluate_rubric_against_benchmark(test_df, scorer=scorer)

    print(f"\nEvaluated on {len(test_df)} test dispute records.")
    print("\n--- PER-CRITERION BREAKDOWN PERFORMANCE ---")
    cb_df = eval_results["criterion_breakdown"]
    for _, row in cb_df.iterrows():
        print(
            f"  {row['criterion']:<28} | Prec: {row['precision']:.3f} | "
            f"Rec: {row['recall']:.3f} | F1: {row['f1']:.3f} | "
            f"PR-AUC: {row['pr_auc']:.3f} | Support: {row['positive_support']}/{row['total_samples']}"
        )

    if eval_results["overall_metrics"]:
        om = eval_results["overall_metrics"]
        print("\n--- OVERALL DISPUTE REPRESENTMENT PREDICTION ---")
        print(f"  PR-AUC:                   {om['pr_auc']:.4f}")
        print(f"  ROC-AUC:                  {om['roc_auc']:.4f}")
        print(f"  Brier Score:              {om['brier_score']:.4f}")
        print(f"  Count-based Win Rate:     {om['count_win_rate']:.4f}")
        print(f"  Amount-weighted Win Rate: {om['amount_weighted_win_rate']:.4f}")
        fin = om["financials"]
        print(f"  Recovered Revenue:        INR {fin['recovered_revenue_inr']:,.2f}")
        print(f"  Net Recovered PnL:        INR {fin['net_pnl_inr']:,.2f}")
        print(f"  Recovery Rate:            {fin['recovery_rate']*100:.2f}%")

    print("\n" + "=" * 65)
    print("Rubric Scorer Evaluation Complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
