"""
Benchmark package for Chargeback Risk Manager (Module 0).
"""

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

__all__ = [
    "SyntheticBenchmarkGenerator",
    "run_baseline_logistic_regression",
    "count_based_win_rate",
    "amount_weighted_win_rate",
    "compute_pr_auc",
    "compute_roc_auc",
    "compute_financial_pnl",
    "evaluate_dispute_model",
    "evaluate_criterion_breakdown",
]
