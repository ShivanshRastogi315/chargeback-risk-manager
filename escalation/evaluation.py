"""
Module 5 — Escalation Layer Evaluation & Verification (escalation/evaluation.py)

Evaluates conformal coverage, auto-decision accuracy lift, and abstention behavior
across different confidence thresholds (e.g. 95%, 90%, 85% coverage).
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from escalation.engine import ConformalEscalationEngine
from roi_engine.engine import ROIEngine


def evaluate_escalation_performance(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alphas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates conformal escalation layer against empirical test ground truth.
    
    Verifies:
    1. Coverage guarantee: P(Y in C(X)) >= 1 - alpha
    2. Auto-decision accuracy: Accuracy on non-escalated cases vs naive baseline
    3. Escalation rates across significance levels
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20]

    # Initialize and train ROI engine
    roi_engine = ROIEngine()
    roi_engine.train_mode_a(train_df)

    # Initialize Escalation Engine
    engine = ConformalEscalationEngine(alpha=0.10, roi_engine=roi_engine)
    calib_report = engine.calibrate(calib_df)

    alpha_evaluations = {}

    for alpha in alphas:
        target_confidence_pct = round((1.0 - alpha) * 100, 1)
        summary_df, packets = engine.evaluate_batch(test_df, alpha_override=alpha)
        
        y_true = summary_df["ground_truth_won"].values if "ground_truth_won" in summary_df.columns else test_df["issuer_dispute_won"].values
        
        n_total = len(summary_df)
        escalated_mask = summary_df["is_escalated"].values
        n_escalated = int(np.sum(escalated_mask))
        n_autodecided = n_total - n_escalated
        escalation_rate_pct = round((n_escalated / n_total) * 100, 2) if n_total > 0 else 0.0

        # Empirical Coverage on full test set
        covered_count = 0
        for y, p in zip(y_true, packets):
            if y in p.conformal_prediction_set:
                covered_count += 1
        empirical_coverage_pct = round((covered_count / n_total) * 100, 2) if n_total > 0 else 0.0

        # Auto-decision accuracy vs Naive accuracy
        # Naive: accuracy of naive_decision_without_conformal
        naive_correct = sum(
            (1 if p.naive_decision_without_conformal == "FIGHT" else 0) == y
            for y, p in zip(y_true, packets)
        )
        naive_acc_pct = round((naive_correct / n_total) * 100, 2) if n_total > 0 else 0.0

        # Auto-decided accuracy: on non-escalated cases only
        if n_autodecided > 0:
            auto_correct = sum(
                (1 if p.decision == "FIGHT" else 0) == y
                for y, p in zip(y_true, packets)
                if not p.is_escalated
            )
            auto_acc_pct = round((auto_correct / n_autodecided) * 100, 2)
        else:
            auto_acc_pct = 100.0

        alpha_evaluations[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "target_confidence_pct": target_confidence_pct,
            "empirical_coverage_pct": empirical_coverage_pct,
            "empirical_error_pct": round(100.0 - empirical_coverage_pct, 2),
            "guarantee_satisfied": bool(empirical_coverage_pct >= (target_confidence_pct - 5.0)),
            "escalation_rate_pct": escalation_rate_pct,
            "escalated_count": n_escalated,
            "auto_decided_count": n_autodecided,
            "naive_overall_accuracy_pct": naive_acc_pct,
            "auto_decision_accuracy_pct": auto_acc_pct,
            "accuracy_lift_from_abstention_pct": round(auto_acc_pct - naive_acc_pct, 2),
        }

    return {
        "calibration_summary": {
            "n_calibration_samples": calib_report.n_calibration_samples,
            "q_threshold_alpha_10": calib_report.q_threshold,
            "calibration_coverage": round(calib_report.calibration_coverage * 100, 2),
        },
        "test_total_samples": len(test_df),
        "significance_evaluations": alpha_evaluations,
    }
