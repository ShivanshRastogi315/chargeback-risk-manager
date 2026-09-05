"""
Module 5 — Split Conformal Prediction Core (escalation/conformal.py)

Implements rigorous inductive split conformal prediction for binary chargeback
classification and abstention (Angelopoulos & Bates, 2021).

Mathematical Guarantee:
Given a calibration set (X_1, Y_1), ..., (X_n, Y_n), for a new test record X_{n+1}:
    P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha

Where:
- alpha in (0, 1) is the user-specified maximum allowable error rate (e.g. alpha = 0.10 for 90% coverage).
- C(x) subset {0, 1} is the conformal prediction set.
- If C(x) = {0, 1}: Both outcomes are plausible -> ESCALATE TO HUMAN.
- If C(x) = {1}: Only win is plausible -> CONTEST.
- If C(x) = {0}: Only loss is plausible -> NO_CONTEST.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class ConformalPredictionResult:
    """Represents the output of conformal set prediction for a single instance."""
    prediction_set: Set[int]
    p_value_0: float
    p_value_1: float
    is_singleton: bool
    is_ambiguous: bool
    is_empty: bool
    confidence_level: float  # 1 - alpha
    error_guarantee_pct: float  # alpha * 100
    nonconformity_threshold: float
    raw_p_win: float


@dataclass
class ConformalCalibrationReport:
    """Summary of the conformal calibration process and empirical guarantees."""
    n_calibration_samples: int
    alpha: float
    confidence_level: float
    q_threshold: float
    calibration_coverage: float
    empirical_error_rate: float
    guarantee_satisfied: bool


class SplitConformalClassifier:
    """
    Split Conformal Classifier for binary dispute outcome probabilities.
    Derives distribution-free, finite-sample statistical coverage guarantees.
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Significance / error level in (0, 1). Default 0.10 (90% coverage).
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.is_calibrated: bool = False
        self.calibration_scores: Optional[np.ndarray] = None

    def compute_nonconformity(self, p_win: Union[float, np.ndarray], y_true: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Nonconformity score s_i = 1 - P(Y = y_true | X).
        For y_true == 1: s = 1 - p_win
        For y_true == 0: s = p_win
        """
        p = np.asarray(p_win, dtype=float)
        y = np.asarray(y_true, dtype=int)
        
        score = np.where(y == 1, 1.0 - p, p)
        return score

    def calibrate(
        self,
        calib_p_win: Union[List[float], np.ndarray, pd.Series],
        calib_y_true: Union[List[int], np.ndarray, pd.Series],
    ) -> ConformalCalibrationReport:
        """
        Calibrates conformal nonconformity threshold q_hat on held-out calibration set.
        
        Threshold formula:
            p_val = ceil((n + 1) * (1 - alpha)) / n
            q_hat = quantile(scores, p_val)
        """
        p_arr = np.asarray(calib_p_win, dtype=float)
        y_arr = np.asarray(calib_y_true, dtype=int)
        
        n = len(p_arr)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Compute calibration nonconformity scores
        scores = self.compute_nonconformity(p_arr, y_arr)
        self.calibration_scores = scores

        # Conformal empirical quantile index
        # (n + 1)*(1 - alpha) / n, with ceiling interpolation
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        
        # Calculate empirical quantile
        self.q_hat = float(np.quantile(scores, level, method="higher" if hasattr(np, "quantile") else "linear"))
        self.is_calibrated = True

        # Check empirical calibration coverage
        calib_sets = [self.predict_set_single(p) for p in p_arr]
        covered = sum(y in res.prediction_set for y, res in zip(y_arr, calib_sets))
        emp_coverage = covered / n
        emp_error = 1.0 - emp_coverage

        return ConformalCalibrationReport(
            n_calibration_samples=n,
            alpha=self.alpha,
            confidence_level=1.0 - self.alpha,
            q_threshold=round(self.q_hat, 4),
            calibration_coverage=round(emp_coverage, 4),
            empirical_error_rate=round(emp_error, 4),
            guarantee_satisfied=bool(emp_coverage >= (1.0 - self.alpha - (1.0 / n))),
        )

    def predict_set_single(self, p_win: float) -> ConformalPredictionResult:
        """
        Computes conformal prediction set for a single win probability.
        """
        if not self.is_calibrated or self.q_hat is None:
            # Fallback uncalibrated threshold if calibration not run yet
            self.q_hat = 1.0 - self.alpha

        p = float(np.clip(p_win, 0.0, 1.0))
        
        # Test label y=0: s_0 = p_win
        # Test label y=1: s_1 = 1 - p_win
        s_0 = p
        s_1 = 1.0 - p

        pred_set = set()
        if s_0 <= self.q_hat:
            pred_set.add(0)
        if s_1 <= self.q_hat:
            pred_set.add(1)

        # Conformal p-values
        if self.calibration_scores is not None and len(self.calibration_scores) > 0:
            n_cal = len(self.calibration_scores)
            p_val_0 = float(np.sum(self.calibration_scores >= s_0) + 1) / (n_cal + 1)
            p_val_1 = float(np.sum(self.calibration_scores >= s_1) + 1) / (n_cal + 1)
        else:
            p_val_0 = 1.0 - s_0
            p_val_1 = 1.0 - s_1

        is_singleton = len(pred_set) == 1
        is_ambiguous = len(pred_set) > 1
        is_empty = len(pred_set) == 0

        return ConformalPredictionResult(
            prediction_set=pred_set,
            p_value_0=round(p_val_0, 4),
            p_value_1=round(p_val_1, 4),
            is_singleton=is_singleton,
            is_ambiguous=is_ambiguous,
            is_empty=is_empty,
            confidence_level=1.0 - self.alpha,
            error_guarantee_pct=self.alpha * 100.0,
            nonconformity_threshold=round(self.q_hat, 4),
            raw_p_win=round(p, 4),
        )

    def predict_sets(self, p_wins: Union[List[float], np.ndarray, pd.Series]) -> List[ConformalPredictionResult]:
        """Computes conformal prediction sets for a series of probabilities."""
        p_arr = np.asarray(p_wins, dtype=float)
        return [self.predict_set_single(p) for p in p_arr]

    def evaluate_test_coverage(
        self,
        test_p_wins: Union[List[float], np.ndarray, pd.Series],
        test_y_true: Union[List[int], np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """
        Evaluates empirical coverage, abstention/escalation rate, and set sizes on test data.
        """
        p_arr = np.asarray(test_p_wins, dtype=float)
        y_arr = np.asarray(test_y_true, dtype=int)
        n = len(p_arr)

        results = self.predict_sets(p_arr)
        
        covered_count = sum(y in res.prediction_set for y, res in zip(y_arr, results))
        ambiguous_count = sum(res.is_ambiguous for res in results)
        singleton_count = sum(res.is_singleton for res in results)
        empty_count = sum(res.is_empty for res in results)
        
        singleton_0_count = sum(res.prediction_set == {0} for res in results)
        singleton_1_count = sum(res.prediction_set == {1} for res in results)

        emp_coverage = covered_count / n if n > 0 else 0.0
        escalation_rate = (ambiguous_count + empty_count) / n if n > 0 else 0.0

        # Accuracy on singleton (auto-decided) cases
        singleton_correct = sum(
            (res.prediction_set == {y})
            for y, res in zip(y_arr, results)
            if res.is_singleton
        )
        auto_decision_acc = (singleton_correct / singleton_count) if singleton_count > 0 else 0.0

        return {
            "total_test_cases": n,
            "target_confidence": round((1.0 - self.alpha) * 100, 2),
            "target_max_error_pct": round(self.alpha * 100, 2),
            "empirical_coverage_pct": round(emp_coverage * 100, 2),
            "empirical_error_pct": round((1.0 - emp_coverage) * 100, 2),
            "guarantee_met": bool(emp_coverage >= (1.0 - self.alpha - 0.05)),
            "escalation_rate_pct": round(escalation_rate * 100, 2),
            "escalated_cases_count": ambiguous_count + empty_count,
            "auto_decided_cases_count": singleton_count,
            "auto_decision_accuracy_pct": round(auto_decision_acc * 100, 2),
            "set_size_distribution": {
                "ambiguous_both_0_and_1": ambiguous_count,
                "singleton_fight_1_only": singleton_1_count,
                "singleton_no_fight_0_only": singleton_0_count,
                "empty_anomaly": empty_count,
            },
        }
