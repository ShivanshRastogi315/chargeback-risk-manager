"""
Module 6 — Rule Drift Detector (drift/detector.py)

Monitors and detects concept drift in dispute evidentiary rule distributions.
Compares live criterion-weight distributions against the baseline training-time
rule configuration to catch silent representment underperformance.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from drift.rules_loader import (
    RuleVersionConfig,
    load_rule_version,
    get_rule_weights,
    REQUIRED_CRITERIA,
)


@dataclass
class CriterionDriftDetail:
    """Detailed drift statistics for a single evidentiary criterion."""
    criterion: str
    baseline_weight: float
    live_weight: float
    absolute_delta: float
    percentage_change: float
    direction: str  # "INCREASED", "DECREASED", "UNCHANGED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "baseline_weight": round(self.baseline_weight, 4),
            "live_weight": round(self.live_weight, 4),
            "absolute_delta": round(self.absolute_delta, 4),
            "percentage_change": round(self.percentage_change, 2),
            "direction": self.direction,
        }


@dataclass
class RuleDriftReport:
    """Comprehensive diagnostic report output by the Rule Drift Detector."""
    baseline_version: str
    live_version_or_source: str
    is_drift_detected: bool
    severity: str  # "OK", "WARNING", "CRITICAL_DRIFT"
    total_variation_distance: float
    l1_distance: float
    cosine_divergence: float
    primary_divergent_criteria: List[CriterionDriftDetail]
    all_criteria_breakdown: Dict[str, CriterionDriftDetail]
    summary_narrative: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_version": self.baseline_version,
            "live_version_or_source": self.live_version_or_source,
            "is_drift_detected": self.is_drift_detected,
            "severity": self.severity,
            "metrics": {
                "total_variation_distance": round(self.total_variation_distance, 4),
                "l1_distance": round(self.l1_distance, 4),
                "cosine_divergence": round(self.cosine_divergence, 4),
            },
            "primary_divergent_criteria": [c.to_dict() for c in self.primary_divergent_criteria],
            "all_criteria": {k: v.to_dict() for k, v in self.all_criteria_breakdown.items()},
            "summary_narrative": self.summary_narrative,
            "recommended_action": self.recommended_action,
        }


class RuleDriftDetector:
    """
    Statistical and Rule-based Drift Detector for Chargeback Rubric Weights.
    Compares live observed criterion importance against training-time rule specifications.
    """

    def __init__(
        self,
        baseline_version: Union[str, RuleVersionConfig, Dict[str, float]] = "ce3_2023",
        warning_threshold: float = 0.06,
        critical_threshold: float = 0.15,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        if isinstance(baseline_version, str):
            self.baseline_name = baseline_version
            self.baseline_weights = get_rule_weights(baseline_version)
        elif isinstance(baseline_version, RuleVersionConfig):
            self.baseline_name = baseline_version.version
            self.baseline_weights = baseline_version.criteria_weights
        elif isinstance(baseline_version, dict):
            self.baseline_name = "custom_baseline"
            total = sum(baseline_version.values())
            self.baseline_weights = {k: v / total for k, v in baseline_version.items()}
        else:
            raise ValueError(f"Unsupported baseline_version type: {type(baseline_version)}")

    def detect_drift_from_weights(
        self,
        live_weights: Dict[str, float],
        live_source_name: str = "live_observed",
    ) -> RuleDriftReport:
        """
        Compares normalized live weights against baseline weights and generates a drift report.
        """
        # Normalize live weights
        tot = sum(live_weights.values())
        norm_live = {k: live_weights.get(k, 0.0) / (tot if tot > 0 else 1.0) for k in REQUIRED_CRITERIA}
        norm_base = {k: self.baseline_weights.get(k, 0.0) for k in REQUIRED_CRITERIA}

        # Calculate distances
        abs_diffs = []
        criteria_details: Dict[str, CriterionDriftDetail] = {}

        for crit in REQUIRED_CRITERIA:
            b_w = norm_base.get(crit, 0.0)
            l_w = norm_live.get(crit, 0.0)
            diff = l_w - b_w
            abs_diff = abs(diff)
            abs_diffs.append(abs_diff)

            pct_change = (diff / b_w * 100.0) if b_w > 0 else (100.0 if l_w > 0 else 0.0)
            if abs(diff) < 1e-4:
                direction = "UNCHANGED"
            elif diff > 0:
                direction = "INCREASED"
            else:
                direction = "DECREASED"

            criteria_details[crit] = CriterionDriftDetail(
                criterion=crit,
                baseline_weight=b_w,
                live_weight=l_w,
                absolute_delta=diff,
                percentage_change=pct_change,
                direction=direction,
            )

        l1_dist = float(sum(abs_diffs))
        tvd = float(0.5 * l1_dist)

        # Cosine divergence
        v_base = np.array([norm_base[k] for k in REQUIRED_CRITERIA])
        v_live = np.array([norm_live[k] for k in REQUIRED_CRITERIA])
        norm_product = float(np.linalg.norm(v_base) * np.linalg.norm(v_live))
        if norm_product > 0:
            cosine_sim = float(np.dot(v_base, v_live) / norm_product)
            cosine_div = float(1.0 - np.clip(cosine_sim, 0.0, 1.0))
        else:
            cosine_div = 1.0

        # Determine severity
        if tvd >= self.critical_threshold:
            severity = "CRITICAL_DRIFT"
            is_drift = True
        elif tvd >= self.warning_threshold:
            severity = "WARNING"
            is_drift = True
        else:
            severity = "OK"
            is_drift = False

        # Rank primary divergent criteria
        sorted_details = sorted(criteria_details.values(), key=lambda c: abs(c.absolute_delta), reverse=True)
        primary_divergent = [c for c in sorted_details if abs(c.absolute_delta) >= 0.04]
        if not primary_divergent and sorted_details:
            primary_divergent = [sorted_details[0]]

        # Construct narrative
        if severity == "OK":
            summary = (
                f"No significant rule drift detected (TVD = {tvd:.4f} < {self.warning_threshold}). "
                f"Live evidentiary weight distribution closely matches baseline '{self.baseline_name}'."
            )
            action = "MAINTAIN_CURRENT_MODEL"
        elif severity == "WARNING":
            top_crit = primary_divergent[0]
            summary = (
                f"Moderate rule drift detected (TVD = {tvd:.4f}, Warning Threshold = {self.warning_threshold}). "
                f"Noticeable shift in '{top_crit.criterion}' ({top_crit.direction} by {abs(top_crit.percentage_change):.1f}%). "
                f"Monitor representment win rates."
            )
            action = "SCHEDULE_RECALIBRATION_REVIEW"
        else:
            div_summary_parts = [
                f"{c.criterion} ({c.direction} {abs(c.percentage_change):.1f}%, delta {c.absolute_delta:+.3f})"
                for c in primary_divergent[:3]
            ]
            summary = (
                f"CRITICAL RULE DRIFT DETECTED (TVD = {tvd:.4f} >= {self.critical_threshold}) between baseline "
                f"'{self.baseline_name}' and live '{live_source_name}'. Key divergent criteria: "
                + "; ".join(div_summary_parts) + ". Silent win-rate degradation is active."
            )
            action = "RECALIBRATE_RUBRIC_TO_CURRENT_VERSION"

        return RuleDriftReport(
            baseline_version=self.baseline_name,
            live_version_or_source=live_source_name,
            is_drift_detected=is_drift,
            severity=severity,
            total_variation_distance=tvd,
            l1_distance=l1_dist,
            cosine_divergence=cosine_div,
            primary_divergent_criteria=primary_divergent,
            all_criteria_breakdown=criteria_details,
            summary_narrative=summary,
            recommended_action=action,
        )

    def detect_drift_from_version(
        self,
        live_version: Union[str, RuleVersionConfig],
    ) -> RuleDriftReport:
        """Compares baseline against a specific target rule configuration version."""
        if isinstance(live_version, str):
            cfg = load_rule_version(live_version)
        else:
            cfg = live_version
        return self.detect_drift_from_weights(
            live_weights=cfg.criteria_weights,
            live_source_name=cfg.version,
        )

    def estimate_empirical_weights_from_disputes(
        self,
        disputes_df: pd.DataFrame,
        target_col: str = "issuer_dispute_won",
    ) -> Dict[str, float]:
        """
        Estimates empirical criterion importance from live dispute outcomes using regularized regression.
        Enables detecting silent rule churn from live dispute telemetry.
        """
        criteria_col_map = {
            "prior_undisputed_in_window": "criterion_prior_window_match",
            "device_match": "criterion_device_match",
            "shipping_match": "criterion_shipping_match",
            "ip_match": "criterion_ip_match",
            "cvv_avs_verified": "criterion_avs_cvv_match",
            "otp_3ds_authenticated": "criterion_3ds_match",
        }

        # Filter available columns
        active_cols = []
        active_crits = []
        for crit, col in criteria_col_map.items():
            if col in disputes_df.columns:
                active_cols.append(col)
                active_crits.append(crit)

        if len(active_cols) < 2 or target_col not in disputes_df.columns:
            # Fallback if ground-truth criterion columns absent
            return dict(self.baseline_weights)

        X = disputes_df[active_cols].astype(float).values
        y = disputes_df[target_col].astype(int).values

        if len(np.unique(y)) < 2:
            return dict(self.baseline_weights)

        clf = LogisticRegression(C=1.0, random_state=42, max_iter=500)
        clf.fit(X, y)

        raw_coefs = np.maximum(clf.coef_[0], 0.001)  # Enforce non-negativity
        total_c = raw_coefs.sum()
        normalized_weights = {}
        for i, crit in enumerate(active_crits):
            normalized_weights[crit] = float(raw_coefs[i] / total_c)

        return normalized_weights
