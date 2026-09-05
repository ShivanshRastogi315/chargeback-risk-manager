"""
Module 1 — Rubric Aggregator & Core Scorer (rubric_scorer/aggregator.py)

Combines per-criterion scores into an overall evidentiary recommendation
while strictly preserving and returning the full criterion breakdown.

Output Contract:
{
    "overall_score": float,
    "recommendation": str,
    "confidence_interval": [lower, upper],
    "ce3_eligible": bool,
    "qualifying_criteria_count": int,
    "criteria": {
        "prior_undisputed_in_window": {...},
        "device_match": {...},
        "shipping_match": {...},
        "ip_match": {...},
        "cvv_avs_verified": {...},
        "otp_3ds_authenticated": {...},
    },
    "summary": str
}
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from rubric_scorer.criteria import (
    BaseCriterionScorer,
    PriorWindowScorer,
    DeviceMatchScorer,
    ShippingMatchScorer,
    IPMatchScorer,
    CVVAVSScorer,
    ThreeDSScorer,
)

# Default CE3.0 / First-Party Trust evidentiary weights
DEFAULT_WEIGHTS = {
    "prior_undisputed_in_window": 0.30,
    "device_match": 0.25,
    "shipping_match": 0.20,
    "ip_match": 0.15,
    "cvv_avs_verified": 0.05,
    "otp_3ds_authenticated": 0.05,
}

CE3_ELIGIBLE_REASON_CODES = {"10.4", "4837"}


class RubricScorer:
    """
    Decomposed Rubric Scorer for Chargeback Evidence Automation.
    Scores each evidentiary criterion independently and aggregates into an explainable verdict.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        contest_threshold: float = 0.60,
        accept_threshold: float = 0.40,
    ):
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        # Normalize weights so they sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.contest_threshold = contest_threshold
        self.accept_threshold = accept_threshold

        self.scorers: Dict[str, BaseCriterionScorer] = {
            "prior_undisputed_in_window": PriorWindowScorer(),
            "device_match": DeviceMatchScorer(),
            "shipping_match": ShippingMatchScorer(),
            "ip_match": IPMatchScorer(),
            "cvv_avs_verified": CVVAVSScorer(),
            "otp_3ds_authenticated": ThreeDSScorer(),
        }

    def score_record(self, record: Union[Dict[str, Any], pd.Series, Any]) -> Dict[str, Any]:
        """
        Score a single dispute transaction record.
        Always returns both overall score and full per-criterion breakdown.
        """
        # Extract reason code and CE3 eligibility
        reason_code = str(record.get("reason_code", "10.4") if isinstance(record, dict) else getattr(record, "reason_code", "10.4"))
        is_ce3_eligible = (reason_code in CE3_ELIGIBLE_REASON_CODES)
        if isinstance(record, dict) and "reason_ce3_eligible" in record:
            is_ce3_eligible = bool(record["reason_ce3_eligible"])
        elif hasattr(record, "reason_ce3_eligible"):
            is_ce3_eligible = bool(getattr(record, "reason_ce3_eligible"))

        criteria_results: Dict[str, Dict[str, Any]] = {}
        weighted_score_sum = 0.0
        ci_lower_sum = 0.0
        ci_upper_sum = 0.0
        qualifying_criteria_count = 0

        for crit_name, scorer in self.scorers.items():
            w = self.weights.get(crit_name, scorer.default_weight)
            res = scorer.score(record)
            res["weight"] = round(w, 4)
            criteria_results[crit_name] = res

            score_val = res["score"]
            ci = res.get("confidence_interval", [score_val, score_val])
            
            weighted_score_sum += score_val * w
            ci_lower_sum += ci[0] * w
            ci_upper_sum += ci[1] * w

            if res["matched"]:
                qualifying_criteria_count += 1

        # Non-CE3 reason code discount (e.g. 13.1 Goods Not Received, 10.5 Counterfeit)
        if not is_ce3_eligible:
            weighted_score_sum *= 0.40
            ci_lower_sum *= 0.40
            ci_upper_sum *= 0.40

        overall_score = float(np.clip(weighted_score_sum, 0.0, 1.0))
        ci_lower = float(np.clip(ci_lower_sum, 0.0, 1.0))
        ci_upper = float(np.clip(ci_upper_sum, 0.0, 1.0))

        # Determine qualitative recommendation
        if overall_score >= self.contest_threshold:
            recommendation = "RECOMMEND_CONTEST"
        elif overall_score < self.accept_threshold:
            recommendation = "RECOMMEND_ACCEPT"
        else:
            recommendation = "AMBIGUOUS_REVIEW"

        # Build readable summary
        matched_names = [k for k, v in criteria_results.items() if v["matched"]]
        summary = (
            f"Evidence evaluation yielded {qualifying_criteria_count}/6 matching criteria "
            f"({', '.join(matched_names) if matched_names else 'none'}). "
            f"Overall representment confidence: {overall_score*100:.1f}%. "
            f"Recommendation: {recommendation}."
        )

        return {
            "overall_score": round(overall_score, 4),
            "recommendation": recommendation,
            "confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)],
            "reason_code": reason_code,
            "ce3_eligible": is_ce3_eligible,
            "qualifying_criteria_count": qualifying_criteria_count,
            "criteria": criteria_results,
            "summary": summary,
        }

    def score_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Scores an entire DataFrame of dispute records.
        Returns:
            summary_df: DataFrame with overall_score, recommendation, and per-criterion scores.
            full_results: List of comprehensive result dictionaries.
        """
        full_results = []
        rows = []

        for idx, row in df.iterrows():
            res = self.score_record(row)
            full_results.append(res)

            row_summary = {
                "dispute_id": row.get("dispute_id", f"dsp_{idx}"),
                "overall_score": res["overall_score"],
                "recommendation": res["recommendation"],
                "ci_lower": res["confidence_interval"][0],
                "ci_upper": res["confidence_interval"][1],
                "ce3_eligible": res["ce3_eligible"],
                "qualifying_criteria_count": res["qualifying_criteria_count"],
            }
            for crit_name, crit_data in res["criteria"].items():
                row_summary[f"score_{crit_name}"] = crit_data["score"]
                row_summary[f"match_{crit_name}"] = int(crit_data["matched"])

            rows.append(row_summary)

        return pd.DataFrame(rows), full_results
