"""
Module 5 — Calibrated Escalation Engine (escalation/engine.py)

Wraps Module 2's ROI Engine and Module 1's Rubric Scorer in an inductive conformal
prediction layer to make three-way routing decisions:
1. 'FIGHT': Statistically certified winnable dispute with positive EV.
2. 'NO_FIGHT': Statistically certified unwinnable dispute (loss fee avoided).
3. 'ESCALATE_TO_HUMAN': Ambiguous dispute where both outcomes are plausible at target confidence.

Outputs a stated statistical error guarantee (e.g. <= 10.0% error on auto-decided disputes).
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from roi_engine.engine import ROIEngine
from rubric_scorer.aggregator import RubricScorer
from escalation.conformal import (
    SplitConformalClassifier,
    ConformalPredictionResult,
    ConformalCalibrationReport,
)


@dataclass
class EscalationDecisionPacket:
    """Structured three-way decision packet with explicit conformal guarantees."""
    dispute_id: str
    merchant_id: str
    amount: float
    currency: str
    reason_code: str
    decision: str                # 'FIGHT' | 'NO_FIGHT' | 'ESCALATE_TO_HUMAN'
    action: str                  # 'AUTOMATED_CONTEST' | 'ACCEPT_CHARGEBACK' | 'ROUTE_TO_HUMAN_ANALYST'
    conformal_prediction_set: List[int]
    conformal_status: str        # 'SINGLETON_WIN' | 'SINGLETON_LOSS' | 'AMBIGUOUS_SET' | 'EMPTY_ANOMALY'
    confidence_level_pct: float  # e.g. 90.0%
    error_guarantee_pct: float   # e.g. 10.0%
    p_win_calibrated: float
    expected_value_inr: float
    rubric_overall_score: float
    qualifying_criteria_count: int
    escalation_reasons: List[str] = field(default_factory=list)
    guarantee_statement: str = ""
    naive_decision_without_conformal: str = ""
    is_escalated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispute_id": self.dispute_id,
            "merchant_id": self.merchant_id,
            "amount": self.amount,
            "currency": self.currency,
            "reason_code": self.reason_code,
            "decision": self.decision,
            "action": self.action,
            "conformal_prediction_set": self.conformal_prediction_set,
            "conformal_status": self.conformal_status,
            "confidence_level_pct": round(self.confidence_level_pct, 1),
            "error_guarantee_pct": round(self.error_guarantee_pct, 1),
            "p_win_calibrated": round(self.p_win_calibrated, 4),
            "expected_value_inr": round(self.expected_value_inr, 2),
            "rubric_overall_score": round(self.rubric_overall_score, 4),
            "qualifying_criteria_count": self.qualifying_criteria_count,
            "is_escalated": self.is_escalated,
            "naive_decision_without_conformal": self.naive_decision_without_conformal,
            "escalation_reasons": self.escalation_reasons,
            "guarantee_statement": self.guarantee_statement,
        }


class ConformalEscalationEngine:
    """
    Main entry point for Module 5.
    Combines EV modeling with conformal abstention guarantees.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        ambiguity_ev_margin: float = 30.0,
        roi_engine: Optional[ROIEngine] = None,
        rubric_scorer: Optional[RubricScorer] = None,
    ):
        """
        Args:
            alpha: Target error rate (default 0.10 -> 90% confidence guarantee).
            ambiguity_ev_margin: Marginal INR EV band around 0 to flag economic ambiguity.
        """
        self.alpha = alpha
        self.ambiguity_ev_margin = ambiguity_ev_margin
        self.conformal_classifier = SplitConformalClassifier(alpha=alpha)
        self.roi_engine = roi_engine or ROIEngine()
        self.rubric_scorer = rubric_scorer or RubricScorer()
        self.is_calibrated: bool = False
        self.calibration_report: Optional[ConformalCalibrationReport] = None

    def calibrate(
        self,
        calibration_df: pd.DataFrame,
    ) -> ConformalCalibrationReport:
        """
        Calibrates the conformal prediction layer on a held-out calibration dataset.
        Extracts calibrated P(win) from Module 2 and aligns with ground truth issuer decisions.
        """
        p_wins = []
        y_trues = []

        for _, row in calibration_df.iterrows():
            roi_res = self.roi_engine.evaluate_dispute(row)
            p_wins.append(roi_res["calibrated_win_probability"])
            y_trues.append(int(row.get("issuer_dispute_won", 0)))

        report = self.conformal_classifier.calibrate(p_wins, y_trues)
        self.calibration_report = report
        self.is_calibrated = True
        return report

    def evaluate_dispute(
        self,
        record: Union[Dict[str, Any], pd.Series, Any],
        alpha_override: Optional[float] = None,
    ) -> EscalationDecisionPacket:
        """
        Evaluates a single dispute record with conformal prediction escalation.
        """
        # Module 1 Rubric Scoring
        rubric_res = self.rubric_scorer.score_record(record)
        
        # Module 2 ROI Engine Evaluation
        roi_res = self.roi_engine.evaluate_dispute(record)
        
        p_win = float(roi_res["calibrated_win_probability"])
        ev_inr = float(roi_res["ev_contest_inr"])
        naive_contest = bool(roi_res["should_contest"])
        naive_decision = "FIGHT" if naive_contest else "NO_FIGHT"

        # Conformal Prediction
        if alpha_override is not None and alpha_override != self.alpha:
            temp_conformal = SplitConformalClassifier(alpha=alpha_override)
            if self.conformal_classifier.is_calibrated and self.conformal_classifier.calibration_scores is not None:
                temp_conformal.calibration_scores = self.conformal_classifier.calibration_scores
                level = min(1.0, np.ceil((len(self.conformal_classifier.calibration_scores) + 1) * (1.0 - alpha_override)) / len(self.conformal_classifier.calibration_scores))
                temp_conformal.q_hat = float(np.quantile(self.conformal_classifier.calibration_scores, level))
                temp_conformal.is_calibrated = True
            conf_res = temp_conformal.predict_set_single(p_win)
            active_alpha = alpha_override
        else:
            conf_res = self.conformal_classifier.predict_set_single(p_win)
            active_alpha = self.alpha

        pred_set = conf_res.prediction_set
        pred_set_sorted = sorted(list(pred_set))

        # Determine Decision and Escalation Reasons
        escalation_reasons = []
        is_escalated = False

        if conf_res.is_ambiguous:
            # Both {0, 1} in conformal prediction set
            decision = "ESCALATE_TO_HUMAN"
            action = "ROUTE_TO_HUMAN_ANALYST"
            conformal_status = "AMBIGUOUS_SET"
            is_escalated = True
            escalation_reasons.append(
                f"Conformal prediction set contains both [NO_FIGHT (0), FIGHT (1)] at {conf_res.confidence_level*100:.1f}% confidence"
            )
        elif conf_res.is_empty:
            # Out-of-distribution anomaly
            decision = "ESCALATE_TO_HUMAN"
            action = "ROUTE_TO_HUMAN_ANALYST"
            conformal_status = "EMPTY_ANOMALY"
            is_escalated = True
            escalation_reasons.append("Empty conformal prediction set: out-of-distribution dispute telemetry anomaly")
        elif pred_set == {1}:
            # Certified winnable
            conformal_status = "SINGLETON_WIN"
            if ev_inr > 0.0:
                decision = "FIGHT"
                action = "AUTOMATED_CONTEST"
            else:
                decision = "NO_FIGHT"
                action = "DECLINE_UNECONOMIC"
                escalation_reasons.append(f"Winnable probability ({p_win*100:.1f}%) but expected recovery is uneconomic (EV = ₹{ev_inr:.2f})")
        elif pred_set == {0}:
            # Certified unwinnable
            conformal_status = "SINGLETON_LOSS"
            decision = "NO_FIGHT"
            action = "ACCEPT_CHARGEBACK"
        else:
            decision = naive_decision
            action = "AUTOMATED_CONTEST" if naive_contest else "ACCEPT_CHARGEBACK"
            conformal_status = "FALLBACK"

        # Additional domain checks that support escalation rationale
        if is_escalated:
            if abs(ev_inr) <= self.ambiguity_ev_margin:
                escalation_reasons.append(f"Borderline commercial expectation: EV = ₹{ev_inr:.2f} lies in uncertainty margin [±₹{self.ambiguity_ev_margin:.2f}]")
            
            # Telemetry split checks
            crit_data = rubric_res.get("criteria", {})
            matched_crit = [k for k, v in crit_data.items() if v.get("matched", False)]
            unmatched_crit = [k for k, v in crit_data.items() if not v.get("matched", False)]
            if len(matched_crit) >= 2 and len(unmatched_crit) >= 2:
                escalation_reasons.append(f"Split evidentiary telemetry: {len(matched_crit)} criteria matched ({', '.join(matched_crit)}), {len(unmatched_crit)} criteria unmatched")

        # Guarantee Statement
        confidence_pct = (1.0 - active_alpha) * 100.0
        error_pct = active_alpha * 100.0
        guarantee_statement = (
            f"Statistically calibrated conformal guarantee: at most {error_pct:.1f}% error rate "
            f"on auto-decided cases ({confidence_pct:.1f}% marginal coverage guarantee)."
        )

        dispute_id = str(record.get("dispute_id", "dsp_unknown") if isinstance(record, dict) else getattr(record, "dispute_id", "dsp_unknown"))
        merchant_id = str(record.get("merchant_id", "merch_unknown") if isinstance(record, dict) else getattr(record, "merchant_id", "merch_unknown"))
        amount = float(record.get("amount", 0.0) if isinstance(record, dict) else getattr(record, "amount", 0.0))
        currency = str(record.get("currency", "INR") if isinstance(record, dict) else getattr(record, "currency", "INR"))
        reason_code = str(record.get("reason_code", "10.4") if isinstance(record, dict) else getattr(record, "reason_code", "10.4"))

        return EscalationDecisionPacket(
            dispute_id=dispute_id,
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            reason_code=reason_code,
            decision=decision,
            action=action,
            conformal_prediction_set=pred_set_sorted,
            conformal_status=conformal_status,
            confidence_level_pct=confidence_pct,
            error_guarantee_pct=error_pct,
            p_win_calibrated=p_win,
            expected_value_inr=ev_inr,
            rubric_overall_score=float(rubric_res["overall_score"]),
            qualifying_criteria_count=int(rubric_res["qualifying_criteria_count"]),
            escalation_reasons=escalation_reasons,
            guarantee_statement=guarantee_statement,
            naive_decision_without_conformal=naive_decision,
            is_escalated=is_escalated,
        )

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        alpha_override: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, List[EscalationDecisionPacket]]:
        """Evaluates a batch of disputes and returns summary DataFrame and detailed decision packets."""
        packets = []
        rows = []

        for idx, row in df.iterrows():
            packet = self.evaluate_dispute(row, alpha_override=alpha_override)
            packets.append(packet)
            row_dict = packet.to_dict()
            if "issuer_dispute_won" in row:
                row_dict["ground_truth_won"] = int(row["issuer_dispute_won"])
            rows.append(row_dict)

        return pd.DataFrame(rows), packets
