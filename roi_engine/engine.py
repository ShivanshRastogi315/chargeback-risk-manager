"""
Module 2 — Fight/No-Fight ROI Engine (roi_engine/engine.py)

Commercial Expected Value (EV) Decision Engine for Dispute Contesting.
Supports automatic mode selection:
- Mode A (Calibrated GBDT): when merchant history >= history_threshold (e.g. 30 cases)
- Mode B (TabPFN Few-Shot): when merchant history < history_threshold (cold start)

EV Formula:
EV(contest) = P(win) * dispute_amount - P(lose) * fight_and_lose_fee - review_cost
Contest iff EV(contest) > EV(no_contest) = 0.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from rubric_scorer.aggregator import RubricScorer
from roi_engine.models import (
    ModeAGBDTModel,
    ModeBTabPFNModel,
    FEATURE_COLUMNS,
    extract_features_from_record,
    prepare_feature_matrix,
)

DEFAULT_REVIEW_COST = 15.0        # ₹15 staff / automated processing cost
DEFAULT_FIGHT_LOSE_FEE = 25.0     # ₹25 network representment loss fee
DEFAULT_HISTORY_THRESHOLD = 30    # Threshold to switch from Mode B to Mode A


class ROIEngine:
    """
    Dual-Mode Fight/No-Fight Commercial ROI Engine.
    Computes calibrated expected value to decide whether contesting is commercially rational.
    """

    def __init__(
        self,
        mode_a_model: Optional[ModeAGBDTModel] = None,
        mode_b_model: Optional[ModeBTabPFNModel] = None,
        rubric_scorer: Optional[RubricScorer] = None,
        review_cost: float = DEFAULT_REVIEW_COST,
        fight_and_lose_fee: float = DEFAULT_FIGHT_LOSE_FEE,
        history_threshold: int = DEFAULT_HISTORY_THRESHOLD,
    ):
        self.review_cost = review_cost
        self.fight_and_lose_fee = fight_and_lose_fee
        self.history_threshold = history_threshold

        self.rubric_scorer = rubric_scorer or RubricScorer()
        self.mode_a = mode_a_model or ModeAGBDTModel()
        self.mode_b = mode_b_model or ModeBTabPFNModel()

    def train_mode_a(self, training_disputes_df: pd.DataFrame) -> "ROIEngine":
        """Trains and calibrates Mode A GBDT model on historical dispute records."""
        X_train = prepare_feature_matrix(training_disputes_df, scorer=self.rubric_scorer)
        y_train = training_disputes_df["issuer_dispute_won"].values
        self.mode_a.fit(X_train, y_train)
        return self

    def train_mode_b(self, context_disputes_df: pd.DataFrame) -> "ROIEngine":
        """Pre-fits / prepares Mode B TabPFN model on thin context records."""
        X_ctx = prepare_feature_matrix(context_disputes_df, scorer=self.rubric_scorer)
        y_ctx = context_disputes_df["issuer_dispute_won"].values
        self.mode_b.fit(X_ctx, y_ctx)
        return self

    def compute_ev(
        self,
        p_win: float,
        amount: float,
        review_cost: Optional[float] = None,
        fight_and_lose_fee: Optional[float] = None,
    ) -> float:
        """
        Compute Commercial Expected Value of contesting:
        EV(contest) = P(win) * amount - (1 - P(win)) * fight_and_lose_fee - review_cost
        """
        rc = self.review_cost if review_cost is None else review_cost
        flf = self.fight_and_lose_fee if fight_and_lose_fee is None else fight_and_lose_fee
        p_win_clipped = float(np.clip(p_win, 0.0, 1.0))
        p_lose = 1.0 - p_win_clipped
        
        ev = (p_win_clipped * amount) - (p_lose * flf) - rc
        return round(float(ev), 2)

    def evaluate_dispute(
        self,
        record: Union[Dict[str, Any], pd.Series],
        merchant_history_count: Optional[int] = None,
        merchant_history_df: Optional[pd.DataFrame] = None,
        force_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single dispute record and return commercial contesting verdict.
        
        Auto-selects:
        - Mode B (TabPFN Cold-Start) if merchant history < history_threshold
        - Mode A (Calibrated GBDT) if merchant history >= history_threshold
        """
        # Determine history size
        if merchant_history_count is None:
            if merchant_history_df is not None:
                merchant_history_count = len(merchant_history_df)
            else:
                merchant_history_count = int(
                    record.get("merchant_prior_dispute_count", 0)
                    if isinstance(record, dict)
                    else getattr(record, "merchant_prior_dispute_count", 0)
                )

        # Mode Selection
        if force_mode:
            selected_mode = force_mode.upper()
        else:
            if merchant_history_count < self.history_threshold:
                selected_mode = "MODE_B_TABPFN_COLDSTART"
            else:
                selected_mode = "MODE_A_GBDT"

        # Rubric evaluation
        rubric_res = self.rubric_scorer.score_record(record)
        feat_dict = extract_features_from_record(record, scorer=self.rubric_scorer)
        feat_df = pd.DataFrame([feat_dict])[FEATURE_COLUMNS]

        # Model Inference
        if selected_mode == "MODE_B_TABPFN_COLDSTART":
            if merchant_history_df is not None and len(merchant_history_df) > 0 and not self.mode_b.is_fitted:
                self.train_mode_b(merchant_history_df)
            elif not self.mode_b.is_fitted:
                # Initialize with a minimal prior context if unfitted
                pass

            if self.mode_b.is_fitted:
                p_win = float(self.mode_b.predict_win_proba(feat_df)[0])
            else:
                # Direct Bayesian fallback to rubric score if no context fitted
                p_win = float(rubric_res["overall_score"])
        else:
            if self.mode_a.is_fitted:
                p_win = float(self.mode_a.predict_win_proba(feat_df)[0])
            else:
                p_win = float(rubric_res["overall_score"])

        amount = float(feat_dict["amount"])
        ev_contest = self.compute_ev(p_win=p_win, amount=amount)
        decision = "CONTEST" if ev_contest > 0.0 else "NO_CONTEST"

        dispute_id = (
            record.get("dispute_id", "dsp_unknown")
            if isinstance(record, dict)
            else getattr(record, "dispute_id", "dsp_unknown")
        )
        merchant_id = (
            record.get("merchant_id", "merch_unknown")
            if isinstance(record, dict)
            else getattr(record, "merchant_id", "merch_unknown")
        )

        reasoning = (
            f"Evaluated via {selected_mode} (Merchant history: {merchant_history_count} prior cases). "
            f"Calibrated P(win) = {p_win*100:.1f}%. "
            f"Dispute Amount: INR {amount:,.2f}. "
            f"Expected Net Value of contesting: INR {ev_contest:,.2f} "
            f"(Review Cost: INR {self.review_cost:.2f}, Loss Fee: INR {self.fight_and_lose_fee:.2f}). "
            f"Decision: {decision}."
        )

        return {
            "dispute_id": dispute_id,
            "merchant_id": merchant_id,
            "selected_mode": selected_mode,
            "merchant_history_count": merchant_history_count,
            "decision": decision,
            "should_contest": bool(decision == "CONTEST"),
            "ev_contest_inr": ev_contest,
            "ev_no_contest_inr": 0.0,
            "calibrated_win_probability": round(p_win, 4),
            "dispute_amount_inr": amount,
            "review_cost_inr": self.review_cost,
            "fight_and_lose_fee_inr": self.fight_and_lose_fee,
            "rubric_breakdown": rubric_res,
            "recommendation_reasoning": reasoning,
        }

    def evaluate_batch(
        self,
        disputes_df: pd.DataFrame,
        merchant_history_counts: Optional[Dict[str, int]] = None,
        force_mode: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Evaluates a batch DataFrame of dispute records.
        Returns:
            results_df: DataFrame with predictions and financial ROI metrics.
            full_packets: List of individual decision dictionaries.
        """
        rows = []
        full_packets = []

        for idx, row in disputes_df.iterrows():
            m_id = row.get("merchant_id", "merch_unknown")
            tier = row.get("merchant_tier", "enterprise")
            
            h_count = None
            if merchant_history_counts and m_id in merchant_history_counts:
                h_count = merchant_history_counts[m_id]
            elif tier == "thin_history":
                h_count = 10  # representative thin history size
            elif tier == "enterprise":
                h_count = 80  # representative enterprise size

            res = self.evaluate_dispute(
                record=row,
                merchant_history_count=h_count,
                force_mode=force_mode,
            )
            full_packets.append(res)

            row_dict = {
                "dispute_id": res["dispute_id"],
                "merchant_id": res["merchant_id"],
                "selected_mode": res["selected_mode"],
                "decision": res["decision"],
                "should_contest": res["should_contest"],
                "calibrated_p_win": res["calibrated_win_probability"],
                "amount": res["dispute_amount_inr"],
                "ev_contest_inr": res["ev_contest_inr"],
                "rubric_overall_score": res["rubric_breakdown"]["overall_score"],
            }
            if "issuer_dispute_won" in row:
                row_dict["ground_truth_won"] = int(row["issuer_dispute_won"])
            if "merchant_tier" in row:
                row_dict["merchant_tier"] = row["merchant_tier"]

            rows.append(row_dict)

        if not rows:
            return pd.DataFrame(columns=[
                "dispute_id", "merchant_id", "selected_mode", "decision",
                "should_contest", "calibrated_p_win", "amount", "ev_contest_inr",
                "rubric_overall_score", "ground_truth_won", "merchant_tier"
            ]), full_packets

        return pd.DataFrame(rows), full_packets
