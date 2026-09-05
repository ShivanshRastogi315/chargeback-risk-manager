"""
Module 4 — Financial Cost Model & Currency Conversions (eval_harness/costs.py)

Converts classification errors (False Positives, False Negatives), representment outcomes,
and operational workflows into explicit currency values (₹ INR), adhering to Section 5
of EXECUTION_PLAYBOOK.md and the non-negotiable Honest Currency Metrics constraint.

Formulas:
- False Positive Cost (₹): Fee incurred from contesting an unwinnable case + review cost
  FP_Cost = N_FP * (fight_and_lose_fee + review_cost)
- False Negative Cost (₹): Unrecovered revenue from erroneously forfeiting a winnable case
  FN_Cost = sum(amount_i - review_cost for i in FN)
- Net Portfolio Recovery (₹):
  Net_PnL = sum(amount_i - review_cost for i in TP) - sum(fight_and_lose_fee + review_cost for i in FP)
- Monthly Merchant ROI Model:
  Net Benefit = (Recovered_improved - Recovered_baseline) + Fees_avoided + Labor_time_saved
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CostModelParameters:
    """Configurable operational and network fee parameters."""
    review_cost_inr: float = 25.0              # Cost per automated dispute review/representment
    fight_and_lose_fee_inr: float = 200.0      # Card network / PSP administrative fee per lost dispute
    manual_review_minutes: float = 35.0        # Time taken per dispute under manual workflow
    automated_review_minutes: float = 5.0      # Time taken per dispute with AI Risk Manager
    loaded_analyst_hourly_cost_inr: float = 400.0  # Fully loaded fraud/dispute analyst hourly cost (₹)


@dataclass
class FinancialEvaluationResult:
    """Detailed currency-denominated breakdown of dispute outcomes."""
    total_disputes: int
    contested_count: int
    uncontested_count: int
    tp_count: int
    fp_count: int
    tn_count: int
    fn_count: int
    tp_gross_amount_inr: float
    tp_net_recovery_inr: float
    fp_cost_inr: float
    fn_cost_inr: float
    tn_fees_saved_inr: float
    total_error_cost_inr: float
    net_realized_pnl_inr: float
    baseline_net_pnl_inr: float
    net_monetary_lift_inr: float
    count_win_rate_contested: float
    amount_win_rate_contested: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_disputes": self.total_disputes,
            "contested_count": self.contested_count,
            "uncontested_count": self.uncontested_count,
            "tp_count": self.tp_count,
            "fp_count": self.fp_count,
            "tn_count": self.tn_count,
            "fn_count": self.fn_count,
            "tp_gross_amount_inr": round(self.tp_gross_amount_inr, 2),
            "tp_net_recovery_inr": round(self.tp_net_recovery_inr, 2),
            "fp_cost_inr": round(self.fp_cost_inr, 2),
            "fn_cost_inr": round(self.fn_cost_inr, 2),
            "tn_fees_saved_inr": round(self.tn_fees_saved_inr, 2),
            "total_error_cost_inr": round(self.total_error_cost_inr, 2),
            "net_realized_pnl_inr": round(self.net_realized_pnl_inr, 2),
            "baseline_net_pnl_inr": round(self.baseline_net_pnl_inr, 2),
            "net_monetary_lift_inr": round(self.net_monetary_lift_inr, 2),
            "count_win_rate_contested": round(self.count_win_rate_contested * 100, 2),
            "amount_win_rate_contested": round(self.amount_win_rate_contested * 100, 2),
        }


class FinancialCostEvaluator:
    """
    Evaluates decisions against financial parameters to calculate FP/FN costs,
    win rates, and net portfolio PnL in Indian Rupees (₹).
    """

    def __init__(self, params: Optional[CostModelParameters] = None):
        self.params = params or CostModelParameters()

    def evaluate_financial_outcomes(
        self,
        should_contest_flags: Union[List[bool], np.ndarray, pd.Series],
        actual_win_labels: Union[List[int], np.ndarray, pd.Series],
        dispute_amounts: Union[List[float], np.ndarray, pd.Series],
    ) -> FinancialEvaluationResult:
        """
        Computes exact monetary realization, FP fee costs, and FN forfeited value.
        """
        contested = np.array(should_contest_flags, dtype=bool)
        actual_won = np.array(actual_win_labels, dtype=int)
        amounts = np.array(dispute_amounts, dtype=float)

        n = len(contested)
        if n == 0:
            return FinancialEvaluationResult(
                0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )

        # Confusion matrix elements for contesting decisions
        tp_mask = contested & (actual_won == 1)   # Contested and won
        fp_mask = contested & (actual_won == 0)   # Contested and lost (unnecessary fees)
        fn_mask = (~contested) & (actual_won == 1)# Declined but was winnable (forfeited revenue)
        tn_mask = (~contested) & (actual_won == 0)# Declined unwinnable (fees saved)

        tp_count = int(np.sum(tp_mask))
        fp_count = int(np.sum(fp_mask))
        fn_count = int(np.sum(fn_mask))
        tn_count = int(np.sum(tn_mask))
        contested_count = int(np.sum(contested))
        uncontested_count = n - contested_count

        # Financial values
        tp_gross_amount = float(np.sum(amounts[tp_mask]))
        tp_net_recovery = float(np.sum(amounts[tp_mask] - self.params.review_cost_inr))

        # FP Cost: Fee for losing + Review cost
        fp_cost = float(fp_count * (self.params.fight_and_lose_fee_inr + self.params.review_cost_inr))

        # FN Cost: Forfeited potential recovery
        fn_cost = float(np.sum(np.maximum(0.0, amounts[fn_mask] - self.params.review_cost_inr)))

        # TN Fees Saved: Avoided lost dispute fee
        tn_fees_saved = float(self.params.fight_and_lose_fee_inr * tn_count)

        # Total Cost of Decision Errors
        total_error_cost = fp_cost + fn_cost

        # Net Realized PnL with AI Risk Manager
        net_realized_pnl = tp_net_recovery - fp_cost

        # Baseline Strategy (Contest All Disputes Naively)
        # All winnable cases won, all unwinnable cases lost
        baseline_won_mask = (actual_won == 1)
        baseline_lost_mask = (actual_won == 0)
        baseline_tp_net = float(np.sum(amounts[baseline_won_mask] - self.params.review_cost_inr))
        baseline_fp_cost = float(np.sum(baseline_lost_mask) * (self.params.fight_and_lose_fee_inr + self.params.review_cost_inr))
        baseline_net_pnl = baseline_tp_net - baseline_fp_cost

        net_monetary_lift = net_realized_pnl - baseline_net_pnl

        # Win Rates
        if contested_count > 0:
            count_win_rate = tp_count / contested_count
            contested_amounts = amounts[contested]
            amount_win_rate = float(np.sum(amounts[tp_mask])) / float(np.sum(contested_amounts)) if np.sum(contested_amounts) > 0 else 0.0
        else:
            count_win_rate = 0.0
            amount_win_rate = 0.0

        return FinancialEvaluationResult(
            total_disputes=n,
            contested_count=contested_count,
            uncontested_count=uncontested_count,
            tp_count=tp_count,
            fp_count=fp_count,
            tn_count=tn_count,
            fn_count=fn_count,
            tp_gross_amount_inr=tp_gross_amount,
            tp_net_recovery_inr=tp_net_recovery,
            fp_cost_inr=fp_cost,
            fn_cost_inr=fn_cost,
            tn_fees_saved_inr=tn_fees_saved,
            total_error_cost_inr=total_error_cost,
            net_realized_pnl_inr=net_realized_pnl,
            baseline_net_pnl_inr=baseline_net_pnl,
            net_monetary_lift_inr=net_monetary_lift,
            count_win_rate_contested=count_win_rate,
            amount_win_rate_contested=amount_win_rate,
        )

    def project_monthly_merchant_roi(
        self,
        monthly_transactions: int = 60000,
        dispute_rate: float = 0.00392,
        avg_disputed_value_inr: float = 2880.0,
        measured_win_rate: float = 0.85,
        baseline_win_rate: float = 0.35,
        contest_rate: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Projects full commercial merchant ROI using measured pipeline metrics
        and the exact formulas from Section 5 of EXECUTION_PLAYBOOK.md.
        """
        disputes_per_month = monthly_transactions * dispute_rate
        contested_disputes = disputes_per_month * contest_rate
        uncontested_disputes = disputes_per_month * (1.0 - contest_rate)

        # Baseline recovery
        baseline_won_cases = contested_disputes * baseline_win_rate
        baseline_lost_cases = contested_disputes * (1.0 - baseline_win_rate)
        baseline_gross_recovery = baseline_won_cases * avg_disputed_value_inr
        baseline_loss_fees = baseline_lost_cases * self.params.fight_and_lose_fee_inr
        baseline_net_recovery = baseline_gross_recovery - baseline_loss_fees

        # System recovery (with measured win rate)
        system_won_cases = contested_disputes * measured_win_rate
        system_lost_cases = contested_disputes * (1.0 - measured_win_rate)
        system_gross_recovery = system_won_cases * avg_disputed_value_inr
        system_loss_fees = system_lost_cases * self.params.fight_and_lose_fee_inr
        system_net_recovery = system_gross_recovery - system_loss_fees

        # Benefits breakdown
        win_rate_lift_benefit = system_net_recovery - baseline_net_recovery
        fees_avoided_benefit = uncontested_disputes * self.params.fight_and_lose_fee_inr
        
        # Labor time savings: (manual - automated) hours * hourly cost
        time_saved_hours_per_case = (self.params.manual_review_minutes - self.params.automated_review_minutes) / 60.0
        labor_savings_benefit = disputes_per_month * time_saved_hours_per_case * self.params.loaded_analyst_hourly_cost_inr

        # Total Net Monthly and Annualized Benefit
        net_monthly_benefit = win_rate_lift_benefit + fees_avoided_benefit + labor_savings_benefit
        net_annual_benefit = net_monthly_benefit * 12.0
        net_annual_lakhs = net_annual_benefit / 100000.0

        return {
            "assumptions": {
                "monthly_transactions": monthly_transactions,
                "dispute_rate": dispute_rate,
                "monthly_disputes": round(disputes_per_month, 1),
                "avg_disputed_value_inr": round(avg_disputed_value_inr, 2),
                "contest_rate": round(contest_rate * 100, 1),
                "baseline_win_rate": round(baseline_win_rate * 100, 1),
                "measured_win_rate": round(measured_win_rate * 100, 1),
                "fight_and_lose_fee_inr": self.params.fight_and_lose_fee_inr,
                "analyst_hourly_cost_inr": self.params.loaded_analyst_hourly_cost_inr,
            },
            "monthly_breakdown_inr": {
                "baseline_net_recovery": round(baseline_net_recovery, 2),
                "system_net_recovery": round(system_net_recovery, 2),
                "win_rate_lift_benefit": round(win_rate_lift_benefit, 2),
                "fees_avoided_benefit": round(fees_avoided_benefit, 2),
                "labor_savings_benefit": round(labor_savings_benefit, 2),
                "net_monthly_benefit": round(net_monthly_benefit, 2),
            },
            "annualized_inr": {
                "net_annual_benefit_inr": round(net_annual_benefit, 2),
                "net_annual_benefit_lakhs": round(net_annual_lakhs, 2),
            },
        }
