"""
Module 6 — Rule Drift Simulator (drift/simulator.py)

Simulates the silent evaluation failure caused by concept drift in chargeback rules
(e.g., model tuned to CE3.0 2023 rules evaluated against CE3.0 2026 rules).
Quantifies the exact performance drop in PR-AUC, Win Rate, False Positive / Negative costs,
and net monetary recovery in INR (₹).
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

from benchmark.generate import SyntheticBenchmarkGenerator
from benchmark.metrics import (
    evaluate_dispute_model,
    compute_pr_auc,
    compute_roc_auc,
    count_based_win_rate,
    amount_weighted_win_rate,
)
from rubric_scorer.aggregator import RubricScorer
from eval_harness.costs import FinancialCostEvaluator, CostModelParameters, FinancialEvaluationResult
from drift.rules_loader import (
    RuleVersionConfig,
    load_rule_version,
    get_rule_weights,
)
from drift.detector import RuleDriftDetector, RuleDriftReport


@dataclass
class RuleEvaluationSliceResult:
    """Evaluation metrics for a specific (model_version, environment_version) pair."""
    model_version: str
    env_version: str
    is_matched: bool
    total_disputes: int
    pr_auc: float
    roc_auc: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    count_win_rate_contested: float
    amount_win_rate_contested: float
    contested_count: int
    contested_ratio: float
    tp_count: int
    fp_count: int
    fn_count: int
    tn_count: int
    tp_net_recovery_inr: float
    fp_cost_inr: float
    fn_cost_inr: float
    tn_fees_saved_inr: float
    total_error_cost_inr: float
    net_realized_pnl_inr: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "env_version": self.env_version,
            "is_matched": self.is_matched,
            "total_disputes": self.total_disputes,
            "performance": {
                "pr_auc": round(self.pr_auc, 4),
                "roc_auc": round(self.roc_auc, 4),
                "accuracy": round(self.accuracy, 4),
                "f1_score": round(self.f1_score, 4),
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
            },
            "win_rates": {
                "count_win_rate_contested_pct": round(self.count_win_rate_contested, 2),
                "amount_win_rate_contested_pct": round(self.amount_win_rate_contested, 2),
                "contested_count": self.contested_count,
                "contested_ratio_pct": round(self.contested_ratio * 100.0, 2),
            },
            "confusion_counts": {
                "tp": self.tp_count,
                "fp": self.fp_count,
                "fn": self.fn_count,
                "tn": self.tn_count,
            },
            "financial_metrics_inr": {
                "tp_net_recovery_inr": round(self.tp_net_recovery_inr, 2),
                "fp_cost_inr": round(self.fp_cost_inr, 2),
                "fn_cost_inr": round(self.fn_cost_inr, 2),
                "tn_fees_saved_inr": round(self.tn_fees_saved_inr, 2),
                "total_error_cost_inr": round(self.total_error_cost_inr, 2),
                "net_realized_pnl_inr": round(self.net_realized_pnl_inr, 2),
            },
        }


@dataclass
class RuleDriftSimulationResult:
    """Complete summary of the rule-drift experiment comparing matched vs mismatched regimes."""
    training_version: str
    live_version: str
    matched_baseline: RuleEvaluationSliceResult
    mismatched_silent_failure: RuleEvaluationSliceResult
    matched_adapted: RuleEvaluationSliceResult
    drift_detector_report: RuleDriftReport
    # Degradation Deltas (Mismatched vs Matched Baseline)
    pr_auc_drop: float
    f1_drop: float
    count_win_rate_drop_pct: float
    fp_cost_increase_inr: float
    fn_cost_increase_inr: float
    total_error_cost_increase_inr: float
    net_pnl_loss_inr: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_version": self.training_version,
            "live_version": self.live_version,
            "drift_detection": self.drift_detector_report.to_dict(),
            "performance_drop_summary": {
                "pr_auc_drop": round(self.pr_auc_drop, 4),
                "f1_drop": round(self.f1_drop, 4),
                "count_win_rate_drop_pct": round(self.count_win_rate_drop_pct, 2),
                "fp_cost_increase_inr": round(self.fp_cost_increase_inr, 2),
                "fn_cost_increase_inr": round(self.fn_cost_increase_inr, 2),
                "total_error_cost_increase_inr": round(self.total_error_cost_increase_inr, 2),
                "net_pnl_loss_inr": round(self.net_pnl_loss_inr, 2),
            },
            "matched_baseline": self.matched_baseline.to_dict(),
            "mismatched_silent_failure": self.mismatched_silent_failure.to_dict(),
            "matched_adapted": self.matched_adapted.to_dict(),
        }


def simulate_issuer_outcomes_for_rules(
    disputes_df: pd.DataFrame,
    rule_config: Union[str, RuleVersionConfig],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Computes simulated issuer representment decisions for a dispute DataFrame
    under a specific rule version's weights, decision threshold, and noise model.
    """
    cfg = load_rule_version(rule_config) if isinstance(rule_config, str) else rule_config
    weights = cfg.criteria_weights
    thresh = cfg.decision_threshold
    sigma = cfg.noise_sigma
    non_ce3_penalty = cfg.non_ce3_penalty_factor

    rng = np.random.RandomState(seed)
    df_out = disputes_df.copy()

    # Extract criteria match values
    c_prior = df_out.get("criterion_prior_window_match", 0).astype(float).values
    c_device = df_out.get("criterion_device_match", 0).astype(float).values
    c_shipping = df_out.get("criterion_shipping_match", 0).astype(float).values
    c_ip = df_out.get("criterion_ip_match", 0).astype(float).values
    c_cvv = df_out.get("criterion_avs_cvv_match", 0).astype(float).values
    c_3ds = df_out.get("criterion_3ds_match", 0).astype(float).values

    ce3_eligible = df_out.get("reason_ce3_eligible", 1).astype(int).values

    raw_scores = (
        weights.get("prior_undisputed_in_window", 0.0) * c_prior +
        weights.get("device_match", 0.0) * c_device +
        weights.get("shipping_match", 0.0) * c_shipping +
        weights.get("ip_match", 0.0) * c_ip +
        weights.get("cvv_avs_verified", 0.0) * c_cvv +
        weights.get("otp_3ds_authenticated", 0.0) * c_3ds
    )

    # Apply non-CE3 penalty
    for i in range(len(raw_scores)):
        if ce3_eligible[i] == 0:
            raw_scores[i] *= non_ce3_penalty

    # Add Gaussian noise
    noise = rng.normal(0.0, sigma, size=len(raw_scores))
    noisy_scores = np.clip(raw_scores + noise, 0.0, 1.0)
    won_labels = (noisy_scores >= thresh).astype(int)

    df_out["issuer_raw_rule_score"] = np.round(raw_scores, 4)
    df_out["issuer_noisy_score"] = np.round(noisy_scores, 4)
    df_out["issuer_dispute_won"] = won_labels
    df_out["rule_version_applied"] = cfg.version

    return df_out


def evaluate_pipeline_under_rules(
    model_rule_version: Union[str, RuleVersionConfig],
    env_rule_version: Union[str, RuleVersionConfig],
    disputes_df: pd.DataFrame,
    cost_evaluator: Optional[FinancialCostEvaluator] = None,
    seed: int = 42,
) -> RuleEvaluationSliceResult:
    """
    Evaluates a Rubric Scorer initialized with model_rule_version against dispute
    records whose ground-truth issuer decisions follow env_rule_version.
    """
    m_cfg = load_rule_version(model_rule_version) if isinstance(model_rule_version, str) else model_rule_version
    e_cfg = load_rule_version(env_rule_version) if isinstance(env_rule_version, str) else env_rule_version

    if cost_evaluator is None:
        cost_evaluator = FinancialCostEvaluator()

    # Ground truth decisions in target environment
    env_disputes = simulate_issuer_outcomes_for_rules(disputes_df, e_cfg, seed=seed)
    y_true = env_disputes["issuer_dispute_won"].values
    amounts = env_disputes["amount"].values

    # Model scorer with model_rule_version weights
    scorer = RubricScorer(weights=m_cfg.criteria_weights, contest_threshold=0.55, accept_threshold=0.40)
    summary_df, _ = scorer.score_dataframe(env_disputes)

    y_scores = summary_df["overall_score"].values
    should_contest = (y_scores >= 0.55).astype(bool)

    # Compute metrics
    pr_auc = compute_pr_auc(y_true, y_scores)
    roc_auc = compute_roc_auc(y_true, y_scores)

    # Standard classification metrics
    y_pred_binary = (y_scores >= 0.55).astype(int)
    tp_mask = (y_pred_binary == 1) & (y_true == 1)
    fp_mask = (y_pred_binary == 1) & (y_true == 0)
    fn_mask = (y_pred_binary == 0) & (y_true == 1)
    tn_mask = (y_pred_binary == 0) & (y_true == 0)

    tp_count = int(np.sum(tp_mask))
    fp_count = int(np.sum(fp_mask))
    fn_count = int(np.sum(fn_mask))
    tn_count = int(np.sum(tn_mask))

    prec = (tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
    rec = (tp_count / (tp_count + fn_count)) if (tp_count + fn_count) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = float(np.mean(y_pred_binary == y_true))

    # Financial and Win Rate metrics
    f_res = cost_evaluator.evaluate_financial_outcomes(
        should_contest_flags=pd.Series(should_contest),
        actual_win_labels=pd.Series(y_true),
        dispute_amounts=pd.Series(amounts),
    )

    is_matched = (m_cfg.version == e_cfg.version)

    return RuleEvaluationSliceResult(
        model_version=m_cfg.version,
        env_version=e_cfg.version,
        is_matched=is_matched,
        total_disputes=len(env_disputes),
        pr_auc=float(pr_auc),
        roc_auc=float(roc_auc),
        accuracy=float(acc),
        f1_score=float(f1),
        precision=float(prec),
        recall=float(rec),
        count_win_rate_contested=float(f_res.count_win_rate_contested),
        amount_win_rate_contested=float(f_res.amount_win_rate_contested),
        contested_count=int(f_res.contested_count),
        contested_ratio=float(f_res.contested_count / len(env_disputes)) if len(env_disputes) > 0 else 0.0,
        tp_count=tp_count,
        fp_count=fp_count,
        fn_count=fn_count,
        tn_count=tn_count,
        tp_net_recovery_inr=float(f_res.tp_net_recovery_inr),
        fp_cost_inr=float(f_res.fp_cost_inr),
        fn_cost_inr=float(f_res.fn_cost_inr),
        tn_fees_saved_inr=float(f_res.tn_fees_saved_inr),
        total_error_cost_inr=float(f_res.total_error_cost_inr),
        net_realized_pnl_inr=float(f_res.net_realized_pnl_inr),
    )


class RuleDriftSimulator:
    """
    Simulates rule-version concept drift across historical and current CE3.0 rule configurations.
    Compares matched training regimes against mismatched 'silent failure' regimes.
    """

    def __init__(
        self,
        training_version: str = "ce3_2023",
        live_version: str = "ce3_2026_04",
        cost_params: Optional[CostModelParameters] = None,
        seed: int = 42,
    ):
        self.training_version = training_version
        self.live_version = live_version
        self.cost_evaluator = FinancialCostEvaluator(params=cost_params)
        self.seed = seed
        self.detector = RuleDriftDetector(baseline_version=training_version)

    def run_simulation(
        self,
        n_transactions: int = 50000,
        disputes_df: Optional[pd.DataFrame] = None,
    ) -> RuleDriftSimulationResult:
        """
        Executes the three evaluation regimes:
        1. Matched Baseline: 2023 Model in 2023 Environment
        2. Mismatched Silent Failure: 2023 Model in 2026 Environment
        3. Matched Modern / Adapted: 2026 Model in 2026 Environment
        """
        if disputes_df is None:
            gen = SyntheticBenchmarkGenerator(seed=self.seed)
            _, disputes_df = gen.generate_benchmark(n_total_transactions=n_transactions)

        # 1. Matched Baseline (e.g. 2023 Model in 2023 Env)
        res_baseline = evaluate_pipeline_under_rules(
            model_rule_version=self.training_version,
            env_rule_version=self.training_version,
            disputes_df=disputes_df,
            cost_evaluator=self.cost_evaluator,
            seed=self.seed,
        )

        # 2. Mismatched Silent Failure (e.g. 2023 Model in 2026 Env)
        res_mismatched = evaluate_pipeline_under_rules(
            model_rule_version=self.training_version,
            env_rule_version=self.live_version,
            disputes_df=disputes_df,
            cost_evaluator=self.cost_evaluator,
            seed=self.seed,
        )

        # 3. Matched Adapted (e.g. 2026 Model in 2026 Env)
        res_adapted = evaluate_pipeline_under_rules(
            model_rule_version=self.live_version,
            env_rule_version=self.live_version,
            disputes_df=disputes_df,
            cost_evaluator=self.cost_evaluator,
            seed=self.seed,
        )

        # Run Drift Detector
        drift_report = self.detector.detect_drift_from_version(self.live_version)

        # Compute degradation deltas (Mismatched in 2026 vs Adapted in 2026 / Baseline)
        pr_auc_drop = res_adapted.pr_auc - res_mismatched.pr_auc
        f1_drop = res_adapted.f1_score - res_mismatched.f1_score
        win_rate_drop = res_adapted.count_win_rate_contested - res_mismatched.count_win_rate_contested
        
        fp_cost_increase = res_mismatched.fp_cost_inr - res_adapted.fp_cost_inr
        fn_cost_increase = res_mismatched.fn_cost_inr - res_adapted.fn_cost_inr
        total_error_cost_inc = res_mismatched.total_error_cost_inr - res_adapted.total_error_cost_inr
        net_pnl_loss = res_adapted.net_realized_pnl_inr - res_mismatched.net_realized_pnl_inr

        return RuleDriftSimulationResult(
            training_version=self.training_version,
            live_version=self.live_version,
            matched_baseline=res_baseline,
            mismatched_silent_failure=res_mismatched,
            matched_adapted=res_adapted,
            drift_detector_report=drift_report,
            pr_auc_drop=pr_auc_drop,
            f1_drop=f1_drop,
            count_win_rate_drop_pct=win_rate_drop,
            fp_cost_increase_inr=fp_cost_increase,
            fn_cost_increase_inr=fn_cost_increase,
            total_error_cost_increase_inr=total_error_cost_inc,
            net_pnl_loss_inr=net_pnl_loss,
        )
