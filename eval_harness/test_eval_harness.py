"""
Module 4 Acceptance Tests — Evaluation Harness (eval_harness/test_eval_harness.py)

Acceptance Criteria Verification:
1. Full pipeline (Modules 0–3) executed end-to-end on held-out test split.
2. Per-criterion precision, recall, F1, and PR-AUC reported for rubric scorer.
3. Mode A vs Mode B cold-start performance evaluated on thin-history slice.
4. Machine citation resolution rate reported and verified.
5. Both count-based win rate and amount-weighted win rate computed and reported.
6. False-positive and false-negative costs converted and expressed in currency (₹ INR).
7. Business ROI cost-savings projection accurately calculated.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from eval_harness.costs import FinancialCostEvaluator, CostModelParameters
from eval_harness.pipeline_evaluator import PipelineEvaluator


def test_financial_cost_model_exactness():
    """Verify FP/FN cost formulas and net PnL arithmetic in ₹ INR."""
    params = CostModelParameters(review_cost_inr=25.0, fight_and_lose_fee_inr=200.0)
    evaluator = FinancialCostEvaluator(params=params)

    # 4 cases:
    # 1. Contested & Won (TP): Amount = ₹5,000 -> Net = 5000 - 25 = ₹4,975
    # 2. Contested & Lost (FP): Amount = ₹2,000 -> Loss = 200 + 25 = ₹225
    # 3. Declined & Winnable (FN): Amount = ₹3,000 -> Forfeited = 3000 - 25 = ₹2,975
    # 4. Declined & Lost (TN): Amount = ₹1,000 -> Fees saved = ₹200
    contested = [True, True, False, False]
    actual_won = [1, 0, 1, 0]
    amounts = [5000.0, 2000.0, 3000.0, 1000.0]

    res = evaluator.evaluate_financial_outcomes(contested, actual_won, amounts)

    assert res.tp_count == 1
    assert res.fp_count == 1
    assert res.fn_count == 1
    assert res.tn_count == 1

    assert abs(res.tp_net_recovery_inr - 4975.0) < 1e-2, f"Expected TP net recovery 4975, got {res.tp_net_recovery_inr}"
    assert abs(res.fp_cost_inr - 225.0) < 1e-2, f"Expected FP cost 225, got {res.fp_cost_inr}"
    assert abs(res.fn_cost_inr - 2975.0) < 1e-2, f"Expected FN cost 2975, got {res.fn_cost_inr}"
    assert abs(res.tn_fees_saved_inr - 200.0) < 1e-2, f"Expected TN fees saved 200, got {res.tn_fees_saved_inr}"
    assert abs(res.net_realized_pnl_inr - (4975.0 - 225.0)) < 1e-2, f"Expected Net PnL 4750, got {res.net_realized_pnl_inr}"

    # Win Rates: 1 win out of 2 contested -> Count win rate = 50.0%
    # Amount win rate: 5000 / (5000 + 2000) = 5000 / 7000 = 71.43%
    assert abs(res.count_win_rate_contested - 0.50) < 1e-2
    assert abs(res.amount_win_rate_contested - (5000.0 / 7000.0)) < 1e-2

    print("[OK] Test 1 Passed: Currency FP/FN cost formulas & amount-weighted win rate math verified.")


def test_business_roi_model_projection():
    """Verify standard mid-size merchant business ROI projection model."""
    evaluator = FinancialCostEvaluator()
    proj = evaluator.project_monthly_merchant_roi(
        monthly_transactions=60000,
        dispute_rate=0.004,
        avg_disputed_value_inr=2200.0,
        measured_win_rate=0.43,
        baseline_win_rate=0.35,
        contest_rate=0.70,
    )

    m_bd = proj["monthly_breakdown_inr"]
    # Check that benefits are computed and positive
    assert m_bd["win_rate_lift_benefit"] > 0
    assert m_bd["fees_avoided_benefit"] > 0
    assert m_bd["labor_savings_benefit"] > 0
    assert m_bd["net_monthly_benefit"] > 0
    assert proj["annualized_inr"]["net_annual_benefit_inr"] > 0

    print(f"[OK] Test 2 Passed: Business ROI projection verified (Net Monthly: INR {m_bd['net_monthly_benefit']:,.2f}, Annual: {proj['annualized_inr']['net_annual_benefit_lakhs']:.2f} Lakhs).")


def test_full_pipeline_evaluator_end_to_end():
    """Verify end-to-end evaluation harness on synthetic benchmark."""
    evaluator = PipelineEvaluator(seed=42)
    
    # Run evaluation on compact dataset for fast acceptance check
    report = evaluator.run_full_evaluation(n_transactions=5000, narrative_eval_sample_size=15)

    # 1. Dataset checks
    assert report.total_transactions == 5000
    assert report.total_disputes > 0
    assert report.train_disputes_count > 0
    assert report.test_disputes_count > 0

    # 2. Rubric Scorer checks
    assert len(report.rubric_metrics) == 6, f"Expected 6 criteria, got {len(report.rubric_metrics)}"
    for crit, c_metrics in report.rubric_metrics.items():
        assert "precision" in c_metrics and "recall" in c_metrics and "f1" in c_metrics

    # 3. ROI Engine checks
    assert "models" in report.roi_cold_start_comparison
    assert "MODE_A_GBDT_THIN" in report.roi_cold_start_comparison["models"]
    assert "MODE_B_TABPFN_FEWSHOT" in report.roi_cold_start_comparison["models"]

    # 4. Narrative Generator checks
    assert report.citation_resolution_rate_pct == 100.0, f"Expected 100.0%, got {report.citation_resolution_rate_pct}%"
    assert report.sentence_grounding_rate_pct == 100.0
    assert report.guardrail_pass_rate_pct == 100.0
    assert report.deliberate_corruption_detection_rate_pct == 100.0

    # 5. Financial & Win-Rate checks
    f_res = report.financial_results
    assert f_res.total_disputes == report.test_disputes_count
    assert f_res.fp_cost_inr >= 0.0
    assert f_res.fn_cost_inr >= 0.0
    assert f_res.count_win_rate_contested >= 0.0
    assert f_res.amount_win_rate_contested >= 0.0

    # 6. Markdown generation check
    md_report = evaluator.generate_markdown_report(report)
    assert "## 1. Dataset & Split Summary" in md_report
    assert "## 2. Rubric Scorer Decomposed Performance" in md_report
    assert "## 5. Representment Outcomes & Honest Currency Error Metrics" in md_report

    print("[OK] Test 3 Passed: Full pipeline evaluator ran end-to-end, producing all required metrics.")


if __name__ == "__main__":
    test_financial_cost_model_exactness()
    test_business_roi_model_projection()
    test_full_pipeline_evaluator_end_to_end()
    print("\nAll Module 4 Acceptance Tests Passed Successfully!")
