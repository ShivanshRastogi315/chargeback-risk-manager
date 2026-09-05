"""
Module 4 — Evaluation Harness Package (eval_harness)

Provides end-to-end multi-module evaluation, honest currency error metrics (₹),
dual win rates (count-based vs amount-weighted), and merchant ROI projections.
"""

from eval_harness.costs import (
    CostModelParameters,
    FinancialEvaluationResult,
    FinancialCostEvaluator,
)
from eval_harness.pipeline_evaluator import (
    FullPipelineEvaluationReport,
    PipelineEvaluator,
)

__all__ = [
    "CostModelParameters",
    "FinancialEvaluationResult",
    "FinancialCostEvaluator",
    "FullPipelineEvaluationReport",
    "PipelineEvaluator",
]
