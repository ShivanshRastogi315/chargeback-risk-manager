"""
Rubric Scorer package for Chargeback Risk Manager (Module 1).
"""

from rubric_scorer.criteria import (
    BaseCriterionScorer,
    PriorWindowScorer,
    DeviceMatchScorer,
    ShippingMatchScorer,
    IPMatchScorer,
    CVVAVSScorer,
    ThreeDSScorer,
)
from rubric_scorer.aggregator import RubricScorer, DEFAULT_WEIGHTS
from rubric_scorer.evaluation import evaluate_rubric_against_benchmark

__all__ = [
    "RubricScorer",
    "DEFAULT_WEIGHTS",
    "BaseCriterionScorer",
    "PriorWindowScorer",
    "DeviceMatchScorer",
    "ShippingMatchScorer",
    "IPMatchScorer",
    "CVVAVSScorer",
    "ThreeDSScorer",
    "evaluate_rubric_against_benchmark",
]
