"""
Module 5 — Calibrated Escalation Package (escalation/)

Implements inductive split conformal prediction for automated chargeback representment,
routing statistically ambiguous cases to human analysts with calibrated error guarantees.
"""

from escalation.conformal import (
    SplitConformalClassifier,
    ConformalPredictionResult,
    ConformalCalibrationReport,
)
from escalation.engine import (
    ConformalEscalationEngine,
    EscalationDecisionPacket,
)
from escalation.evaluation import evaluate_escalation_performance
from escalation.demo_case import (
    create_curated_ambiguous_dispute_case,
    generate_and_save_demo_case,
)

__all__ = [
    "SplitConformalClassifier",
    "ConformalPredictionResult",
    "ConformalCalibrationReport",
    "ConformalEscalationEngine",
    "EscalationDecisionPacket",
    "evaluate_escalation_performance",
    "create_curated_ambiguous_dispute_case",
    "generate_and_save_demo_case",
]
