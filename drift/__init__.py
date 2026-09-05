"""
Module 6 — Rule-Version Drift Simulator (drift/)

Simulates and detects concept drift in dispute evidentiary rule distributions
(Visa CE3.0 / Mastercard First-Party Trust rule churn).
"""

from drift.rules_loader import (
    RuleVersionConfig,
    load_rule_version,
    get_rule_weights,
    list_available_rule_versions,
    REQUIRED_CRITERIA,
)
from drift.detector import (
    RuleDriftDetector,
    RuleDriftReport,
    CriterionDriftDetail,
)
from drift.simulator import (
    RuleDriftSimulator,
    RuleDriftSimulationResult,
    RuleEvaluationSliceResult,
    simulate_issuer_outcomes_for_rules,
    evaluate_pipeline_under_rules,
)
from drift.evaluation import (
    run_rule_drift_experiment,
    format_drift_markdown_report,
)

__all__ = [
    "RuleVersionConfig",
    "load_rule_version",
    "get_rule_weights",
    "list_available_rule_versions",
    "REQUIRED_CRITERIA",
    "RuleDriftDetector",
    "RuleDriftReport",
    "CriterionDriftDetail",
    "RuleDriftSimulator",
    "RuleDriftSimulationResult",
    "RuleEvaluationSliceResult",
    "simulate_issuer_outcomes_for_rules",
    "evaluate_pipeline_under_rules",
    "run_rule_drift_experiment",
    "format_drift_markdown_report",
]
