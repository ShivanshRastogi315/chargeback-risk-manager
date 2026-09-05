"""
ROI Engine package for Chargeback Risk Manager (Module 2).
"""

from roi_engine.models import (
    ModeAGBDTModel,
    ModeBTabPFNModel,
    FEATURE_COLUMNS,
    extract_features_from_record,
    prepare_feature_matrix,
)
from roi_engine.engine import (
    ROIEngine,
    DEFAULT_REVIEW_COST,
    DEFAULT_FIGHT_LOSE_FEE,
    DEFAULT_HISTORY_THRESHOLD,
)
from roi_engine.evaluation import run_roi_cold_start_comparison

__all__ = [
    "ROIEngine",
    "ModeAGBDTModel",
    "ModeBTabPFNModel",
    "FEATURE_COLUMNS",
    "extract_features_from_record",
    "prepare_feature_matrix",
    "run_roi_cold_start_comparison",
    "DEFAULT_REVIEW_COST",
    "DEFAULT_FIGHT_LOSE_FEE",
    "DEFAULT_HISTORY_THRESHOLD",
]
