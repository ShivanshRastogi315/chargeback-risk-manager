"""
Module 2 — Mode A (Calibrated GBDT) and Mode B (TabPFN Few-Shot) Models (roi_engine/models.py)

Implements:
- Mode A: Gradient-Boosted Decision Trees with isotonic / Platt probability calibration for established merchants.
- Mode B: TabPFN in-context few-shot tabular classifier for cold-start and thin-history merchants.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    _has_lgb = True
except ImportError:
    _has_lgb = False

try:
    from tabpfn import TabPFNClassifier
    _has_tabpfn = True
except Exception:
    _has_tabpfn = False

from rubric_scorer.aggregator import RubricScorer


FEATURE_COLUMNS = [
    "rubric_overall_score",
    "score_prior_undisputed_in_window",
    "score_device_match",
    "score_shipping_match",
    "score_ip_match",
    "score_cvv_avs_verified",
    "score_otp_3ds_authenticated",
    "qualifying_criteria_count",
    "amount",
    "prior_transaction_count",
    "prior_undisputed_window_count",
    "prior_transaction_age_min_days",
    "prior_transaction_age_max_days",
    "reason_ce3_eligible",
    "cvv_avs_matched",
    "otp_3ds_matched",
]


def extract_features_from_record(
    record: Union[Dict[str, Any], pd.Series],
    scorer: Optional[RubricScorer] = None,
) -> Dict[str, float]:
    """
    Extracts tabular numerical features combining Module 1 Rubric breakdown
    with transaction metadata and economics.
    """
    if scorer is None:
        scorer = RubricScorer()

    rubric_res = scorer.score_record(record)
    crit = rubric_res["criteria"]

    def _get(key, default=0.0):
        if isinstance(record, dict):
            return record.get(key, default)
        elif isinstance(record, pd.Series):
            return record.get(key, default)
        return getattr(record, key, default)

    return {
        "rubric_overall_score": float(rubric_res["overall_score"]),
        "score_prior_undisputed_in_window": float(crit["prior_undisputed_in_window"]["score"]),
        "score_device_match": float(crit["device_match"]["score"]),
        "score_shipping_match": float(crit["shipping_match"]["score"]),
        "score_ip_match": float(crit["ip_match"]["score"]),
        "score_cvv_avs_verified": float(crit["cvv_avs_verified"]["score"]),
        "score_otp_3ds_authenticated": float(crit["otp_3ds_authenticated"]["score"]),
        "qualifying_criteria_count": float(rubric_res["qualifying_criteria_count"]),
        "amount": float(_get("amount", 2000.0)),
        "prior_transaction_count": float(_get("prior_transaction_count", 0)),
        "prior_undisputed_window_count": float(_get("prior_undisputed_window_count", 0)),
        "prior_transaction_age_min_days": float(_get("prior_transaction_age_min_days", -1.0)),
        "prior_transaction_age_max_days": float(_get("prior_transaction_age_max_days", -1.0)),
        "reason_ce3_eligible": float(1.0 if _get("reason_ce3_eligible", 1) else 0.0),
        "cvv_avs_matched": float(1.0 if _get("cvv_avs_matched", False) else 0.0),
        "otp_3ds_matched": float(1.0 if _get("otp_3ds_matched", False) else 0.0),
    }


def prepare_feature_matrix(
    df: pd.DataFrame,
    scorer: Optional[RubricScorer] = None,
) -> pd.DataFrame:
    """Prepares feature matrix DataFrame for model training or inference."""
    if scorer is None:
        scorer = RubricScorer()
    
    rows = []
    for _, row in df.iterrows():
        feat = extract_features_from_record(row, scorer=scorer)
        rows.append(feat)
    return pd.DataFrame(rows)[FEATURE_COLUMNS]


class ModeAGBDTModel:
    """
    Mode A: Calibrated Gradient-Boosted Trees Classifier for merchants with sufficient history.
    Calibrates probabilities via Isotonic Regression or Platt Scaling (Sigmoid).
    """

    def __init__(
        self,
        calibration_method: str = "isotonic",  # 'isotonic' or 'sigmoid'
        random_state: int = 42,
    ):
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.is_fitted = False

        if _has_lgb:
            base_estimator = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                random_state=random_state,
                verbose=-1,
            )
        else:
            base_estimator = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=5,
                random_state=random_state,
            )

        self.model = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.calibration_method,
            cv=3,
        )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "ModeAGBDTModel":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        unique_classes, counts = np.unique(y_arr, return_counts=True)
        min_class_count = min(counts) if len(counts) > 0 else 0

        # When class count or sample size is too small for standard 3-fold CV:
        if len(unique_classes) < 2 or min_class_count < 3:
            # Fallback for thin history: fit uncalibrated base estimator directly
            if _has_lgb:
                self.model = lgb.LGBMClassifier(
                    n_estimators=30,
                    max_depth=3,
                    learning_rate=0.05,
                    min_child_samples=max(1, min_class_count),
                    random_state=self.random_state,
                    verbose=-1,
                )
            else:
                self.model = HistGradientBoostingClassifier(
                    max_iter=30,
                    max_depth=3,
                    min_samples_leaf=1,
                    random_state=self.random_state,
                )
            
            if len(unique_classes) < 2:
                # Augment with dummy boundary if completely 1-class
                dummy_x = np.ones((1, X_arr.shape[1]))
                X_arr = np.vstack([X_arr, dummy_x])
                y_arr = np.concatenate([y_arr, [1 if unique_classes[0] == 0 else 0]])

            self.model.fit(X_arr, y_arr)
        else:
            # Standard calibrated GBDT
            cv_folds = min(3, min_class_count)
            if _has_lgb:
                base_estimator = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=max(2, min_class_count // 2),
                    random_state=self.random_state,
                    verbose=-1,
                )
            else:
                base_estimator = HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=4,
                    learning_rate=0.05,
                    min_samples_leaf=max(2, min_class_count // 2),
                    random_state=self.random_state,
                )

            self.model = CalibratedClassifierCV(
                estimator=base_estimator,
                method=self.calibration_method,
                cv=cv_folds,
            )
            self.model.fit(X_arr, y_arr)

        self.is_fitted = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=float)
        probs = self.model.predict_proba(X_arr)
        if probs.shape[1] == 1:
            # Degenerate single-class output
            p1 = np.full(len(X_arr), float(probs[0, 0]))
            return np.column_stack([1.0 - p1, p1])
        return probs

    def predict_win_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs[:, 1]


class ModeBTabPFNModel:
    """
    Mode B: In-Context Pretrained Tabular Foundation Model (TabPFN)
    designed specifically for thin-history / cold-start merchants (5-25 cases).
    """

    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = 4,
        random_state: int = 42,
    ):
        self.device = device
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.is_fitted = False
        self.use_fallback = False

        if _has_tabpfn:
            try:
                self.model = TabPFNClassifier(
                    device=self.device,
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                )
            except Exception:
                self.use_fallback = True
                self.model = HistGradientBoostingClassifier(
                    max_iter=30,
                    max_depth=3,
                    min_samples_leaf=2,
                    l2_regularization=1.0,
                    random_state=self.random_state,
                )
        else:
            self.use_fallback = True
            self.model = HistGradientBoostingClassifier(
                max_iter=30,
                max_depth=3,
                min_samples_leaf=2,
                l2_regularization=1.0,
                random_state=self.random_state,
            )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "ModeBTabPFNModel":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        if len(np.unique(y_arr)) < 2:
            # If thin history only contains 1 class, augment with dummy reference boundary
            dummy_x0 = np.zeros((1, X_arr.shape[1]))
            dummy_x1 = np.ones((1, X_arr.shape[1]))
            X_arr = np.vstack([X_arr, dummy_x0, dummy_x1])
            y_arr = np.concatenate([y_arr, [0, 1]])

        try:
            self.model.fit(X_arr, y_arr)
        except Exception:
            self.use_fallback = True
            self.model = HistGradientBoostingClassifier(
                max_iter=30,
                max_depth=3,
                min_samples_leaf=2,
                random_state=self.random_state,
            )
            self.model.fit(X_arr, y_arr)

        self.is_fitted = True
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=float)
        try:
            probs = self.model.predict_proba(X_arr)
        except Exception:
            probs = np.column_stack([np.full(len(X_arr), 0.5), np.full(len(X_arr), 0.5)])

        if probs.shape[1] == 1:
            p1 = np.full(len(X_arr), float(probs[0, 0]))
            return np.column_stack([1.0 - p1, p1])
        return probs

    def predict_win_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs[:, 1]
