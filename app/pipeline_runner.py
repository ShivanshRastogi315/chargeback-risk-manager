"""
app/pipeline_runner.py — Cached execution manager for end-to-end pipeline execution.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

from benchmark.generate import SyntheticBenchmarkGenerator
from rubric_scorer.aggregator import RubricScorer
from roi_engine.engine import ROIEngine
from narrative_gen.generator import NarrativeGenerator, RepresentmentPacket
from escalation.engine import ConformalEscalationEngine, EscalationDecisionPacket
from drift.detector import RuleDriftDetector
from drift.rules_loader import load_rule_version


class PipelineManager:
    """
    Central pipeline orchestrator for the Streamlit UI.
    Maintains pre-trained and calibrated instances of Modules 0 through 6.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.benchmark_gen = SyntheticBenchmarkGenerator(seed=seed)
        self.rubric_scorer = RubricScorer()
        self.roi_engine = ROIEngine()
        self.narrative_gen = NarrativeGenerator()
        self.conformal_engine: Optional[ConformalEscalationEngine] = None
        self.is_initialized = False
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.calib_df: Optional[pd.DataFrame] = None
        self.rules_2023 = load_rule_version("ce3_2023")
        self.rules_2026 = load_rule_version("ce3_2026_04")
        self.drift_detector = RuleDriftDetector(baseline_version=self.rules_2023)

    def initialize(self, n_transactions: int = 20000):
        """Initializes benchmark splits, trains ROI engine, and calibrates conformal predictor."""
        if self.is_initialized:
            return

        # 1. Generate benchmark dataset
        _, df_dsp = self.benchmark_gen.generate_benchmark(n_total_transactions=n_transactions)
        train_df, test_df = self.benchmark_gen.create_stratified_split(df_dsp, test_size=0.40)
        calib_df, eval_test_df = self.benchmark_gen.create_stratified_split(test_df, test_size=0.50)

        self.train_df = train_df
        self.test_df = eval_test_df
        self.calib_df = calib_df

        # 2. Train ROI Engine Mode A on training split
        self.roi_engine.train_mode_a(train_df)

        # 3. Fit Conformal Escalation Engine on calibration split
        self.conformal_engine = ConformalEscalationEngine(alpha=0.10, roi_engine=self.roi_engine)
        self.conformal_engine.calibrate(calib_df)

        self.is_initialized = True

    def process_single_dispute(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single dispute transaction through Modules 1, 2, 5, 3, and 6.
        Returns complete diagnostic outputs for side-by-side visualization.
        """
        if not self.is_initialized:
            self.initialize()

        # 1. Module 1: Rubric Scorer
        rubric_dict = self.rubric_scorer.score_record(record)

        # 2. Module 2: ROI Engine
        roi_eval = self.roi_engine.evaluate_dispute(record)

        # 3. Module 5: Calibrated Conformal Escalation
        if self.conformal_engine is not None:
            escalation_packet = self.conformal_engine.evaluate_dispute(record)
            escalation_dict = escalation_packet.to_dict()
        else:
            escalation_dict = {}

        # 4. Module 3: Grounded Narrative Generation with Machine Citation Validation
        narrative_packet = self.narrative_gen.generate_packet(
            record=record,
            rubric_result=rubric_dict,
            roi_result=roi_eval,
        )
        narrative_dict = narrative_packet.to_dict()

        # 5. Module 6: Rule Drift Check
        drift_check = self.drift_detector.detect_drift_from_weights(
            self.rules_2026.criteria_weights,
            live_source_name="ce3_2026_04",
        ).to_dict()

        return {
            "record": record,
            "module_1_rubric": rubric_dict,
            "module_2_roi": roi_eval,
            "module_5_escalation": escalation_dict,
            "module_3_narrative": narrative_dict,
            "module_3_packet": narrative_packet,
            "module_6_drift": drift_check,
        }
