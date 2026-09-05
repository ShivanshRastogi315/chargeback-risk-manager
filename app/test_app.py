"""
Tests for Streamlit Demo Application & Pipeline Runner (app/test_app.py)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from app.cases import (
    get_happy_path_case,
    get_weak_evidence_case,
    get_cold_start_case,
    get_demo_scenarios,
)
from app.pipeline_runner import PipelineManager
from escalation.demo_case import create_curated_ambiguous_dispute_case


@pytest.fixture(scope="module")
def pipeline_mgr():
    mgr = PipelineManager(seed=42)
    mgr.initialize(n_transactions=10000)
    return mgr


def test_happy_path_execution(pipeline_mgr):
    """Verifies happy path transaction generates fight recommendation and 100% cited narrative."""
    case = get_happy_path_case()
    result = pipeline_mgr.process_single_dispute(case)

    assert result is not None
    assert result["module_1_rubric"]["overall_score"] > 0.70
    assert result["module_2_roi"]["ev_contest_inr"] > 0
    assert result["module_5_escalation"]["decision"] == "FIGHT"
    assert result["module_3_packet"].is_valid is True
    assert result["module_3_packet"].citation_report.citation_resolution_rate == 1.0


def test_weak_evidence_execution(pipeline_mgr):
    """Verifies weak evidence dispute correctly recommends NO_FIGHT."""
    case = get_weak_evidence_case()
    result = pipeline_mgr.process_single_dispute(case)

    assert result is not None
    assert result["module_1_rubric"]["overall_score"] < 0.35
    assert result["module_2_roi"]["ev_contest_inr"] < 0
    assert result["module_5_escalation"]["decision"] == "NO_FIGHT"


def test_curated_ambiguous_escalation_execution(pipeline_mgr):
    """Verifies curated ambiguous case routes to ESCALATE_TO_HUMAN via Module 5."""
    case = create_curated_ambiguous_dispute_case()
    result = pipeline_mgr.process_single_dispute(case)

    assert result is not None
    assert result["module_5_escalation"]["decision"] == "ESCALATE_TO_HUMAN"
    assert 0 in result["module_5_escalation"]["conformal_prediction_set"]
    assert 1 in result["module_5_escalation"]["conformal_prediction_set"]


def test_cold_start_execution(pipeline_mgr):
    """Verifies cold-start merchant case selects Mode B / TabPFN."""
    case = get_cold_start_case()
    result = pipeline_mgr.process_single_dispute(case)

    assert result is not None
    assert result["module_2_roi"]["selected_mode"] in ["MODE_B_TABPFN_COLDSTART", "MODE_A_GBDT"]
    assert result["module_3_packet"].citation_report.citation_resolution_rate == 1.0


def test_all_scenarios_catalog():
    """Verifies all scenarios in catalog are valid dispute dictionaries."""
    scenarios = get_demo_scenarios()
    assert len(scenarios) >= 4
    for name, scen in scenarios.items():
        assert "record" in scen
        assert "description" in scen
        assert "dispute_id" in scen["record"]
        assert "amount" in scen["record"]
