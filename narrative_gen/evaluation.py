"""
Module 3 — Narrative Generator Evaluation Harness (narrative_gen/evaluation.py)

Performs automated evaluation of narrative generation on benchmark datasets:
1. Citation resolution rate across generated claims (Target: 100.0%).
2. Sentence grounding rate across all substantive narrative sentences (Target: 100.0%).
3. Guardrail pass rate (PII check, prohibited language check, required fields check).
4. Deliberate failure detection rate across injected hallucinations & corrupted citations.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np

from narrative_gen.generator import NarrativeGenerator, RepresentmentPacket
from rubric_scorer.aggregator import RubricScorer
from roi_engine.engine import ROIEngine


def evaluate_narrative_generator_on_dataset(
    disputes_df: pd.DataFrame,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates the narrative generator end-to-end against a benchmark disputes DataFrame.
    """
    eval_df = disputes_df.copy()
    if sample_size and len(eval_df) > sample_size:
        eval_df = eval_df.sample(n=sample_size, random_state=random_state)

    rubric_scorer = RubricScorer()
    roi_engine = ROIEngine()
    generator = NarrativeGenerator()

    total_records = len(eval_df)
    packets: List[RepresentmentPacket] = []

    total_sentences_all = 0
    grounded_sentences_all = 0
    total_tags_all = 0
    valid_tags_all = 0
    total_claims_all = 0
    valid_claims_all = 0
    guardrails_passed_count = 0
    packets_fully_valid_count = 0

    reason_code_breakdown = {}

    for _, row in eval_df.iterrows():
        rubric_res = rubric_scorer.score_record(row)
        roi_res = roi_engine.evaluate_dispute(row)
        packet = generator.generate_packet(row, rubric_result=rubric_res, roi_result=roi_res)
        packets.append(packet)

        # Aggregate stats
        c_rep = packet.citation_report
        g_rep = packet.guardrail_report

        total_sentences_all += c_rep.total_sentences
        grounded_sentences_all += c_rep.grounded_sentences
        total_tags_all += c_rep.total_tags
        valid_tags_all += c_rep.valid_tags
        total_claims_all += c_rep.total_claims
        valid_claims_all += c_rep.valid_claims

        if g_rep.passed:
            guardrails_passed_count += 1
        if packet.is_valid:
            packets_fully_valid_count += 1

        rcode = packet.reason_code
        if rcode not in reason_code_breakdown:
            reason_code_breakdown[rcode] = {
                "count": 0,
                "valid_count": 0,
                "claims": 0,
                "valid_claims": 0,
            }
        reason_code_breakdown[rcode]["count"] += 1
        if packet.is_valid:
            reason_code_breakdown[rcode]["valid_count"] += 1
        reason_code_breakdown[rcode]["claims"] += c_rep.total_claims
        reason_code_breakdown[rcode]["valid_claims"] += c_rep.valid_claims

    # Compute overall rates
    overall_citation_res_rate = (valid_claims_all / total_claims_all) if total_claims_all > 0 else 0.0
    overall_sentence_grounding_rate = (grounded_sentences_all / total_sentences_all) if total_sentences_all > 0 else 0.0
    overall_guardrail_pass_rate = (guardrails_passed_count / total_records) if total_records > 0 else 0.0
    overall_packet_validity_rate = (packets_fully_valid_count / total_records) if total_records > 0 else 0.0

    # Test Deliberate Hallucination / Corruption Detection
    corruption_types = [
        "hallucinated_device_id",
        "fabricated_amount",
        "missing_citation",
        "pii_leak",
        "prohibited_language",
    ]
    corruption_results = {}
    sample_record = eval_df.iloc[0]
    rubric_sample = rubric_scorer.score_record(sample_record)
    roi_sample = roi_engine.evaluate_dispute(sample_record)

    for c_type in corruption_types:
        corrupted_packet = generator.generate_corrupted_packet_for_demo(
            sample_record,
            corruption_type=c_type,
            rubric_result=rubric_sample,
            roi_result=roi_sample,
        )
        caught = not corrupted_packet.is_valid
        detected_by = []
        if not corrupted_packet.citation_report.is_100_percent_faithful:
            detected_by.append("citation_validator")
        if not corrupted_packet.guardrail_report.passed:
            detected_by.append("guardrail_runner")

        corruption_results[c_type] = {
            "caught": caught,
            "detected_by": detected_by,
            "unresolved_claims_count": len(corrupted_packet.citation_report.unresolved_claims),
            "guardrail_violations_count": (
                len(corrupted_packet.guardrail_report.pii_violations)
                + len(corrupted_packet.guardrail_report.prohibited_language_violations)
                + len(corrupted_packet.guardrail_report.missing_required_fields)
            ),
        }

    all_corruptions_caught = all(r["caught"] for r in corruption_results.values())

    results = {
        "total_disputes_evaluated": total_records,
        "total_sentences_generated": total_sentences_all,
        "grounded_sentences_count": grounded_sentences_all,
        "sentence_grounding_rate": round(overall_sentence_grounding_rate * 100, 2),
        "total_citation_tags_parsed": total_tags_all,
        "valid_citation_tags_count": valid_tags_all,
        "total_factual_claims": total_claims_all,
        "valid_factual_claims": valid_claims_all,
        "citation_resolution_rate": round(overall_citation_res_rate * 100, 2),
        "guardrail_pass_rate": round(overall_guardrail_pass_rate * 100, 2),
        "fully_valid_packet_rate": round(overall_packet_validity_rate * 100, 2),
        "target_100_percent_faithfulness_met": (overall_citation_res_rate >= 0.9999 and overall_guardrail_pass_rate >= 0.9999),
        "deliberate_corruption_detection_rate": 100.0 if all_corruptions_caught else 0.0,
        "corruption_tests": corruption_results,
        "reason_code_breakdown": reason_code_breakdown,
        "sample_packet": packets[0].to_dict() if packets else {},
    }

    return results
