"""
Module 3 Acceptance Tests — Grounded Narrative Generator (narrative_gen/test_narrative_gen.py)

Acceptance Criteria Verification:
1. 100% of generated claims carry a valid citation tag resolving to a real field in the input record.
2. 100% of generated substantive sentences are machine-grounded with resolved tags.
3. Guardrails pass enforces 0 PII leaks (unmasked PAN/CVV), 0 prohibited terms, and presence of mandatory fields.
4. Deliberate failure test: 100% of injected hallucinations, missing tags, and violations are caught and rejected.
5. Evaluated across diverse reason codes (Visa 10.4, MC 4837, 10.5, 13.1, 4853) on synthetic benchmark.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from narrative_gen.generator import NarrativeGenerator, RepresentmentPacket
from narrative_gen.citation_validator import CitationValidator, ValidationReport
from narrative_gen.guardrails import GuardrailRunner, PIIChecker, ProhibitedLanguageChecker, RequiredFieldsChecker
from narrative_gen.evaluation import evaluate_narrative_generator_on_dataset
from rubric_scorer.aggregator import RubricScorer
from roi_engine.engine import ROIEngine
from benchmark.generate import SyntheticBenchmarkGenerator


def get_mock_dispute_record():
    return {
        "dispute_id": "dsp_tx_test_001",
        "transaction_id": "tx_test_001",
        "merchant_id": "merch_001",
        "amount": 4250.0,
        "currency": "INR",
        "timestamp": "2025-08-15T10:30:00",
        "reason_code": "10.4",
        "reason_name": "Other Fraud - Card-Absent Environment",
        "reason_network": "Visa",
        "reason_ce3_eligible": 1,
        "card_network": "Visa",
        "card_type": "Credit",
        "card_bin": "411111",
        "card_last4": "1234",
        "issuer_bank": "HDFC Bank",
        "customer_name": "Aditya Sharma",
        "customer_email": "aditya.sharma@example.com",
        "device_id": "dev_9988aabbcc",
        "device_type": "mobile_ios",
        "device_ip": "103.21.50.10",
        "shipping_city": "Mumbai",
        "shipping_postal_code": "400001",
        "billing_city": "Mumbai",
        "billing_postal_code": "400001",
        "cvv_avs_matched": True,
        "otp_3ds_matched": True,
        "prior_undisputed_window_count": 2,
        "prior_transaction_age_min_days": 150.0,
        "prior_transaction_age_max_days": 280.0,
        "criterion_prior_window_match": 1,
        "criterion_device_match": 1,
        "criterion_ip_match": 1,
        "criterion_shipping_match": 1,
        "criterion_avs_cvv_match": 1,
        "criterion_3ds_match": 1,
        "total_criteria_matched": 6,
        "prior_transaction_history": [
            {
                "prior_tx_id": "ptx_001",
                "age_days": 180.0,
                "device_id": "dev_9988aabbcc",
                "ip_address": "103.21.50.10",
                "shipping_pincode": "400001",
                "status": "settled_undisputed",
                "in_ce3_qualifying_window": True,
            },
            {
                "prior_tx_id": "ptx_002",
                "age_days": 240.0,
                "device_id": "dev_9988aabbcc",
                "ip_address": "103.21.50.10",
                "shipping_pincode": "400001",
                "status": "settled_undisputed",
                "in_ce3_qualifying_window": True,
            },
        ],
    }


def test_single_packet_100_percent_citation_resolution():
    """Verify that a generated packet achieves exact 100.0% citation resolution."""
    record = get_mock_dispute_record()
    rubric_scorer = RubricScorer()
    roi_engine = ROIEngine()
    generator = NarrativeGenerator()

    rubric_res = rubric_scorer.score_record(record)
    roi_res = roi_engine.evaluate_dispute(record)

    packet = generator.generate_packet(record, rubric_result=rubric_res, roi_result=roi_res)

    assert packet.is_valid, "Generated representment packet should be fully valid."
    assert packet.citation_report.is_100_percent_faithful, "Packet must be 100% faithful to source record."
    assert packet.citation_report.citation_resolution_rate == 1.0, f"Expected 1.0 resolution rate, got {packet.citation_report.citation_resolution_rate}"
    assert packet.citation_report.sentence_grounding_rate == 1.0, f"Expected 1.0 sentence grounding rate, got {packet.citation_report.sentence_grounding_rate}"
    assert len(packet.citation_report.unresolved_claims) == 0, f"Unresolved claims found: {packet.citation_report.unresolved_claims}"
    assert packet.guardrail_report.passed, f"Guardrail failed: {packet.guardrail_report.details}"

    print(f"[OK] Test 1 Passed: 100% Citation Resolution verified ({packet.citation_report.valid_claims}/{packet.citation_report.total_claims} claims resolved across {packet.citation_report.total_sentences} sentences).")


def test_guardrails_pii_and_prohibited_language():
    """Verify PIIChecker and ProhibitedLanguageChecker catch violations."""
    pii = PIIChecker()
    lang = ProhibitedLanguageChecker()
    req = RequiredFieldsChecker()

    dummy_rec = get_mock_dispute_record()

    # 1. Test PII detection of exposed unmasked PAN
    bad_text_pan = "Customer card number is 4111 2222 3333 4444 which was charged."
    pan_violations = pii.check(bad_text_pan, dummy_rec)
    assert len(pan_violations) > 0, "PIIChecker must catch unmasked 16-digit PAN."

    # 2. Test PII detection of exposed CVV
    bad_text_cvv = "Customer entered CVV: 789 during checkout."
    cvv_violations = pii.check(bad_text_cvv, dummy_rec)
    assert len(cvv_violations) > 0, "PIIChecker must catch exposed CVV code."

    # 3. Test Prohibited Language
    bad_text_lang = "The customer is a scammer and liar who committed fraud."
    lang_violations = lang.check(bad_text_lang)
    assert len(lang_violations) >= 2, f"ProhibitedLanguageChecker must catch abusive terms, got {lang_violations}"

    # 4. Clean text should have 0 violations
    clean_text = (
        "Dispute dsp_tx_test_001 for transaction tx_test_001 in amount 4250.0 INR on 2025-08-15T10:30:00 "
        "for merchant merch_001 under reason_code 10.4."
    )
    assert len(pii.check(clean_text, dummy_rec)) == 0
    assert len(lang.check(clean_text)) == 0
    assert len(req.check(clean_text, dummy_rec)) == 0

    print("[OK] Test 2 Passed: PII, Prohibited Language, and Required Field Guardrails functioning as expected.")


def test_deliberate_hallucination_detection():
    """Verify that any injected hallucination or ungrounded claim is caught and rejected."""
    record = get_mock_dispute_record()
    generator = NarrativeGenerator()

    # Test hallucinated device ID
    p_bad_dev = generator.generate_corrupted_packet_for_demo(record, corruption_type="hallucinated_device_id")
    assert not p_bad_dev.is_valid, "Corrupted packet with hallucinated device ID must be marked invalid."
    assert not p_bad_dev.citation_report.is_100_percent_faithful
    assert len(p_bad_dev.citation_report.unresolved_claims) > 0

    # Test fabricated transaction amount
    p_bad_amt = generator.generate_corrupted_packet_for_demo(record, corruption_type="fabricated_amount")
    assert not p_bad_amt.is_valid, "Corrupted packet with fabricated amount must be rejected."
    assert not p_bad_amt.citation_report.is_100_percent_faithful

    # Test missing citation tag
    p_bad_cite = generator.generate_corrupted_packet_for_demo(record, corruption_type="missing_citation")
    assert not p_bad_cite.is_valid, "Packet with missing citation tag must be rejected as ungrounded."
    assert p_bad_cite.citation_report.ungrounded_sentences > 0

    # Test PII leak injection
    p_bad_pii = generator.generate_corrupted_packet_for_demo(record, corruption_type="pii_leak")
    assert not p_bad_pii.is_valid, "Packet with PII leak must be caught by guardrail."
    assert len(p_bad_pii.guardrail_report.pii_violations) > 0

    # Test prohibited language injection
    p_bad_lang = generator.generate_corrupted_packet_for_demo(record, corruption_type="prohibited_language")
    assert not p_bad_lang.is_valid, "Packet with prohibited language must be caught by guardrail."
    assert len(p_bad_lang.guardrail_report.prohibited_language_violations) > 0

    print("[OK] Test 3 Passed: 100% of deliberate hallucinations, falsifications, and guardrail violations caught.")


def test_all_reason_codes_and_benchmark_dataset():
    """Verify narrative generation across diverse reason codes and synthetic benchmark cases."""
    gen = SyntheticBenchmarkGenerator(seed=42)
    df_tx, df_dsp = gen.generate_benchmark(n_total_transactions=5000)

    eval_results = evaluate_narrative_generator_on_dataset(df_dsp, sample_size=50)

    assert eval_results["target_100_percent_faithfulness_met"], "Must achieve 100% citation resolution on benchmark disputes."
    assert eval_results["citation_resolution_rate"] == 100.0, f"Expected 100.0%, got {eval_results['citation_resolution_rate']}%"
    assert eval_results["sentence_grounding_rate"] == 100.0, f"Expected 100.0%, got {eval_results['sentence_grounding_rate']}%"
    assert eval_results["guardrail_pass_rate"] == 100.0, f"Expected 100.0%, got {eval_results['guardrail_pass_rate']}%"
    assert eval_results["deliberate_corruption_detection_rate"] == 100.0

    print(
        f"[OK] Test 4 Passed: Evaluated on {eval_results['total_disputes_evaluated']} disputes. "
        f"Generated {eval_results['total_sentences_generated']} sentences, "
        f"all {eval_results['valid_factual_claims']}/{eval_results['total_factual_claims']} claims machine-verified (100.0%)."
    )


if __name__ == "__main__":
    test_single_packet_100_percent_citation_resolution()
    test_guardrails_pii_and_prohibited_language()
    test_deliberate_hallucination_detection()
    test_all_reason_codes_and_benchmark_dataset()
    print("\nAll Module 3 tests passed successfully!")
