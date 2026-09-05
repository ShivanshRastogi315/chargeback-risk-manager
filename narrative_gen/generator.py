"""
Module 3 — Grounded Narrative Generator Core (narrative_gen/generator.py)

Orchestrates template-constrained narrative generation, machine citation validation,
and pre-flight guardrail checks into a unified representment packet generator.

Guarantees:
- Every factual claim carries a machine-resolvable citation tag [criterion: field=value].
- Citation validator checks 100% of tags against the underlying dispute and rubric data.
- Guardrail pass checks PII leaks, network tone compliance, and mandatory fields.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import json

from narrative_gen.templates import StructuredTemplateEngine, NarrativeSection
from narrative_gen.citation_validator import CitationValidator, ValidationReport
from narrative_gen.guardrails import GuardrailRunner, GuardrailReport


@dataclass
class RepresentmentPacket:
    """The complete auditable dispute representment packet."""
    dispute_id: str
    transaction_id: str
    reason_code: str
    merchant_id: str
    amount: float
    currency: str
    sections: List[NarrativeSection]
    full_text: str
    citation_report: ValidationReport
    guardrail_report: GuardrailReport
    is_valid: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispute_id": self.dispute_id,
            "transaction_id": self.transaction_id,
            "reason_code": self.reason_code,
            "merchant_id": self.merchant_id,
            "amount": self.amount,
            "currency": self.currency,
            "is_valid": self.is_valid,
            "citation_resolution_rate": round(self.citation_report.citation_resolution_rate * 100, 2),
            "sentence_grounding_rate": round(self.citation_report.sentence_grounding_rate * 100, 2),
            "guardrail_passed": self.guardrail_report.passed,
            "total_sentences": self.citation_report.total_sentences,
            "total_tags": self.citation_report.total_tags,
            "total_claims": self.citation_report.total_claims,
            "sections_count": len(self.sections),
            "full_text": self.full_text,
        }


class NarrativeGenerator:
    """
    Main generator interface for Module 3.
    Synthesizes compliant, verifiable evidence packets from dispute facts and rubric breakdowns.
    """

    def __init__(
        self,
        float_tolerance: float = 0.05,
        custom_prohibited_terms: Optional[List[str]] = None,
    ):
        self.template_engine = StructuredTemplateEngine()
        self.validator = CitationValidator(float_tolerance=float_tolerance)
        self.guardrail_runner = GuardrailRunner(custom_prohibited=custom_prohibited_terms)

    def generate_packet(
        self,
        record: Union[Dict[str, Any], pd.Series, Any],
        rubric_result: Optional[Dict[str, Any]] = None,
        roi_result: Optional[Dict[str, Any]] = None,
    ) -> RepresentmentPacket:
        """
        Generates, validates, and guardrails a complete representment packet for a dispute record.
        """
        if isinstance(record, pd.Series):
            rec_dict = record.to_dict()
        elif isinstance(record, dict):
            rec_dict = dict(record)
        else:
            rec_dict = {k: getattr(record, k) for k in dir(record) if not k.startswith("_")}

        # 1. Build structured sections
        sections = self.template_engine.build_packet_sections(
            record_dict=rec_dict,
            rubric_result=rubric_result,
            roi_result=roi_result,
        )

        # 2. Assemble full document
        full_text = self.template_engine.assemble_full_narrative(sections)

        # 3. Machine Citation Validation
        citation_report = self.validator.validate_narrative(
            narrative_text=full_text,
            record=rec_dict,
            rubric_result=rubric_result,
            roi_result=roi_result,
        )

        # 4. Guardrail Verification Pass
        guardrail_report = self.guardrail_runner.run_guardrails(
            narrative_text=full_text,
            record=rec_dict,
        )

        # Overall validity: 100% citation resolution AND all guardrails passed
        is_valid = citation_report.is_100_percent_faithful and guardrail_report.passed

        return RepresentmentPacket(
            dispute_id=str(rec_dict.get("dispute_id", "dsp_unknown")),
            transaction_id=str(rec_dict.get("transaction_id", "tx_unknown")),
            reason_code=str(rec_dict.get("reason_code", "10.4")),
            merchant_id=str(rec_dict.get("merchant_id", "merch_unknown")),
            amount=float(rec_dict.get("amount", 0.0)),
            currency=str(rec_dict.get("currency", "INR")),
            sections=sections,
            full_text=full_text,
            citation_report=citation_report,
            guardrail_report=guardrail_report,
            is_valid=is_valid,
            metadata={
                "ce3_eligible": bool(rec_dict.get("reason_ce3_eligible", 1)),
                "rubric_score": rubric_result.get("overall_score") if rubric_result else None,
                "roi_recommendation": roi_result.get("recommendation") if roi_result else None,
            },
        )

    def generate_batch(
        self,
        df: pd.DataFrame,
        rubric_results: Optional[List[Dict[str, Any]]] = None,
        roi_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RepresentmentPacket]:
        """Generates representment packets for a collection/DataFrame of disputes."""
        packets = []
        for idx, row in df.iterrows():
            rub = rubric_results[idx] if rubric_results and idx < len(rubric_results) else None
            roi = roi_results[idx] if roi_results and idx < len(roi_results) else None
            packet = self.generate_packet(row, rubric_result=rub, roi_result=roi)
            packets.append(packet)
        return packets

    def generate_corrupted_packet_for_demo(
        self,
        record: Union[Dict[str, Any], pd.Series, Any],
        corruption_type: str = "hallucinated_device_id",
        rubric_result: Optional[Dict[str, Any]] = None,
        roi_result: Optional[Dict[str, Any]] = None,
    ) -> RepresentmentPacket:
        """
        Deliberately injects an ungrounded claim or corrupted citation tag into the packet.
        Used to demonstrate live to evaluators and judges how the system catches hallucinations.
        
        Supported corruption types:
        - 'hallucinated_device_id': references a non-existent device ID 'dev_fabricated_9999'
        - 'fabricated_amount': claims a false transaction amount '₹999,999.00'
        - 'missing_citation': removes citation tag from an evidentiary sentence
        - 'pii_leak': introduces an unmasked 16-digit PAN
        - 'prohibited_language': introduces aggressive/unprofessional language
        """
        packet = self.generate_packet(record, rubric_result=rubric_result, roi_result=roi_result)
        corrupted_text = packet.full_text

        import re
        if corruption_type == "hallucinated_device_id":
            corrupted_text = re.sub(
                r"\[device_match:\s*current_device_id=[^,]+",
                "[device_match: current_device_id=dev_fabricated_9999_NONEXISTENT",
                corrupted_text,
                count=1,
            )
        elif corruption_type == "fabricated_amount":
            corrupted_text = re.sub(
                r"amount=\d+(\.\d+)?",
                "amount=999999.0",
                corrupted_text,
                count=1,
            )
        elif corruption_type == "missing_citation":
            # Strip out one of the bracketed tags entirely
            corrupted_text = re.sub(r"\[device_match:[^\]]+\]", "", corrupted_text, count=1)
        elif corruption_type == "pii_leak":
            corrupted_text += "\nExposed cardholder full PAN: 4111 2222 3333 4444 [unmasked: pan=4111222233334444]."
        elif corruption_type == "prohibited_language":
            corrupted_text += "\nThe cardholder is an absolute scammer and liar who stole our money [remark: note=complaint]."

        # Re-validate corrupted text
        if isinstance(record, pd.Series):
            rec_dict = record.to_dict()
        elif isinstance(record, dict):
            rec_dict = dict(record)
        else:
            rec_dict = {k: getattr(record, k) for k in dir(record) if not k.startswith("_")}

        citation_report = self.validator.validate_narrative(
            narrative_text=corrupted_text,
            record=rec_dict,
            rubric_result=rubric_result,
            roi_result=roi_result,
        )
        guardrail_report = self.guardrail_runner.run_guardrails(
            narrative_text=corrupted_text,
            record=rec_dict,
        )

        is_valid = citation_report.is_100_percent_faithful and guardrail_report.passed

        return RepresentmentPacket(
            dispute_id=packet.dispute_id,
            transaction_id=packet.transaction_id,
            reason_code=packet.reason_code,
            merchant_id=packet.merchant_id,
            amount=packet.amount,
            currency=packet.currency,
            sections=packet.sections,
            full_text=corrupted_text,
            citation_report=citation_report,
            guardrail_report=guardrail_report,
            is_valid=is_valid,
            metadata={"corruption_type": corruption_type, "deliberate_failure_demo": True},
        )
