"""
Module 3 — Evidence Guardrails Pass (narrative_gen/guardrails.py)

Performs a lightweight pre-flight compliance & safety pass on representment packets:
1. PII Leak Check: Disallows exposed unmasked 13-19 digit card PANs, raw CVVs, or unpermitted sensitive data.
2. Prohibited Language Check: Enforces formal, objective network-compliant representment tone,
   blocking aggressive, defamatory, colloquial, or accusatory language.
3. Required Fields Check: Verifies all mandatory transaction and case identification fields are present.
"""

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
import re
import pandas as pd


@dataclass
class GuardrailReport:
    """Structured report returned by the guardrail runner."""
    passed: bool
    pii_violations: List[str] = field(default_factory=list)
    prohibited_language_violations: List[str] = field(default_factory=list)
    missing_required_fields: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "pii_violations_count": len(self.pii_violations),
            "prohibited_language_count": len(self.prohibited_language_violations),
            "missing_fields_count": len(self.missing_required_fields),
            "pii_violations": self.pii_violations,
            "prohibited_language_violations": self.prohibited_language_violations,
            "missing_required_fields": self.missing_required_fields,
        }


class PIIChecker:
    """
    Scans text for raw, unmasked credit card PANs and exposed CVV security codes.
    Allows properly masked card tokens (e.g. '**** **** **** 1234' or '...1234')
    and legitimate customer identifiers from the record.
    """

    # Unmasked 13 to 19 digit card PAN pattern (with optional spaces or dashes)
    RAW_PAN_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    # Raw CVV pattern e.g. "CVV: 123" or "CVC 456"
    RAW_CVV_REGEX = re.compile(r"\b(?:cvv|cvc|cvv2|security code)\s*[:=]\s*\b(\d{3,4})\b", re.IGNORECASE)

    def check(self, text: str, record_dict: Dict[str, Any]) -> List[str]:
        violations = []

        # Check for unmasked PANs
        for match in self.RAW_PAN_REGEX.finditer(text):
            candidate = match.group(0).strip()
            digits_only = re.sub(r"\D", "", candidate)
            # Filter out timestamps, IP-like numbers, or allowed 13-19 digit strings
            if 13 <= len(digits_only) <= 19:
                # Check if it's an ISO timestamp or date
                if "-" in candidate and ":" in candidate:
                    continue
                # Check if it's a known non-PAN hash or ID (e.g. dev_0000000000000)
                if any(k in candidate for k in ["dev_", "dsp_", "tx_", "merch_"]):
                    continue
                # If Luhn-valid or standard continuous digit sequence, flag it!
                if digits_only.isdigit() and len(digits_only) in (15, 16):
                    violations.append(f"Exposed unmasked PAN detected: '{candidate[:4]}...{candidate[-4:]}'")

        # Check for raw CVV exposure
        for match in self.RAW_CVV_REGEX.finditer(text):
            violations.append(f"Exposed raw CVV/CVC value detected: '{match.group(0)}'")

        return violations


class ProhibitedLanguageChecker:
    """
    Enforces card network representment decorum and objective dispute tone.
    Flags defamatory, combative, colloquial, threatening, or abusive terms.
    """

    PROHIBITED_TERMS = [
        "thief",
        "thieves",
        "liar",
        "liars",
        "scammer",
        "scammers",
        "fraudster customer",
        "criminal",
        "con artist",
        "crook",
        "cheat",
        "stealing",
        "stole our money",
        "sue you",
        "we will sue",
        "take to court",
        "police arrest",
        "lock them up",
        "idiot",
        "stupid",
        "nonsense",
        "bogus claim",
        "total joke",
        "bullshit",
        "bastard",
        "scamming us",
    ]

    def __init__(self, custom_prohibited: Optional[List[str]] = None):
        self.prohibited = list(self.PROHIBITED_TERMS)
        if custom_prohibited:
            self.prohibited.extend(custom_prohibited)
        # Precompile pattern
        pattern_str = r"\b(" + "|".join(re.escape(t) for t in self.prohibited) + r")\b"
        self.regex = re.compile(pattern_str, re.IGNORECASE)

    def check(self, text: str) -> List[str]:
        violations = []
        for match in self.regex.finditer(text):
            term = match.group(0)
            violations.append(f"Prohibited/unprofessional language detected: '{term}'")
        return violations


class RequiredFieldsChecker:
    """
    Verifies that all mandatory transaction and dispute identification parameters
    are explicitly cited or stated in the representment packet.
    """

    MANDATORY_FIELDS = [
        "dispute_id",
        "transaction_id",
        "amount",
        "currency",
        "reason_code",
        "timestamp",
        "merchant_id",
    ]

    def check(self, text: str, record_dict: Dict[str, Any]) -> List[str]:
        missing = []
        for req_field in self.MANDATORY_FIELDS:
            # Field key or its value should be referenced in the text
            val = str(record_dict.get(req_field, "")).strip()
            if not val:
                continue
            
            # Check if field name or value exists in text
            field_cited = (f"{req_field}=" in text) or (req_field in text)
            val_present = (val in text) or (isinstance(record_dict.get(req_field), (int, float)) and f"{float(record_dict.get(req_field)):.2f}" in text)

            if not (field_cited or val_present):
                missing.append(f"Mandatory representment field '{req_field}' (value '{val}') is not referenced in packet")

        return missing


class GuardrailRunner:
    """
    Combined guardrail validator executing PII checks, tone compliance, and field completeness.
    """

    def __init__(self, custom_prohibited: Optional[List[str]] = None):
        self.pii_checker = PIIChecker()
        self.language_checker = ProhibitedLanguageChecker(custom_prohibited=custom_prohibited)
        self.fields_checker = RequiredFieldsChecker()

    def run_guardrails(
        self,
        narrative_text: str,
        record: Union[Dict[str, Any], pd.Series, Any],
    ) -> GuardrailReport:
        """Executes all guardrail passes on the generated narrative."""
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
        elif isinstance(record, dict):
            record_dict = dict(record)
        else:
            record_dict = {k: getattr(record, k) for k in dir(record) if not k.startswith("_")}

        pii_violations = self.pii_checker.check(narrative_text, record_dict)
        lang_violations = self.language_checker.check(narrative_text)
        missing_fields = self.fields_checker.check(narrative_text, record_dict)

        all_passed = (
            len(pii_violations) == 0
            and len(lang_violations) == 0
            and len(missing_fields) == 0
        )

        return GuardrailReport(
            passed=all_passed,
            pii_violations=pii_violations,
            prohibited_language_violations=lang_violations,
            missing_required_fields=missing_fields,
            details={
                "pii_passed": len(pii_violations) == 0,
                "language_passed": len(lang_violations) == 0,
                "fields_passed": len(missing_fields) == 0,
            },
        )
