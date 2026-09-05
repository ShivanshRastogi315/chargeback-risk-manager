"""
Module 3 — Auditable Citation Validator (narrative_gen/citation_validator.py)

Extracts and verifies 100% machine-resolvable citation tags from generated
representment narratives against the original dispute record, rubric scores,
and ROI verdicts.

Tag Format:
    [<criterion_or_scope>: <key1>=<val1>, <key2>=<val2>, ...]
    e.g. [device_match: current_device_id=dev_abc123, prior_device_id=dev_abc123, match=True]
    e.g. [transaction_details: transaction_id=tx_0001, amount=2500.0, currency=INR]

Hard Constraint:
    Every factual claim sentence must contain at least one valid citation tag.
    100% of key-value pairs in each tag must resolve to real fields and values in the source record.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import re
import json
import numpy as np
import pandas as pd


@dataclass
class CitationTag:
    """Represents an individual parsed citation tag from a sentence."""
    raw_tag: str
    scope: str
    key_values: Dict[str, str]
    sentence_idx: int
    is_valid: bool = False
    unresolved_fields: List[str] = field(default_factory=list)
    resolution_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentenceTrace:
    """Represents a sentence from the narrative and its citation resolution status."""
    sentence_idx: int
    text: str
    section_name: str
    tags: List[CitationTag] = field(default_factory=list)
    is_grounded: bool = False
    error_message: Optional[str] = None


@dataclass
class ValidationReport:
    """Structured report returned by the CitationValidator."""
    total_sentences: int
    grounded_sentences: int
    ungrounded_sentences: int
    total_tags: int
    valid_tags: int
    invalid_tags: int
    total_claims: int
    valid_claims: int
    unresolved_claims: List[Dict[str, Any]]
    citation_resolution_rate: float
    sentence_grounding_rate: float
    is_100_percent_faithful: bool
    sentence_traces: List[SentenceTrace] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "total_sentences": self.total_sentences,
            "grounded_sentences": self.grounded_sentences,
            "sentence_grounding_rate": round(self.sentence_grounding_rate * 100, 2),
            "total_tags": self.total_tags,
            "valid_tags": self.valid_tags,
            "total_claims": self.total_claims,
            "valid_claims": self.valid_claims,
            "citation_resolution_rate": round(self.citation_resolution_rate * 100, 2),
            "is_100_percent_faithful": self.is_100_percent_faithful,
            "unresolved_count": len(self.unresolved_claims),
        }


class CitationValidator:
    """
    Plain-Python regex and schema parser that resolves citation tags against
    the underlying transaction, dispute record, prior history, rubric breakdown,
    and ROI engine outputs.
    """

    TAG_REGEX = re.compile(r"\[([a-zA-Z0-9_]+):\s*([^\]]+)\]")
    KV_PAIR_REGEX = re.compile(r'([a-zA-Z0-9_]+)\s*=\s*("[^"]*"|\'[^\']*\'|[^,]+)')

    def __init__(self, float_tolerance: float = 0.05):
        self.float_tolerance = float_tolerance

    def extract_tags(self, text: str, sentence_idx: int = 0) -> List[CitationTag]:
        """Extracts all citation tags from a given text snippet or sentence."""
        tags = []
        for match in self.TAG_REGEX.finditer(text):
            raw_tag = match.group(0)
            scope = match.group(1).strip()
            kv_content = match.group(2).strip()

            key_values = {}
            for kv in self.KV_PAIR_REGEX.finditer(kv_content):
                k = kv.group(1).strip()
                v = kv.group(2).strip().strip("'\"")
                key_values[k] = v

            tags.append(
                CitationTag(
                    raw_tag=raw_tag,
                    scope=scope,
                    key_values=key_values,
                    sentence_idx=sentence_idx,
                )
            )
        return tags

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into discrete claim sentences while handling citation tags
        and abbreviations cleanly without splitting tags from their host sentences.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sentences = []

        for line in lines:
            if line.startswith("#"):
                sentences.append(line)
                continue

            current = []
            in_tag = False
            i = 0
            while i < len(line):
                ch = line[i]
                if ch == '[':
                    in_tag = True
                elif ch == ']':
                    in_tag = False

                current.append(ch)

                if not in_tag and (ch in ('.', ']', '!', '?')):
                    rest = line[i+1:]
                    match = re.match(r'^\s+(?=[A-Z0-9])', rest)
                    tag_following = re.match(r'^\s*\[[a-zA-Z0-9_]+:', rest)
                    if match and not tag_following:
                        s_text = "".join(current).strip()
                        if s_text:
                            sentences.append(s_text)
                        current = []
                        i += len(match.group(0))
                        continue
                i += 1

            remainder = "".join(current).strip()
            if remainder:
                sentences.append(remainder)

        return sentences

    def _normalize_val(self, val: Any) -> Any:
        """Helper to normalize data types for fuzzy/exact value comparison."""
        if val is None:
            return ""
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            return float(val)
        return str(val).strip()

    def _values_match(self, claimed_val_str: str, actual_val: Any) -> bool:
        """Determines if a claimed string value matches the actual record value."""
        claimed_clean = claimed_val_str.strip()
        actual_norm = self._normalize_val(actual_val)

        # Boolean comparison
        if claimed_clean.lower() in ("true", "false", "yes", "no", "1", "0"):
            claimed_bool = claimed_clean.lower() in ("true", "yes", "1")
            if isinstance(actual_norm, str) and actual_norm.lower() in ("true", "false", "1", "0"):
                return claimed_bool == (actual_norm.lower() in ("true", "1"))
            if isinstance(actual_norm, (int, float)):
                return claimed_bool == bool(actual_norm)

        # Numeric comparison
        try:
            c_float = float(claimed_clean.replace("₹", "").replace(",", "").replace("%", ""))
            if isinstance(actual_norm, (int, float)):
                if "%" in claimed_clean and actual_norm <= 1.0 and c_float > 1.0:
                    # e.g. 88.0% vs 0.88
                    return abs(c_float / 100.0 - float(actual_norm)) <= self.float_tolerance
                return abs(c_float - float(actual_norm)) <= self.float_tolerance
            if isinstance(actual_norm, str):
                try:
                    a_float = float(actual_norm.replace("₹", "").replace(",", "").replace("%", ""))
                    return abs(c_float - a_float) <= self.float_tolerance
                except ValueError:
                    pass
        except ValueError:
            pass

        # String exact or substring comparison
        actual_str = str(actual_norm).lower()
        claimed_str = claimed_clean.lower()
        if claimed_str == actual_str:
            return True
        # Allow masked string matches (e.g. card_last4 '1234' matching '****1234')
        if actual_str.endswith(claimed_str) or claimed_str.endswith(actual_str):
            return True

        return False

    def _resolve_field_in_context(
        self,
        scope: str,
        field_name: str,
        claimed_value: str,
        record_dict: Dict[str, Any],
        rubric_dict: Optional[Dict[str, Any]],
        roi_dict: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        Attempts to resolve a specific field and value against the combined evidentiary context.
        """
        # Check 1: Direct field in record
        if field_name in record_dict:
            actual_val = record_dict[field_name]
            if self._values_match(claimed_value, actual_val):
                return True, f"Matched direct field '{field_name}' in record"
            return False, f"Value mismatch for '{field_name}': claimed='{claimed_value}', actual='{actual_val}'"

        # Check 2: Direct field in rubric result
        if rubric_dict:
            if field_name in rubric_dict:
                actual_val = rubric_dict[field_name]
                if self._values_match(claimed_value, actual_val):
                    return True, f"Matched rubric top-level field '{field_name}'"
            # In rubric criteria dict
            criteria = rubric_dict.get("criteria", {})
            if scope in criteria:
                crit_data = criteria[scope]
                if field_name in crit_data:
                    actual_val = crit_data[field_name]
                    if self._values_match(claimed_value, actual_val):
                        return True, f"Matched rubric criterion '{scope}.{field_name}'"
                details = crit_data.get("details", {})
                if field_name in details:
                    actual_val = details[field_name]
                    if self._values_match(claimed_value, actual_val):
                        return True, f"Matched rubric details '{scope}.details.{field_name}'"
                if field_name == "match" or field_name == "matched":
                    actual_val = crit_data.get("matched")
                    if self._values_match(claimed_value, actual_val):
                        return True, f"Matched rubric criterion match flag '{scope}.matched'"
                if field_name == "score":
                    actual_val = crit_data.get("score")
                    if self._values_match(claimed_value, actual_val):
                        return True, f"Matched rubric criterion score '{scope}.score'"

        # Check 3: Direct field in ROI result
        if roi_dict:
            if field_name in roi_dict:
                actual_val = roi_dict[field_name]
                if self._values_match(claimed_value, actual_val):
                    return True, f"Matched ROI field '{field_name}'"

        # Check 4: Prior transaction history list
        prior_history = record_dict.get("prior_transaction_history", [])
        if isinstance(prior_history, str):
            try:
                prior_history = json.loads(prior_history)
            except Exception:
                prior_history = []

        if isinstance(prior_history, list) and len(prior_history) > 0:
            # Search across all prior history items
            for ptx in prior_history:
                if isinstance(ptx, dict):
                    # Check field aliases (e.g. prior_device_id -> device_id)
                    aliases = [field_name, field_name.replace("prior_", ""), field_name.replace("current_", "")]
                    for alias in aliases:
                        if alias in ptx:
                            if self._values_match(claimed_value, ptx[alias]):
                                return True, f"Matched prior transaction history item '{alias}' in '{ptx.get('prior_tx_id', 'ptx')}'"

        # Check 5: Aliased / transformed fields
        field_aliases = {
            "current_device_id": ["device_id"],
            "prior_device_id": ["device_id"],
            "current_device_ip": ["device_ip"],
            "prior_device_ip": ["device_ip", "ip_address"],
            "current_shipping_postal_code": ["shipping_postal_code"],
            "prior_shipping_pincode": ["shipping_postal_code", "shipping_pincode"],
            "current_billing_postal_code": ["billing_postal_code"],
            "qualifying_window_count": ["prior_undisputed_window_count"],
            "qualifying_criteria_count": ["total_criteria_matched", "qualifying_criteria_count"],
            "ce3_eligible": ["reason_ce3_eligible", "ce3_eligible"],
            "expected_value": ["ev_contest_inr", "expected_value", "ev_contest"],
            "mode": ["selected_mode", "mode"],
            "recommendation": ["decision", "recommendation", "action", "should_contest"],
            "decision": ["decision", "recommendation", "action"],
            "p_win": ["calibrated_win_probability", "p_win", "win_probability"],
        }
        if field_name in field_aliases:
            for alt_name in field_aliases[field_name]:
                if alt_name in record_dict:
                    actual_val = record_dict[alt_name]
                    if self._values_match(claimed_value, actual_val):
                        return True, f"Matched aliased field '{alt_name}' for '{field_name}' in record"
                if roi_dict and alt_name in roi_dict:
                    actual_val = roi_dict[alt_name]
                    if self._values_match(claimed_value, actual_val):
                        return True, f"Matched aliased field '{alt_name}' for '{field_name}' in ROI"
                if rubric_dict and alt_name in rubric_dict:
                    actual_val = rubric_dict[alt_name]
                    if self._values_match(claimed_value, actual_val):
                        return True, f"Matched aliased field '{alt_name}' for '{field_name}' in Rubric"

        return False, f"Field '{field_name}' not found in any record/rubric/ROI context or value mismatched"

    def validate_narrative(
        self,
        narrative_text: str,
        record: Union[Dict[str, Any], pd.Series, Any],
        rubric_result: Optional[Dict[str, Any]] = None,
        roi_result: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        """
        Validates an entire representment narrative packet against source records.
        Verifies every claim sentence has valid citation tags that 100% resolve.
        """
        # Convert record to clean dictionary
        if isinstance(record, pd.Series):
            record_dict = record.to_dict()
        elif isinstance(record, dict):
            record_dict = dict(record)
        else:
            record_dict = {k: getattr(record, k) for k in dir(record) if not k.startswith("_")}

        sentences = self.split_into_sentences(narrative_text)
        total_sentences = 0
        grounded_sentences = 0
        ungrounded_sentences = 0

        total_tags = 0
        valid_tags = 0
        invalid_tags = 0

        total_claims = 0
        valid_claims = 0
        unresolved_claims = []
        sentence_traces = []

        current_section = "General"

        for s_idx, sentence_text in enumerate(sentences):
            # Check if this line is a section header (e.g. #, ##, or SECTION X)
            if sentence_text.startswith("#") or (
                sentence_text.isupper() and len(sentence_text) < 40 and not "[" in sentence_text
            ):
                current_section = sentence_text.lstrip("#").strip()
                continue

            total_sentences += 1
            tags = self.extract_tags(sentence_text, sentence_idx=s_idx)

            if not tags:
                # No citation tags found on this substantive sentence!
                ungrounded_sentences += 1
                unresolved_claims.append({
                    "sentence_idx": s_idx,
                    "sentence_text": sentence_text,
                    "reason": "Missing citation tag (ungrounded factual claim)",
                })
                sentence_traces.append(
                    SentenceTrace(
                        sentence_idx=s_idx,
                        text=sentence_text,
                        section_name=current_section,
                        tags=[],
                        is_grounded=False,
                        error_message="Missing citation tag",
                    )
                )
                continue

            # Process tags in this sentence
            sentence_all_tags_valid = True
            sentence_error_msg = None

            for tag in tags:
                total_tags += 1
                tag_valid = True
                tag_unresolved_fields = []
                tag_res_details = {}

                if not tag.key_values:
                    tag_valid = False
                    tag_unresolved_fields.append("<empty_tag>")
                    unresolved_claims.append({
                        "sentence_idx": s_idx,
                        "tag": tag.raw_tag,
                        "reason": "Tag contains no key-value pairs",
                    })

                for k, v in tag.key_values.items():
                    total_claims += 1
                    is_res, msg = self._resolve_field_in_context(
                        scope=tag.scope,
                        field_name=k,
                        claimed_value=v,
                        record_dict=record_dict,
                        rubric_dict=rubric_result,
                        roi_dict=roi_result,
                    )
                    tag_res_details[k] = {"resolved": is_res, "detail": msg}
                    if is_res:
                        valid_claims += 1
                    else:
                        tag_valid = False
                        tag_unresolved_fields.append(k)
                        unresolved_claims.append({
                            "sentence_idx": s_idx,
                            "scope": tag.scope,
                            "field": k,
                            "claimed_value": v,
                            "reason": msg,
                            "tag": tag.raw_tag,
                        })

                tag.is_valid = tag_valid
                tag.unresolved_fields = tag_unresolved_fields
                tag.resolution_details = tag_res_details

                if tag_valid:
                    valid_tags += 1
                else:
                    invalid_tags += 1
                    sentence_all_tags_valid = False
                    sentence_error_msg = f"Unresolved fields: {', '.join(tag_unresolved_fields)}"

            if sentence_all_tags_valid:
                grounded_sentences += 1
                sentence_traces.append(
                    SentenceTrace(
                        sentence_idx=s_idx,
                        text=sentence_text,
                        section_name=current_section,
                        tags=tags,
                        is_grounded=True,
                        error_message=None,
                    )
                )
            else:
                ungrounded_sentences += 1
                sentence_traces.append(
                    SentenceTrace(
                        sentence_idx=s_idx,
                        text=sentence_text,
                        section_name=current_section,
                        tags=tags,
                        is_grounded=False,
                        error_message=sentence_error_msg,
                    )
                )

        citation_rate = (valid_claims / total_claims) if total_claims > 0 else 0.0
        grounding_rate = (grounded_sentences / total_sentences) if total_sentences > 0 else 0.0
        is_100_percent = (
            total_sentences > 0
            and ungrounded_sentences == 0
            and invalid_tags == 0
            and len(unresolved_claims) == 0
        )

        return ValidationReport(
            total_sentences=total_sentences,
            grounded_sentences=grounded_sentences,
            ungrounded_sentences=ungrounded_sentences,
            total_tags=total_tags,
            valid_tags=valid_tags,
            invalid_tags=invalid_tags,
            total_claims=total_claims,
            valid_claims=valid_claims,
            unresolved_claims=unresolved_claims,
            citation_resolution_rate=citation_rate,
            sentence_grounding_rate=grounding_rate,
            is_100_percent_faithful=is_100_percent,
            sentence_traces=sentence_traces,
        )
