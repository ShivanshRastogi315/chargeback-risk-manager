"""
Module 1 — Per-Criterion Evidentiary Scorers (rubric_scorer/criteria.py)

Implements individual explainable scorers for Visa CE3.0 and Mastercard
First-Party Trust evidentiary requirements:
1. PriorUndisputedWindowScorer (120-365 days window requirement)
2. DeviceMatchScorer (Hardware/Browser device fingerprint match)
3. ShippingMatchScorer (Physical delivery address and PIN code match)
4. IPMatchScorer (IP address / subnet / geo match)
5. CVVAVSScorer (Card verification value & AVS match)
6. ThreeDSScorer (3D-Secure / OTP cardholder step-up authentication)
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import json
import numpy as np


class BaseCriterionScorer:
    """Base class for individual evidentiary criterion scorers."""
    
    criterion_name: str = "base_criterion"
    default_weight: float = 0.0

    def score(self, record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Evaluate record against criterion.
        Returns dictionary containing:
        - score: float in [0.0, 1.0]
        - matched: bool (thresholded at >= 0.5)
        - confidence_interval: [lower, upper]
        - evidence_strength: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'
        - weight: float
        - details: Dict[str, Any] explainable metadata
        """
        raise NotImplementedError


def _parse_prior_history(record: Union[Dict[str, Any], Any]) -> List[Dict[str, Any]]:
    """Helper to safely extract prior transaction history list."""
    history = record.get("prior_transaction_history") if isinstance(record, dict) else getattr(record, "prior_transaction_history", None)
    if history is None:
        return []
    if isinstance(history, str):
        try:
            return json.loads(history)
        except Exception:
            return []
    if isinstance(history, list):
        return history
    return []


class PriorWindowScorer(BaseCriterionScorer):
    """
    Evaluates Visa CE3.0 / Mastercard FPT prerequisite:
    At least 1 (or 2) settled, undisputed transactions between 120 and 365 days prior.
    """
    criterion_name = "prior_undisputed_in_window"
    default_weight = 0.30

    def score(self, record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        history = _parse_prior_history(record)
        
        # Fallback to tabular summary columns if raw JSON history is absent
        prior_window_count = 0
        matching_tx_ids = []
        matching_ages = []

        if history:
            for p in history:
                age = float(p.get("age_days", 0.0))
                status = p.get("status", "settled_undisputed")
                in_window = p.get("in_ce3_qualifying_window", False) or (120.0 <= age <= 365.0)
                if in_window and status == "settled_undisputed":
                    prior_window_count += 1
                    matching_tx_ids.append(p.get("prior_tx_id", "ptx_unknown"))
                    matching_ages.append(age)
        else:
            if isinstance(record, dict):
                prior_window_count = int(record.get("prior_undisputed_window_count", 0))
                min_age = float(record.get("prior_transaction_age_min_days", -1.0))
                max_age = float(record.get("prior_transaction_age_max_days", -1.0))
            else:
                prior_window_count = int(getattr(record, "prior_undisputed_window_count", 0))
                min_age = float(getattr(record, "prior_transaction_age_min_days", -1.0))
                max_age = float(getattr(record, "prior_transaction_age_max_days", -1.0))
            
            if min_age >= 0:
                matching_ages = [min_age, max_age]

        if prior_window_count >= 2:
            score = 0.98
            strength = "HIGH"
            ci = [0.95, 1.00]
        elif prior_window_count == 1:
            score = 0.92
            strength = "HIGH"
            ci = [0.85, 0.98]
        elif len(history) > 0 or (isinstance(record, dict) and record.get("prior_transaction_count", 0) > 0):
            score = 0.20
            strength = "LOW"
            ci = [0.10, 0.35]
        else:
            score = 0.02
            strength = "NONE"
            ci = [0.00, 0.08]

        return {
            "score": round(score, 4),
            "matched": bool(score >= 0.50),
            "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
            "evidence_strength": strength,
            "weight": self.default_weight,
            "details": {
                "qualifying_tx_count": prior_window_count,
                "qualifying_tx_ids": matching_tx_ids,
                "qualifying_ages_days": matching_ages,
                "rule_requirement": "Prior undisputed settled transaction between 120 and 365 days prior",
            },
        }


class DeviceMatchScorer(BaseCriterionScorer):
    """
    Evaluates Core Identifier 1: Device fingerprint matching.
    Checks whether the transaction device ID matches prior undisputed orders.
    """
    criterion_name = "device_match"
    default_weight = 0.25

    def score(self, record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        curr_device = record.get("device_id") if isinstance(record, dict) else getattr(record, "device_id", "")
        curr_type = record.get("device_type") if isinstance(record, dict) else getattr(record, "device_type", "")
        history = _parse_prior_history(record)

        exact_match = False
        matched_prior_txs = []
        same_device_type = False

        for p in history:
            p_device = p.get("device_id", "")
            if curr_device and p_device and curr_device == p_device:
                exact_match = True
                matched_prior_txs.append(p.get("prior_tx_id", "ptx_unknown"))

        if exact_match:
            score = 0.96
            strength = "HIGH"
            ci = [0.92, 1.00]
        elif history and same_device_type:
            score = 0.30
            strength = "LOW"
            ci = [0.20, 0.45]
        elif history:
            score = 0.05
            strength = "NONE"
            ci = [0.00, 0.12]
        else:
            score = 0.02
            strength = "NONE"
            ci = [0.00, 0.08]

        return {
            "score": round(score, 4),
            "matched": bool(score >= 0.50),
            "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
            "evidence_strength": strength,
            "weight": self.default_weight,
            "details": {
                "current_device_id": curr_device,
                "current_device_type": curr_type,
                "exact_match_found": exact_match,
                "matched_prior_tx_ids": matched_prior_txs,
                "rule_requirement": "Exact device fingerprint match with prior undisputed transaction",
            },
        }


class ShippingMatchScorer(BaseCriterionScorer):
    """
    Evaluates Core Identifier 2: Physical delivery / shipping address match.
    Checks whether the delivery postal code and city match prior delivered orders.
    """
    criterion_name = "shipping_match"
    default_weight = 0.20

    def score(self, record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        curr_pin = record.get("shipping_postal_code") if isinstance(record, dict) else getattr(record, "shipping_postal_code", "")
        curr_city = record.get("shipping_city") if isinstance(record, dict) else getattr(record, "shipping_city", "")
        bill_pin = record.get("billing_postal_code") if isinstance(record, dict) else getattr(record, "billing_postal_code", "")
        history = _parse_prior_history(record)

        exact_pin_match = False
        matched_prior_txs = []

        for p in history:
            p_pin = p.get("shipping_pincode", "")
            if curr_pin and p_pin and curr_pin == p_pin:
                exact_pin_match = True
                matched_prior_txs.append(p.get("prior_tx_id", "ptx_unknown"))

        bill_ship_match = bool(curr_pin and bill_pin and curr_pin == bill_pin)

        if exact_pin_match:
            score = 0.95
            strength = "HIGH"
            ci = [0.90, 1.00]
        elif bill_ship_match and history:
            score = 0.40
            strength = "MEDIUM"
            ci = [0.25, 0.55]
        elif history:
            score = 0.06
            strength = "NONE"
            ci = [0.00, 0.15]
        else:
            score = 0.03
            strength = "NONE"
            ci = [0.00, 0.10]

        return {
            "score": round(score, 4),
            "matched": bool(score >= 0.50),
            "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
            "evidence_strength": strength,
            "weight": self.default_weight,
            "details": {
                "shipping_postal_code": curr_pin,
                "shipping_city": curr_city,
                "billing_postal_code": bill_pin,
                "exact_postal_match": exact_pin_match,
                "matched_prior_tx_ids": matched_prior_txs,
                "billing_matches_shipping": bill_ship_match,
                "rule_requirement": "Physical shipping address / postal code match with prior delivered transaction",
            },
        }


class IPMatchScorer(BaseCriterionScorer):
    """
    Evaluates Supporting Identifier: IP Address and Subnet / Geolocation match.
    """
    criterion_name = "ip_match"
    default_weight = 0.15

    def score(self, record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        curr_ip = record.get("device_ip") if isinstance(record, dict) else getattr(record, "device_ip", "")
        history = _parse_prior_history(record)

        exact_ip_match = False
        subnet_match = False
        matched_prior_txs = []

        curr_octets = curr_ip.split(".") if curr_ip else []

        for p in history:
            p_ip = p.get("ip_address", "")
            if curr_ip and p_ip:
                if curr_ip == p_ip:
                    exact_ip_match = True
                    matched_prior_txs.append(p.get("prior_tx_id", "ptx_unknown"))
                else:
                    p_octets = p_ip.split(".")
                    if len(curr_octets) == 4 and len(p_octets) == 4:
                        if curr_octets[:2] == p_octets[:2]:
                            subnet_match = True

        if exact_ip_match:
            score = 0.95
            strength = "HIGH"
            ci = [0.90, 1.00]
        elif subnet_match:
            score = 0.40
            strength = "MEDIUM"
            ci = [0.25, 0.55]
        elif history:
            score = 0.05
            strength = "NONE"
            ci = [0.00, 0.12]
        else:
            score = 0.02
            strength = "NONE"
            ci = [0.00, 0.08]

        return {
            "score": round(score, 4),
            "matched": bool(score >= 0.50),
            "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
            "evidence_strength": strength,
            "weight": self.default_weight,
            "details": {
                "device_ip": curr_ip,
                "exact_ip_match": exact_ip_match,
                "subnet_match": subnet_match,
                "matched_prior_tx_ids": matched_prior_txs,
                "rule_requirement": "IP address or regional subnet match with prior undisputed transaction",
            },
        }


class CVVAVSScorer(BaseCriterionScorer):
    """
    Evaluates Auxiliary Security Criterion: CVV and AVS verification at authorization.
    """
    criterion_name = "cvv_avs_verified"
    default_weight = 0.05

    def score(self, record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        matched = bool(record.get("cvv_avs_matched", False) if isinstance(record, dict) else getattr(record, "cvv_avs_matched", False))
        
        if matched:
            score = 0.95
            strength = "HIGH"
            ci = [0.90, 1.00]
        else:
            score = 0.05
            strength = "NONE"
            ci = [0.00, 0.15]

        return {
            "score": round(score, 4),
            "matched": matched,
            "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
            "evidence_strength": strength,
            "weight": self.default_weight,
            "details": {
                "avs_cvv_status": "MATCHED" if matched else "MISMATCH_OR_UNVERIFIED",
                "rule_requirement": "CVV2 and Address Verification match at time of authorization",
            },
        }


class ThreeDSScorer(BaseCriterionScorer):
    """
    Evaluates Auxiliary Security Criterion: 3DS / OTP step-up authentication.
    """
    criterion_name = "otp_3ds_authenticated"
    default_weight = 0.05

    def score(self, record: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        matched = bool(record.get("otp_3ds_matched", False) if isinstance(record, dict) else getattr(record, "otp_3ds_matched", False))

        if matched:
            score = 0.95
            strength = "HIGH"
            ci = [0.90, 1.00]
        else:
            score = 0.05
            strength = "NONE"
            ci = [0.00, 0.15]

        return {
            "score": round(score, 4),
            "matched": matched,
            "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
            "evidence_strength": strength,
            "weight": self.default_weight,
            "details": {
                "otp_3ds_status": "AUTHENTICATED" if matched else "NOT_AUTHENTICATED",
                "rule_requirement": "Two-factor authentication step-up completed via 3D Secure / OTP",
            },
        }
