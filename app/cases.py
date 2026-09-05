"""
app/cases.py — Curated dispute records for demonstration scenarios.
"""

from typing import Dict, Any, List
from escalation.demo_case import create_curated_ambiguous_dispute_case


def get_happy_path_case() -> Dict[str, Any]:
    """
    Strong First-Party Fraud Case (Visa CE3.0 Compelling Evidence Compliant).
    - Prior qualifying order at 145 days (settled, undisputed).
    - Device ID match (+0.30), IP match (+0.20), Shipping address match (+0.25).
    - AVS/CVV matched (+0.05), 3DS OTP authenticated (+0.20).
    - P(win) > 98%, EV > +INR 3,000.
    """
    return {
        "dispute_id": "dsp_demo_win_001",
        "transaction_id": "tx_demo_win_001",
        "merchant_id": "merch_d2c_001",
        "merchant_tier": "mid_market",
        "merchant_prior_dispute_count": 142,
        "amount": 3499.0,
        "currency": "INR",
        "timestamp": "2025-08-15T11:20:00",
        "reason_code": "10.4",
        "reason_name": "Fraud - Card-Absent Environment / Non-Recognition",
        "reason_network": "Visa",
        "reason_ce3_eligible": 1,
        "card_network": "Visa",
        "card_type": "Credit",
        "card_bin": "411111",
        "card_last4": "4321",
        "issuer_bank": "HDFC Bank",
        "customer_name": "Aarav Sharma",
        "customer_email": "aarav.sharma@example.com",
        "device_id": "dev_trusted_ios_99a",
        "device_type": "mobile_ios",
        "device_ip": "49.207.55.12",
        "billing_city": "Bengaluru",
        "billing_state": "Karnataka",
        "billing_postal_code": "560034",
        "shipping_city": "Bengaluru",
        "shipping_state": "Karnataka",
        "shipping_postal_code": "560034",
        "cvv_avs_matched": True,
        "otp_3ds_matched": True,
        "prior_undisputed_window_count": 2,
        "prior_transaction_age_min_days": 145.0,
        "prior_transaction_age_max_days": 210.0,
        "criterion_prior_window_match": 1,
        "criterion_device_match": 1,
        "criterion_ip_match": 1,
        "criterion_shipping_match": 1,
        "criterion_avs_cvv_match": 1,
        "criterion_3ds_match": 1,
        "total_criteria_matched": 6,
        "issuer_dispute_won": 1,
        "prior_transaction_history": [
            {
                "prior_tx_id": "ptx_hist_881",
                "age_days": 145.0,
                "device_id": "dev_trusted_ios_99a",
                "ip_address": "49.207.55.12",
                "shipping_pincode": "560034",
                "status": "settled_undisputed",
                "in_ce3_qualifying_window": True,
            },
            {
                "prior_tx_id": "ptx_hist_880",
                "age_days": 210.0,
                "device_id": "dev_trusted_ios_99a",
                "ip_address": "49.207.55.12",
                "shipping_pincode": "560034",
                "status": "settled_undisputed",
                "in_ce3_qualifying_window": True,
            },
        ],
    }


def get_weak_evidence_case() -> Dict[str, Any]:
    """
    Legitimate Fraud / Stolen Card (No-Fight Case).
    - Zero prior undisputed orders in 120-365d window.
    - Unrecognized device, foreign IP geolocation, mismatched shipping pincode.
    - Reason Code 10.5 (Counterfeit / Stolen, non-CE3).
    - P(win) ~ 2.0%, EV < 0 -> System recommends NO_FIGHT, saving loss fees.
    """
    return {
        "dispute_id": "dsp_demo_lose_002",
        "transaction_id": "tx_demo_lose_002",
        "merchant_id": "merch_d2c_002",
        "merchant_tier": "mid_market",
        "merchant_prior_dispute_count": 85,
        "amount": 450.0,
        "currency": "INR",
        "timestamp": "2025-08-18T09:45:00",
        "reason_code": "10.5",
        "reason_name": "Fraud - Cardholder Counterfeit / Stolen",
        "reason_network": "Visa",
        "reason_ce3_eligible": 0,
        "card_network": "Visa",
        "card_type": "Debit",
        "card_bin": "422222",
        "card_last4": "9876",
        "issuer_bank": "ICICI Bank",
        "customer_name": "Pooja Hegde",
        "customer_email": "pooja.h99@example.com",
        "device_id": "dev_anon_bot_33x",
        "device_type": "desktop_windows",
        "device_ip": "185.220.101.5",
        "billing_city": "Mumbai",
        "billing_state": "Maharashtra",
        "billing_postal_code": "400050",
        "shipping_city": "Kolkata",
        "shipping_state": "West Bengal",
        "shipping_postal_code": "700001",
        "cvv_avs_matched": True,
        "otp_3ds_matched": False,
        "prior_undisputed_window_count": 0,
        "prior_transaction_age_min_days": -1.0,
        "prior_transaction_age_max_days": -1.0,
        "criterion_prior_window_match": 0,
        "criterion_device_match": 0,
        "criterion_ip_match": 0,
        "criterion_shipping_match": 0,
        "criterion_avs_cvv_match": 1,
        "criterion_3ds_match": 0,
        "total_criteria_matched": 1,
        "issuer_dispute_won": 0,
        "prior_transaction_history": [],
    }


def get_cold_start_case() -> Dict[str, Any]:
    """
    Cold-Start Merchant Case (Thin History - 8 prior disputes).
    - Demonstrates TabPFN (Mode B) in-context few-shot prediction vs Mode A GBDT.
    - 2 qualifying prior window orders, matched device & IP, unmatched shipping.
    - P(win) ~ 78.4%, EV = +INR 1,650.00.
    """
    return {
        "dispute_id": "dsp_demo_coldstart_003",
        "transaction_id": "tx_demo_coldstart_003",
        "merchant_id": "merch_coldstart_099",
        "merchant_tier": "thin_history",
        "merchant_prior_dispute_count": 8,
        "amount": 2200.0,
        "currency": "INR",
        "timestamp": "2025-08-22T16:10:00",
        "reason_code": "4837",
        "reason_name": "No Cardholder Authorization",
        "reason_network": "Mastercard",
        "reason_ce3_eligible": 1,
        "card_network": "Mastercard",
        "card_type": "Credit",
        "card_bin": "512345",
        "card_last4": "1122",
        "issuer_bank": "Axis Bank",
        "customer_name": "Vikram Malhotra",
        "customer_email": "vikram.m@example.com",
        "device_id": "dev_chrome_mac_44c",
        "device_type": "desktop_macos",
        "device_ip": "115.111.200.45",
        "billing_city": "Delhi",
        "billing_state": "Delhi",
        "billing_postal_code": "110001",
        "shipping_city": "Gurugram",
        "shipping_state": "Haryana",
        "shipping_postal_code": "122002",
        "cvv_avs_matched": True,
        "otp_3ds_matched": True,
        "prior_undisputed_window_count": 1,
        "prior_transaction_age_min_days": 180.0,
        "prior_transaction_age_max_days": 180.0,
        "criterion_prior_window_match": 1,
        "criterion_device_match": 1,
        "criterion_ip_match": 1,
        "criterion_shipping_match": 0,
        "criterion_avs_cvv_match": 1,
        "criterion_3ds_match": 1,
        "total_criteria_matched": 5,
        "issuer_dispute_won": 1,
        "prior_transaction_history": [
            {
                "prior_tx_id": "ptx_hist_701",
                "age_days": 180.0,
                "device_id": "dev_chrome_mac_44c",
                "ip_address": "115.111.200.45",
                "shipping_pincode": "110001",
                "status": "settled_undisputed",
                "in_ce3_qualifying_window": True,
            }
        ],
    }


def get_demo_scenarios() -> Dict[str, Dict[str, Any]]:
    """Returns catalog of all predefined demonstration cases."""
    return {
        "🌟 Case 1: Happy Path — Strong Visa CE3.0 Defense (Clear Win)": {
            "record": get_happy_path_case(),
            "description": "High-confidence first-party fraud defense. All 6 CE3.0 evidentiary criteria satisfied. EV > +₹3,000, 100% cited machine narrative generated.",
            "type": "happy_path",
        },
        "⚠️ Case 2: Scripted Deliberate Failure Moment (Borderline Ambiguity)": {
            "record": create_curated_ambiguous_dispute_case(),
            "description": "Borderline dispute with split telemetry (IP matches, Device new, 121-day order at edge). Shows what broke in V1 (naive EV fight) and how Module 5 routes to human review.",
            "type": "ambiguous_failure",
        },
        "🛑 Case 3: Weak Evidence / Legitimate Fraud (Correct No-Fight)": {
            "record": get_weak_evidence_case(),
            "description": "Third-party fraud or zero prior history. EV is negative. System protects merchant by correctly declining to contest, avoiding ₹200 fee penalty.",
            "type": "no_fight",
        },
        "❄️ Case 4: Cold-Start Merchant (<30 Disputes, Mode B TabPFN)": {
            "record": get_cold_start_case(),
            "description": "Merchant with only 8 prior cases. Automatically invokes Mode B TabPFN Few-Shot in-context learning to overcome thin training history.",
            "type": "cold_start",
        },
    }
