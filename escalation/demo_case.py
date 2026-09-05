"""
Module 5 — Live Demo Scripted Failure & Ambiguity Case (escalation/demo_case.py)

Generates and saves the curated ambiguous dispute case for the live hackathon demo
(Slide 5: "The Scripted Deliberate Failure Moment"):
1. Shows what broke in Version 1: Naive ROI engine blindly recommends 'FIGHT' on a 50/50 case.
2. Shows how we got out in Module 5: Conformal prediction detects set {0, 1} and routes to
   'ESCALATE_TO_HUMAN' with a machine-verifiable statistical error guarantee.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from escalation.engine import ConformalEscalationEngine
from roi_engine.engine import ROIEngine
from benchmark.generate import SyntheticBenchmarkGenerator


def create_curated_ambiguous_dispute_case() -> dict:
    """
    Constructs a realistic, borderline dispute case designed for the live demo.
    Features:
    - Split Telemetry: IP matches prior order (+0.15), but Device ID is disparate/unseen (0.0).
    - Historical Window: Single settled order at 121.0 days (barely in 120-365d window, +0.30).
    - Delivery: Newly added shipping PIN code (unmatched to billing PIN, 0.0).
    - Security: CVV matched (+0.05), but 3DS frictionless / unauthenticated (0.0).
    - Economics: Dispute Amount ₹850.00, P(win) ~ 49.3% -> EV ~ +₹194.17 (borderline uncertainty).
    - Result: Conformal set is {0, 1} -> Routes to ESCALATE_TO_HUMAN.
    """
    return {
        "dispute_id": "dsp_demo_ambiguous_007",
        "transaction_id": "tx_demo_ambiguous_007",
        "merchant_id": "merch_d2c_012",
        "merchant_tier": "thin_history",
        "merchant_prior_dispute_count": 8,
        "amount": 850.0,
        "currency": "INR",
        "timestamp": "2025-08-20T14:15:30",
        "reason_code": "10.4",
        "reason_name": "Fraud - Card-Absent Environment / Non-Recognition",
        "reason_network": "Visa",
        "reason_ce3_eligible": 1,
        "card_network": "Visa",
        "card_type": "Credit",
        "card_bin": "411111",
        "card_last4": "5678",
        "issuer_bank": "HDFC Bank",
        "customer_name": "Rohan Deshmukh",
        "customer_email": "rohan.deshmukh@example.com",
        "device_id": "dev_new_mobile_88bb",
        "device_type": "mobile_android",
        "device_ip": "103.21.50.10",  # Matched IP
        "billing_city": "Mumbai",
        "billing_state": "Maharashtra",
        "billing_postal_code": "400001",
        "shipping_city": "Pune",      # Alternate delivery address
        "shipping_state": "Maharashtra",
        "shipping_postal_code": "411038",
        "cvv_avs_matched": True,
        "otp_3ds_matched": False,
        "prior_undisputed_window_count": 1,
        "prior_transaction_age_min_days": 121.0,
        "prior_transaction_age_max_days": 121.0,
        "criterion_prior_window_match": 1,
        "criterion_device_match": 0,
        "criterion_ip_match": 1,
        "criterion_shipping_match": 0,
        "criterion_avs_cvv_match": 1,
        "criterion_3ds_match": 0,
        "total_criteria_matched": 3,
        "issuer_dispute_won": 0,  # Borderline outcome that fails adjudicator scrutiny
        "prior_transaction_history": [
            {
                "prior_tx_id": "ptx_hist_991",
                "age_days": 121.0,
                "device_id": "dev_matched_browser_77aa",
                "ip_address": "103.21.50.10",
                "shipping_pincode": "400001",
                "status": "settled_undisputed",
                "in_ce3_qualifying_window": True,
            }
        ],
    }


def generate_and_save_demo_case(
    output_path: str = "escalation/ambiguous_escalation_case.json"
) -> dict:
    """
    Calibrates the engine on benchmark data, evaluates the curated ambiguous case,
    and saves the before/after demonstration artifact.
    """
    # 1. Generate benchmark calibration data
    gen = SyntheticBenchmarkGenerator(seed=42)
    _, df_dsp = gen.generate_benchmark(n_total_transactions=20000)
    train_df, test_df = gen.create_stratified_split(df_dsp, test_size=0.40)
    calib_df, eval_test_df = gen.create_stratified_split(test_df, test_size=0.50)

    # 2. Train and calibrate engine
    roi_engine = ROIEngine()
    roi_engine.train_mode_a(train_df)

    engine = ConformalEscalationEngine(alpha=0.10, roi_engine=roi_engine)
    calib_report = engine.calibrate(calib_df)

    # 3. Evaluate the ambiguous demo dispute
    ambiguous_record = create_curated_ambiguous_dispute_case()
    decision_packet = engine.evaluate_dispute(ambiguous_record)

    # 4. Construct side-by-side presentation artifact
    demo_artifact = {
        "case_summary": {
            "dispute_id": ambiguous_record["dispute_id"],
            "disputed_amount_inr": ambiguous_record["amount"],
            "reason_code": f"{ambiguous_record['reason_code']} ({ambiguous_record['reason_name']})",
            "qualifying_criteria_matched": f"{decision_packet.qualifying_criteria_count}/6 criteria matched",
            "rubric_confidence_score": f"{decision_packet.rubric_overall_score*100:.1f}%",
            "model_win_probability": f"{decision_packet.p_win_calibrated*100:.1f}%",
            "expected_net_value": f"INR {decision_packet.expected_value_inr:.2f}",
        },
        "telemetry_split_details": {
            "device_match": "UNMATCHED (0.00) — Device dev_new_mobile_88bb is an unrecognized mobile browser",
            "ip_match": "MATCHED (+0.15) — IP 103.21.50.10 matches prior settled order #ptx_hist_991",
            "shipping_match": "UNMATCHED (0.00) — Delivery PIN 411038 differs from billing PIN 400001",
            "historical_window": "BORDERLINE (+0.30) — Prior order age 121.0 days (threshold: 120.0 days)",
            "authentication": "PARTIAL (+0.05) — CVV verified, 3DS step-up challenge not completed",
        },
        "slide_5_demo_comparison": {
            "naive_system_without_conformal (What Broke)": {
                "decision": decision_packet.naive_decision_without_conformal,
                "action": "AUTOMATED_CONTEST",
                "flaw_narrative": (
                    f"The naive ROI engine saw EV = +INR {decision_packet.expected_value_inr:.2f} (> 0) and blindly forced an automated 'FIGHT' "
                    "verdict with zero uncertainty awareness. In reality, this borderline case loses at "
                    "adjudication, costing the merchant a ₹200 loss penalty + ₹25 review cost."
                ),
            },
            "ai_risk_manager_with_conformal (How We Got Out)": {
                "decision": decision_packet.decision,
                "action": decision_packet.action,
                "conformal_prediction_set": decision_packet.conformal_prediction_set,
                "confidence_guarantee": f"{decision_packet.confidence_level_pct:.1f}% marginal coverage guarantee",
                "error_rate_bound": f"At most {decision_packet.error_guarantee_pct:.1f}% error rate on auto-decisions",
                "escalation_reasons": decision_packet.escalation_reasons,
                "guarantee_statement": decision_packet.guarantee_statement,
                "success_narrative": (
                    "Conformal calibration detects the prediction set contains BOTH {NO_FIGHT (0), FIGHT (1)}. "
                    "The system declines to guess and gracefully routes the case to a human analyst, "
                    "backing its decision with a mathematically proven coverage guarantee."
                ),
            },
        },
        "presenter_script": (
            "Presenter Script (Slide 5 Deliberate Failure): 'Here is a genuinely ambiguous dispute where "
            "the IP matched, but the device is new and the prior order sits right at the 121-day boundary. "
            f"Without our escalation layer, the system blindly guessed FIGHT because EV was +INR {decision_packet.expected_value_inr:.2f}. "
            "With Module 5 active, the conformal predictor recognizes that both win and loss are plausible "
            "at 90% confidence and routes it to human review. That\\'s how we got out — not by claiming "
            "infallibility, but by knowing when the system does not know.'"
        ),
        "raw_record": ambiguous_record,
        "decision_packet": decision_packet.to_dict(),
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(demo_artifact, f, indent=2)

    return demo_artifact


if __name__ == "__main__":
    artifact = generate_and_save_demo_case()
    print("\n" + "=" * 70)
    print("MODULE 5 — AMBIGUOUS ESCALATION DEMO CASE SAVED SUCCESSFULLY")
    print("=" * 70)
    print(f"Case ID : {artifact['case_summary']['dispute_id']}")
    print(f"Amount  : INR {artifact['case_summary']['disputed_amount_inr']}")
    print(f"Naive Decision (Without Conformal) : {artifact['slide_5_demo_comparison']['naive_system_without_conformal (What Broke)']['decision']}")
    print(f"Module 5 Decision (With Conformal) : {artifact['slide_5_demo_comparison']['ai_risk_manager_with_conformal (How We Got Out)']['decision']}")
    print(f"Prediction Set                     : {artifact['slide_5_demo_comparison']['ai_risk_manager_with_conformal (How We Got Out)']['conformal_prediction_set']}")
    print(f"Statistical Guarantee              : {artifact['slide_5_demo_comparison']['ai_risk_manager_with_conformal (How We Got Out)']['guarantee_statement']}")
    print("=" * 70 + "\n")
