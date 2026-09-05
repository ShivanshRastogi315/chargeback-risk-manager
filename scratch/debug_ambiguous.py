import sys
from pathlib import Path
REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from benchmark.generate import SyntheticBenchmarkGenerator
from roi_engine.engine import ROIEngine
from escalation.engine import ConformalEscalationEngine
def create_curated_ambiguous_dispute_case() -> dict:
    return {
        "dispute_id": "dsp_demo_ambiguous_007",
        "transaction_id": "tx_demo_ambiguous_007",
        "merchant_id": "merch_d2c_012",
        "merchant_tier": "thin_history",
        "merchant_prior_dispute_count": 8,
        "amount": 450.0,
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
        "issuer_dispute_won": 0,
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

gen = SyntheticBenchmarkGenerator(seed=42)
_, df_dsp = gen.generate_benchmark(n_total_transactions=10000)
train_df, test_df = gen.create_stratified_split(df_dsp, test_size=0.40)
calib_df, eval_test_df = gen.create_stratified_split(test_df, test_size=0.50)

roi_engine = ROIEngine()
roi_engine.train_mode_a(train_df)

engine = ConformalEscalationEngine(alpha=0.10, roi_engine=roi_engine)
calib_report = engine.calibrate(calib_df)

amb_rec = create_curated_ambiguous_dispute_case()
packet = engine.evaluate_dispute(amb_rec)

print(f"Calibration q_hat: {engine.conformal_classifier.q_hat}")
print(f"P_win: {packet.p_win_calibrated}")
print(f"EV: {packet.expected_value_inr}")
print(f"Prediction set: {packet.conformal_prediction_set}")
print(f"Decision: {packet.decision}")
print(f"Conformal status: {packet.conformal_status}")
print(f"Rubric breakdown: {engine.rubric_scorer.score_record(amb_rec)}")
