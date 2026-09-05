"""
Module 0 — Synthetic Benchmark Generator (benchmark/generate.py)

Generates calibrated, reproducible synthetic transaction and dispute datasets
for chargeback representment evaluation and evidence automation.

Calibrated against:
- ULB Credit Card Fraud Dataset: ~0.172% fraud/dispute rate
- PaySim Synthetic Dataset: ~0.13% fraud rate
- IEEE-CIS Fraud Detection: realistic device/IP/email matching correlations
- IBM TabFormer: longitudinal customer transaction history and MCC structure
- Visa CE3.0 / Mastercard First-Party Trust evidentiary guidelines
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import json
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import sys
from pathlib import Path

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from faker import Faker
    _has_faker = True
except ImportError:
    _has_faker = False

from benchmark.metrics import (
    evaluate_dispute_model,
    evaluate_criterion_breakdown,
    count_based_win_rate,
    amount_weighted_win_rate,
    compute_pr_auc,
    compute_roc_auc,
)

# Reference BIN / Issuer mappings (India & Global major issuers)
ISSUER_REFERENCE = [
    {"bin": "411111", "network": "Visa", "issuer": "HDFC Bank", "type": "Credit"},
    {"bin": "422222", "network": "Visa", "issuer": "ICICI Bank", "type": "Debit"},
    {"bin": "433333", "network": "Visa", "issuer": "State Bank of India", "type": "Debit"},
    {"bin": "512345", "network": "Mastercard", "issuer": "Axis Bank", "type": "Credit"},
    {"bin": "523456", "network": "Mastercard", "issuer": "Kotak Mahindra Bank", "type": "Debit"},
    {"bin": "607123", "network": "RuPay", "issuer": "State Bank of India", "type": "Debit"},
    {"bin": "652150", "network": "RuPay", "issuer": "Punjab National Bank", "type": "Debit"},
    {"bin": "454321", "network": "Visa", "issuer": "Standard Chartered", "type": "Credit"},
    {"bin": "548765", "network": "Mastercard", "issuer": "Citibank", "type": "Credit"},
    {"bin": "401288", "network": "Visa", "issuer": "Federal Bank", "type": "Debit"},
]

MAJOR_INDIAN_CITIES = [
    {"city": "Mumbai", "state": "Maharashtra", "pincode_prefix": "400"},
    {"city": "Bengaluru", "state": "Karnataka", "pincode_prefix": "560"},
    {"city": "Delhi", "state": "Delhi", "pincode_prefix": "110"},
    {"city": "Hyderabad", "state": "Telangana", "pincode_prefix": "500"},
    {"city": "Chennai", "state": "Tamil Nadu", "pincode_prefix": "600"},
    {"city": "Pune", "state": "Maharashtra", "pincode_prefix": "411"},
    {"city": "Kolkata", "state": "West Bengal", "pincode_prefix": "700"},
    {"city": "Ahmedabad", "state": "Gujarat", "pincode_prefix": "380"},
    {"city": "Jaipur", "state": "Rajasthan", "pincode_prefix": "302"},
    {"city": "Lucknow", "state": "Uttar Pradesh", "pincode_prefix": "226"},
]

DEVICE_TYPES = ["mobile_ios", "mobile_android", "desktop_windows", "desktop_macos", "mobile_browser"]

REASON_CODES = [
    {
        "code": "10.4",
        "network": "Visa",
        "name": "Fraud - Card-Absent Environment / Non-Recognition",
        "description": "Cardholder claims they did not participate in or authorize the transaction.",
        "ce3_eligible": True,
        "weight": 0.65,
    },
    {
        "code": "4837",
        "network": "Mastercard",
        "name": "No Cardholder Authorization",
        "description": "First-party fraud / non-recognition claim for e-commerce purchase.",
        "ce3_eligible": True,
        "weight": 0.20,
    },
    {
        "code": "10.5",
        "network": "Visa",
        "name": "Fraud - Cardholder Counterfeit / Stolen",
        "description": "True third-party unauthorized fraud claim.",
        "ce3_eligible": False,
        "weight": 0.05,
    },
    {
        "code": "13.1",
        "network": "Visa",
        "name": "Merchandise / Services Not Received",
        "description": "Cardholder claims ordered goods were not delivered.",
        "ce3_eligible": False,
        "weight": 0.05,
    },
    {
        "code": "4853",
        "network": "Mastercard",
        "name": "Cardholder Dispute - Defective/Not as Described",
        "description": "Cardholder disputes quality/fulfillment terms.",
        "ce3_eligible": False,
        "weight": 0.05,
    },
]

# CE3.0 / First-Party Trust Rule Weights for Simulated Issuer Decision Function
ISSUER_RULE_WEIGHTS = {
    "prior_undisputed_in_window": 0.30,  # Foundational CE3.0 rule: undisputed tx 120-365 days prior
    "device_match": 0.25,                # Device ID exact match
    "shipping_match": 0.20,              # Shipping address/PIN match
    "ip_match": 0.15,                    # IP / Geo / Subnet match
    "cvv_avs_verified": 0.05,            # CVV + AVS verification
    "otp_3ds_authenticated": 0.05,       # 3DS / OTP step-up verification
}

ISSUER_DECISION_THRESHOLD = 0.55
ISSUER_NOISE_SIGMA = 0.08


class SyntheticBenchmarkGenerator:
    """
    Calibrated Synthetic Transaction and Dispute Dataset Generator.
    Produces reproducible datasets with fixed random seed.
    """

    def __init__(
        self,
        seed: int = 42,
        dispute_rate: float = 0.0038,  # Calibrated to 0.38% (0.13% - 0.40% industry range)
        noise_sigma: float = ISSUER_NOISE_SIGMA,
        n_merchants: int = 40,
        thin_merchant_ratio: float = 0.40,  # 40% of merchants have thin/cold-start history
    ):
        self.seed = seed
        self.dispute_rate = dispute_rate
        self.noise_sigma = noise_sigma
        self.n_merchants = n_merchants
        self.thin_merchant_ratio = thin_merchant_ratio
        
        self.rng = np.random.RandomState(seed)
        if _has_faker:
            Faker.seed(seed)
            self.fake = Faker("en_IN")
        else:
            self.fake = None

    def _sample_amount(self) -> float:
        """
        Calibrated right-skewed lognormal distribution for e-commerce transactions in INR (₹).
        Mean ~₹3,200, median ~₹1,450, 95th percentile ~₹18,500.
        """
        # log(amount) with mean 7.3 (~1480) and std 1.15
        raw_amt = float(np.exp(self.rng.normal(7.3, 1.15)))
        amt = np.clip(raw_amt, 150.0, 85000.0)
        return round(float(amt), 2)

    def _generate_ip(self, city_info: Dict[str, str], is_match: bool, prior_ip: Optional[str] = None) -> str:
        if is_match and prior_ip is not None:
            return prior_ip
        # Subnet simulation: first two octets by city region
        prefix_map = {
            "Mumbai": "103.21",
            "Bengaluru": "115.112",
            "Delhi": "122.160",
            "Hyderabad": "182.73",
            "Chennai": "106.51",
            "Pune": "114.143",
            "Kolkata": "117.201",
            "Ahmedabad": "125.17",
            "Jaipur": "157.34",
            "Lucknow": "14.139",
        }
        prefix = prefix_map.get(city_info["city"], "49.204")
        return f"{prefix}.{self.rng.randint(1, 254)}.{self.rng.randint(1, 254)}"

    def _generate_pincode(self, city_info: Dict[str, str], is_match: bool, prior_pincode: Optional[str] = None) -> str:
        if is_match and prior_pincode is not None:
            return prior_pincode
        suffix = self.rng.randint(100, 999)
        return f"{city_info['pincode_prefix']}{suffix:03d}"

    def generate_benchmark(
        self,
        n_total_transactions: int = 50000,
        start_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates both full transaction log and enriched dispute dataset.
        Returns:
            transactions_df: Full dataset with class imbalance (~0.38% dispute rate)
            disputes_df: Enriched dispute dataset with per-criterion labels and issuer outcomes
        """
        if start_date is None:
            start_date = datetime(2025, 1, 1, 0, 0, 0)

        # Setup merchant tiers (Enterprise vs Thin/Cold-start)
        merchants = []
        n_thin = int(self.n_merchants * self.thin_merchant_ratio)
        for i in range(self.n_merchants):
            is_thin = i < n_thin
            merchants.append({
                "merchant_id": f"merch_{i+1:03d}",
                "tier": "thin_history" if is_thin else "enterprise",
                "traffic_weight": 0.25 if is_thin else 1.0,
            })
        merchant_weights = np.array([m["traffic_weight"] for m in merchants])
        merchant_weights /= merchant_weights.sum()

        records = []
        dispute_records = []

        # Customer pool
        n_customers = max(5000, int(n_total_transactions * 0.35))
        customer_profiles = []
        for cid in range(n_customers):
            city_info = MAJOR_INDIAN_CITIES[self.rng.randint(0, len(MAJOR_INDIAN_CITIES))]
            base_pincode = f"{city_info['pincode_prefix']}{self.rng.randint(100, 999):03d}"
            base_device_id = f"dev_{self.rng.bytes(6).hex()}"
            base_ip = self._generate_ip(city_info, False)
            issuer_ref = ISSUER_REFERENCE[self.rng.randint(0, len(ISSUER_REFERENCE))]
            
            if self.fake:
                name = self.fake.name()
                email = f"{name.lower().replace(' ', '.')}{self.rng.randint(10, 99)}@gmail.com"
            else:
                name = f"Cardholder {cid+1}"
                email = f"user{cid+1}@example.in"

            customer_profiles.append({
                "customer_id": f"cust_{cid+1:06d}",
                "name": name,
                "email": email,
                "city_info": city_info,
                "base_pincode": base_pincode,
                "base_device_id": base_device_id,
                "base_ip": base_ip,
                "issuer_ref": issuer_ref,
                "card_last4": f"{self.rng.randint(1000, 9999):04d}",
                "device_type": DEVICE_TYPES[self.rng.randint(0, len(DEVICE_TYPES))],
            })

        # Generate transactions
        for tx_idx in range(n_total_transactions):
            tx_id = f"tx_{tx_idx+1:08d}"
            cust = customer_profiles[self.rng.randint(0, n_customers)]
            merch_idx = self.rng.choice(len(merchants), p=merchant_weights)
            merch = merchants[merch_idx]

            # Timestamp within a 1-year evaluation window (days 120 to 485)
            day_offset = self.rng.uniform(120, 485)
            hour_offset = self.rng.uniform(0, 24)
            tx_time = start_date + timedelta(days=day_offset, hours=hour_offset)

            amount = self._sample_amount()
            is_dispute = bool(self.rng.rand() < self.dispute_rate)

            # Prior transaction history for cardholder
            # For cold-start merchants or new users, history is thinner
            if merch["tier"] == "thin_history":
                has_history_prob = 0.70
            else:
                has_history_prob = 0.85

            has_prior_history = bool(self.rng.rand() < has_history_prob)
            prior_history = []
            
            # CE3.0 requires prior undisputed transactions in 120-365 days window
            has_qualifying_prior_window = False
            device_matches = False
            ip_matches = False
            shipping_matches = False

            if has_prior_history:
                # Number of prior transactions (1 to 6)
                n_prior = self.rng.randint(1, 7 if merch["tier"] == "enterprise" else 3)
                for p_idx in range(n_prior):
                    # Sample age of prior transaction relative to current tx
                    # Distribute some inside [120, 365] days window and some recent [5, 119] days
                    in_ce3_window = bool(self.rng.rand() < 0.60)
                    if in_ce3_window:
                        p_age_days = float(self.rng.uniform(120.0, 365.0))
                        has_qualifying_prior_window = True
                    else:
                        p_age_days = float(self.rng.uniform(5.0, 119.0))

                    p_time = tx_time - timedelta(days=p_age_days)
                    p_amt = self._sample_amount()
                    
                    prior_history.append({
                        "prior_tx_id": f"ptx_{tx_idx+1:06d}_{p_idx+1}",
                        "timestamp": p_time.isoformat(),
                        "age_days": round(p_age_days, 1),
                        "amount": p_amt,
                        "device_id": cust["base_device_id"],
                        "ip_address": cust["base_ip"],
                        "shipping_pincode": cust["base_pincode"],
                        "status": "settled_undisputed",
                        "in_ce3_qualifying_window": in_ce3_window,
                    })

            # For current transaction: assign device, IP, shipping with realistic correlation
            # In genuine/friendly fraud cases, cardholder used their real device/IP/address
            # In stolen card fraud cases, attacker used different device/IP/address
            if is_dispute:
                # Determine underlying nature of dispute:
                # ~75% friendly fraud / first-party non-recognition (CE3.0 defendable)
                # ~25% true fraud / unauthorized takeover (hard to defend)
                is_first_party_friendly = bool(self.rng.rand() < 0.75)
                
                if is_first_party_friendly:
                    # High probability of genuine matches
                    device_matches = bool(has_prior_history and (self.rng.rand() < 0.82))
                    ip_matches = bool(has_prior_history and (self.rng.rand() < 0.70))
                    shipping_matches = bool(has_prior_history and (self.rng.rand() < 0.88))
                    cvv_avs_matched = bool(self.rng.rand() < 0.95)
                    otp_3ds_matched = bool(self.rng.rand() < 0.92)
                else:
                    # Low probability of genuine matches (true fraud)
                    device_matches = bool(has_prior_history and (self.rng.rand() < 0.10))
                    ip_matches = bool(has_prior_history and (self.rng.rand() < 0.12))
                    shipping_matches = bool(has_prior_history and (self.rng.rand() < 0.15))
                    cvv_avs_matched = bool(self.rng.rand() < 0.75)
                    otp_3ds_matched = bool(self.rng.rand() < 0.30)
            else:
                # Regular non-disputed transaction
                device_matches = has_prior_history and bool(self.rng.rand() < 0.90)
                ip_matches = has_prior_history and bool(self.rng.rand() < 0.80)
                shipping_matches = has_prior_history and bool(self.rng.rand() < 0.95)
                cvv_avs_matched = bool(self.rng.rand() < 0.98)
                otp_3ds_matched = bool(self.rng.rand() < 0.96)

            # Assign concrete transaction attribute values based on match decisions
            current_device_id = cust["base_device_id"] if device_matches else f"dev_alt_{self.rng.bytes(6).hex()}"
            current_ip = cust["base_ip"] if ip_matches else self._generate_ip(cust["city_info"], False)
            current_shipping_pincode = cust["base_pincode"] if shipping_matches else self._generate_pincode(cust["city_info"], False)
            billing_pincode = cust["base_pincode"]

            # Per-criterion ground truth flags
            crit_prior_window = bool(has_qualifying_prior_window)
            crit_device = bool(device_matches and has_prior_history)
            crit_ip = bool(ip_matches and has_prior_history)
            crit_shipping = bool(shipping_matches and has_prior_history)
            crit_avs_cvv = bool(cvv_avs_matched)
            crit_3ds = bool(otp_3ds_matched)

            # Assign Reason Code for disputed transactions
            reason_info = None
            issuer_won = 0
            issuer_raw_score = 0.0
            issuer_noisy_score = 0.0

            if is_dispute:
                reason_p = np.array([r["weight"] for r in REASON_CODES])
                reason_p /= reason_p.sum()
                r_idx = self.rng.choice(len(REASON_CODES), p=reason_p)
                reason_info = REASON_CODES[r_idx]

                # Simulated Issuer Decision Function:
                # Rule score based on CE3.0 evidentiary weight table + Gaussian noise
                issuer_raw_score = (
                    ISSUER_RULE_WEIGHTS["prior_undisputed_in_window"] * (1.0 if crit_prior_window else 0.0) +
                    ISSUER_RULE_WEIGHTS["device_match"] * (1.0 if crit_device else 0.0) +
                    ISSUER_RULE_WEIGHTS["shipping_match"] * (1.0 if crit_shipping else 0.0) +
                    ISSUER_RULE_WEIGHTS["ip_match"] * (1.0 if crit_ip else 0.0) +
                    ISSUER_RULE_WEIGHTS["cvv_avs_verified"] * (1.0 if crit_avs_cvv else 0.0) +
                    ISSUER_RULE_WEIGHTS["otp_3ds_authenticated"] * (1.0 if crit_3ds else 0.0)
                )

                # If reason code is not CE3.0 eligible (e.g. 10.5 counterfeit or 13.1 goods not received),
                # issuer decision criteria differ (penalized for CE3.0-only evidence)
                if not reason_info["ce3_eligible"]:
                    issuer_raw_score *= 0.40

                # Inject Gaussian noise representing issuer unpredictability / human adjudicator variance
                noise = float(self.rng.normal(0.0, self.noise_sigma))
                issuer_noisy_score = float(np.clip(issuer_raw_score + noise, 0.0, 1.0))
                issuer_won = 1 if (issuer_noisy_score >= ISSUER_DECISION_THRESHOLD) else 0

            # Base transaction row
            tx_row = {
                "transaction_id": tx_id,
                "merchant_id": merch["merchant_id"],
                "merchant_tier": merch["tier"],
                "customer_id": cust["customer_id"],
                "customer_name": cust["name"],
                "customer_email": cust["email"],
                "timestamp": tx_time.isoformat(),
                "amount": amount,
                "currency": "INR",
                "card_bin": cust["issuer_ref"]["bin"],
                "card_network": cust["issuer_ref"]["network"],
                "issuer_bank": cust["issuer_ref"]["issuer"],
                "card_type": cust["issuer_ref"]["type"],
                "card_last4": cust["card_last4"],
                "device_id": current_device_id,
                "device_type": cust["device_type"],
                "device_ip": current_ip,
                "billing_city": cust["city_info"]["city"],
                "billing_state": cust["city_info"]["state"],
                "billing_postal_code": billing_pincode,
                "billing_country": "IN",
                "shipping_city": cust["city_info"]["city"],
                "shipping_state": cust["city_info"]["state"],
                "shipping_postal_code": current_shipping_pincode,
                "shipping_country": "IN",
                "cvv_avs_matched": crit_avs_cvv,
                "otp_3ds_matched": crit_3ds,
                "prior_transaction_count": len(prior_history),
                "is_disputed": int(is_dispute),
            }
            records.append(tx_row)

            # If disputed, construct enriched dispute record
            if is_dispute:
                # Extract ages of prior transactions
                prior_ages = [p["age_days"] for p in prior_history]
                prior_window_count = sum(1 for p in prior_history if p["in_ce3_qualifying_window"])
                min_prior_age = min(prior_ages) if prior_ages else -1.0
                max_prior_age = max(prior_ages) if prior_ages else -1.0

                dispute_row = {
                    **tx_row,
                    "dispute_id": f"dsp_{tx_id}",
                    "reason_code": reason_info["code"],
                    "reason_name": reason_info["name"],
                    "reason_network": reason_info["network"],
                    "reason_ce3_eligible": int(reason_info["ce3_eligible"]),
                    "prior_transaction_history": json.dumps(prior_history),
                    "prior_undisputed_window_count": prior_window_count,
                    "prior_transaction_age_min_days": min_prior_age,
                    "prior_transaction_age_max_days": max_prior_age,
                    # Per-criterion ground truth match labels
                    "criterion_prior_window_match": int(crit_prior_window),
                    "criterion_device_match": int(crit_device),
                    "criterion_ip_match": int(crit_ip),
                    "criterion_shipping_match": int(crit_shipping),
                    "criterion_avs_cvv_match": int(crit_avs_cvv),
                    "criterion_3ds_match": int(crit_3ds),
                    # Criteria summary count
                    "total_criteria_matched": int(
                        crit_prior_window + crit_device + crit_ip + crit_shipping + crit_avs_cvv + crit_3ds
                    ),
                    # Simulated Issuer Outcome
                    "issuer_raw_rule_score": round(issuer_raw_score, 4),
                    "issuer_noisy_score": round(issuer_noisy_score, 4),
                    "issuer_dispute_won": int(issuer_won),  # 1 = Merchant wins representment, 0 = Lost
                }
                dispute_records.append(dispute_row)

        tx_df = pd.DataFrame(records)
        disp_df = pd.DataFrame(dispute_records)

        return tx_df, disp_df

    def create_stratified_split(
        self,
        df: pd.DataFrame,
        target_col: str = "issuer_dispute_won",
        test_size: float = 0.25,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates a stratified train/test split preserving label distribution and merchant tiers.
        """
        strat_key = df[target_col].astype(str) + "_" + df["merchant_tier"].astype(str)
        
        # In case some composite stratum has < 2 samples, fallback to target_col stratification
        value_counts = strat_key.value_counts()
        if (value_counts < 2).any():
            strat_key = df[target_col]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        train_idx, test_idx = next(sss.split(df, strat_key))

        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        return train_df, test_df


def run_baseline_logistic_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluates a naive Logistic Regression baseline on raw dispute features.
    Verifies that the dataset produces a sane, non-trivial PR-AUC.
    """
    feature_cols = [
        "amount",
        "prior_transaction_count",
        "prior_undisputed_window_count",
        "prior_transaction_age_min_days",
        "prior_transaction_age_max_days",
        "cvv_avs_matched",
        "otp_3ds_matched",
        "reason_ce3_eligible",
    ]

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["issuer_dispute_won"].values
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["issuer_dispute_won"].values
    amounts_test = test_df["amount"].values

    clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
    clf.fit(X_train, y_train)

    y_score_test = clf.predict_proba(X_test)[:, 1]
    
    eval_results = evaluate_dispute_model(
        y_true=y_test,
        y_score=y_score_test,
        amounts=amounts_test,
        threshold=0.5,
    )

    return {
        "model": "Naive Logistic Regression Baseline",
        "features": feature_cols,
        "eval_results": eval_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Module 0: Calibrated Synthetic Benchmark Generator")
    parser.add_argument("--n-samples", type=int, default=50000, help="Total transactions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="benchmark/data", help="Output directory for datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating synthetic benchmark with N={args.n_samples}, seed={args.seed}...")

    generator = SyntheticBenchmarkGenerator(seed=args.seed)
    tx_df, disp_df = generator.generate_benchmark(n_total_transactions=args.n_samples)

    total_tx = len(tx_df)
    total_disputes = len(disp_df)
    dispute_pct = (total_disputes / total_tx) * 100
    won_disputes = disp_df["issuer_dispute_won"].sum()
    win_pct = (won_disputes / total_disputes) * 100 if total_disputes > 0 else 0

    print(f"Total transactions generated: {total_tx:,}")
    print(f"Total disputes generated: {total_disputes:,} ({dispute_pct:.3f}% dispute rate)")
    print(f"Merchant representment wins: {won_disputes:,}/{total_disputes:,} ({win_pct:.2f}% win rate)")

    # Stratified split on disputes
    train_disp, test_disp = generator.create_stratified_split(disp_df, target_col="issuer_dispute_won", test_size=0.25)
    print(f"Disputes Split -> Train: {len(train_disp):,}, Test: {len(test_disp):,}")

    # Export datasets
    tx_df.to_parquet(os.path.join(args.output_dir, "transactions.parquet"), index=False)
    disp_df.to_parquet(os.path.join(args.output_dir, "disputes.parquet"), index=False)
    train_disp.to_parquet(os.path.join(args.output_dir, "train_disputes.parquet"), index=False)
    test_disp.to_parquet(os.path.join(args.output_dir, "test_disputes.parquet"), index=False)

    # Also save CSV for portability
    disp_df.to_csv(os.path.join(args.output_dir, "disputes.csv"), index=False)
    train_disp.to_csv(os.path.join(args.output_dir, "train_disputes.csv"), index=False)
    test_disp.to_csv(os.path.join(args.output_dir, "test_disputes.csv"), index=False)

    print(f"Datasets saved to {args.output_dir}/")

    # Run baseline model evaluation
    print("\nRunning Naive Logistic Regression Baseline...")
    baseline_out = run_baseline_logistic_regression(train_disp, test_disp)
    res = baseline_out["eval_results"]
    print(f"Baseline Results on Test Set (N={len(test_disp)}):")
    print(f"  PR-AUC:                   {res['pr_auc']:.4f}")
    print(f"  ROC-AUC:                  {res['roc_auc']:.4f}")
    print(f"  Count-based Win Rate:     {res['count_win_rate']:.4f}")
    print(f"  Amount-weighted Win Rate: {res['amount_weighted_win_rate']:.4f}")
    print(f"  Net Recovered PnL:        INR {res['financials']['net_pnl_inr']:,.2f}")
    print(f"  Recovery Rate:            {res['financials']['recovery_rate']*100:.2f}%")


if __name__ == "__main__":
    main()
