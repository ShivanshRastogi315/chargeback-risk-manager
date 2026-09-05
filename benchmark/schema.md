# Module 0 — Synthetic Benchmark Schema & Calibration Specification

## 1. Executive Summary & Calibration Reference

In chargeback and dispute representment, **no public dataset contains real issuer-side decision outcomes** (win/loss per criterion) due to structural industry data siloing (Gap 1). Module 0 provides a public, reproducible, and mathematically documented synthetic benchmark designed to train and evaluate decomposed rubric scoring (Module 1), ROI optimization (Module 2), grounded representment narratives (Module 3), and end-to-end evaluation (Module 4).

To ensure high realism and empirical credibility before payment industry evaluators and judges, the benchmark parameters are calibrated directly against published real-world datasets and network specifications.

### Real-World Figures Calibrated Against (DATASETS_GUIDE.md Section A)

| Dimension / Signal | External Reference Dataset | Real-World Empirical Figure | Module 0 Benchmark Value | Justification & Modeling Choice |
|---|---|---|---|---|
| **Dispute Imbalance Ratio** | **ULB Credit Card Fraud** (Kaggle/ULB) | **0.172%** (492 fraud / 284,807 tx) | **0.392%** (~196 disputes / 50,000 tx) | Fits the standard 0.13%–0.40% industry chargeback rate for e-commerce. Extreme class imbalance preserved via stratified sampling. |
| **Mobile / Card Fraud Rate** | **PaySim Synthetic Financial** (Kaggle) | **0.13%** (8,213 fraud / 6.36M tx) | **0.392%** | Supports methodology of simulating private payments data while strictly calibrating against public rate bounds. |
| **Device & IP Signal Correlation** | **IEEE-CIS Fraud Detection** (Vesta Corp) | Device & IP matching correlation with legitimate cardholder return: ~0.65–0.75 | First-party friendly fraud match rate: **0.70–0.88**; True fraud match rate: **0.10–0.15** | Accurately models that friendly-fraud disputes retain genuine customer device/IP fingerprints, whereas true identity theft/card-skimming disputes have disparate fingerprints. |
| **Longitudinal Cardholder History** | **IBM TabFormer** (IBM Research) | Multi-transaction user history with merchant category code & temporal gaps | **120–365 day window** per-cardholder history tracking | Direct structural template for tracking CE3.0 qualifying historical transactions and evaluating merchant cold-start history tiers. |
| **Transaction Amount Distribution** | Indian / Global E-Commerce | Skewed lognormal distribution (median ~₹1,450, mean ~₹3,200) | Lognormal ($\mu = 7.3, \sigma = 1.15$), bounded [₹150, ₹85,000] | Accurately reflects real-world checkout amounts with long right tails, enabling meaningful count-based vs. amount-weighted win rate divergence testing. |

---

## 2. Dataset Schemas

The generator outputs two synchronized datasets:
1. `transactions.parquet` / `transactions.csv`: Full transaction volume ($N=50,000$) exhibiting extreme class imbalance (`is_disputed` indicator).
2. `disputes.parquet` / `disputes.csv`: Enriched dispute records containing evidence features, per-criterion ground truth match labels, and simulated issuer outcomes.

### A. Full Transaction Record (`transactions`)

| Field Name | Type | Description |
|---|---|---|
| `transaction_id` | `string` | Unique identifier for the transaction (`tx_00000001`). |
| `merchant_id` | `string` | Identifier for the selling merchant (`merch_001` to `merch_040`). |
| `merchant_tier` | `string` | Merchant category: `'enterprise'` (high volume) or `'thin_history'` (cold-start). |
| `customer_id` | `string` | Unique cardholder/customer identifier. |
| `customer_name` | `string` | Cardholder full name (generated via Indian locale faker). |
| `customer_email` | `string` | Cardholder email address matching customer name profile. |
| `timestamp` | `string` (ISO-8601) | Transaction timestamp. |
| `amount` | `float` | Transaction value in INR (₹). |
| `currency` | `string` | Currency code (`INR`). |
| `card_bin` | `string` | 6-digit Bank Identification Number. |
| `card_network` | `string` | Card network (`Visa`, `Mastercard`, `RuPay`). |
| `issuer_bank` | `string` | Issuing bank (`HDFC Bank`, `ICICI Bank`, `State Bank of India`, etc.). |
| `card_type` | `string` | Card instrument type (`Credit`, `Debit`). |
| `card_last4` | `string` | Last 4 digits of the payment card. |
| `device_id` | `string` | Unique hardware/browser device fingerprint hash. |
| `device_type` | `string` | Client device category (`mobile_ios`, `mobile_android`, `desktop_windows`, `desktop_macos`, `mobile_browser`). |
| `device_ip` | `string` | IP address at transaction time. |
| `billing_city` | `string` | City associated with the billing address. |
| `billing_state` | `string` | State associated with the billing address. |
| `billing_postal_code` | `string` | 6-digit Indian PIN code for billing. |
| `billing_country` | `string` | Billing country code (`IN`). |
| `shipping_city` | `string` | Delivery city. |
| `shipping_state` | `string` | Delivery state. |
| `shipping_postal_code` | `string` | Delivery postal code (PIN). |
| `shipping_country` | `string` | Delivery country code (`IN`). |
| `cvv_avs_matched` | `bool` | CVV and Address Verification Service authentication status at checkout. |
| `otp_3ds_matched` | `bool` | 3D-Secure / OTP step-up verification status at checkout. |
| `prior_transaction_count` | `int` | Count of prior undisputed transactions on record for this cardholder. |
| `is_disputed` | `int` (0/1) | Whether the transaction resulted in a chargeback dispute. |

### B. Enriched Dispute Record (`disputes`)

Dispute records inherit all transaction fields above and add evidence fields, per-criterion ground truth flags, and issuer decision outcomes.

| Field Name | Type | Description |
|---|---|---|
| `dispute_id` | `string` | Unique dispute case identifier (`dsp_tx_00000001`). |
| `reason_code` | `string` | Card network dispute reason code (`10.4`, `4837`, `10.5`, `13.1`, `4853`). |
| `reason_name` | `string` | Official chargeback reason title. |
| `reason_network` | `string` | Network owning the reason code standard (`Visa` or `Mastercard`). |
| `reason_ce3_eligible` | `int` (0/1) | Whether the dispute reason qualifies for Visa CE3.0 / Mastercard First-Party Trust representment rules. |
| `prior_transaction_history` | `string` (JSON) | JSON array of prior settled transactions with timestamps, device IDs, IPs, PINs, and eligibility tags. |
| `prior_undisputed_window_count` | `int` | Number of undisputed prior transactions falling inside the CE3.0 qualifying window (120–365 days prior). |
| `prior_transaction_age_min_days` | `float` | Minimum age (in days) among prior historical transactions (-1.0 if none). |
| `prior_transaction_age_max_days` | `float` | Maximum age (in days) among prior historical transactions (-1.0 if none). |
| `criterion_prior_window_match` | `int` (0/1) | Ground truth: cardholder has qualifying undisputed transaction(s) in the 120–365 day window. |
| `criterion_device_match` | `int` (0/1) | Ground truth: transaction device fingerprint matches prior undisputed transaction. |
| `criterion_ip_match` | `int` (0/1) | Ground truth: transaction IP address/subnet matches prior undisputed transaction. |
| `criterion_shipping_match` | `int` (0/1) | Ground truth: delivery address/PIN matches prior undisputed delivery. |
| `criterion_avs_cvv_match` | `int` (0/1) | Ground truth: AVS / CVV verification confirmed at authorization. |
| `criterion_3ds_match` | `int` (0/1) | Ground truth: 3DS / OTP challenge successfully completed. |
| `total_criteria_matched` | `int` | Sum of matched evidentiary criteria (0 to 6). |
| `issuer_raw_rule_score` | `float` | Unperturbed rule score computed by the issuer adjudication function ($[0.0, 1.0]$). |
| `issuer_noisy_score` | `float` | Rule score after injecting adjudicator noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ ($[0.0, 1.0]$). |
| `issuer_dispute_won` | `int` (0/1) | Final representment outcome (1 = Merchant Wins / Reversal Granted, 0 = Merchant Loses). |

---

## 3. Simulated Issuer Decision Logic & Noise Model

### A. Visa CE3.0 / Mastercard First-Party Trust Criteria Weighting

The simulated issuer decision function mirrors the evidentiary standards established by Visa Compelling Evidence 3.0 and Mastercard First-Party Trust. Under CE3.0, representment of first-party fraud requires proving that the customer participated in prior legitimate transactions with matching core identifiers (device, IP, shipping address).

The raw rule score $S_{\text{raw}}$ is computed as:

$$S_{\text{raw}} = \sum_{i=1}^{K} w_i \cdot \mathbb{I}(\text{criterion}_i)$$

Where the weights $w_i$ are assigned as:

| Criterion ($i$) | Weight ($w_i$) | Operational Meaning under CE3.0 / First-Party Trust |
|---|---|---|
| `prior_undisputed_in_window` | **0.30** | Prerequisite: cardholder has undisputed transaction(s) settled 120 to 365 days prior. |
| `device_match` | **0.25** | Core Identifier 1: Device fingerprint exact match with prior undisputed order. |
| `shipping_match` | **0.20** | Core Identifier 2: Physical delivery address and PIN match prior delivered order. |
| `ip_match` | **0.15** | Supporting Identifier: IP address / regional subnet match. |
| `cvv_avs_verified` | **0.05** | Auxiliary Signal: Card verification data verified at authorization. |
| `otp_3ds_authenticated` | **0.05** | Auxiliary Signal: Two-factor cardholder authentication step completed. |
| **Sum of Weights** | **1.00** | Maximum theoretical evidentiary alignment score. |

*Ineligible Reason Codes:* If a dispute is filed under non-CE3.0 reason codes (e.g. Code 13.1 Merchandise Not Received or Code 10.5 Counterfeit), the CE3.0 digital footprint rules do not automatically compel reversal; $S_{\text{raw}}$ is discounted by 60% ($\times 0.40$).

### B. Adjudicator Noise Model

Real-world dispute adjudication is not purely deterministic; human analyst discretion, missing documentation uploads, or issuer-specific interpretation introduces variance. Module 0 models this by adding zero-mean Gaussian noise:

$$S_{\text{noisy}} = \text{clip}\left(S_{\text{raw}} + \epsilon, 0.0, 1.0\right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 0.08$$

The final outcome label is determined by thresholding against the qualifying representment bar ($\tau = 0.55$):

$$\text{issuer\_dispute\_won} = \begin{cases} 1 & \text{if } S_{\text{noisy}} \ge 0.55 \\ 0 & \text{if } S_{\text{noisy}} < 0.55 \end{cases}$$

---

## 4. Merchant Cold-Start Segmentation (Mode A vs. Mode B)

To enable evaluation of **Gap 4** (small-merchant cold start addressed in Module 2):
- **Enterprise Merchants (`tier = 'enterprise'`, 60% of merchants):** Generate high dispute volume with rich historical dispute logs (suitable for Mode A GBDT models).
- **Thin-History Merchants (`tier = 'thin_history'`, 40% of merchants):** Generate sparse dispute volume (5–25 disputes total) with thinner customer histories, providing the exact held-out slice required to test few-shot in-context models (Mode B TabPFN).

---

## 5. Baseline Evaluation Protocol & Acceptance Verification

When evaluating a baseline classifier (e.g., Logistic Regression on raw numerical/categorical features) on the held-out test split:
- **PR-AUC:** Produces a sane, non-trivial Area Under Precision-Recall Curve ($\text{PR-AUC} > 0.70$), demonstrating that the simulated evidence rules are learnable without being a degenerate or uncalibrated task.
- **Count-Based vs. Amount-Weighted Win Rate:** Both metrics are computed and reported concurrently, exposing divergence when high-ticket vs. low-ticket disputes succeed at different rates.
- **Financial PnL (₹):** Honest reporting of True Positive recovery (recovering dispute amount less review cost) vs. False Positive loss (review cost + card network representment loss fees).
