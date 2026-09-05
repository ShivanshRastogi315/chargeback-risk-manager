# Module 6 — Rule-Version Concept Drift Simulation Report
**Silent Evaluation Failure Analysis & Drift Detection (Visa CE3.0 Evolution)**

> **APPROXIMATION NOTICE:** All rule-version files (`ce3_2023.yaml`, `ce3_2025_10.yaml`, `ce3_2026_04.yaml`) are **approximated for demonstration** based on publicly available Visa CE3.0 documentation and industry research.

---

## 1. Executive Summary & Drift Detector Verdict
- **Baseline Training Version:** `ce3_2023`
- **Live Production Environment Version:** `ce3_2026_04`
- **Drift Detection Status:** **`CRITICAL_DRIFT`** (TVD = `0.4000` | Cosine Div = `0.3121`)
- **Recommended Remediation:** `RECALIBRATE_RUBRIC_TO_CURRENT_VERSION`
- **Diagnostic Narrative:** CRITICAL RULE DRIFT DETECTED (TVD = 0.4000 >= 0.15) between baseline 'ce3_2023' and live 'ce3_2026_04'. Key divergent criteria: otp_3ds_authenticated (INCREASED 1400.0%, delta +0.280); prior_undisputed_in_window (DECREASED 57.1%, delta -0.200); cvv_avs_verified (INCREASED 400.0%, delta +0.120). Silent win-rate degradation is active.

### Key Divergent Criteria Breakdown
| Criterion | Baseline Weight (2023) | Live Weight (2026) | Absolute Delta | Relative Shift (%) | Direction |
|---|---|---|---|---|---|
| `otp_3ds_authenticated` | 0.02 | 0.30 | +0.280 | +1400.0% | **INCREASED** |
| `prior_undisputed_in_window` | 0.35 | 0.15 | -0.200 | -57.1% | **DECREASED** |
| `cvv_avs_verified` | 0.03 | 0.15 | +0.120 | +400.0% | **INCREASED** |
| `device_match` | 0.30 | 0.20 | -0.100 | -33.3% | **DECREASED** |
| `shipping_match` | 0.20 | 0.15 | -0.050 | -25.0% | **DECREASED** |
| `ip_match` | 0.10 | 0.05 | -0.050 | -50.0% | **DECREASED** |

---

## 2. Quantitative Performance Drop (Silent Failure Demonstration)
Comparing model performance across the three operating regimes on the same dispute records:

| Metric / Business Dimension | Regime 1: Matched Baseline (`2023` in `2023`) | Regime 2: Mismatched Silent Failure (`2023` in `2026`) | Regime 3: Adapted Modern (`2026` in `2026`) | Degradation Impact (Regime 2 vs 3) |
|---|---|---|---|---|
| **PR-AUC** | `0.9905` | `0.9751` | `0.9968` | **-0.0216** (-2.2%) |
| **ROC-AUC** | `0.9901` | `0.9571` | `0.9958` | **-0.0388** |
| **F1 Score** | `0.9623` | `0.9323` | `0.9805` | **-0.0483** |
| **Contested Win Rate (Count)** | `0.94%` | `0.96%` | `0.98%` | **-0.03%** |
| **Contested Win Rate (Amount)**| `0.96%` | `0.98%` | `0.99%` | **-0.01%** |
| **False Positives (Lost Contests)** | 7 cases | 5 cases | 2 cases | **+3 excess FP cases** |
| **False Negatives (Missed Wins)** | 2 cases | 12 cases | 3 cases | **+9 excess FN cases** |
| **False Positive Cost (₹ INR)** | ₹1,575.00 | ₹1,125.00 | ₹450.00 | **+₹675.00** |
| **False Negative Cost (₹ INR)** | ₹6,975.05 | ₹32,614.58 | ₹2,761.78 | **+₹29,852.80** |
| **Total Monetary Error Cost (₹)** | ₹8,550.05 | ₹33,739.58 | ₹3,211.78 | **+₹30,527.80** |
| **Net Realized Portfolio PnL (₹)**| **₹312,954.02** | **₹319,208.54** | **₹349,736.34** | **-₹30,527.80 Net Loss** |

---

## 3. Why the Silent Failure Occurs (§2.6 Literature Review Insight)
1. **IP Match Deprecation:** The 2023 model relies on static IP matches (+0.10 weight). In 2026, mobile proxies and CGNAT rendered IP matching unreliable, causing issuers to downweight IP (+0.05). The 2023 model overconfidently contests fraudulent transactions that merely share an IP subnet.
2. **3DS OTP Step-Up Expansion:** In 2026, 3DS OTP step-up authentication became a decisive criterion (+0.30 weight, +1400% surge vs 2023). The 2023 model discounts 3DS (+0.02) and declines to contest winnable disputes where shipping was ambiguous but strong OTP authentication was present.
3. **Drift Detector Intervention:** The `RuleDriftDetector` tracks Total Variation Distance and flags the exact criteria responsible, prompting automatic recalibration before merchant revenue is forfeited.

---