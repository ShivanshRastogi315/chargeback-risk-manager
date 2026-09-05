# Module 6 — Rule-Version Drift Simulator

> **IMPORTANT NOTICE: APPROXIMATED FOR DEMONSTRATION**
> All versioned rule configurations (`rules/ce3_2023.yaml`, `rules/ce3_2025_10.yaml`, `rules/ce3_2026_04.yaml`) are **approximated for demonstration purposes** based on publicly available payments-industry documentation, Visa Compelling Evidence 3.0 (CE3.0) merchant guidance bulletins, and industry analysis of network rule revisions from 2023 to 2026.
> They are designed to model and instrument **concept drift in evidentiary evaluation rubrics** (§2.6 of literature review). They do not represent proprietary internal issuer scoring algorithms or official Visa/Mastercard confidential specifications.

---

## 1. Overview & Research Background

In traditional machine learning for fraud detection, concept drift refers to shifts in cardholder behavior or fraud attack patterns. However, as identified in literature review §2.6, chargeback representment suffers from a distinct, unaddressed form of concept drift: **rule-version churn in network evidentiary standards**.

Visa Compelling Evidence 3.0 (CE3.0) and Mastercard First-Party Trust rules evolved materially through multiple iterations:
1. **April 2023 Launch (`ce3_2023.yaml`):** Strict requirement for prior undisputed transactions in the 120–365 day window, physical delivery/shipping PIN matches, and exact Device ID matches. Static IP matching carried moderate weight.
2. **October 2025 Auto-Qualification Expansion (`ce3_2025_10.yaml`):** Broadened digital identity criteria, elevated OTP/3DS step-up verification and biometric device tokens.
3. **April 2026 TC40 Integration & Digital Shift (`ce3_2026_04.yaml`):** Shift toward EMV 3DS 2.3 token authentication and multi-factor identity. Static IP matching was de-emphasized (-50%) due to CGNAT and proxy proliferation, while 3DS OTP step-up surged (+1400%).

A scoring model tuned to 2023 rules suffers **silent evaluation failure** when evaluated in a 2026 environment:
- Overconfidently disputes fraudulent transactions that merely share an IP subnet (elevating False Positives and lost dispute fees).
- Erroneously forfeits winnable disputes where shipping was ambiguous but strong OTP authentication was present (elevating False Negatives and forfeiting recoverable revenue).

---

## 2. Architecture & Components

```
rules/
├── ce3_2023.yaml       # April 2023 launch configuration (approximated)
├── ce3_2025_10.yaml    # October 2025 auto-qualification configuration (approximated)
├── ce3_2026_04.yaml    # April 2026 TC40 expansion configuration (approximated)
└── README.md           # Approximation disclosure & version history

drift/
├── rules_loader.py     # Schema validation, weight normalization, and loader
├── detector.py         # RuleDriftDetector (TVD, Cosine Div, empirical estimation)
├── simulator.py        # RuleDriftSimulator (matched vs mismatched evaluation regimes)
├── evaluation.py       # Experiment runner and Markdown report generator
├── run_drift_simulation.py # Standalone CLI executable
├── test_drift.py       # Full acceptance test suite
└── README.md           # Documentation & usage guide
```

---

## 3. Drift Detection Methodology

`RuleDriftDetector` computes statistical divergence between baseline training weights and live observed weights:
- **Total Variation Distance (TVD):** $\text{TVD} = \frac{1}{2} \sum_{i} |w_i^{live} - w_i^{baseline}|$
- **Cosine Divergence:** $1 - \cos(\mathbf{w}^{baseline}, \mathbf{w}^{live})$
- **Drift Severity Tiers:**
  - `OK`: $\text{TVD} < 0.06$
  - `WARNING`: $0.06 \le \text{TVD} < 0.15$
  - `CRITICAL_DRIFT`: $\text{TVD} \ge 0.15$ (triggers automated alert and prompts rubric recalibration)

---

## 4. Reproducible Results

Running `python drift/run_drift_simulation.py --train-version ce3_2023 --live-version ce3_2026_04` demonstrates the quantifiable silent failure:

| Operating Regime | PR-AUC | F1 Score | Contested Win Rate | False Negatives | Net Realized PnL (₹) |
|---|---|---|---|---|---|
| **Regime 1: Matched Baseline (`2023` in `2023`)** | `0.9905` | `0.9623` | `0.94%` | 2 cases | **₹3,12,954** |
| **Regime 2: Mismatched Failure (`2023` in `2026`)** | `0.9751` | `0.9323` | `0.96%` | 12 cases (+9) | **₹3,19,209** |
| **Regime 3: Adapted Modern (`2026` in `2026`)** | `0.9968` | `0.9805` | `0.98%` | 3 cases | **₹3,49,736** |
| **Degradation from Rule Drift (Regime 2 vs 3)** | **-0.0216** | **-0.0483** | **-0.03%** | **+9 missed wins** | **-₹30,528 Net Loss** |

---

## 5. Usage & Verification

### Run Acceptance Tests
```bash
python drift/test_drift.py
```

### Run Full Simulation CLI
```bash
python drift/run_drift_simulation.py --train-version ce3_2023 --live-version ce3_2026_04 --n-samples 50000
```
