# AI Risk Manager — Chargeback Evidence Automation & Representment Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-32%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Defense-Only](https://img.shields.io/badge/compliance-Defense--Only%20(No%20Offense)-blueviolet.svg)](docs/00_START_HERE.md)
[![Auditability](https://img.shields.io/badge/citation--validity-100%25%20machine--verified-success.svg)](docs/FINAL_RESULTS.md)
[![Streamlit UI](https://img.shields.io/badge/demo-Streamlit%20App-red.svg)](app/main.py)

**AI Risk Manager** is a defense-only, explainable-by-construction chargeback representment system designed for merchants (particularly mid-market and cold-start merchants) to systematically dispute illegitimate first-party fraud claims (e.g. Visa 10.4 / Mastercard First-Party Trust).

Instead of outputting an opaque win-probability, the platform:
1. Decomposes dispute evidence **criterion-by-criterion** against card network specifications (Visa CE3.0 / Mastercard FPT).
2. Calculates calibrated **Fight / No-Fight ROI expected value (EV)** even with thin merchant history via pretrained tabular foundation models (TabPFN).
3. Generates **auditable, machine-verifiable evidence narratives** where **100% of claims cite verified transaction fields**.
4. Wraps decisions in a **conformal prediction layer** providing statistical error guarantees with 3-way routing (*Fight*, *Don't Fight*, *Escalate to Human Review*).
5. Detects **rule-version concept drift** across card network standard updates.

---

## 🏛️ System Architecture

```
┌───────────────────────────┐     ┌────────────────────────────┐     ┌─────────────────────────────┐
│  Module 0: Benchmark      │ ──▶ │  Module 1: Rubric Scorer   │ ──▶ │  Module 2: ROI Engine       │
│  • 50k Synthetic Tx       │     │  • Decomposed 6 CE3.0 Crit │     │  • Mode A: GBDT (History)   │
│  • Realistic Imbalance    │     │  • PR-AUC: 0.9978          │     │  • Mode B: TabPFN Cold-Start│
│  • Dual Win Rate Metrics  │     │  • Decomposed JSON Output  │     │  • Exact Expected Value EV  │
└───────────────────────────┘     └────────────────────────────┘     └─────────────────────────────┘
              │                                                                     │
              ▼                                                                     ▼
┌───────────────────────────┐     ┌────────────────────────────┐     ┌─────────────────────────────┐
│  Module 6: Rule Drift     │     │  Module 3: Narrative Gen   │ ◀── │  Module 5: Conformal Guard  │
│  • CE3.0 '23/'25/'26 YAMLs│     │  • 100% Citation Tags      │     │  • Split Conformal Bound    │
│  • Silent Failure Analysis│     │  • 100% Grounding Rate     │     │  • 3-Way Routing + Escalate │
│  • TVD Drift Detector     │     │  • Zero-PII Guardrails     │     │  • Graceful Failure Beat    │
└───────────────────────────┘     └────────────────────────────┘     └─────────────────────────────┘
              │                                  │                                  │
              └──────────────────────────────────┴──────────────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │  Module 4: Evaluation Harness         │
                             │  • Honest Currency Metrics (₹ INR)   │
                             │  • Count & Amount-Weighted Win Rates  │
                             │  • Automated JSON / Markdown Reports  │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │  Streamlit Full Pipeline Demo App     │
                             │  • 6-Tab Interactive Dashboard        │
                             │  • Side-by-Side Diagnosis & Narrative │
                             │  • Interactive Business ROI Calculator│
                             └───────────────────────────────────────┘
```

---

## 🚀 Key Novelties & Gaps Closed

| Open Industry Gap | Standard Industry Limitation | AI Risk Manager Solution | Measured Result |
|---|---|---|---|
| **Gap 1: Reproducible Benchmark** | Unaudited vendor win rates on private data; single misleading accuracy metric. | Public, reproducible 50k benchmark with stratified train/test splits & dual metrics. | Empirical dispute rate: `0.406%`; dual count vs. amount tracking. |
| **Gap 2: Explainable Rubrics** | Black-box win probability float (`0.74`). | Decomposed scoring across 6 CE3.0 criteria with per-criterion confidence intervals. | PR-AUC: `0.9978`, ROC-AUC: `0.9969`, P/R/F1: `1.000` across all criteria. |
| **Gap 3: Grounded Auditability** | LLM hallucinations and unverified claims in dispute letters. | Template & LLM engine requiring strict `[criterion: field=value]` citations + guardrails. | **100.00%** machine citation resolution; **100.00%** hallucination catch rate. |
| **Gap 4: Cold-Start Small Merchant** | Enterprise models require 10,000+ historical merchant disputes. | Mode B TabPFN few-shot tabular engine for thin-history merchants. | F1 = `1.0000`, Brier = `0.0008` with only 15 prior disputes (matching full GBDT). |
| **Extension 5: Calibrated Escalation** | Overconfident guesses on ambiguous boundary disputes. | Split conformal prediction layer with guaranteed error rate ($\alpha = 0.05$). | Safe 3-way routing: automatically flags coin-flip disputes for human spot-check. |
| **Extension 6: Rule-Version Drift** | Undetected loss drops when Visa/Mastercard update rules. | Versioned YAML rule configs (`2023`, `2025-10`, `2026-04`) + TVD drift detector. | Quantified silent failure drop (-₹30,528 net loss); real-time alert trigger. |

---

## 💰 Measured Business ROI & Impact

Evaluated on a mid-size D2C merchant profile (60,000 monthly transactions, ~244 disputes/month, ₹2,952.40 avg dispute value):

```
Net Monthly Benefit = (Recovered_Improved − Recovered_Baseline) 
                      + Fees_Avoided (Unwinnable cases declined) 
                      + Labor_Time_Saved
```

- **Win-Rate Lift Benefit:** **+₹1,88,969.61 / month** (from 35.0% baseline to **61.70%** measured win rate)
- **Fight-and-Lose Fees Avoided:** **+₹3,821.18 / month** (19 unwinnable disputes declined × ₹200 fee)
- **Analyst Labor Time Saved:** **+₹48,720.00 / month** (35 min → 5 min spot-check @ ₹400/hr, 121.8 hrs saved)
- **Total Net Monthly Merchant Benefit:** **₹2,41,510.79 / month** (+119.2% net recovery lift)
- **Annualized Merchant Bottom-Line Benefit:** **₹28,98,129.45 / year (~₹28.98 Lakhs / year)**

*See the full detailed derivation and step-by-step arithmetic in [`docs/FINAL_RESULTS.md`](docs/FINAL_RESULTS.md).*

---

## 📦 Quick Start & How to Run

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/ShivanshRastogi315/chargeback-risk-manager.git
cd chargeback-risk-manager

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Test Suite
Verify that all 32 unit and integration tests across all modules pass:
```bash
python -m pytest
```

### 3. Run the End-to-End Evaluation Harness (Module 4)
Execute the complete pipeline evaluation on 50,000 synthetic transactions:
```bash
python eval_harness/run_evaluation.py --n_transactions 50000 --seed 42
```
Outputs:
- `evaluation_results.json`: Comprehensive machine-readable metrics.
- `EVALUATION_REPORT.md`: Formatted executive evaluation summary.

### 4. Run the Rule-Version Drift Simulation (Module 6)
Demonstrate the silent failure phenomenon across CE3.0 rule versions:
```bash
python drift/run_drift_simulation.py
```

### 5. Launch the Streamlit Interactive Demo UI
Launch the interactive 6-tab dashboard:
```bash
streamlit run app/main.py
```

---

## 🖥️ Demo Dashboard Features

The Streamlit demo interface (`app/main.py`) provides:
1. **Pipeline Diagnosis & Narrative (Tab 1):** Select preset dispute cases (Clear Win, Unwinnable, Thin-History Cold-Start, Ambiguous Edge-Case) to inspect decomposed rubric scores, ROI decision trees, and verifiable cited evidence packets side-by-side.
2. **Benchmark Explorer (Tab 2):** Explore class imbalance distributions, dual count-based vs. amount-weighted win rates, and transaction feature distributions.
3. **Rule Drift Simulator (Tab 3):** Compare performance of models trained on CE3.0 2023 rules evaluated on 2025-10 and 2026-04 rules, demonstrating the silent failure drop and Total Variation Distance drift alerts.
4. **Conformal Escalation (Tab 4):** Live demonstration of the **scripted failure beat** — showcasing how ambiguous coin-flip cases trigger conformal abstention and route to human review.
5. **Business ROI Calculator (Tab 5):** Interactive sliders for transaction volume, dispute rate, hourly analyst cost, and loss fees to compute live merchant ROI projections.
6. **Deck Summary & Results (Tab 6):** Executive summary of all measured metrics for pitch deck preparation.

---

## 🛡️ Defense-Only Boundary & Compliance Statement

This repository adheres strictly to a **defense-only** ethical boundary:
- **No Evidence Fabrication:** The narrative generator and rubric scorer only summarize and cite real facts present in verified transaction records.
- **Zero Offense Capability:** No tools or prompts are provided to construct fraudulent chargebacks or facilitate first-party fraud.
- **Honest Metric Reporting:** All evaluations report count-based win rates, amount-weighted win rates, and honest false-positive currency costs (₹ INR).

---

## 📂 Project Structure

```
chargeback-risk-manager/
├── app/                      # Streamlit interactive demo dashboard
│   ├── main.py               # Main multi-tab Streamlit application
│   ├── cases.py              # Synthetic case library (Clear Win, Thin History, Ambiguous)
│   ├── pipeline_runner.py    # End-to-end inference execution pipeline
│   ├── styles.py             # Premium CSS styles, cards, and badges
│   └── test_app.py           # Streamlit app automated unit tests
├── benchmark/                # Module 0: Synthetic Benchmark
│   ├── generate.py           # 50k transaction generator with CE3.0 ground truth
│   ├── metrics.py            # Dual win rate and error cost calculators
│   ├── schema.md             # Field schema and label generation documentation
│   └── test_benchmark.py     # Benchmark test suite
├── rubric_scorer/            # Module 1: Decomposed Rubric Scorer
│   ├── criteria.py           # Per-criterion scoring functions (6 CE3.0 criteria)
│   ├── aggregator.py         # Confidence aggregator preserving decomposed breakdown
│   ├── evaluation.py         # Per-criterion precision, recall, and PR-AUC eval
│   └── test_rubric_scorer.py # Rubric scorer test suite
├── roi_engine/               # Module 2: Fight/No-Fight ROI Engine
│   ├── models.py             # Mode A (GBDT) and Mode B (TabPFN Few-Shot) models
│   ├── engine.py             # Expected value decision engine
│   ├── evaluation.py         # Cold-start thin-history evaluation runner
│   └── test_roi_engine.py    # ROI engine test suite
├── narrative_gen/            # Module 3: Grounded Narrative Generator
│   ├── generator.py          # Evidence narrative builder with structured headers
│   ├── citation_validator.py # Machine regex validator verifying [criterion: field=val]
│   ├── guardrails.py         # Zero-PII and prohibited terminology filter
│   ├── templates.py          # Visa 10.4 representment packet templates
│   └── test_narrative_gen.py # Narrative generator test suite
├── eval_harness/             # Module 4: Full Pipeline Evaluation Harness
│   ├── costs.py              # ₹ INR currency cost model & ROI projection
│   ├── pipeline_evaluator.py # Multi-module end-to-end evaluation runner
│   ├── run_evaluation.py     # CLI execution script generating reports
│   └── test_eval_harness.py  # Harness test suite
├── escalation/               # Module 5: Calibrated Conformal Escalation
│   ├── conformal.py          # Split conformal prediction engine
│   ├── engine.py             # 3-way routing engine (Fight / No-Fight / Escalate)
│   ├── demo_case.py          # Scripted ambiguous edge-case failure demo generator
│   └── test_escalation.py    # Escalation test suite
├── drift/                    # Module 6: Rule-Version Drift Simulator
│   ├── rules_loader.py       # YAML configuration loader
│   ├── detector.py           # Total Variation Distance (TVD) drift detector
│   ├── simulator.py          # Multi-year evaluation comparator
│   ├── run_drift_simulation.py # CLI simulation runner
│   └── test_drift.py         # Drift simulator test suite
├── rules/                    # Versioned CE3.0 rule configurations
│   ├── ce3_2023.yaml         # Original CE3.0 baseline rules
│   ├── ce3_2025_10.yaml      # October 2025 revised rules
│   └── ce3_2026_04.yaml      # April 2026 stricter rules
├── docs/                     # Architecture, specifications, and reports
│   ├── 00_START_HERE.md      # Ground rules and documentation pointer table
│   ├── FINAL_RESULTS.md      # Measured metrics & recomputed Section 5 cost model
│   ├── PROGRESS.md           # Master phase completion checklist
│   ├── PROJECT_SPEC.md       # Architecture specs and acceptance criteria
│   ├── EXECUTION_PLAYBOOK.md # Business case formulas, tech stack & demo pitch
│   └── DECISIONS_LOG.md      # Log of technical decisions and fixes
├── requirements.txt          # Python dependencies
└── README.md                 # Master repository guide
```

---

## 📜 License
MIT License. Built strictly for defensive, legitimate merchant representment and chargeback risk management.