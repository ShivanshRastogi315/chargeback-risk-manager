# AI Risk Manager — End-to-End Pipeline Evaluation Report
**Evaluation Harness (Module 4) — Benchmark, Rubric, ROI, Narrative & Financial Verification**

---

## 1. Dataset & Split Summary (Module 0 Synthetic Benchmark)
- **Total Simulated Transactions:** 50,000
- **Total Chargeback Disputes:** 203 (Dispute Rate: **0.406%**)
- **Stratified Split:** Train = 152 disputes | Held-Out Test = 51 disputes

---

## 2. Rubric Scorer Decomposed Performance (Module 1)
**Overall Rubric Scorer PR-AUC:** `0.9978` | **ROC-AUC:** `0.9969`

| Evidentiary Criterion | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Test Support |
|---|---|---|---|---|---|---|
| `prior_undisputed_in_window` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 33 |
| `device_match` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 30 |
| `shipping_match` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 34 |
| `ip_match` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 22 |
| `cvv_avs_verified` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 50 |
| `otp_3ds_authenticated` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 35 |

---

## 3. Fight/No-Fight ROI Engine & Cold-Start Comparison (Module 2)
- **Full Test Set PR-AUC:** `0.9978` | **Contested Ratio:** `92.2%`

### Thin-History Merchant Slice Evaluation (Cold-Start Gap Analysis)
| Modeling Approach | Strategy / Data Regime | PR-AUC | ROC-AUC | F1 Score | Brier Score | Net Realized PnL (₹) |
|---|---|---|---|---|---|---|
| **Mode A (GBDT - Thin History Only)** | `MODE_A_GBDT_THIN` | 1.0000 | 1.0000 | 0.8571 | 0.0482 | **₹6,962.80** |
| **Mode B (TabPFN - Few-Shot Cold Start)** | `MODE_B_TABPFN_FEWSHOT` | 1.0000 | 1.0000 | 1.0000 | 0.0008 | **₹7,002.80** |
| **Mode A (GBDT - Full Cross-Merchant History)** | `MODE_A_GBDT_FULL` | 1.0000 | 1.0000 | 1.0000 | 0.0001 | **₹7,002.80** |

---

## 4. Grounded Narrative Generator & Auditability Verification (Module 3)
- **Machine Citation Resolution Rate:** **`100.00%`** (100% verifiable field grounding)
- **Sentence Grounding Rate:** **`100.00%`** (all claim sentences cited)
- **Compliance Guardrails Pass Rate:** **`100.00%`** (0 PII leaks, 0 prohibited terms)
- **Deliberate Hallucination / Corruption Catch Rate:** **`100.00%`** (100% of injected errors rejected)

---

## 5. Representment Outcomes & Honest Currency Error Metrics (Module 4)

### A. Dual Win-Rate Metrics
- **Count-Based Win Rate (Contested Cases):** **`0.62%`** (29 won / 47 contested)
- **Amount-Weighted Win Rate (Contested Cases):** **`0.57%`** (₹89,813.98 won / ₹93,413.98 disputed)

### B. Explicit Currency Error Costs (₹ INR)
| Error Type / Financial Outcome | Case Count | Monetary Impact (₹ INR) | Economic Meaning |
|---|---|---|---|
| **True Positives (TP)** | 29 | **+₹89,088.98** | Contested & Won (Net Recovered Revenue) |
| **False Positives (FP) Cost** | 18 | **-₹4,050.00** | Contested & Lost (Wasted Review + Loss Fee Penalty) |
| **False Negatives (FN) Cost** | 0 | **-₹0.00** | Erroneously Declined Winnable (Forfeited Revenue) |
| **True Negatives (TN) Fees Saved** | 4 | **+₹800.00** | Unwinnable Disputes Correctly Declined (Loss Fees Avoided) |
| **Total Classification Error Cost** | 18 | **₹4,050.00** | Total Monetary Loss from FP + FN Mistakes |
| **Net Realized Portfolio PnL** | — | **₹85,038.98** | AI Risk Manager Net Realized Recovery |
| **Baseline Net PnL (Contest All)** | — | **₹84,138.98** | Naive Contest-All Strategy |
| **Net Monetary Lift over Baseline** | — | **+₹900.00** | Direct Merchant Bottom-Line Value Added |

---

## 6. Business ROI Projection (Section 5 Standard Mid-Size Merchant Model)
- **Profile:** 60,000 monthly tx @ 0.41% dispute rate (~244 disputes/mo)
- **Dispute Recovery Value:** Baseline: ₹202,794.41 → AI Risk Manager: **₹391,764.02**
- **Win Rate Lift Value:** +₹188,969.61/month
- **Unwinnable Dispute Fees Avoided:** +₹3,821.18/month
- **Analyst Labor Time Saved (35m → 5m):** +₹48,720.00/month
- **Net Monthly Bottom-Line Benefit:** **₹241,510.79/month**
- **Net Annual Merchant Savings:** **₹2,898,129.45/year** (**₹28.98 Lakhs/year**)

---