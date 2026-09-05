# Demo UI — Streamlit Application

Interactive multi-module demonstration web application connecting Modules 0 through 6 into an auditable, side-by-side defense interface.

---

## Quick Start

To launch the demo application:
```bash
streamlit run app/main.py
```

---

## Application Modules & Tab Structure

| Tab | Focus & Features | Modules Demonstrated |
|---|---|---|
| **⚡ 1. Live Pipeline Run** | End-to-end processing of a single transaction: intake telemetry, decomposed rubric breakdown, commercial EV decision, conformal routing, and grounded citation narrative. | Modules 0, 1, 2, 3, 5 |
| **⚠️ 2. Slide 5 Scripted Failure** | The deliberate ambiguity test case (`dsp_demo_ambiguous_007`): side-by-side comparison of Version 1 naive EV contest (What Broke) vs Module 5 Conformal Escalation to Human with 90% coverage guarantee (How We Got Out). | Module 5 |
| **📊 3. Module 4 Evaluation Harness** | Full benchmark offline evaluation results: per-criterion P/R/F1, Mode A vs Mode B cold-start comparison, honest ₹ FP/FN currency costs, and dual win rates. | Module 4 |
| **💰 4. Cost Model & Business ROI** | Interactive merchant ROI calculator recomputing monthly/annual net INR savings from win-rate lift, loss fees avoided, and labor time reduction. | Section 5 Business Model |
| **🔄 5. Module 6 Rule Drift Simulator** | Interactive concept drift experiment between Visa CE3.0 rule versions (`ce3_2023`, `ce3_2025_10`, `ce3_2026_04`) demonstrating silent financial loss and TVD detection. | Module 6 |
| **🛡️ 6. Defense-Only & Architecture** | Compliance declaration, architecture flowchart, and mapped literature gaps table. | System Governance |

---

## Testing

Run the test suite:
```bash
python -m pytest app/test_app.py -v
```
