# AI Risk Manager — Chargeback Evidence Automation
### Build Spec (hand this to your coding agent, one phase at a time)

**One-line pitch:** a defense-only system that scores each chargeback's evidentiary
strength criterion-by-criterion (not one opaque number), decides whether it's worth
contesting, and generates an auditable evidence narrative where every claim is
traceable to a source field — evaluated against a documented, reproducible benchmark.

**Non-negotiable constraint:** defense-only. No functionality that fabricates evidence,
falsifies transaction records, or helps a bad actor construct a fraudulent chargeback
or a fraudulent representment. The rubric-scoring and narrative layers only ever
summarize/cite data that's actually present in the transaction record.

---

## Architecture

```
┌─────────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌────────────────────┐
│ Module 0        │   │ Module 1         │   │ Module 2           │   │ Module 3           │
│ Synthetic        │──▶│ Rubric Scorer    │──▶│ Fight/No-Fight     │──▶│ Grounded Narrative  │
│ Benchmark (Gap 1)│   │ (Gap 2)          │   │ ROI Engine (Gap 4) │   │ Generator (Gap 3)   │
└─────────────────┘   └──────────────────┘   └───────────────────┘   └────────────────────┘
        │                                                                        │
        └───────────────────────── Module 4: Evaluation Harness ◀───────────────┘
                                    (closes the loop back to Module 0's labels)
```

Data flows one direction; each module has a clean I/O contract so the agent can build
and test them independently before wiring them together.

---

## Module 0 — Synthetic Benchmark (Gap 1)

**Purpose:** the shared, documented dataset + eval protocol nobody in the space has
published. This is also what Modules 1–2 train/test against, so build it first.

**Deliverable:** `benchmark/` package with:
- `generate.py` — produces N synthetic transaction+dispute records. Each record has:
  - transaction fields (amount, timestamp, device ID, IP, shipping address, BIN/issuer, prior-transaction history for that cardholder)
  - a reason code (Visa 10.4 style, i.e. "cardholder claims non-recognition")
  - a **simulated issuer decision function** — a documented, rule-based function (not a black box) that decides win/loss based on how many CE3.0/First-Party-Trust-style criteria match (device match, IP match, shipping match, prior-transaction age 120–365 days, etc.), plus injected noise to represent real-world unpredictability
  - ground-truth **per-criterion** labels (which specific criteria matched) — this is what makes Gap 2 trainable/testable at all
- `schema.md` — documents every field, the label-generation logic, and the injected noise model, so it's defensible as "synthetic but realistic" in front of judges
- Stratified train/test split preserving extreme class imbalance (aim for <1% dispute rate, consistent with published industry figures)
- `metrics.py` — implements **both** count-based win rate and amount-weighted win rate (the review flagged that these disagree in real papers — report both, always)

**Acceptance criteria:** running `generate.py` produces a reproducible dataset with a fixed seed; `schema.md` fully explains the label logic; a naive baseline model (logistic regression on raw features) gets a sane, non-trivial PR-AUC on it.

---

## Module 1 — Rubric Scorer (Gap 2)

**Purpose:** replace the single opaque win-probability with a **decomposed, per-criterion
confidence score** — directly explainable by construction since CE3.0/First-Party-Trust
criteria are public.

**Deliverable:** `rubric_scorer/` package with:
- One scoring function/small model per criterion (device match confidence, IP match confidence, shipping match confidence, transaction-age-window confidence, etc.) — start with simple rule-based/statistical scorers, upgrade to learned scorers only if time permits
- An aggregator that combines per-criterion scores into an overall recommendation *while preserving and returning the breakdown* — the output object should always look like `{overall: 0.72, criteria: {device_match: 0.9, ip_match: 0.4, ...}}`, never just a single float
- Trained/evaluated against Module 0's per-criterion ground truth

**Acceptance criteria:** for any input record, the module returns both an overall score and a criterion-level breakdown; breakdown accuracy is evaluated separately per criterion (not just overall accuracy).

---

## Module 2 — Fight/No-Fight ROI Engine (Gap 4)

**Purpose:** decide whether contesting is worth it, with two modes so the small-merchant
gap is explicitly addressed rather than assumed away.

**Deliverable:** `roi_engine/` package with:
- **Mode A (merchant-history available):** standard classifier (e.g. gradient-boosted trees) trained on that merchant's own historical dispute outcomes, combined with Module 1's rubric score and the dispute's fee/amount economics (contest cost, fight-and-lose fee if applicable, dispute amount) to output expected value of contesting.
- **Mode B (cold-start / thin history):** few-shot approach using an open pretrained tabular model (e.g. TabPFN) so a merchant with very little history still gets a usable score — this is the part that directly differentiates from enterprise vendors who require large per-merchant history.
- A single interface that auto-selects mode based on how much merchant history is available.

**Acceptance criteria:** Mode B's output quality is evaluated specifically on a held-out slice of Module 0's benchmark simulating "thin-history" merchants (few prior disputes), and compared against Mode A on the same slice to demonstrate the cold-start gap being closed.

---

## Module 3 — Grounded Narrative Generator (Gap 3)

**Purpose:** generate the representment evidence text such that every claim is
traceable to a specific structured field — solving the auditability gap the review
flagged (no vendor publishes a faithfulness metric; this module makes faithfulness
checkable by construction).

**Deliverable:** `narrative_gen/` package with:
- Template- or LLM-based generation of the evidence narrative from Module 1's structured breakdown
- **Hard constraint:** every generated sentence carries a citation tag back to the specific field/criterion it's derived from (e.g. `[device_match: device_id=X matched prior order #Y]`) — no free-floating claims
- A lightweight guardrail pass (PII leak check, prohibited-language check, required-field-present check) before returning final output — a scaled-down version of the Best-of-N guardrail pattern from the review, doesn't need multiple candidates for a hackathon MVP, one pass with a validator is enough
- Output format matching how a real representment packet is structured (evidence category headers per reason code)

**Acceptance criteria:** an automated check confirms 100% of generated claims have a valid citation tag resolving to a real field in the input record (this *is* your auditability metric — report it as a number in your demo).

---

## Module 4 — Evaluation Harness

**Purpose:** close the loop back to Module 0 and produce the numbers you'll actually present to judges.

**Deliverable:** a single script/notebook that runs the full pipeline against the held-out test set and reports:
- Rubric scorer: per-criterion precision/recall + overall PR-AUC
- ROI engine: Mode A vs Mode B performance on thin-history slice
- Narrative generator: citation-validity rate (the auditability number)
- Both count-based and amount-weighted "win rate" if you simulate representment outcomes end-to-end

---

## Suggested build order for the agent (one phase per session/prompt)

1. **Phase 0:** Module 0 only. Get the benchmark generating and documented before touching anything else — everything downstream depends on it being right.
2. **Phase 1:** Module 1 against Module 0's test data.
3. **Phase 2:** Module 2, both modes, evaluated on the thin-history slice.
4. **Phase 3:** Module 3, with the citation-validity check as a hard test.
5. **Phase 4:** Module 4 + a thin demo UI (Streamlit is fastest for a hackathon) that runs one transaction through the whole pipeline and shows the breakdown, ROI decision, and cited narrative side by side.

Give the agent one phase per prompt with its acceptance criteria pasted in — this keeps
scope bounded and testable, which matters more for agent output quality than handing
over the whole spec at once.

## Suggested stack
Python, scikit-learn/XGBoost for tabular scoring, `tabpfn` package for Module 2 Mode B,
a small LLM call (or pure templating if you want to avoid API dependency) for Module 3,
Streamlit for the demo UI, pytest for the acceptance-criteria checks above.
