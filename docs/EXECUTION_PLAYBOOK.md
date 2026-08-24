# AI Risk Manager — Chargeback Evidence Automation
## Full Execution Playbook: Build, Prove, Present, Win

This document sits on top of `PROJECT_SPEC.md` (your architecture) and
`chargeback_evidence_automation_review.md` (your literature review). It doesn't
repeat their content — it adds implementation depth, two stretch modules that
close gaps even your own review flags as unsolved, a hardware-mapped tech
stack, a cost-savings model you can put on a slide, and a presentation script
built specifically to answer "what broke, and how you got out."

---

## 1. The Winning Narrative — one paragraph you should be able to say from memory

*"Everyone in this space — Justt, Chargeflow, Kount — publishes a single win/loss
number on private data. We built the thing the field is missing: a public,
documented benchmark; a rubric scorer that explains **which specific criterion**
failed instead of one opaque probability; a narrative generator where every
sentence is cited back to a real field, checkable by machine; and a fight/no-fight
engine that works even for a merchant with almost no history. Then we broke it
on purpose, in front of you, and showed you exactly how the system catches its
own mistake instead of hiding it."*

Everything below exists to make that paragraph true and demonstrable.

### The judges' question, mapped gap-by-gap

| Gap (from your review) | What breaks in the field today | What breaks in *your first version* | How you get out |
|---|---|---|---|
| Gap 1 — no shared benchmark | Vendors report "80% win rate" on private, cherry-picked data | Your naive baseline (logistic regression) gets a misleadingly high accuracy but near-zero PR-AUC on the imbalanced split | Report PR-AUC + both count-based and amount-weighted win rate, always, on a documented held-out split |
| Gap 2 — opaque win/loss label | Issuers never tell merchants *which* criterion failed | Your rubric scorer initially collapses all criteria into one score (a natural first-pass mistake) | Refactor to per-criterion breakdown; show the before/after: one float → an inspectable object |
| Gap 3 — unauditable AI narratives | No vendor publishes a faithfulness metric | Your first LLM narrative pass hallucinates a claim not present in the input record | Add the citation-validator; show a caught hallucination getting rejected live |
| Gap 4 — small-merchant cold start | Enterprise vendors need 500+ data points/case; small merchants don't have them | Mode A (history-based) fails/degrades badly on your thin-history test slice | Mode B (TabPFN v2 few-shot) kicks in automatically; show the accuracy gap closing on that slice |
| §2.6 — rule churn as drift *(unaddressed in literature)* | CE3.0 changed 3 times in 3 years; nobody treats this as concept drift | A model tuned to 2023-era criteria silently underperforms on 2026 criteria | Module 6: rule-version-aware evaluation that *detects* this silent failure |
| §3.5 — no chargeback application of conformal abstention *(unaddressed)* | Escalation-to-human decisions use ad hoc confidence thresholds everywhere | Your ROI engine confidently recommends "fight" on a genuinely ambiguous case | Module 5: calibrated abstention routes it to human review instead, with a statistical guarantee |

The last two rows are your differentiators. Nobody else in the room will have them, because your own review shows nobody in the *published literature* has them either for this specific problem.

---

## 2. Build Guide: Modules 0–4 (implementation depth added to your spec)

Your spec already has clean I/O contracts and acceptance criteria — don't
duplicate that here. This section is the "how" underneath each "what."

### Module 0 — Synthetic Benchmark

- **Field generation:** use `faker` for realistic names/addresses/emails, but
  make the *fraud-relevant* fields (device ID reuse, IP geolocation drift,
  shipping-vs-billing mismatch, prior-transaction age) deliberately correlated
  with the label — a benchmark where features are independent of outcome is
  useless for demonstrating anything.
- **Simulated issuer decision function:** implement it as an explicit weighted
  rule table (not a black box) — e.g. device match = +0.3, IP match = +0.2,
  transaction-age-in-window = +0.25, shipping match = +0.25, then add Gaussian
  noise (σ tunable) before thresholding to win/loss. Document the exact weights
  in `schema.md` — a judge should be able to read your label logic in five
  minutes and *believe* it's a fair test, not a rigged one.
- **Class imbalance:** target ~0.3–0.5% positive (dispute) rate, consistent
  with the industry case study your review cites. Use `imbalanced-learn`'s
  `StratifiedKFold` for splits, never a random split — random splits on 0.3%
  imbalance can produce test folds with near-zero positives by chance.
- **Two label layers, not one:** (a) overall win/loss, (b) per-criterion match
  ground truth. Layer (b) is what makes Module 1 trainable/testable at all —
  build it first, treat it as the benchmark's actual novel contribution.

### Module 1 — Rubric Scorer

- Start every criterion scorer as a **simple, explainable rule or logistic
  regression** on 2–4 relevant features (e.g. device-match confidence =
  function of device-ID exact match + device-fingerprint similarity). Only
  upgrade a criterion to a learned gradient-boosted model if the rule-based
  version's per-criterion F1 is clearly weak in Module 4's harness — this
  keeps the system explainable-by-default and lets you *justify* every place
  you added ML complexity instead of using it everywhere reflexively.
- **Aggregator design:** don't just average per-criterion scores — weight them
  by how much CE3.0/First-Party-Trust rules actually weight them (some
  criteria are near-mandatory, others are supporting). Document these weights
  next to the ones in Module 0's issuer-decision function — if they're
  similar, that's a sanity check your rubric scorer is learning something
  real, not fitting noise.
- **Output contract:** always `{overall, criteria: {...}, confidence_interval}` —
  add a confidence interval per criterion (bootstrap over the training folds
  is enough) since Module 5 needs it.

### Module 2 — Fight/No-Fight ROI Engine

- **EV formula (put this exact formula on a slide):**
  `EV(contest) = P(win) × dispute_amount − P(lose) × fight_and_lose_fee − review_cost`
  Contest iff `EV(contest) > EV(no_contest) = 0`.
- **Mode A:** GBDT (XGBoost/LightGBM) trained on merchant's own dispute
  history + Module 1's rubric score as input features. Calibrate probabilities
  with Platt scaling or isotonic regression — an uncalibrated classifier's
  `predict_proba` is not a real probability, and your EV formula needs a real
  one to be honest.
- **Mode B (cold start):** `tabpfn` package, in-context / few-shot — feed it
  the merchant's thin history (even 5–20 prior cases) plus the cross-merchant
  synthetic benchmark as context. This is the literal differentiator your
  review names: an open pretrained tabular model doing what Mastercard's
  proprietary one does, but accessible to a small merchant.
- **Mode selection rule:** simple threshold on merchant history size (e.g.
  <30 prior disputes → Mode B), documented and tunable, not hidden.

### Module 3 — Grounded Narrative Generator

- **Prompt architecture:** don't let the LLM see raw free text and improvise —
  feed it *only* Module 1's structured criterion breakdown plus a fixed
  template of allowed sentence types per reason code. Constrain generation to
  "fill in this structured template" rather than "write freely," which
  dramatically reduces hallucination surface before the validator even runs.
- **Citation validator (your headline metric):** a plain Python regex/parser —
  extract every `[criterion: field=value]` tag, check the field+value actually
  exists in the input record, reject/flag any sentence whose tag doesn't
  resolve. Report this as a single number: *"100% of claims machine-verified
  against source fields"* — this is your answer to Gap 3 and it's genuinely
  simple to build and impossible for a judge to dispute, since it's checkable
  in front of them live.
- **Guardrail pass:** PII leak check (regex for card numbers, emails not
  already in the allowed output fields), prohibited-language check (simple
  keyword list), required-field-present check per reason code. One pass is
  enough for MVP — you don't need full best-of-N orchestration to demonstrate
  the *concept* the field is missing.

### Module 4 — Evaluation Harness

Report, every time, without exception:
- Rubric scorer: per-criterion precision/recall/F1 (not just overall)
- ROI engine: Mode A vs Mode B accuracy on the thin-history slice, side by side
- Narrative generator: citation-validity rate + guardrail pass/fail rate
- Both count-based *and* amount-weighted win rate (your review shows these
  can diverge on the same intervention — report both or a judge who's read
  the same papers you have will ask why you didn't)
- **False-positive cost, in currency, not just a rate** — this is explicitly
  what the track's own bar asks for. Convert every FP/FN into ₹ using the
  cost model in Section 4.

---

## 3. Stretch Modules — the two things that make this "unbeatable" rather than "solid"

These are optional relative to your original 4-gap spec, but they're the
highest-leverage additions available to you, because your own review names
them as genuinely open in the published literature. Closing an open gap you
found yourself, and can cite by arXiv number, is a different tier of
submission than closing a gap the whole industry is already racing on.

### Module 5 — Calibrated Escalation (closes §3.5)

**What it is:** instead of Module 2's ROI engine outputting a bare
fight/no-fight recommendation, wrap it in a **conformal prediction** layer
that outputs one of three states: *fight*, *don't fight*, or *escalate to
human review* — with a statistical guarantee (e.g. "at most 5% of
auto-decided cases are wrong, calibrated on held-out data") rather than an
arbitrary confidence threshold.

**How to build it:**
- Use `mapie` (Model Agnostic Prediction Interval Estimator) or hand-roll
  split conformal prediction: hold out a calibration set, compute
  nonconformity scores, derive a threshold that guarantees your target error
  rate.
- Wire it after Module 2: if the conformal prediction set contains both
  "fight" and "no-fight" as plausible labels at your target confidence level,
  route to escalate instead of picking one.
- **This is your live "graceful failure" demo moment** (see Section 6) — feed
  a genuinely ambiguous synthetic case, show the system decline to guess and
  route to human review instead, with the guarantee stated on screen.

**Why judges will find this credible:** conformal prediction is a real,
citable technique (Angelopoulos & Bates) already used in production ML at
scale elsewhere — you're not inventing statistics, you're being the first (per
your own literature review) to apply an existing rigorous technique to this
specific decision point.

### Module 6 — Rule-Version Drift Simulator (closes §2.6)

**What it is:** your review's most original finding — CE3.0 changed
materially three times in three years, and that's a form of concept drift
nobody treats as drift, because it's drift in the *evaluation rubric*, not in
underlying fraud behavior.

**How to build it:**
- Store Module 1's criterion weights as versioned config files:
  `rules/ce3_2023.yaml`, `rules/ce3_2025_10.yaml`, `rules/ce3_2026_04.yaml`
  (approximate the actual rule changes described in your review as best you
  can from public CE3.0 documentation — label clearly as "approximated for
  demonstration").
- Run Module 4's evaluation harness against a model trained/tuned on the
  2023 rule version, but *evaluated* against the 2026 rule version's criteria
  weights. Show the performance drop — this is your demonstrated "silent
  failure" that the literature doesn't currently name.
- Then show the fix: a simple drift *detector* (compare live criterion-weight
  distribution against the training-time rule version; alert when they
  diverge past a threshold) — you're not solving rule-churn robustness fully
  in a hackathon, you're being the first to **name and instrument** it, which
  is itself the contribution.

**How to present this without overclaiming:** be explicit that you are
demonstrating the *existence and mechanism* of this failure mode, not
claiming a production-grade drift-robust model — that honesty is exactly what
the track's bar rewards, and overclaiming here is the fastest way to lose
credibility with an ML-literate judge.

---

## 4. Tech Stack — mapped to your hardware

Your Intel Ultra 7 + RTX 5060 (8GB VRAM class) is genuinely enough for this
entire project run locally. Here's the split:

| Component | Tool | Runs on | Notes |
|---|---|---|---|
| Tabular scoring (Modules 1, 2 Mode A) | `xgboost`, `lightgbm`, `catboost`, `scikit-learn` | CPU (Ultra 7) | Trivial at your data scale — seconds per training run on 50–100K synthetic rows |
| Cold-start tabular (Module 2 Mode B) | `tabpfn` (TabPFN v2) | GPU (RTX 5060) or CPU fallback | 8GB VRAM is comfortably enough for TabPFN v2 inference; test CPU fallback too in case of driver issues on demo day |
| Synthetic benchmark generation (Module 0) | `faker`, `sdv` (Synthetic Data Vault) if you want CTGAN-style augmentation later | CPU | No GPU needed for the rule-based generator; `sdv`'s CTGAN benefits from GPU if you add it |
| Conformal calibration (Module 5) | `mapie` or hand-rolled split conformal | CPU | Pure statistics, negligible compute |
| Narrative generation (Module 3) | **Primary:** small local model via `Ollama` (Llama 3.1 8B-Instruct or Qwen2.5-7B-Instruct, Q4 quantized) — **Fallback/higher quality:** Anthropic or OpenAI API | GPU for local (fits in 8GB at Q4) / cloud for API | Run local for demo-day reliability (no wifi dependency); use API version during development for higher-quality baseline comparison |
| Citation validator / guardrails (Module 3) | Hand-written Python (regex + field-resolution check) | CPU | Deliberately not a black-box library — inspectable code is a *feature* here, judges can read it |
| Demo UI | `Streamlit` | CPU | Fastest path to a working, presentable UI; use `plotly`/`altair` for the PR-curve and criterion-breakdown visuals |
| Evaluation/experiment tracking | Plain JSON/CSV logs, or `Weights & Biases` free tier if you want a polished experiment dashboard | CPU | W&B is a nice-to-have signal of rigor, not required |
| Testing | `pytest` against every module's acceptance criteria | CPU | Already specified in your spec — keep it |
| Version control | Git/GitHub | — | **Deliberately structure your commit history as a narrative** — commit messages like `fix: rubric scorer collapsing per-criterion scores into one float` or `feat: add conformal abstention after ambiguous-case failure` turn your git log into literal, checkable evidence of "what broke, how you got out" |

**When to reach for cloud:** only if you want to explore something explicitly
out of local-hardware range — e.g. a small-scale nod to the transaction
foundation model direction (§3.3 of your review) as a "future work" appendix
experiment, not a demo-day dependency. Google Colab (free T4) or Kaggle (free
P100/T4, 30 hrs/week) are enough for that; keep it out of your live demo path
entirely so a bad wifi connection on stage can't sink you.

---

## 5. The Cost-Savings / ROI Model (your business-case slide)

This is the section that turns "we built a model" into "we save merchants
money" — and it's a chance to directly contrast your rigor against the
field's own weakness (Gap 1): vendors market unaudited numbers, you show your
work.

**General formula:**

```
Monthly disputes           = transaction_volume × dispute_rate
Revenue at stake            = disputes × avg_disputed_value
Recovered (baseline)        = disputes × contest_rate × win_rate_baseline × avg_disputed_value
Recovered (with system)     = disputes × contest_rate × win_rate_improved × avg_disputed_value
Fight-and-lose fees avoided = (cases correctly redirected to "no-fight") × fee_per_lost_case
Analyst time saved          = disputes × (manual_minutes − automated_minutes) / 60 × loaded_hourly_cost

Net monthly benefit = (Recovered_improved − Recovered_baseline)
                       + Fees_avoided
                       + Time_saved_value
```

**Illustrative worked example** *(label this explicitly as illustrative —
replace with your Module 4 harness's real numbers before the final deck)*:

| Assumption | Value | Source/justification |
|---|---|---|
| Monthly transactions | 60,000 | mid-size D2C merchant, plausible Razorpay customer profile |
| Dispute rate | 0.4% | consistent with the 0.31% figure your review cites from the 2026 industry case study |
| Avg disputed transaction value | ₹2,200 | disputed orders skew slightly higher-value |
| Contest rate | 70% of disputes | conservative — some go uncontested due to low value or missed deadlines today |
| Baseline win rate | 35% | conservative independent estimate — **not** vendor-marketed "80%" figures, which your review shows are unaudited |
| Improved win rate | 43% (+8pt) | conservative, within the ~11pt count-based lift your review cites from the Best-of-N paper — deliberately understated |
| Fight-and-lose fee | ₹200/case | illustrative PSP administrative fee |
| Manual case-handling time | 35 min → 5 min | evidence compilation automated, human does a spot-check review |
| Loaded analyst cost | ₹400/hour | illustrative |

**Resulting monthly numbers:**
- Disputes/month: 240 → contested: 168
- Baseline recovery: 168 × 0.35 × ₹2,200 ≈ **₹1,29,800**, minus fight-and-lose fees on 109 losses (₹21,800) → net **₹1,08,000**
- With system: 168 × 0.43 × ₹2,200 ≈ **₹1,58,400**, minus fees on 96 losses (₹19,200) → net **₹1,39,200**
- Win-rate-lift benefit: **≈ ₹31,200/month**
- Fee-avoidance benefit (ROI engine correctly declining ~20 unwinnable cases): **≈ ₹4,000/month**
- Labor-time benefit: 240 disputes × 30 min saved ÷ 60 × ₹400 ≈ **≈ ₹48,000/month**
- **Total illustrative benefit: ≈ ₹83,000/month (~₹10 lakh/year) for one mid-size merchant**

**How to present this:** show the formula first (so it reads as a real model,
not a made-up number), then this worked example labeled "illustrative,"
then — critically — **recompute the win-rate-lift row using your actual
Module 4 harness output** before the final deck, so the number you present
live is your own measured number, not a literature-borrowed estimate. That
swap is the difference between "we estimated" and "we measured," and judges
notice which one you're doing.

---

## 6. Presentation & Demo Script

### Deck structure (aim for ~10 slides, tight)

1. **The problem, in numbers** — chargeback volume/value growth, first-party
   fraud now the leading category (from your earlier research brief)
2. **Why existing solutions don't fully close it** — the gap table from
   Section 1 of this doc, visually
3. **Architecture** — the module diagram from your spec, now showing 6 boxes
   (add Modules 5 & 6 visually distinct as "extensions")
4. **Live demo, part 1 — the happy path** — one transaction through the full
   pipeline: rubric breakdown → ROI decision → cited narrative, end to end
5. **Live demo, part 2 — the deliberate failure** *(this is your answer to
   the judging question — script this precisely, see below)*
6. **Results table** — Module 4's numbers: per-criterion P/R, both win-rate
   metrics, citation-validity rate, Mode A vs B on thin-history slice
7. **Cost-savings model** — the formula + your real (not illustrative)
   recomputed number
8. **Defense-only compliance statement** — explicit, one slide, no ambiguity
9. **Limitations & future work** — stated out loud, not hidden (Section 8)
10. **Close** — the one-paragraph narrative from Section 1, verbatim if you can

### The scripted failure moment (slide 5) — write this out and rehearse it

1. Feed the system a **genuinely ambiguous** synthetic case — one where the
   rubric criteria are split (e.g. device matches, but IP doesn't, and the
   transaction sits right at the edge of the 120–365 day window).
2. Show Module 2's raw EV calculation landing close to zero — a coin-flip case.
3. **Without Module 5**, show what the naive system would have done: picked a
   side confidently, with no signal that it was uncertain. Narrate: *"This is
   what broke in our first version — a bare recommendation with no sense of
   its own uncertainty."*
4. **With Module 5 active**, show the same case now routed to "escalate to
   human review" with the calibration guarantee displayed on screen.
5. Close the moment with one sentence: *"That's how we got out — not by being
   right every time, but by knowing when we don't know."*

This is a genuinely strong demo beat because it's *live*, it's *specific*, and
it directly answers the exact question you told me matters most.

---

## 7. Judging Criteria Alignment Checklist

Track 02's stated bar: *"Honest metrics including false-positive cost. Strictly
defense-only: anything offense-capable is disqualified."*

- [ ] Held-out test set with realistic (not rebalanced-away) class imbalance
- [ ] Multiple metrics reported, not one accuracy number (PR-AUC, per-criterion P/R, both win-rate definitions)
- [ ] False positives and false negatives both converted to ₹ cost, explicitly
- [ ] A documented, reproducible benchmark (Module 0) — not "trust us"
- [ ] A stated, visible defense-only boundary (Module 3's narrative generator only ever cites data that's actually present — never fabricates or falsifies)
- [ ] One live, scripted failure shown and explained (Section 6, slide 5)
- [ ] Explicit limitations section — nothing overclaimed
- [ ] Every generated claim in the evidence narrative machine-verifiable against source fields (your citation-validity %)

---

## 8. Anticipated Judge Questions (prep answers now, not on stage)

**"Isn't this just re-implementing what Justt/Chargeflow already do?"**
No — they output a single win/loss recommendation on private data; you output
a per-criterion, auditable breakdown on a public, documented benchmark, plus
two capabilities (calibrated escalation, rule-drift detection) that don't
exist in the reviewed literature at all for this problem.

**"Your synthetic data isn't real — how do you know this generalizes?"**
Be honest: it doesn't guarantee production performance, and you should say so.
What it does prove is that your evaluation *protocol* — the metrics, the
splits, the honesty about count-based vs. amount-weighted disagreement — is
sound and reusable the moment real data is available. That protocol, not the
specific numbers, is Gap 1's actual contribution.

**"Why not just use a bigger/newer model for everything?"**
Because explainability is a requirement here, not a nice-to-have — a bigger
black-box model would make Gap 2 and Gap 3 (the ones with the clearest
regulatory/trust stakes) *harder* to solve, not easier. You chose
interpretable-by-construction where it mattered and reserved learned models
for narrow, checkable roles.

**"Isn't the conformal/drift stuff overkill for a hackathon?"**
It's scoped deliberately small — Module 5 is a calibration wrapper, Module 6
is versioned config files plus a comparison run. Neither required new
infrastructure; both directly close gaps your own literature review names as
open. Small implementation, real contribution.

---

## 9. Explicit Limitations to State Out Loud

State these proactively, don't wait to be asked — it's the single fastest way
to build credibility with an ML-literate judge, and it mirrors exactly the
critique your own review levels at the commercial field (nobody publishes
their weaknesses):

- Synthetic benchmark, not production issuer data — labeled as such throughout
- Rule-version files for Module 6 are approximated from public CE3.0
  documentation, not verified against actual issuer implementation
- Conformal guarantees (Module 5) hold under the calibration set's
  distribution — a real deployment would need ongoing recalibration
- Local LLM narrative quality (if used for the live demo) is a step below
  what a frontier API model would produce — note this explicitly if you
  demo the local model for reliability reasons

---

## 10. Suggested Build Order (extends your spec's phases)

1. Module 0 (benchmark) — everything depends on this being right
2. Module 1 (rubric scorer) against Module 0
3. Module 2 (ROI engine, both modes) on the thin-history slice
4. Module 3 (narrative generator) with citation-validator as a hard test
5. Module 4 (evaluation harness) — get real numbers flowing early, not last
6. **Module 5 (conformal escalation)** — wire in once Module 2 is stable
7. **Module 6 (rule-drift simulator)** — build once Module 1's rule config is finalized
8. Streamlit demo UI wiring all six modules together
9. Recompute Section 5's cost model with real Module 4 numbers
10. Deck + script the failure moment (Section 6) + full rehearsal, at least twice, with someone playing skeptical judge
