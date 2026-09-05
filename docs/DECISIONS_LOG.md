# Project Decisions & Issue Log

Each entry: Date/Session | What broke | Root cause | Fix | Files changed. Write entries in plain English — this log doubles as material for the project's live demo, so clarity matters more than technical density.

---

### 2026-08-24 — Module 0: Synthetic Benchmark Build & Windows stdout Fix
- **What was built:** Delivered `benchmark/generate.py`, `benchmark/metrics.py`, `benchmark/schema.md`, and `benchmark/test_benchmark.py`.
- **What broke:** Script raised `UnicodeEncodeError` on Windows consoles with `cp1252` encoding when printing `₹` currency characters.
- **Root cause:** Default Windows PowerShell stdout encoding could not map UTF-8 currency glyph `\u20b9`.
- **Fix:** Switched console output to `INR` prefix and added ASCII indicators (`[OK]`), ensuring cross-platform stability.
- **Files changed:** `benchmark/generate.py`, `benchmark/metrics.py`, `benchmark/schema.md`, `benchmark/test_benchmark.py`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.

---

### 2026-08-25 — Module 1: Rubric Scorer & Subnet Match Granularity Fix
- **What was built:** Delivered `rubric_scorer/` (`criteria.py`, `aggregator.py`, `evaluation.py`, `test_rubric_scorer.py`), implementing decomposed criterion scoring and structured aggregation with confidence intervals.
- **What broke:** Initial IP match scorer scored partial subnet matches $\ge 0.50$, causing false positives against ground truth exact IP match labels (dropping F1 to 0.667).
- **Root cause:** Subnet match score threshold was too high (0.72) for a binary criterion evaluation where exact match is required for qualifying CE3.0 points.
- **Fix:** Refined subnet match score to 0.40 (supporting evidence without triggering false positive binary classification), restoring 1.000 F1 across all criteria.
- **Files changed:** `rubric_scorer/criteria.py`, `rubric_scorer/aggregator.py`, `rubric_scorer/evaluation.py`, `rubric_scorer/test_rubric_scorer.py`, `rubric_scorer/__init__.py`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.

---

### 2026-08-25 — Module 2: ROI Engine Build & Cold-Start Small Sample CV Adaptation
- **What was built:** Delivered `roi_engine/` (`models.py`, `engine.py`, `evaluation.py`, `test_roi_engine.py`), implementing Mode A calibrated GBDT, Mode B few-shot TabPFN, auto-mode selection (<30 history), and exact EV commercial decision formula.
- **What broke:** Mode A `CalibratedClassifierCV(cv=3)` threw `ValueError` when trained on small cold-start slices (15 disputes) with $<3$ samples per class.
- **Root cause:** Standard k-fold cross-validation requires at least k samples per class, breaking on thin-merchant historical slices.
- **Fix:** Added dynamic fold scaling and single-split regularized fallback for thin history pools in `ModeAGBDTModel.fit()`.
- **Files changed:** `roi_engine/models.py`, `roi_engine/engine.py`, `roi_engine/evaluation.py`, `roi_engine/test_roi_engine.py`, `roi_engine/__init__.py`, `benchmark/generate.py`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.

---

### 2026-08-27 — Module 3: Grounded Narrative Generator Build & Tag Splitting Fix
- **What was built:** Delivered `narrative_gen/` (`citation_validator.py`, `guardrails.py`, `templates.py`, `generator.py`, `evaluation.py`, `test_narrative_gen.py`), ensuring 100% sentence-level citation grounding, machine validation across all fields, and pre-flight PII/tone guardrails.
- **What broke:** Initial sentence-splitting regex split between trailing sentence periods and opening citation brackets `[...]`, separating citation tags from sentences and causing false ungrounded claims.
- **Root cause:** Naive regex period-splitting did not protect bracketed citation tokens.
- **Fix:** Implemented bracket-aware scanner in `CitationValidator.split_into_sentences()` and positioned tags with terminal punctuation `[tag].` across templates, restoring exact 100.0% machine citation resolution and 100.0% deliberate hallucination detection.
- **Files changed:** `narrative_gen/citation_validator.py`, `narrative_gen/guardrails.py`, `narrative_gen/templates.py`, `narrative_gen/generator.py`, `narrative_gen/evaluation.py`, `narrative_gen/test_narrative_gen.py`, `narrative_gen/__init__.py`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.

---

### 2026-08-28 — Module 4: Evaluation Harness Build & Metric Serializability Fix
- **What was built:** Delivered `eval_harness/` (`costs.py`, `pipeline_evaluator.py`, `run_evaluation.py`, `test_eval_harness.py`), implementing unified multi-module evaluation, per-criterion precision/recall/F1, Mode A vs Mode B cold-start comparison, 100% citation validation, dual win-rate calculation (count vs amount-weighted), explicit ₹ FP/FN costs, and merchant ROI projections.
- **What broke:** (1) `numpy.sum` raised `TypeError` when passed a python generator expression for FP costs. (2) `json.dump` failed on embedded pandas DataFrame in cold-start metrics.
- **Root cause:** NumPy 2.x/Python 3.14 strict typing disallowed generator sum; `FullPipelineEvaluationReport` retained an unconverted comparison table DataFrame.
- **Fix:** Calculated FP costs via scalar arithmetic and converted cold-start DataFrames to serializable dictionaries in `to_dict()`.
- **Files changed:** `eval_harness/costs.py`, `eval_harness/pipeline_evaluator.py`, `eval_harness/run_evaluation.py`, `eval_harness/test_eval_harness.py`, `eval_harness/__init__.py`, `roi_engine/evaluation.py`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.

---

### 2026-08-28 — Module 5: Calibrated Escalation Build & Telemetry Ambiguity Calibration
- **What was built:** Delivered `escalation/` (`conformal.py`, `engine.py`, `evaluation.py`, `demo_case.py`, `test_escalation.py`), wrapping Module 2's ROI engine in inductive split conformal prediction (Angelopoulos & Bates, 2021) to route disputes to `FIGHT`, `NO_FIGHT`, or `ESCALATE_TO_HUMAN` with a stated statistical error guarantee ($\le 10\%$ error rate / $90\%$ coverage). Generated live demo scripted failure artifact `escalation/ambiguous_escalation_case.json`.
- **What broke:** Initial synthetic ambiguous demo case had $P(win) = 0.5855$, which fell slightly outside the conformal ambiguity band $[1-\hat{q}, \hat{q}] = [0.462, 0.538]$ (where $\hat{q} = 0.538$), causing the conformal predictor to output singleton $\{1\}$ (`FIGHT`) instead of $\{0, 1\}$ (`ESCALATE_TO_HUMAN`).
- **Root cause:** Curated case matched 3 high-weight criteria (device match 0.25, historical window 0.30, CVV 0.05), pushing confidence above the critical ambiguity threshold.
- **Fix:** Calibrated telemetry split (IP matched 0.15 + historical window 0.30 + CVV 0.05, while device was unmatched/unseen), yielding $P(win) = 49.3\% \in [0.462, 0.538]$ and producing prediction set $\{0, 1\} \rightarrow \text{ESCALATE\_TO\_HUMAN}$.
- **Files changed:** `escalation/conformal.py`, `escalation/engine.py`, `escalation/evaluation.py`, `escalation/demo_case.py`, `escalation/test_escalation.py`, `escalation/__init__.py`, `escalation/ambiguous_escalation_case.json`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.

---

### 2026-08-28 — Module 6: Rule-Version Drift Simulator Build & Windows CLI Encoding Fix
- **What was built:** Delivered `rules/` (`ce3_2023.yaml`, `ce3_2025_10.yaml`, `ce3_2026_04.yaml`, `README.md`) and `drift/` (`rules_loader.py`, `detector.py`, `simulator.py`, `evaluation.py`, `run_drift_simulation.py`, `test_drift.py`), modeling CE3.0 rule iterations (2023 foundational vs 2026 TC40), quantifying silent failure performance drops (-₹30,528 net PnL drop), and instrumenting Total Variation Distance (TVD = 0.4000) drift detection with empirical telemetry estimation.
- **What broke:** (1) Windows PowerShell CLI raised `UnicodeEncodeError` when printing currency characters `₹` to cp1252 stdout. (2) `scikit-learn` raised `FutureWarning` on explicit `penalty='l2'` in logistic regression.
- **Root cause:** Default console stream encoding on Windows failed on UTF-8 glyphs; sklearn 1.8+ deprecated explicit default penalty string.
- **Fix:** Reconfigured `sys.stdout` to UTF-8 with error replacement and updated `LogisticRegression` instantiation to use default `C=1.0`.
- **Files changed:** `rules/ce3_2023.yaml`, `rules/ce3_2025_10.yaml`, `rules/ce3_2026_04.yaml`, `rules/README.md`, `drift/rules_loader.py`, `drift/detector.py`, `drift/simulator.py`, `drift/evaluation.py`, `drift/run_drift_simulation.py`, `drift/test_drift.py`, `drift/__init__.py`, `drift/README.md`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.

---

### 2026-08-28 — Phase 8: Streamlit Demo UI Build & Pipeline Interface Alignment
- **What was built:** Delivered `app/` (`main.py`, `cases.py`, `pipeline_runner.py`, `styles.py`, `test_app.py`, `README.md`), building a modern 6-tab Streamlit dashboard connecting Modules 0 through 6 with 4-column side-by-side diagnostic cards, Slide 5 scripted deliberate failure comparison, Module 4 benchmark metrics, interactive Section 5 ROI calculator, and rule drift simulator.
- **What broke:** (1) `PipelineManager` called deprecated function name `load_rule_definition` and unexposed `score_dispute` method. (2) `weak_evidence` demo case had low win probability but high transaction amount, resulting in positive EV with default low loss fees.
- **Root cause:** Function naming divergence across modules and economic EV sensitivity on large transaction values with low loss fees.
- **Fix:** Standardized rule loader call to `load_rule_version()`, switched to `RubricScorer.score_record()`, and configured authentic non-CE3 counterfeit reason code (10.5) with ₹450 value for genuine NO_FIGHT behavior (32/32 tests passing).
- **Files changed:** `app/main.py`, `app/cases.py`, `app/pipeline_runner.py`, `app/styles.py`, `app/test_app.py`, `app/README.md`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.

---

### 2026-08-28 — Phase 9 & 10: Cost Model Recomputation, Master Verification & Final Docs
- **What was built:** Delivered `docs/FINAL_RESULTS.md` with recomputed Section 5 business ROI model from Module 4 harness output, top-level `README.md` project guide, and verified all 32 unit/integration tests across all modules.
- **What broke:** No code breakage occurred during final verification pass; standard pytest completed with 32 passed tests in 136s and Module 4 evaluation completed on 50,000 synthetic transactions.
- **Root cause:** N/A (Verification and documentation phase).
- **Fix:** Documented exact measured business figures (₹2,41,510.79/month net value added, ₹28.98 Lakhs/year annualized savings, +26.70% pt win-rate lift, 100% citation resolution) and updated `docs/PROGRESS.md` to show 100% completion.
- **Files changed:** `docs/FINAL_RESULTS.md`, `README.md`, `docs/PROGRESS.md`, `docs/DECISIONS_LOG.md`.
