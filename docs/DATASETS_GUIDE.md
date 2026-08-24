# Datasets Guide — Track 02: Chargeback Evidence Automation

## The honest starting point

No public dataset contains real chargeback **outcomes** (win/loss, or which
evidentiary criterion succeeded) — that's literally Gap 1 from your own
review: issuer-side decision data is structurally siloed away from merchants,
and no consortium has pooled it. So there is nothing to "download and train
on" for the actual problem you're solving. Your Module 0 synthetic generator
*is* the dataset.

What real datasets *are* useful for is **calibration and credibility** — pulling
real, published statistics (imbalance ratios, amount distributions, feature
cardinalities) into your synthetic generator so that when a judge asks "how
do you know your synthetic data is realistic," you can point to specific
external numbers instead of "we guessed." Use everything below as *reference
inputs to Module 0*, not as training data for the actual dispute-outcome task.

---

## A. Fraud/transaction datasets — to calibrate Module 0's realism

| Dataset | What it has | What it's missing for you | How to actually use it | Link |
|---|---|---|---|---|
| **IEEE-CIS Fraud Detection** (Kaggle, Vesta Corp) | ~590K real e-commerce transactions, 400+ features incl. device type, browser, email domain, card info | No dispute/chargeback outcome labels, only fraud/not-fraud | Richest source for calibrating your **device-match / IP-match / browser-fingerprint** feature realism — check real correlation strength between device signals and fraud label, use similar magnitude in your simulated issuer-decision function | kaggle.com/c/ieee-fraud-detection |
| **Credit Card Fraud Detection (ULB)** | 284,807 European transactions, 492 fraud (0.172%), PCA-anonymized features | Anonymized (V1–V28), no raw fields, no dispute labels | Use only for **imbalance-ratio calibration** — cite this as one of three independent real-world sources justifying your <1% dispute rate choice in Module 0's `schema.md` | kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **PaySim** (synthetic, Kaggle) | 6.36M synthetic mobile-money transactions, 0.13% fraud rate, transaction types (CASH-IN/OUT, TRANSFER, PAYMENT, DEBIT), balances | Not card/chargeback-specific, no dispute outcomes | Use as a **methodology reference** — it's the field's standard example of "simulate because real data is too private to release," which is exactly your Module 0 justification; cite its imbalance rate as a third calibration point | kaggle.com/datasets/ealaxi/paysim1 |
| **IBM TabFormer** | 24M synthetic-but-realistic card transactions, 20K users, merchant category code, merchant city, chip/swipe/online type, per-user longitudinal history | No dispute/representment outcome labels | Best structural template for **schema design** — its per-user transaction history structure is a good model for your "prior-transaction age 120–365 days" CE3.0 criterion field | github.com/IBM/TabFormer |
| **IBM AML-Data** (companion dataset) | Synthetic multi-agent-generated transactions with money-laundering labels, bank transfers/purchases/checks | Not chargeback-related | Optional — only relevant if you want a structural reference for multi-entity relationship data (useful if you ever extend toward Track 02's "Abuse-ring sentinel" direction) | github.com/IBM/AML-Data |

**What to actually pull from these into `schema.md`:** the imbalance rate
(0.13–0.4% across all three real/near-real sources above — this is your
justification for Module 0's target rate), and rough feature-cardinality
ranges (e.g. how many distinct merchant categories, device types) so your
synthetic categorical fields don't look arbitrarily invented.

---

## B. The closest thing to a "shared benchmark" (and why it still isn't enough)

| Resource | What it argues | Why it's not sufficient on its own |
|---|---|---|
| **Amazon Fraud Dataset Benchmark** (Grover et al., arXiv:2208.14417; github.com/amazon-science/fraud-dataset-benchmark) | Standardizes evaluation across multiple fraud datasets, argues fraud data needs benchmark-specific handling (imbalance, drift, mixed types) | No dispute/representment outcome labels — general fraud detection, not chargeback-specific. Worth citing directly in your deck as "the closest prior art to what we built for Gap 1, and even it doesn't cover dispute outcomes" |
| **TabArena** | General tabular ML benchmark suite for model comparison | Not fraud/dispute-specific at all — useful only if you want a neutral reference for comparing GBDT vs. TabPFN performance claims |

Citing both of these *and* explaining precisely what they don't cover is a
strong, specific way to substantiate Gap 1 in your presentation — it shows
you didn't just assert the gap, you checked the closest available work and
confirmed it.

---

## C. Reference data needed to generate realistic auxiliary fields

These aren't ML training data — they're lookup/reference sources so your
synthetic records don't have obviously fake-looking device/location/issuer
fields.

| Need | Source | Notes |
|---|---|---|
| **BIN/IIN ranges** (so simulated card numbers map to plausible issuer/network/country) | Public BIN list databases (e.g. binlist.net's free API, or community-maintained BIN CSV datasets on GitHub — search "BIN list database github") | Use to assign realistic issuer names and card networks to synthetic transactions; needed for the "issuer-specific criteria weighting" detail in your rubric scorer |
| **IP geolocation** (so "IP mismatch" features are plausible, not random) | MaxMind GeoLite2 (free tier, City/Country database, requires free account signup at maxmind.com) | Generate synthetic IPs, geolocate them, then deliberately inject mismatches (billing country ≠ IP country) at your chosen noise rate for the fraud-labeled subset |
| **Postal/address data** (India-focused, since this is a Razorpay merchant context) | data.gov.in's PIN code dataset, or India Post's official PIN code list | Use for realistic shipping/billing address generation; deliberately mismatch shipping vs. billing PIN codes for a controlled fraction of records to drive the "shipping match" criterion |
| **Realistic names/emails/generic addresses** | `faker` Python library (`pip install faker`, use `Faker('en_IN')` locale) | Not a dataset — a generator. Use for the non-fraud-signal fields (names, emails) so the record set looks populated and real without needing external data |

---

## D. Rules/reference documents (not datasets, but required "config data" for Modules 1 & 6)

| Document | Why you need it | Where to get it |
|---|---|---|
| **Visa Compelling Evidence 3.0 (CE3.0) criteria** | Defines the exact evidentiary categories your rubric scorer (Module 1) and simulated issuer-decision function (Module 0) should mirror | Visa's public merchant/acquirer documentation on CE3.0 (search "Visa Compelling Evidence 3.0 requirements") |
| **Mastercard First-Party Trust criteria** | Same purpose, second network's rules — useful if you want to show your rubric generalizes across networks, not just Visa | Mastercard's public merchant documentation |
| **Chargeback reason code lists** (Visa 10.4, Mastercard 4837, etc.) | Needed for Module 3's reason-code-to-template mapping | Publicly published reason-code reference guides (Chargebacks911, PXP, and similar merchant-education sites publish full lists) |
| **Historical CE3.0 rule-version timeline** | This *is* Module 6's raw material — you need to approximate what changed at each version (April 2023 launch, October 2025 auto-qualification expansion, April 2026 TC40 expansion, per your review) | Visa's own changelog/bulletins if available, or reconstruct approximate rule-set differences from network communications and reputable payments-industry coverage; **label your reconstruction as approximated, explicitly, in `rules/README.md`** |

---

## E. What NOT to bother sourcing

- **Real chargeback/dispute-outcome data** — doesn't exist publicly, don't spend time looking; this is the gap you're demonstrating, not a dataset you're missing due to insufficient search effort.
- **Real merchant PII or actual cardholder data** — never use even if you somehow found a leak or scrape; this would also violate the defense-only/no-fabrication constraint in spirit even if not in your own model's output.
- **Paid/licensed fraud-signal data brokers** (device fingerprinting vendors, etc.) — out of scope and unnecessary; your synthetic device-ID field with injected reuse patterns demonstrates the same concept without a vendor relationship.

---

## Suggested order of operations

1. Pull imbalance-rate and rough feature-cardinality numbers from IEEE-CIS, ULB creditcard, and PaySim → write these into Module 0's `schema.md` as your calibration justification.
2. Set up MaxMind GeoLite2 + a BIN list source → wire into your `faker`-based generator for device/IP/issuer realism.
3. Pull India Post PIN code data if you want shipping/billing address realism specific to an Indian merchant context.
4. Compile CE3.0 + First-Party-Trust criteria into `rules/ce3_current.yaml` for Module 1, and your best reconstruction of the version history into `rules/ce3_2023.yaml` / `rules/ce3_2025_10.yaml` / `rules/ce3_2026_04.yaml` for Module 6.
5. Pull the reason-code list into Module 3's template mapping.
6. Cite the Amazon Fraud Dataset Benchmark paper directly in your deck as "the closest prior art, and here's specifically what it still doesn't cover" — this is a free credibility point.
