# Versioned Rule Configurations (Visa CE3.0 & Mastercard First-Party Trust)

> **IMPORTANT NOTICE: APPROXIMATED FOR DEMONSTRATION**
> The rule configurations in this directory (`ce3_2023.yaml`, `ce3_2025_10.yaml`, `ce3_2026_04.yaml`) are **approximated for demonstration purposes** based on publicly available payments-industry documentation, Visa Compelling Evidence 3.0 (CE3.0) merchant guidance bulletins, and industry analysis of network rule revisions from 2023 to 2026.
> They are designed to model and simulate **concept drift in evidentiary evaluation rubrics** (§2.6 of literature review). They do not represent proprietary internal issuer scoring algorithms or official Visa/Mastercard confidential specifications.

---

## Historical Rule Versions

### 1. `ce3_2023.yaml` — Visa CE3.0 Initial Launch (April 2023)
- **Context:** Initial rollout of Visa Compelling Evidence 3.0 for Reason Code 10.4 (Card-Absent Fraud / Non-Recognition).
- **Core Focus:** Heavy emphasis on historical undisputed transactions in the 120–365 day window, physical delivery/shipping PIN matches, and exact Device ID matches. Static IP matching carried moderate weight.

### 2. `ce3_2025_10.yaml` — Expanded Auto-Qualification (October 2025)
- **Context:** Expansion of digital identity auto-qualification and biometric device tokens.
- **Core Focus:** Increased evidentiary weight given to strong 3DS/OTP authentication and modern biometric device fingerprints; reduced reliance on exact physical shipping matches for digital/omnichannel transactions.

### 3. `ce3_2026_04.yaml` — TC40 Integration & Digital Identity Shift (April 2026)
- **Context:** Network-wide integration with TC40 issuer fraud reports and EMV 3DS 2.3 token authentication.
- **Core Focus:** Heavy emphasis on 3DS OTP step-up verification and multi-factor CVV/AVS authentication (+1400% relative weight increase vs 2023). Near-deprecation of static IP matching (collapsed by 50%) due to widespread CGNAT, dynamic IP reallocation, and mobile VPN/proxy adoption.
