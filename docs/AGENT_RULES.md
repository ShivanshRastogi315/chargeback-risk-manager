# Agent Operating Rules & Multi-Session Protocols

These operational rules apply to every AI agent session working on this repository:

- **SESSION START:** Always read [00_START_HERE.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/00_START_HERE.md) and [PROGRESS.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/PROGRESS.md) in full before doing anything else. Only read [PROJECT_SPEC.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/PROJECT_SPEC.md) / [EXECUTION_PLAYBOOK.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/EXECUTION_PLAYBOOK.md) / [DATASETS_GUIDE.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/DATASETS_GUIDE.md) for the specific section relevant to the current phase — never re-read them in full.
- **SESSION END:** Update [PROGRESS.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/PROGRESS.md)'s status table and the `CURRENT PHASE` line at top. If anything broke and got fixed this session, append an entry to [DECISIONS_LOG.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/DECISIONS_LOG.md) before ending.
- **DOCS INTEGRITY:** Do not modify `docs/` files other than `PROGRESS.md` and `DECISIONS_LOG.md` unless explicitly asked by the user.
- **DECISION AUTONOMY:** If an implementation choice is small/reversible, make a reasonable decision yourself and log it — don't stop to ask. If it changes the architecture, an acceptance criterion, or the defense-only boundary, stop and ask.
- **LOG ENTRY STYLE:** Keep [DECISIONS_LOG.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/DECISIONS_LOG.md) entries terse (2-4 sentences per field) — bullet points, not essays.
- **VERIFY ACCEPTANCE CRITERIA:** Never proceed past a phase's acceptance criteria without explicitly checking them against [PROJECT_SPEC.md](file:///c:/Users/Shivansh/Desktop/chargeback-risk-manager/docs/PROJECT_SPEC.md)'s stated criteria for that module.
