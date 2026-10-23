# 02: Import all Job Postings + Target Role tagging

**What to build:** A one-time import of all Kaggle Job Postings and companies into SQLite, plus a reviewed Target Role title list and a tagging / re-tag command (v0.1). See ADR 0001 and the spec (Importer, Schema).

**Blocked by:** 01

**Status:** ready-for-agent

- [x] All postings (~123,849) and companies from the downloaded Kaggle dataset are stored in SQLite, keyed by job ID / company ID
- [x] Re-running the import creates no duplicates
- [x] A reviewed title-variant list exists for the four Target Roles (AI Engineer, Grant Management Specialist, AI Innovation Manager, Quantum Software Developer); quantum matching excludes SEO / "optimization" noise; the candidate reviews the list before it is final
- [x] Tagging sets each posting's Target Role or none; re-tag after editing the list takes seconds and needs no re-import
- [x] The command reports totals and tagged counts per Target Role (expected roughly 1,081 tagged)
- [x] Seam-1 tests on a ~20-row fixture: storage, idempotency, title-variant matches and non-matches, re-tag

## Comments

**2026-10-23, implemented (v0.1).** 123,849 postings and 24,473 companies imported (~36 s); re-tag ~10 s.
Candidate review of `candidate/title_variants.toml` decided:
- AI Engineer includes Data Scientist, plus data science, research/applied scientist, analytics engineer and statistician titles.
- Grant Management Specialist includes proposal and fundraising/development roles; excludes grant writers, grant accountants and volunteer posts.
- AI Innovation Manager: AI/tech innovation only (no consumer-brand innovation).
- Quantum Software Developer: all 8 "quantum" titles kept.

Result: 1,108 tagged (AI Engineer 776, Grant Management Specialist 312, AI Innovation Manager 12, Quantum Software Developer 8). Matched titles: `.scratch/job-letter-bot/title-review.md`.
