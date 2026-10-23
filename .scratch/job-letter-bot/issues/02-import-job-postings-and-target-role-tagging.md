# 02: Import all Job Postings + Target Role tagging

**What to build:** A one-time import of all Kaggle Job Postings and companies into SQLite, plus a reviewed Target Role title list and a tagging / re-tag command (v0.1). See ADR 0001 and the spec (Importer, Schema).

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] All postings (~123,849) and companies from the downloaded Kaggle dataset are stored in SQLite, keyed by job ID / company ID
- [ ] Re-running the import creates no duplicates
- [ ] A reviewed title-variant list exists for the four Target Roles (AI Engineer, Grant Management Specialist, AI Innovation Manager, Quantum Software Developer); quantum matching excludes SEO / "optimization" noise; the candidate reviews the list before it is final
- [ ] Tagging sets each posting's Target Role or none; re-tag after editing the list takes seconds and needs no re-import
- [ ] The command reports totals and tagged counts per Target Role (expected roughly 1,081 tagged)
- [ ] Seam-1 tests on a ~20-row fixture: storage, idempotency, title-variant matches and non-matches, re-tag
