# 09: Cover Letter (AI template) + downloads

**What to build:** "Write the cover letter for #n" produces a checked Cover Letter from the AI Letter Template, downloadable as .docx and PDF (v0.6).

**Blocked by:** 08

**Status:** ready-for-agent

- [ ] `write_cover_letter` picks the Letter Template by the posting's Target Role and fills only its gaps from the Job Info
- [ ] The motivation sentence uses only the role summary and facts already in the Letter Template
- [ ] Deterministic checks: job name, company and work type in the letter equal the Job Info
- [ ] Judge checks the generated sentences; one automatic rewrite on "fabricated"; if it still fails, the letter is shown with a warning and the reason
- [ ] Check results shown next to the letter; letter saved with checks and date
- [ ] .docx and PDF downloads
- [ ] Seam-2 tests: wrong company / title / work type fails, rewrite path, final warning, only gaps change
