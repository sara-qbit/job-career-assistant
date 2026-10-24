# 09: Cover Letter (AI template) + downloads

**What to build:** "Write the cover letter for #n" produces a checked Cover Letter from the AI Letter Template, downloadable as .docx and PDF (v0.6).

**Blocked by:** 08

**Status:** ready-for-agent

- [x] `write_cover_letter` picks the Letter Template by the posting's Target Role and fills only its gaps from the Job Info
- [x] No separate motivation sentence (candidate decision 2026-10-24): the two generated gaps each complete their template sentence coherently, using only the Job Info. "...in achieving the [company vision]" gets the company vision reworded to fit, and "I am highly motivated to [role]" gets the role summary reworded as a verb phrase. All other gaps are copied exactly
- [x] `candidate/templates/ai.txt` updated to the candidate's text of 2026-10-24 ("in achieving the {{company_vision}}", "The current Projects")
- [x] Deterministic checks: job name, company and work type in the letter equal the Job Info
- [x] Judge checks the generated sentences; one automatic rewrite on "fabricated"; if it still fails, the letter is shown with a warning and the reason
- [x] Check results shown next to the letter; letter saved with checks and date
- [x] .docx and PDF downloads
- [x] Seam-2 tests: wrong company / title / work type fails, rewrite path, final warning, only gaps change
