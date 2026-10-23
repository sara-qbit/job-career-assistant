# 08: Job Info + judge

**What to build:** "Tell me about #n" shows a verified Job Info card (v0.5).

**Blocked by:** 07

**Status:** ready-for-agent

- [x] `create_job_info` retrieves the posting and company by job ID (no similarity search)
- [x] Job name, company name and work type copied exactly from the data
- [x] Role summary and company vision produced via a schema; "not stated in the posting" when the source lacks it
- [x] RAGAs `DiscreteMetric` judge (`anthropic/claude-haiku-5.5`) returns supported / fabricated plus a reason for both summaries; verdicts shown on the card
- [x] Job Info saved and reused (not regenerated)
- [x] Seam-2 tests with fake LLM and judge: exact fields, "not stated", verdict storage, reuse
