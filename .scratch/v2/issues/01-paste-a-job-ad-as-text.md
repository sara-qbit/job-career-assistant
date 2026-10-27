# 01: Paste a job ad as text

**What to build:** The candidate pastes the full text of a job ad (often 3,000–8,000 characters) and it becomes a pasted Job Posting with its own job ID. "Show the Job Description" and "Job Info" then work on it as for a Kaggle Job Posting. See spec: `.scratch/v2/spec.md` (Pasting a job, Job Info and Job Description for pasted jobs).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] The chat send accepts an attached job ad (text) separate from the typed message; the ad bypasses the 2,000-character message limit and has its own limit (about 30,000 characters, with a clear message when exceeded)
- [x] Typed messages keep the 2,000-character limit
- [x] The ad text goes through the injection check (flagged and logged, not blocked) and reaches the model only wrapped as data
- [x] The UI offers a way to paste a long ad that reaches this input
- [x] A pasted Job Posting is stored with source "pasted", the date pasted, and a job ID that cannot collide with Kaggle job IDs; its text is stored unchanged
- [x] Pasting the same ad again (ignoring whitespace differences) reuses the existing Job Posting
- [x] Text that is not a job ad gets a clear answer and nothing is stored
- [x] Pasted Job Postings never enter the Knowledge Base or the market statistics
- [x] "Show the Job Description" shows the pasted ad verbatim; Job Info works on it, with the company vision taken from the ad text only
- [x] A pasted Job Posting stays stored after the Conversation and is still usable when the Conversation is reopened
- [x] Tests through the chat service with the fake model; full suite and ruff green
