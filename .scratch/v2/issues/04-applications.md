# 04: Applications

**What to build:** When the candidate says "save it" or "I applied", the assistant records an Application for the pasted Job Posting, dated that day. Repeats are caught, a wrong Application can be removed, and Kaggle Job Postings are refused. See spec: Applications.

**Blocked by:** 01

**Status:** ready-for-agent

- [x] "Save it" / "I applied" records an Application with today's local date and confirms it
- [x] A second request for the same Job Posting answers "Already applied on <DD.MM.YYYY>" and adds nothing
- [x] "Remove the application" / "I didn't apply" removes the Application; the Job Posting stays stored
- [x] An Application for a Kaggle Job Posting is refused with an explanation that only pasted jobs go on the Application Table
- [x] An Application can be recorded for a pasted Job Posting from an earlier Conversation
- [x] The system prompt says to record an Application only when the candidate says they applied or asks to save
- [x] Tests through the chat service; full suite and ruff green
