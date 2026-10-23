# 07: Memory and "#n"

**What to build:** Conversations persist and the candidate can refer to Search Results as "#n" reliably (v0.4, part 2).

**Blocked by:** 06

**Status:** ready-for-agent

- [x] LangGraph SQLite checkpointer, one thread per Conversation; full history stored
- [x] Only a sliding window (~20 recent messages) is sent to the LLM
- [x] Structured agent state holds the latest numbered Search Results and the current job ID; "#n" resolves to the exact job ID even after older messages leave the window
- [x] Reopening a Conversation (also after app restart) restores history and Search Results
- [x] Tool-call cap (6 per question) stops a looping agent with a helpful answer
- [x] Seam-3 tests: "#2" after trimming, restore after reopen, cap reached
