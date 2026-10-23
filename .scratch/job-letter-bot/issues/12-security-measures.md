# 12: Security measures

**What to build:** The six security measures are active and demonstrable (v0.7, part 2).

**Blocked by:** 09

**Status:** ready-for-agent

- [ ] Input validation: non-empty, at most 2,000 characters, numeric and existing job IDs, clear messages
- [ ] Prompt-injection defence: posting text in delimited data blocks plus a system-prompt rule; pattern-based flagging of user input
- [ ] Scope guard politely refuses off-topic requests
- [ ] Personal data only passed to the LLM in `write_cover_letter` and never logged
- [ ] Secrets only in the environment file; startup check
- [ ] Tool-call cap confirmed; security events written to a security log
- [ ] Seam-3 tests for each measure
