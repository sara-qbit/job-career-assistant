# 06: First chat: agent + Chat service + MAIN page

**What to build:** The candidate can chat in a ChatGPT-style page, search jobs and read a verbatim Job Description (v0.4, part 1).

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] LangChain `create_agent` with `openai/gpt-6-luna` via OpenRouter, a domain system prompt from a prompt file, tools `search_jobs` and `get_job_description`
- [ ] `get_job_description` returns the Job Description byte-identical from SQLite, no LLM; shown in its own box with a link to the original posting
- [ ] Chat service: one turn returns a stream of answer events (text, status, tool calls with arguments and results, citations, Job Description, token usage)
- [ ] MAIN page: chat messages, bottom input, streamed answers, status lines while tools run, `[n]` citation markers linking to postings
- [ ] Tool errors are returned to the LLM as tool results; API calls retried with backoff; the UI never shows a stack trace
- [ ] Token usage and cost recorded per LLM call
- [ ] Seam-3 tests with a scripted fake LLM; one smoke test that the Streamlit app starts
