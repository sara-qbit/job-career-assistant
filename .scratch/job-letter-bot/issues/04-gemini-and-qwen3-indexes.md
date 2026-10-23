# 04: Gemini + Qwen3 indexes

**What to build:** Two more Knowledge Base collections using Gemini Embedding 2 and Qwen3-Embedding-8B via OpenRouter embeddings (v0.2, part 2).

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Both cloud models selectable by name through the same embedding interface as MiniLM
- [ ] Requests are batched; rate limits and transient failures are retried with backoff; persistent failure raises the API error type
- [ ] All three collections contain the same tagged postings
- [ ] Tests with a fake embedding client: batching, retry, error mapping
