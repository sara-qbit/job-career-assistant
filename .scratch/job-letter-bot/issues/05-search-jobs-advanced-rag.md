# 05: search_jobs (self-query + multi-query + fusion)

**What to build:** The advanced RAG Retriever behind the `search_jobs` tool, usable from the command line (v0.3).

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Self-query: the LLM returns search text plus metadata filters (remote, work type, location, Target Role, experience level) via a schema, not prompt-described JSON
- [ ] Multi-query: 3–4 query variants via a schema
- [ ] Filtered similarity search per variant; Reciprocal Rank Fusion; de-duplication by job ID; top 5 numbered Search Results with title, company, location, work type, link
- [ ] Returns a trace of every stage (filters, variants, per-variant results, fusion scores) for later display
- [ ] Clear "no match" result with a broaden-search hint
- [ ] Embedding model selectable by name
- [ ] Seam-2 tests with fake LLM and embedder: filters honoured, fusion ranking, de-duplication, numbering, empty result
