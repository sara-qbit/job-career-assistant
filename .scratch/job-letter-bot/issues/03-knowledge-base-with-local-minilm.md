# 03: Knowledge Base with local MiniLM

**What to build:** Tagged Job Postings are chunked and indexed into a ChromaDB collection with the local MiniLM embedding model (v0.2, part 1).

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] Recursive splitting (~1,000 characters, ~150 overlap); every chunk starts with a context header (title, company, work type, location)
- [ ] Every chunk carries metadata: job ID, Target Role, work type, remote flag, location, experience level, salary
- [ ] Only tagged postings are indexed; one collection per embedding model
- [ ] Progress indicator during indexing
- [ ] Re-index only embeds postings not yet in the collection
- [ ] Seam-1 tests with a fake embedder: header, metadata, tagged-only, incremental
