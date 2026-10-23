# 13: Sidebar: Conversations, embedding dropdown, reranker

**What to build:** A ChatGPT-like left sidebar with past Conversations, the embedding dropdown and the reranker toggle (v0.8).

**Blocked by:** 04, 07

**Status:** ready-for-agent

- [ ] "New chat" and a list of past Conversations; clicking one reopens it fully
- [ ] Embedding dropdown with Gemini Embedding 2 preselected, Qwen3 and local MiniLM as alternatives
- [ ] Local cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the top ~20 after fusion; toggle on by default; scores added to the search trace
- [ ] Seam-2/3 tests: reranker on/off changes order as scripted; model selection is passed through
