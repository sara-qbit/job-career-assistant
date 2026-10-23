# 15: RAGAs evaluation

**What to build:** A measured evaluation of the RAG pipeline and the agent, shown on the Evaluation tab (v1.0).

**Blocked by:** 14

**Status:** ready-for-agent

- [ ] RAGAs test-set generation from tagged postings (~25), filtered, then approved by the candidate (~20); plus 10 hand-written agent cases
- [ ] RAGAs metrics (faithfulness, context precision, context recall) per embedding model on the same tagged postings, and with / without reranker
- [ ] Tool Selection Accuracy and Parameter Accuracy on the hand-written cases; `DiscreteMetric` judge verdicts for reasonable alternatives
- [ ] Field accuracy (job name, company, work type) over saved Cover Letters
- [ ] Results stored per run and shown with charts on the Evaluation tab
- [ ] Tests for the accuracy computations with fixed data and a fake judge
