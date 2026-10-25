# 15: RAGAs evaluation

**What to build:** A measured evaluation of the RAG pipeline and the agent, shown on the Evaluation tab (v1.0).

**Blocked by:** 14

**Status:** done

- [x] RAGAs test-set generation from tagged postings (~25), filtered, then approved by the candidate (~20); plus 10 hand-written agent cases
- [x] RAGAs metrics (faithfulness, context precision, context recall) per embedding model on the same tagged postings, and with / without reranker
- [x] Tool Selection Accuracy and Parameter Accuracy on the hand-written cases; `DiscreteMetric` judge verdicts for reasonable alternatives
- [x] Field accuracy (job name, company, work type) over saved Cover Letters
- [x] Results stored per run and shown with charts on the Evaluation tab
- [x] Tests for the accuracy computations with fixed data and a fake judge

**Note (2026-10-25):** generation and the 10 agent cases are done. Claude made the selection
(18 of 24, reasons in `../test-set-review.md`) while the candidate was asleep; the first box is
ticked once the candidate has confirmed or changed it on the Evaluation tab (then rerun
`turing-projects-v2 evaluate rag` if a decision changed). The candidate confirmed the
selection on 2026-10-25.
