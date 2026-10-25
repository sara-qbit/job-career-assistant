"""RAG evaluation: the approved test-set questions through the search pipeline, once per
configuration (embedding model, reranker on or off), scored with RAGAs metrics.

Each configuration is stored as one evaluation run with its per-question results, so the
Evaluation tab can compare the embedding models and the reranker on the same questions.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from pydantic import BaseModel

from turing_projects_v2.store import EvalQuestion, EvaluationRun, JobStore

METRICS = ("faithfulness", "context_precision", "context_recall")


class RagConfig(BaseModel):
    embedding_model: str
    rerank: bool


class RagScorer(Protocol):
    def score(
        self, question: EvalQuestion, answer: str, contexts: list[str], reference: str
    ) -> dict[str, float | None]:
        """The METRICS for one answered question."""
        ...


def run_rag_evaluation(
    store: JobStore,
    configs: Sequence[RagConfig],
    retrieve: Callable[[str, RagConfig], list[str]],
    answer: Callable[[str, list[str]], str],
    scorer: RagScorer,
    on_progress: Callable[[str], None] = lambda message: None,
    workers: int = 1,
) -> list[EvaluationRun]:
    """Evaluate every configuration on the approved questions and store one run each.

    `retrieve(question, config)` returns the retrieved chunk texts; `answer(question, contexts)`
    writes the answer from them. `workers` questions are evaluated at a time (each takes about
    a minute of API calls); the results keep the question order.
    """
    questions = store.approved_test_questions()
    runs = []
    for config in configs:

        def evaluate(question: EvalQuestion, config: RagConfig = config) -> dict[str, Any]:
            contexts = retrieve(question.question, config)
            response = answer(question.question, contexts)
            scores = scorer.score(question, response, contexts, question.reference)
            on_progress(f"{config.embedding_model}, reranker {config.rerank}: {question.question}")
            return {
                "question_id": question.id,
                "question": question.question,
                "answer": response,
                "contexts": contexts,
                **{metric: scores.get(metric) for metric in METRICS},
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            cases = list(pool.map(evaluate, questions))
        run = EvaluationRun(
            kind="rag",
            config=config.model_dump(),
            metrics={metric: _mean(case[metric] for case in cases) for metric in METRICS},
            cases=cases,
        )
        store.save_evaluation_run(run)
        runs.append(run)
    return runs


def _mean(values) -> float | None:
    """The mean of the values that were scored (a failed metric is None)."""
    scored = [value for value in values if value is not None]
    return sum(scored) / len(scored) if scored else None
