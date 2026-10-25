"""The real parts of the evaluation: RAGAs metrics and judges on OpenRouter, the search
pipeline as the retriever, and test-set generation from tagged postings.

Everything here calls paid APIs; the computations they feed are tested with fakes instead.
RAGAs takes about 13 s to import, so it is imported inside the functions that need it.
"""

import os
import random
import threading
from collections.abc import Callable
from typing import Any

import openai
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from turing_projects_v2.config import Settings
from turing_projects_v2.evaluation.accuracy import AgentCase, ToolCall, ToolVerdict
from turing_projects_v2.evaluation.rag import RagConfig
from turing_projects_v2.prompts import load_prompt
from turing_projects_v2.rag.indexer import KnowledgeBase, chunk_texts
from turing_projects_v2.rag.reranker import Reranker
from turing_projects_v2.rag.retriever import QueryTranslation, search_jobs, translate_query
from turing_projects_v2.store import EvalQuestion, JobStore


def ragas_llm(settings: Settings, model: str, *, for_metrics: bool = False) -> Any:
    """A RAGAs (Instructor) LLM on OpenRouter. The RAGAs metric collection calls the LLM
    asynchronously, so `for_metrics` gives it an async client; `DiscreteMetric` needs a sync one."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    from ragas.llms import llm_factory

    client_class = openai.AsyncOpenAI if for_metrics else openai.OpenAI
    client = client_class(
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        timeout=120,
        max_retries=3,
    )
    # Context recall lists every sentence of the reference answer with a verdict; 4,096
    # tokens cut off about 1 in 40 of those answers.
    return llm_factory(model, client=client, temperature=0, max_tokens=16384)


class RagasScorer:
    """Faithfulness, context precision and context recall with RAGAs; a metric that fails
    (e.g. an unparsable judge answer) scores None instead of stopping the run.

    `make_llm()` gives a RAGAs LLM with an async client. Each question gets its own: the
    questions are scored on several threads, each with its own event loop, and an async
    client must not be shared between event loops.
    """

    def __init__(
        self, make_llm: Callable[[], Any], on_error: Callable[[str], None] = print
    ) -> None:
        self._make_llm = make_llm
        self._on_error = on_error

    def score(
        self, question: EvalQuestion, answer: str, contexts: list[str], reference: str
    ) -> dict[str, float | None]:
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

        llm = self._make_llm()
        common = {"user_input": question.question, "retrieved_contexts": contexts or [""]}
        return {
            "faithfulness": self._value(Faithfulness(llm=llm), response=answer, **common),
            "context_precision": self._value(
                ContextPrecision(llm=llm), reference=reference, **common
            ),
            "context_recall": self._value(ContextRecall(llm=llm), reference=reference, **common),
        }

    def _value(self, metric: Any, **inputs: Any) -> float | None:
        try:
            return float(metric.score(**inputs).value)
        except Exception as error:  # one failed metric must not end the run
            self._on_error(f"{metric.name} failed: {type(error).__name__}: {error}")
            return None


class RagasToolJudge:
    """The tool-choice judge as a RAGAs `DiscreteMetric` (reasonable / unreasonable)."""

    def __init__(self, llm: Any) -> None:
        from ragas.metrics import DiscreteMetric

        self._llm = llm
        self._metric = DiscreteMetric(
            name="reasonable_tool",
            allowed_values=["reasonable", "unreasonable"],
            prompt=load_prompt("tool_judge"),
        )

    def judge(self, case: AgentCase, call: ToolCall) -> ToolVerdict:
        result = self._metric.score(
            llm=self._llm,
            question=case.question,
            expected=f"{case.expected_tool} {case.expected_args}",
            called=f"{call.name} {call.args}",
        )
        return ToolVerdict(value=result.value, reason=result.reason or "")


def search_retriever(
    llm: BaseChatModel,
    knowledge_base: Callable[[str], KnowledgeBase],
    reranker: Reranker | None,
) -> Callable[[str, RagConfig], list[str]]:
    """Retrieval as the chatbot does it: the matched chunk of each Search Result.

    Each question's query translation (self-query and multi-query) is made once and reused for
    every configuration, so the configurations differ only in embedding model and reranker.
    """
    translations: dict[str, QueryTranslation] = {}
    lock = threading.Lock()

    def retrieve(question: str, config: RagConfig) -> list[str]:
        with lock:
            translation = translations.get(question)
        if translation is None:
            translation = translate_query(question, llm)
            with lock:
                translation = translations.setdefault(question, translation)
        collection, embedder = knowledge_base(config.embedding_model)
        outcome = search_jobs(
            question,
            llm,
            collection,
            embedder,
            reranker=reranker if config.rerank else None,
            translation=translation,
        )
        return [source.text for source in outcome.trace.sources]

    return retrieve


def answer_writer(llm: BaseChatModel) -> Callable[[str, list[str]], str]:
    """Answers a question from the retrieved chunks only (what faithfulness is measured on)."""

    def answer(question: str, contexts: list[str]) -> str:
        excerpts = "\n\n".join(
            f"<excerpt_data>\n{context}\n</excerpt_data>" for context in contexts
        )
        message = llm.invoke(
            [
                SystemMessage(load_prompt("rag_answer")),
                HumanMessage(f"{excerpts}\n\nQuestion: {question}"),
            ]
        )
        return message.text

    return answer


def generate_test_questions(
    store: JobStore,
    llm: BaseChatModel,
    embedder: Any,
    size: int = 25,
    postings: int = 40,
    seed: int = 15,
) -> list[EvalQuestion]:
    """About `size` RAGAs test-set questions with reference answers, generated from a sample of
    tagged postings (spread over the Target Roles), saved undecided for approval."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator

    tagged = store.tagged_postings()
    by_role: dict[str, list] = {}
    for posting in tagged:
        by_role.setdefault(posting.target_role or "", []).append(posting)
    rng = random.Random(seed)
    sample = []
    per_role = max(1, postings // len(by_role))
    for role_postings in by_role.values():
        sample += rng.sample(role_postings, min(per_role, len(role_postings)))
    documents = [
        Document(
            page_content="\n".join(chunk_texts(posting)),
            metadata={"job_id": posting.job_id, "target_role": posting.target_role},
        )
        for posting in sample
    ]
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embedder)
    # Each posting is one chunk: RAGAs' own splitter needs headlines, which postings do not have.
    # Few parallel requests with patient retries: RAGAs' default of 16 drops connections.
    run_config = RunConfig(max_workers=4, max_retries=10, max_wait=60, timeout=180)
    # A single unparsable LLM answer (e.g. one posting's summary) skips that piece instead
    # of failing the whole generation.
    testset = generator.generate_with_chunks(
        documents, testset_size=size, run_config=run_config, raise_exceptions=False
    )
    questions = [
        EvalQuestion(
            question=sample_.eval_sample.user_input,
            reference=sample_.eval_sample.reference or "",
            reference_contexts=list(sample_.eval_sample.reference_contexts or []),
        )
        for sample_ in testset.samples
    ]
    store.save_test_questions(questions)
    return questions
