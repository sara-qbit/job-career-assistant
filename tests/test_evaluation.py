import time
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage

from tests.kaggle_fixture import posting as posting_row
from tests.kaggle_fixture import write_kaggle_dataset
from turing_projects_v2.evaluation.accuracy import (
    AgentCase,
    ToolCall,
    field_accuracy,
    score_agent_cases,
)
from turing_projects_v2.evaluation.agent import run_agent_evaluation
from turing_projects_v2.evaluation.fields import run_field_evaluation
from turing_projects_v2.evaluation.rag import RagConfig, run_rag_evaluation
from turing_projects_v2.fakes import FakeChatModel, FakeToolJudge
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.store import CoverLetter, EvalQuestion, JobPosting, JobStore

SEARCH = AgentCase(
    question="remote AI engineer jobs in Texas",
    expected_tool="search_jobs",
    expected_args={"question": "remote AI engineer jobs in Texas"},
)
DESCRIPTION = AgentCase(
    question="show me the job description of job 7",
    expected_tool="get_job_description",
    expected_args={"job_id": 7},
)
STATS = AgentCase(
    question="how many grant jobs are remote?",
    expected_tool="job_market_stats",
    expected_args={"statistic": "postings_by_remote", "target_role": "Grant Management Specialist"},
)


def test_tool_selection_and_parameter_accuracy_on_fixed_cases():
    calls = {
        SEARCH.question: ToolCall(name="search_jobs", args={"question": SEARCH.question}),
        DESCRIPTION.question: ToolCall(name="get_job_description", args={"job_id": 8}),
        STATS.question: ToolCall(
            name="job_market_stats",
            args={"statistic": "postings_by_remote", "target_role": "Grant Management Specialist"},
        ),
    }

    report = score_agent_cases([SEARCH, DESCRIPTION, STATS], calls, FakeToolJudge())

    assert report.tool_selection_accuracy == 1.0
    # search 1/1, description 0/1 (wrong job ID), stats 2/2
    assert report.parameter_accuracy == pytest.approx((1 + 0 + 1) / 3)
    assert [result.parameters_correct for result in report.results] == [1.0, 0.0, 1.0]


def test_wrong_tool_counts_as_reasonable_only_when_the_judge_says_so():
    calls = {
        SEARCH.question: ToolCall(name="job_market_stats", args={"statistic": "postings"}),
        DESCRIPTION.question: ToolCall(name="create_job_info", args={"job_id": 7}),
        STATS.question: None,  # the agent answered without a tool
    }
    judge = FakeToolJudge({"create_job_info": "reasonable"})

    report = score_agent_cases([SEARCH, DESCRIPTION, STATS], calls, judge)

    assert report.tool_selection_accuracy == 0.0
    assert report.tool_selection_accuracy_with_alternatives == pytest.approx(1 / 3)
    assert [r.alternative.value if r.alternative else None for r in report.results] == [
        "unreasonable",
        "reasonable",
        None,
    ]
    # Parameters are only scored when the expected tool was called.
    assert report.parameter_accuracy is None
    assert [call for call, _ in judge.judged] == ["job_market_stats", "create_job_info"]


def test_string_arguments_match_ignoring_case_and_spacing():
    calls = {
        STATS.question: ToolCall(
            name="job_market_stats",
            args={"statistic": "postings_by_remote", "target_role": " grant management specialist"},
        )
    }

    report = score_agent_cases([STATS], calls, FakeToolJudge())

    assert report.parameter_accuracy == 1.0


def posting(job_id, title, company, work_type):
    return JobPosting(
        **dict.fromkeys(JobPosting.model_fields)
        | {
            "job_id": job_id,
            "title": title,
            "company_name": company,
            "source": "kaggle",
            "formatted_work_type": work_type,
        }
    )


def letter(job_id, text):
    return CoverLetter(
        job_id=job_id,
        target_role="AI Engineer",
        template_name="ai",
        text=text,
        checks=[],
        company_vision_verdict=None,
        role_summary_verdict=None,
        rewritten=False,
        created_at=datetime(2026, 10, 24, tzinfo=UTC),
    )


def test_field_accuracy_is_the_share_of_letters_containing_each_exact_field():
    postings = {
        1: posting(1, "AI Engineer", "Acme Corp", "Full-time"),
        2: posting(2, "Grant Writer", "Hope Fund", "Contract"),
    }
    letters = [
        letter(1, "I apply for AI Engineer (Full-time) at Acme Corp."),
        letter(2, "I apply for Grant Writer (Part-time) at Hope Foundation."),
    ]

    report = field_accuracy(letters, postings.get)

    assert report.letters == 2
    assert report.accuracy == {"job name": 1.0, "company": 0.5, "work type": 0.5}


def test_field_not_stated_in_the_posting_is_not_counted():
    postings = {1: posting(1, "AI Engineer", None, "Full-time")}

    report = field_accuracy([letter(1, "AI Engineer, Full-time")], postings.get)

    assert report.accuracy == {"job name": 1.0, "company": None, "work type": 1.0}


def eval_question(question):
    return EvalQuestion(
        question=question,
        reference=f"Answer to {question}",
        reference_contexts=[f"Context of {question}"],
    )


def test_only_approved_test_questions_are_used_for_evaluation():
    store = JobStore(":memory:")
    store.save_test_questions([eval_question("q1"), eval_question("q2"), eval_question("q3")])
    first, second, third = store.test_questions()
    assert [c.approved for c in (first, second, third)] == [None, None, None]

    store.set_test_approval(first.id, True)
    store.set_test_approval(second.id, False)
    store.set_test_approval(third.id, True)

    assert [q.question for q in store.approved_test_questions()] == ["q1", "q3"]
    assert store.approved_test_questions()[1].reference_contexts == ["Context of q3"]


class ScriptedScorer:
    """Scores 1.0 when the answer mentions the reference's job, else 0.5; records calls."""

    def __init__(self):
        self.scored = []

    def score(self, question, answer, contexts, reference):
        self.scored.append((question.question, tuple(contexts)))
        hit = 1.0 if question.question in answer else 0.5
        return {"faithfulness": hit, "context_precision": hit, "context_recall": 1.0}


def test_each_configuration_is_stored_as_a_run_with_per_question_scores():
    store = JobStore(":memory:")
    store.save_test_questions([eval_question("q1"), eval_question("q2")])
    for question in store.test_questions():
        store.set_test_approval(question.id, True)
    configs = [
        RagConfig(embedding_model="gemini", rerank=True),
        RagConfig(embedding_model="minilm", rerank=False),
    ]

    def retrieve(question, config):
        return [f"{config.embedding_model} chunk for {question}"]

    def answer(question, contexts):
        # minilm "finds" only q1, so its q2 answer misses
        return f"about {question}" if "gemini" in contexts[0] or question == "q1" else "no idea"

    scorer = ScriptedScorer()
    run_rag_evaluation(store, configs, retrieve, answer, scorer)

    gemini, minilm = store.evaluation_runs()
    assert gemini.kind == "rag" and gemini.config == {"embedding_model": "gemini", "rerank": True}
    assert gemini.metrics == {"faithfulness": 1.0, "context_precision": 1.0, "context_recall": 1.0}
    assert minilm.metrics["faithfulness"] == pytest.approx(0.75)
    assert [case["question"] for case in minilm.cases] == ["q1", "q2"]
    assert minilm.cases[1]["answer"] == "no idea"
    assert minilm.cases[1]["contexts"] == ["minilm chunk for q2"]
    assert len(scorer.scored) == 4


def test_agent_evaluation_asks_the_agent_model_for_its_first_tool_call_and_stores_a_run():
    store = JobStore(":memory:")
    llm = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "search_jobs", "args": SEARCH.expected_args, "id": "1"}],
            ),
            "I can answer that without a tool.",
        ]
    )

    run = run_agent_evaluation(store, [SEARCH, DESCRIPTION], llm, tools=[], judge=FakeToolJudge())

    assert run.kind == "agent"
    assert run.metrics == {
        "tool_selection_accuracy": 0.5,
        "tool_selection_accuracy_with_alternatives": 0.5,
        "parameter_accuracy": 1.0,
    }
    assert [case["call"] for case in run.cases] == [
        {"name": "search_jobs", "args": SEARCH.expected_args},
        None,
    ]
    assert store.evaluation_runs() == [
        run.model_copy(update={"id": 1, "created_at": store.evaluation_runs()[0].created_at})
    ]
    assert llm.calls[0][-1].text == SEARCH.question


def test_field_evaluation_covers_every_saved_letter_and_is_stored(tmp_path):
    store = JobStore(":memory:")
    rows = [posting_row(1, "AI Engineer"), posting_row(2, "Grant Writer", company_name="Hope Fund")]
    import_kaggle(write_kaggle_dataset(tmp_path / "raw", rows), store)
    store.save_cover_letter(letter(1, "AI Engineer (Full-time) at Acme Corp"))
    store.save_cover_letter(letter(2, "Grant Writer (Full-time) at Hope Foundation"))

    run = run_field_evaluation(store)

    assert run.kind == "fields"
    assert run.config == {"letters": 2}
    assert run.metrics == {"job name": 1.0, "company": 0.5, "work type": 1.0}
    assert [(case["job_id"], case["company"]) for case in run.cases] == [(1, True), (2, False)]
    assert [run.model_dump(exclude={"id", "created_at"})] == [
        r.model_dump(exclude={"id", "created_at"}) for r in store.evaluation_runs()
    ]


def test_parallel_evaluation_keeps_the_question_order():
    store = JobStore(":memory:")
    store.save_test_questions([eval_question(f"q{n}") for n in range(1, 7)])
    for question in store.test_questions():
        store.set_test_approval(question.id, True)

    def retrieve(question, config):
        time.sleep(0.05 if question in ("q1", "q2") else 0)  # early questions finish last
        return [f"chunk for {question}"]

    (run,) = run_rag_evaluation(
        store,
        [RagConfig(embedding_model="gemini", rerank=True)],
        retrieve,
        lambda question, contexts: f"about {question}",
        ScriptedScorer(),
        workers=3,
    )

    assert [case["question"] for case in run.cases] == ["q1", "q2", "q3", "q4", "q5", "q6"]
    assert run.metrics["faithfulness"] == 1.0
