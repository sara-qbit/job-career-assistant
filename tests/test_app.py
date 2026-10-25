from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.test_chat import (
    DESCRIPTION,
    FakeKnowledgeBases,
    call,
    chat,
    chat_store,
    chat_with,
    of_kind,
    search_steps,
)
from turing_projects_v2.agent import chat as chat_module
from turing_projects_v2.agent.chat import CitationsEvent
from turing_projects_v2.fakes import FakeChatModel, FakeReranker
from turing_projects_v2.store import EvalQuestion, EvaluationRun

APP = Path(__file__).parents[1] / "src" / "turing_projects_v2" / "ui" / "app.py"


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    """A test key, and a Chat service without answers unless a test sets its own."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    service = chat(tmp_path, chat_store(tmp_path), [])
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    st.cache_resource.clear()


def test_app_starts_and_shows_the_chat_page():
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Job Career Assistant"
    assert len(app.chat_input) == 1


def test_answers_show_linked_citations_and_the_verbatim_job_description(tmp_path, monkeypatch):
    responses = [
        *search_steps(),
        "#1 Python LLM Engineer [1]",
        call("get_job_description", {"job_id": 1}),
        "See the box.",
    ]
    service = chat(tmp_path, chat_store(tmp_path), responses)
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    app.chat_input[0].set_value("python jobs").run()
    app.chat_input[0].set_value("show me #1").run()

    assert not app.exception
    markdown = [element.value for element in app.markdown]
    assert any("[\\[1\\]](https://www.linkedin.com/jobs/view/" in text for text in markdown)
    assert DESCRIPTION.removesuffix("\n") in [element.value for element in app.code]
    assert len(app.chat_message) == 4


def test_reopening_the_page_restores_the_conversation(tmp_path, monkeypatch):
    service = chat(tmp_path, chat_store(tmp_path), [*search_steps(), "#1 Python LLM Engineer [1]"])
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    first = AppTest.from_file(str(APP), default_timeout=30).run()
    first.chat_input[0].set_value("python jobs").run()

    reopened = AppTest.from_file(str(APP), default_timeout=30)
    reopened.query_params["c"] = first.query_params["c"]
    reopened.run()

    assert not reopened.exception
    assert len(reopened.chat_message) == 2
    markdown = [element.value for element in reopened.markdown]
    assert "python jobs" in markdown
    assert any(r"[\[1\]](https://www.linkedin.com/jobs/view/" in text for text in markdown)


def test_sidebar_starts_with_gemini_and_the_reranker_on():
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not app.exception
    assert app.sidebar.button(key="new_chat").label == "New chat"
    assert app.sidebar.selectbox(key="embedding_model").value == "gemini"
    assert app.sidebar.selectbox(key="embedding_model").options == [
        "Gemini Embedding 2",
        "Qwen3-Embedding-8B",
        "MiniLM (local)",
    ]
    assert app.sidebar.toggle(key="rerank").value is True


def test_sidebar_choices_are_used_for_the_search(tmp_path, monkeypatch):
    store = chat_store(tmp_path)
    knowledge_bases = FakeKnowledgeBases(tmp_path, store)
    reranker = FakeReranker({"Grants Officer": 0.9})
    llm = FakeChatModel(responses=[*search_steps(), "Found [1]"])
    service = chat_with(tmp_path, store, llm, knowledge_bases=knowledge_bases, reranker=reranker)
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    app.sidebar.selectbox(key="embedding_model").select("qwen3").run()
    app.sidebar.toggle(key="rerank").set_value(False).run()
    app.chat_input[0].set_value("python jobs").run()

    assert not app.exception
    assert knowledge_bases.opened == ["qwen3"]
    assert reranker.scored == []


def test_past_conversation_is_listed_and_reopens_on_click(tmp_path, monkeypatch):
    responses = [*search_steps(), "#1 Python LLM Engineer [1]", "Hello!"]
    service = chat(tmp_path, chat_store(tmp_path), responses)
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.chat_input[0].set_value("python jobs").run()
    first_id = app.query_params["c"]

    app.sidebar.button(key="new_chat").click().run()
    assert len(app.chat_message) == 0
    app.chat_input[0].set_value("hi").run()
    assert app.query_params["c"] != first_id

    app.sidebar.button(key=f"conversation-{first_id}").click().run()

    assert not app.exception
    assert app.query_params["c"] == first_id
    assert [element.value for element in app.chat_message[0].markdown] == ["python jobs"]
    assert len(app.chat_message) == 2
    titles = [button.label for button in app.sidebar.button if button.key != "new_chat"]
    assert titles == ["hi", "python jobs"]


def test_reopened_conversation_still_resolves_result_numbers(tmp_path, monkeypatch):
    store = chat_store(tmp_path)
    llm = FakeChatModel(responses=[*search_steps(), "Found [1] [2]", "Hello!", "Sure."])
    service = chat_with(tmp_path, store, llm)
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.chat_input[0].set_value("python jobs").run()
    first_id = app.query_params["c"]
    app.sidebar.button(key="new_chat").click().run()
    app.chat_input[0].set_value("hi").run()

    app.sidebar.button(key=f"conversation-{first_id}").click().run()
    app.chat_input[0].set_value("tell me about #1").run()

    assert not app.exception
    (citations,) = of_kind(service.conversation(first_id).turns[0].events, CitationsEvent)
    assert f"#1 = job ID {citations.results[0].job_id}" in llm.calls[-1][0].text


EVALUATION_SECTIONS = [
    "Conversation history",
    "RAG process",
    "Sources",
    "Token usage and cost",
    "Tool calls",
    "Security",
    "System prompts",
    "RAGAs test set",
    "Evaluation results",
]


def test_evaluation_page_shows_every_section_for_the_conversation(tmp_path, monkeypatch):
    service = chat(tmp_path, chat_store(tmp_path), [*search_steps(), "Found [1] [2]"])
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.chat_input[0].set_value("python jobs").run()

    app.switch_page("app_pages/evaluation.py").run()

    assert not app.exception
    assert [header.value for header in app.subheader] == EVALUATION_SECTIONS
    downloads = app.get("download_button")
    assert [button.proto.label for button in downloads] == ["Markdown", "JSON"]
    text = " ".join(element.value for element in app.markdown)
    assert "python engineer" in text  # a query variant of the RAG process


def test_evaluation_page_of_a_new_conversation_explains_that_it_is_empty():
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    app.switch_page("app_pages/evaluation.py").run()

    assert not app.exception
    assert [header.value for header in app.subheader] == EVALUATION_SECTIONS
    assert any("No questions in this Conversation yet" in info.value for info in app.info)


def evaluated_service(tmp_path):
    service = chat(tmp_path, chat_store(tmp_path), [])
    service.store.save_test_questions(
        [
            EvalQuestion(question=f"question {n}", reference=f"answer {n}", reference_contexts=[])
            for n in (1, 2)
        ]
    )
    for kind, config, metrics in [
        ("rag", {"embedding_model": "gemini", "rerank": True}, {"faithfulness": 0.9}),
        ("rag", {"embedding_model": "gemini", "rerank": False}, {"faithfulness": 0.7}),
        ("agent", {"cases": 10}, {"tool_selection_accuracy": 1.0}),
        ("fields", {"letters": 6}, {"job name": 1.0}),
    ]:
        service.store.save_evaluation_run(
            EvaluationRun(kind=kind, config=config, metrics=metrics, cases=[])
        )
    return service


def test_evaluation_results_are_shown_with_charts(tmp_path, monkeypatch):
    service = evaluated_service(tmp_path)
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    app.switch_page("app_pages/evaluation.py").run()

    assert not app.exception
    assert [header.value for header in app.subheader][-2:] == EVALUATION_SECTIONS[-2:]
    assert len(app.get("vega_lite_chart")) >= 1
    text = " ".join(element.value for element in app.markdown)
    assert "Tool Selection Accuracy" in text


def test_ticking_a_test_question_approves_it(tmp_path, monkeypatch):
    service = evaluated_service(tmp_path)
    monkeypatch.setattr(chat_module, "open_chat_service", lambda settings: service)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.switch_page("app_pages/evaluation.py").run()
    _first, second = service.store.test_questions()

    app.checkbox(key=f"approve-{second.id}").check().run()

    assert not app.exception
    assert [q.question for q in service.store.approved_test_questions()] == ["question 2"]
    app.checkbox(key=f"approve-{second.id}").uncheck().run()
    assert service.store.approved_test_questions() == []
