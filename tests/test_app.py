from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.test_chat import DESCRIPTION, call, chat, chat_store, search_steps
from turing_projects_v2.agent import chat as chat_module

APP = Path(__file__).parents[1] / "src" / "turing_projects_v2" / "ui" / "app.py"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    st.cache_resource.clear()


def test_app_starts_and_shows_the_chat_page():
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Job Letter Assistant"
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
