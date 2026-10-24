import json

from langchain_core.messages import AIMessage

from tests.test_chat import chat, chat_store, search_steps, with_usage
from turing_projects_v2.agent.chat import Conversation
from turing_projects_v2.export import conversation_json, conversation_markdown


def two_turn_conversation(tmp_path):
    first, *retriever_steps = search_steps()
    service = chat(
        tmp_path,
        chat_store(tmp_path),
        [
            with_usage(first, 100, 10, 0.001),
            *retriever_steps,
            with_usage(AIMessage(content="Found [1] and [2]."), 200, 20, 0.002),
            with_usage(AIMessage(content="You're welcome."), 50, 5, 0.0005),
        ],
    )
    list(service.send("c1", "python jobs"))
    list(service.send("c1", "thanks"))
    return service.conversation("c1")


def test_markdown_export_has_questions_answers_sources_tools_and_costs(tmp_path):
    markdown = conversation_markdown(two_turn_conversation(tmp_path), title="python jobs")

    assert markdown.startswith("# python jobs\n")
    assert "## 1. python jobs\n" in markdown
    assert "Found [1] and [2]." in markdown
    assert "## 2. thanks\n" in markdown
    assert "You're welcome." in markdown
    assert "1. [Python LLM Engineer at Acme Corp](https://www.linkedin.com/jobs/view/1)" in markdown
    assert '- `search_jobs` {"question": "python jobs"}' in markdown
    assert "330 tokens · $0.0030" in markdown
    assert "55 tokens · $0.0005" in markdown
    assert markdown.rstrip().endswith("**Total:** 385 tokens · $0.0035")


def test_json_export_reads_back_as_the_same_conversation(tmp_path):
    conversation = two_turn_conversation(tmp_path)

    exported = conversation_json(conversation)

    assert Conversation.model_validate_json(exported) == conversation
    assert json.loads(exported)["total"] == {"tokens": 385, "cost": 0.0035, "cost_known": True}


def test_cost_not_reported_by_a_provider_keeps_the_known_part(tmp_path):
    service = chat(
        tmp_path,
        chat_store(tmp_path),
        [
            with_usage(AIMessage(content="Hi"), 100, 10, 0.001),
            with_usage(AIMessage(content="Hello again"), 10, 2, None),
        ],
    )
    list(service.send("c1", "hello"))
    list(service.send("c1", "hello again"))

    markdown = conversation_markdown(service.conversation("c1"), title="hello")

    assert "110 tokens · $0.0010" in markdown
    assert "12 tokens · cost not reported" in markdown
    assert markdown.rstrip().endswith("**Total:** 122 tokens · $0.0010 + not reported")
