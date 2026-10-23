import json

import httpx
from langchain_core.messages import HumanMessage

from turing_projects_v2.llm import OpenRouterChat

USAGE = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15, "cost": 0.00042}


def openrouter(handler):
    return OpenRouterChat(
        model="openai/gpt-6-luna",
        api_key="test-key",
        base_url="https://openrouter.test/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        stream_usage=True,
    )


def test_cost_reported_by_openrouter_is_kept():
    def answer(request):
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-6-luna",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": USAGE,
            },
        )

    message = openrouter(answer).invoke([HumanMessage("hello")])

    assert message.response_metadata["cost"] == 0.00042
    assert message.usage_metadata["input_tokens"] == 12


def test_cost_reported_by_openrouter_is_kept_when_streaming():
    def chunk(delta, finish_reason=None, usage=None):
        body = {
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "openai/gpt-6-luna",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage:
            body |= {"choices": [], "usage": usage}
        return f"data: {json.dumps(body)}\n\n"

    def answer(request):
        events = [
            chunk({"role": "assistant", "content": "Hi"}),
            chunk({}, finish_reason="stop"),
            chunk({}, usage=USAGE),
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content="".join(events)
        )

    chunks = list(openrouter(answer).stream([HumanMessage("hello")]))
    message = sum(chunks[1:], chunks[0])

    assert message.text == "Hi"
    assert message.response_metadata["cost"] == 0.00042
    assert message.usage_metadata["output_tokens"] == 3
