import json
import time

import httpx
import openai
import pytest

from turing_projects_v2.config import Settings
from turing_projects_v2.errors import ConfigError, ExternalAPIError
from turing_projects_v2.rag.embeddings import OpenRouterEmbeddings, load_embedder


def vector_of(text):
    return [float(len(text)), 1.0]


def embedding_api(requests, failures=()):
    """An OpenRouter embeddings endpoint that records requests and answers `failures` first."""
    failures = list(failures)

    def answer(request):
        body = json.loads(request.content)
        requests.append(body)
        if failures:
            return failures.pop(0)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": n, "embedding": vector_of(text)}
                    for n, text in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    return answer


def embedder(handler, batch_size=3, sleeps=None, request_timeout=10):
    client = openai.OpenAI(
        api_key="test-key",
        base_url="https://openrouter.test/api/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return OpenRouterEmbeddings(
        client,
        "qwen/qwen3-embedding-8b",
        batch_size=batch_size,
        sleep=(sleeps if sleeps is not None else []).append,
        request_timeout=request_timeout,
    )


def test_documents_are_sent_in_batches_and_come_back_in_order():
    requests = []
    texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"]

    vectors = embedder(embedding_api(requests), batch_size=3).embed_documents(texts)

    assert [request["input"] for request in requests] == [
        ["a", "bb", "ccc"],
        ["dddd", "eeeee", "ffffff"],
        ["ggggggg"],
    ]
    assert all(request["model"] == "qwen/qwen3-embedding-8b" for request in requests)
    assert vectors == [
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [5.0, 1.0],
        [6.0, 1.0],
        [7.0, 1.0],
    ]


def test_rate_limits_and_server_errors_are_retried_with_growing_waits():
    requests, sleeps = [], []
    rate_limited = httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
    unavailable = httpx.Response(503, json={"error": {"message": "Provider unavailable"}})
    api = embedding_api(requests, failures=[rate_limited, unavailable])

    vectors = embedder(api, sleeps=sleeps).embed_documents(["a", "bb"])

    assert vectors == [[1.0, 1.0], [2.0, 1.0]]
    assert len(requests) == 3
    assert len(sleeps) == 2
    assert 0 < sleeps[0] < sleeps[1]


def test_persistent_rate_limit_raises_the_api_error_type():
    requests = []
    rate_limited = httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
    api = embedding_api(requests, failures=[rate_limited] * 10)

    with pytest.raises(ExternalAPIError):
        embedder(api).embed_documents(["a"])
    assert 1 < len(requests) < 10


def test_rejected_request_raises_the_api_error_type_without_retrying():
    requests = []
    unauthorized = httpx.Response(401, json={"error": {"message": "No auth credentials found"}})

    with pytest.raises(ExternalAPIError):
        embedder(embedding_api(requests, failures=[unauthorized])).embed_documents(["a"])
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("name", "model"),
    [("gemini", "google/gemini-embedding-2"), ("qwen3", "qwen/qwen3-embedding-8b")],
)
def test_cloud_models_are_selectable_by_name(name, model):
    settings = Settings(openrouter_api_key="test-key", _env_file=None)

    selected = load_embedder(settings, name)

    assert isinstance(selected, OpenRouterEmbeddings)
    assert selected.model == model


def test_unknown_model_name_is_a_config_error():
    settings = Settings(openrouter_api_key="test-key", _env_file=None)

    with pytest.raises(ConfigError, match="gemini, qwen3"):
        load_embedder(settings, "word2vec")


def ok_response(data):
    return httpx.Response(
        200, json={"object": "list", "model": "qwen/qwen3-embedding-8b", "data": data}
    )


MALFORMED_RESPONSES = {
    "error body with status 200": httpx.Response(
        200, json={"error": {"message": "Provider returned error", "code": 502}}
    ),
    "fewer vectors than texts": ok_response(
        [{"object": "embedding", "index": 0, "embedding": [1.0, 1.0]}]
    ),
    "duplicate indexes": ok_response(
        [
            {"object": "embedding", "index": 0, "embedding": [1.0, 1.0]},
            {"object": "embedding", "index": 0, "embedding": [2.0, 1.0]},
        ]
    ),
}


@pytest.mark.parametrize("malformed", MALFORMED_RESPONSES.values(), ids=MALFORMED_RESPONSES)
def test_malformed_response_is_retried(malformed):
    requests = []

    vectors = embedder(embedding_api(requests, failures=[malformed])).embed_documents(["a", "bb"])

    assert vectors == [[1.0, 1.0], [2.0, 1.0]]
    assert len(requests) == 2


@pytest.mark.parametrize("malformed", MALFORMED_RESPONSES.values(), ids=MALFORMED_RESPONSES)
def test_persistently_malformed_response_raises_the_api_error_type(malformed):
    api = embedding_api([], failures=[malformed] * 10)

    with pytest.raises(ExternalAPIError):
        embedder(api).embed_documents(["a", "bb"])


@pytest.mark.parametrize("status", [408, 409])
def test_request_timeout_and_conflict_are_retried(status):
    requests = []
    failure = httpx.Response(status, json={"error": {"message": "Try again"}})

    vectors = embedder(embedding_api(requests, failures=[failure])).embed_documents(["a"])

    assert vectors == [[1.0, 1.0]]
    assert len(requests) == 2


def test_rate_limit_waits_as_long_as_retry_after_asks():
    sleeps = []
    rate_limited = httpx.Response(
        429, headers={"Retry-After": "30"}, json={"error": {"message": "Rate limit exceeded"}}
    )

    embedder(embedding_api([], failures=[rate_limited]), sleeps=sleeps).embed_documents(["a"])

    assert sleeps == [30]


def test_request_that_hangs_is_abandoned_and_retried():
    requests = []
    answer = embedding_api(requests)

    def hang_once(request):
        if not requests:
            requests.append("hung")
            time.sleep(2)
        return answer(request)

    started = time.perf_counter()
    vectors = embedder(hang_once, request_timeout=0.2).embed_documents(["a"])

    assert vectors == [[1.0, 1.0]]
    assert time.perf_counter() - started < 1.5
