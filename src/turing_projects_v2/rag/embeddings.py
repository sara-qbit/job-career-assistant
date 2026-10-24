"""Embedding providers behind one interface (LangChain `Embeddings`), selected by name."""

import threading
import time
from collections.abc import Callable

import openai
from langchain_core.embeddings import Embeddings

from turing_projects_v2.config import Settings
from turing_projects_v2.errors import ConfigError, ExternalAPIError

MAX_ATTEMPTS = 5
"""Tries per batch; the waits between them are 1, 2, 4 and 8 seconds (or what Retry-After asks)."""

TRANSIENT_STATUS_CODES = {408, 409, 429}
"""Besides these, every 5xx status is retried too."""

MAX_RETRY_AFTER = 60
"""Longest wait (seconds) taken from a `Retry-After` header."""


class MalformedResponseError(Exception):
    """An embeddings response that does not hold exactly one vector per text sent."""


class RequestTimeoutError(Exception):
    """An embeddings request that got no complete answer within the request timeout."""


class OpenRouterEmbeddings(Embeddings):
    """An embedding model served by OpenRouter; texts are sent `batch_size` at a time.

    Retries are done here rather than by the OpenAI client, so that they can be tested without
    waiting and so that a request is given up after `request_timeout` seconds in total: the
    client's own timeout restarts whenever a byte arrives, so a slow answer can take forever.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        batch_size: int = 64,
        sleep: Callable[[float], None] = time.sleep,
        request_timeout: float = 120,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.sleep = sleep
        self.request_timeout = request_timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors += self._embed_batch(texts[start : start + self.batch_size])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Vectors for one request's worth of texts; transient failures are retried.

        A failure that persists, or one that retrying cannot fix, raises ExternalAPIError.
        """
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._request_in_time(texts)
            except (openai.APIError, MalformedResponseError, RequestTimeoutError) as error:
                if not _is_transient(error):
                    raise ExternalAPIError(f"{self.model} embeddings failed: {error}") from error
                if attempt == MAX_ATTEMPTS - 1:
                    raise ExternalAPIError(
                        f"{self.model} embeddings failed after {MAX_ATTEMPTS} tries: {error}"
                    ) from error
                self.sleep(_wait_before_retry(error, attempt))
        raise AssertionError("unreachable")

    def _request_in_time(self, texts: list[str]) -> list[list[float]]:
        """`_request`, given up after `request_timeout` seconds.

        The request runs on a daemon thread; one that is given up is left to finish on its own.
        """
        vectors: list[list[float]] = []
        failures: list[BaseException] = []

        def run() -> None:
            try:
                vectors.extend(self._request(texts))
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(self.request_timeout)
        if worker.is_alive():
            raise RequestTimeoutError(f"no answer within {self.request_timeout:g}s")
        if failures:
            raise failures[0]
        return vectors

    def _request(self, texts: list[str]) -> list[list[float]]:
        """One embeddings request; a response without exactly one vector per text is malformed.

        OpenRouter reports some provider failures as status 200 with an error body.
        """
        response = self.client.embeddings.create(
            model=self.model, input=texts, encoding_format="float"
        )
        data = getattr(response, "data", None) or []
        if sorted(item.index for item in data) != list(range(len(texts))):
            raise MalformedResponseError(
                f"expected {len(texts)} vectors, got {len(data)}: {str(response)[:200]}"
            )
        return [item.embedding for item in sorted(data, key=lambda item: item.index)]


def _is_transient(error: Exception) -> bool:
    """Whether trying the same request again may succeed."""
    if isinstance(error, openai.APIStatusError):
        return error.status_code in TRANSIENT_STATUS_CODES or error.status_code >= 500
    return isinstance(
        error, (openai.APIConnectionError, MalformedResponseError, RequestTimeoutError)
    )


def _wait_before_retry(error: Exception, attempt: int) -> float:
    """Seconds to wait after failed try `attempt` (0-based): 1, 2, 4, 8, or longer if the
    server's `Retry-After` header asks for it (up to MAX_RETRY_AFTER)."""
    backoff = 2**attempt
    if isinstance(error, openai.APIStatusError):
        try:
            return max(backoff, min(float(error.response.headers["retry-after"]), MAX_RETRY_AFTER))
        except KeyError, ValueError:
            pass
    return backoff


def _minilm(settings: Settings) -> Embeddings:
    # Imported here: loading sentence-transformers (and torch) takes several seconds.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def _openrouter(model: str) -> Callable[[Settings], Embeddings]:
    def factory(settings: Settings) -> Embeddings:
        client = openai.OpenAI(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            timeout=60,
            max_retries=0,  # OpenRouterEmbeddings retries itself
        )
        return OpenRouterEmbeddings(client, model)

    return factory


EMBEDDING_MODELS: dict[str, Callable[[Settings], Embeddings]] = {
    "minilm": _minilm,
    "gemini": _openrouter("google/gemini-embedding-2"),
    "qwen3": _openrouter("qwen/qwen3-embedding-8b"),
}


def load_embedder(settings: Settings, name: str) -> Embeddings:
    """The embedding model called `name` (one of EMBEDDING_MODELS)."""
    try:
        factory = EMBEDDING_MODELS[name]
    except KeyError:
        raise ConfigError(
            f"Unknown embedding model '{name}' ({', '.join(EMBEDDING_MODELS)})"
        ) from None
    return factory(settings)
