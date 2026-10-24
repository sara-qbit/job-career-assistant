"""Embedding providers behind one interface (LangChain `Embeddings`), selected by name."""

import time
from collections.abc import Callable

import openai
from langchain_core.embeddings import Embeddings

from turing_projects_v2.config import Settings
from turing_projects_v2.errors import ConfigError, ExternalAPIError

MAX_ATTEMPTS = 5
"""Tries per batch; the waits between them are 1, 2, 4 and 8 seconds."""

TRANSIENT_ERRORS = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)


class OpenRouterEmbeddings(Embeddings):
    """An embedding model served by OpenRouter; texts are sent `batch_size` at a time."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        batch_size: int = 64,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.sleep = sleep

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
                response = self.client.embeddings.create(
                    model=self.model, input=texts, encoding_format="float"
                )
            except TRANSIENT_ERRORS as error:
                if attempt == MAX_ATTEMPTS - 1:
                    raise ExternalAPIError(
                        f"{self.model} embeddings failed after {MAX_ATTEMPTS} tries: {error}"
                    ) from error
                self.sleep(2**attempt)
                continue
            except openai.APIError as error:
                raise ExternalAPIError(f"{self.model} embeddings failed: {error}") from error
            return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        raise AssertionError("unreachable")


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
