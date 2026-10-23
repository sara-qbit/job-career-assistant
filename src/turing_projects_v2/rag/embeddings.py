"""Embedding providers behind one interface (LangChain `Embeddings`), selected by name."""

from collections.abc import Callable

from langchain_core.embeddings import Embeddings

from turing_projects_v2.errors import ConfigError


def _minilm() -> Embeddings:
    # Imported here: loading sentence-transformers (and torch) takes several seconds.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


EMBEDDING_MODELS: dict[str, Callable[[], Embeddings]] = {
    "minilm": _minilm,
}


def load_embedder(name: str) -> Embeddings:
    """The embedding model called `name` (one of EMBEDDING_MODELS)."""
    try:
        factory = EMBEDDING_MODELS[name]
    except KeyError:
        raise ConfigError(
            f"Unknown embedding model '{name}' ({', '.join(EMBEDDING_MODELS)})"
        ) from None
    return factory()
