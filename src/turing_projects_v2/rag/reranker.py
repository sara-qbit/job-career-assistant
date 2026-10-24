"""Reranker: re-scores the best fused Search Results by reading query and posting text together."""

from typing import Protocol

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """How well each passage answers `query`; higher is better."""
        ...


class CrossEncoderReranker:
    """The local cross-encoder; downloaded (about 90 MB) on first use."""

    def __init__(self) -> None:
        # Imported here: loading sentence-transformers (and torch) takes several seconds.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(CROSS_ENCODER_MODEL)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(score) for score in self._model.predict([(query, p) for p in passages])]
