"""Deterministic stand-ins for the LLM, the embedder and the judge.

Tests inject these so they run free, offline and repeatably.
"""

import hashlib
from collections.abc import Iterable
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr

from turing_projects_v2.judge import Verdict


class FakeChatModel(BaseChatModel):
    """Chat model that returns scripted responses in order and records every call.

    A response may be a string, an AIMessage (e.g. with tool calls), or a Pydantic
    object returned from `with_structured_output`.
    """

    responses: list[Any]
    _next_index: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """The messages sent on each call, in order."""
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        response = self._next_response(messages)
        message = AIMessage(content=response) if isinstance(response, str) else response
        if not isinstance(message, AIMessage):
            raise TypeError(f"Scripted response {message!r} is structured output, not a message")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def respond(prompt: Any) -> Any:
            messages = prompt.to_messages() if hasattr(prompt, "to_messages") else prompt
            return self._next_response(messages)

        return RunnableLambda(respond)

    def _next_response(self, messages: Any) -> str | AIMessage | BaseModel:
        if self._next_index >= len(self.responses):
            raise AssertionError("FakeChatModel ran out of scripted responses")
        self._calls.append(messages)
        response = self.responses[self._next_index]
        self._next_index += 1
        return response


class FakeEmbedder(Embeddings):
    """Embeds text to a fixed vector derived from its hash; equal texts get equal vectors.

    Records every document text it embeds in `embedded`.
    """

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255 for i in range(self.dimensions)]


class FakeJudge:
    """Judge that returns scripted verdicts in order, then `default` once they run out."""

    def __init__(
        self,
        verdicts: Iterable[Verdict] = (),
        default: Verdict = Verdict(value="supported", reason="fake judge default"),  # noqa: B008
    ) -> None:
        self._verdicts = list(verdicts)
        self.default = default
        self.checked: list[tuple[str, str]] = []

    def check(self, statement: str, source: str) -> Verdict:
        self.checked.append((statement, source))
        return self._verdicts.pop(0) if self._verdicts else self.default
