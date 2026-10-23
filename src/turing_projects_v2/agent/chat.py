"""Chat service: one turn sends a message to a Conversation and yields a stream of answer events.

This is the seam the UI talks to; tests drive it with a scripted fake LLM.
"""

import queue
from collections.abc import Iterator
from typing import Annotated, Any, Literal
from uuid import UUID

import openai
from chromadb.api.models.Collection import Collection
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from turing_projects_v2.agent.agent import build_agent
from turing_projects_v2.config import Settings
from turing_projects_v2.errors import AppError, ExternalAPIError
from turing_projects_v2.llm import chat_model
from turing_projects_v2.rag.embeddings import load_embedder
from turing_projects_v2.rag.indexer import open_collection
from turing_projects_v2.rag.retriever import SearchOutcome, SearchResult
from turing_projects_v2.store import JobStore, TokenUsage
from turing_projects_v2.tools.jobs import JobDescription, job_tools

TOOL_STATUS = {
    "search_jobs": "Searching the Knowledge Base",
    "get_job_description": "Fetching the Job Description",
}


class TextEvent(BaseModel):
    """A piece of the answer text; may contain [n] citation markers."""

    kind: Literal["text"] = "text"
    text: str


class StatusEvent(BaseModel):
    """What the agent is doing right now (shown while a tool runs)."""

    kind: Literal["status"] = "status"
    text: str


class ToolCallEvent(BaseModel):
    kind: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    args: dict[str, Any]


class ToolResultEvent(BaseModel):
    kind: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    content: str
    is_error: bool


class CitationsEvent(BaseModel):
    """The numbered Search Results that [n] markers in the answer refer to."""

    kind: Literal["citations"] = "citations"
    results: list[SearchResult]


class JobDescriptionEvent(BaseModel):
    kind: Literal["job_description"] = "job_description"
    description: JobDescription


class UsageEvent(BaseModel):
    kind: Literal["usage"] = "usage"
    usage: TokenUsage


class ErrorEvent(BaseModel):
    """The turn failed; `message` is safe to show the candidate."""

    kind: Literal["error"] = "error"
    message: str


type AnswerEvent = Annotated[
    TextEvent
    | StatusEvent
    | ToolCallEvent
    | ToolResultEvent
    | CitationsEvent
    | JobDescriptionEvent
    | UsageEvent
    | ErrorEvent,
    Field(discriminator="kind"),
]


class ChatService:
    def __init__(
        self, llm: BaseChatModel, store: JobStore, collection: Collection, embedder: Embeddings
    ) -> None:
        self._store = store
        self._agent = build_agent(llm, job_tools(store, llm, collection, embedder), InMemorySaver())

    def send(self, conversation_id: str, message: str) -> Iterator[AnswerEvent]:
        """Run one turn of the Conversation; never raises for expected failures.

        Token usage of every LLM call (also those inside tools) is recorded in the store and
        yielded as a UsageEvent.
        """
        usage_events: queue.SimpleQueue[UsageEvent] = queue.SimpleQueue()
        recorder = _UsageRecorder(conversation_id, self._store, usage_events)
        config = {"configurable": {"thread_id": conversation_id}, "callbacks": [recorder]}
        stream = self._agent.stream(
            {"messages": [HumanMessage(message)]}, config, stream_mode=["messages", "updates"]
        )
        try:
            for mode, data in stream:
                while not usage_events.empty():
                    yield usage_events.get()
                if mode == "messages":
                    yield from _text_events(*data)
                else:
                    for update in data.values():
                        for update_message in (update or {}).get("messages", []):
                            yield from _update_events(update_message)
        except AppError as error:
            yield ErrorEvent(message=error.user_message)
        except openai.APIError:
            yield ErrorEvent(message=ExternalAPIError.user_message)
        except GraphRecursionError:
            yield ErrorEvent(
                message="I could not finish this request. Please rephrase it or split it up."
            )
        while not usage_events.empty():
            yield usage_events.get()


def open_chat_service(settings: Settings) -> ChatService:
    """The Chat service on the real store, Knowledge Base and OpenRouter agent model."""
    embedder = load_embedder(settings.embedding_model)
    return ChatService(
        chat_model(settings, settings.agent_model),
        JobStore(settings.database_path),
        open_collection(settings.chroma_dir, settings.embedding_model),
        embedder,
    )


def _text_events(chunk: Any, metadata: dict[str, Any]) -> Iterator[TextEvent]:
    # LLM calls inside tools (e.g. the Retriever's) stream too; only the agent's answer counts.
    if metadata.get("langgraph_node") != "model":
        return
    if isinstance(chunk, AIMessageChunk | AIMessage) and chunk.text:
        yield TextEvent(text=chunk.text)


def _update_events(message: Any) -> Iterator[AnswerEvent]:
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            yield StatusEvent(text=f"{TOOL_STATUS.get(call['name'], call['name'])} …")
            yield ToolCallEvent(call_id=call["id"] or "", name=call["name"], args=call["args"])
    elif isinstance(message, ToolMessage):
        yield ToolResultEvent(
            call_id=message.tool_call_id,
            name=message.name or "",
            content=message.text,
            is_error=message.status == "error",
        )
        if message.artifact is None or message.status == "error":
            return
        if message.name == "search_jobs":
            outcome = SearchOutcome.model_validate(message.artifact)
            yield CitationsEvent(results=outcome.results)
        elif message.name == "get_job_description":
            yield JobDescriptionEvent(description=JobDescription.model_validate(message.artifact))


class _UsageRecorder(BaseCallbackHandler):
    """Records the token usage and cost of every LLM call of one turn."""

    def __init__(
        self, conversation_id: str, store: JobStore, events: queue.SimpleQueue[UsageEvent]
    ) -> None:
        self._conversation_id = conversation_id
        self._store = store
        self._events = events
        self._models: dict[UUID, str] = {}

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        model = (kwargs.get("metadata") or {}).get("ls_model_name")
        if model:
            self._models[run_id] = model

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        message = getattr(response.generations[0][0], "message", None)
        if not isinstance(message, AIMessage) or message.usage_metadata is None:
            return
        model = (
            self._models.pop(run_id, None)
            or message.response_metadata.get("model_name")
            or "unknown"
        )
        usage = TokenUsage(
            conversation_id=self._conversation_id,
            model=model,
            input_tokens=message.usage_metadata["input_tokens"],
            output_tokens=message.usage_metadata["output_tokens"],
            cost=message.response_metadata.get("cost"),
        )
        self._store.record_token_usage(usage)
        self._events.put(UsageEvent(usage=usage))
