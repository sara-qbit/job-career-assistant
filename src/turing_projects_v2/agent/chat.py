"""Chat service: one turn sends a message to a Conversation and yields a stream of answer events.

This is the seam the UI talks to; tests drive it with a scripted fake LLM.
"""

import queue
from collections.abc import Callable, Iterator
from functools import cache
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field, computed_field

from turing_projects_v2.agent.agent import TOOL_CALL_CAP_RESULT, build_agent
from turing_projects_v2.agent.memory import open_checkpointer
from turing_projects_v2.config import Settings
from turing_projects_v2.errors import AppError, ExternalAPIError
from turing_projects_v2.judge import Judge, open_judge
from turing_projects_v2.llm import chat_model
from turing_projects_v2.profile import CandidateProfile, load_profile
from turing_projects_v2.rag.embeddings import load_embedder
from turing_projects_v2.rag.indexer import KnowledgeBase, open_collection
from turing_projects_v2.rag.reranker import CrossEncoderReranker, Reranker
from turing_projects_v2.rag.retriever import (
    SearchOutcome,
    SearchResult,
    SearchSettings,
    SearchTrace,
)
from turing_projects_v2.security import (
    EXCERPT_LENGTH,
    LLMScopeGuard,
    ScopeGuard,
    SecurityEventType,
    input_problem,
    looks_like_injection,
    redacted,
)
from turing_projects_v2.store import (
    ConversationSummary,
    CoverLetter,
    JobInfo,
    JobStore,
    SecurityLogEntry,
    TokenUsage,
    UsageTotal,
)
from turing_projects_v2.tools.jobs import JobDescription, job_tools
from turing_projects_v2.tools.stats import MarketStats, market_stats_tool

TOOL_STATUS = {
    "search_jobs": "Searching the Knowledge Base",
    "get_job_description": "Fetching the Job Description",
    "create_job_info": "Creating the Job Info and checking it",
    "write_cover_letter": "Writing the Cover Letter and checking it",
    "job_market_stats": "Computing job market statistics",
}


TITLE_LENGTH = 60
"""Longest Conversation title; the first question is cut at a word to fit."""


OFF_TOPIC_ANSWER = (
    "Sorry, I can only help with job search and job applications: finding Job Postings, "
    "explaining jobs and companies, writing Cover Letters and job market statistics."
)


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
    """The numbered Search Results that [n] markers in the answer refer to, and how the search
    found them (for the Evaluation tab)."""

    kind: Literal["citations"] = "citations"
    results: list[SearchResult]
    trace: SearchTrace | None = None


class JobDescriptionEvent(BaseModel):
    kind: Literal["job_description"] = "job_description"
    description: JobDescription


class JobInfoEvent(BaseModel):
    kind: Literal["job_info"] = "job_info"
    info: JobInfo


class CoverLetterEvent(BaseModel):
    kind: Literal["cover_letter"] = "cover_letter"
    letter: CoverLetter


class MarketStatsEvent(BaseModel):
    kind: Literal["market_stats"] = "market_stats"
    stats: MarketStats


class SecurityEvent(BaseModel):
    """A security measure acted on the candidate's message; `message` explains it to them."""

    kind: Literal["security"] = "security"
    event_type: SecurityEventType
    message: str


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
    | JobInfoEvent
    | CoverLetterEvent
    | MarketStatsEvent
    | SecurityEvent
    | UsageEvent
    | ErrorEvent,
    Field(discriminator="kind"),
]


class Turn(BaseModel):
    """One question of a stored Conversation, the events of its answer and their token usage."""

    question: str
    events: list[AnswerEvent]
    usage: list[TokenUsage] = []

    @computed_field
    @property
    def total(self) -> UsageTotal:
        return UsageTotal.of(self.usage)


class Conversation(BaseModel):
    """A stored Conversation as it is restored on reopening."""

    turns: list[Turn]
    search_results: list[SearchResult]
    """The latest numbered Search Results, which "#n" refers to."""
    current_job_id: int | None
    usage: list[TokenUsage] = []
    """Every LLM call of the Conversation, also those of refused messages."""

    @computed_field
    @property
    def total(self) -> UsageTotal:
        return UsageTotal.of(self.usage)


class ChatService:
    def __init__(
        self,
        llm: BaseChatModel,
        store: JobStore,
        knowledge_base: Callable[[str], KnowledgeBase],
        judge: Judge,
        profile: CandidateProfile,
        *,
        scope_guard: ScopeGuard,
        reranker: Reranker | None = None,
        default_search: SearchSettings | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        history_window: int = 20,
        max_tool_calls: int = 6,
    ) -> None:
        """`knowledge_base(name)` opens the Knowledge Base of embedding model `name`;
        `default_search` is used for turns sent without search settings (SearchSettings() if
        None); `checkpointer` stores the Conversations (in memory if None)."""
        self._default_search = default_search or SearchSettings()
        self._store = store
        self._profile = profile
        self._scope_guard = scope_guard
        self._agent = build_agent(
            llm,
            [
                *job_tools(store, llm, knowledge_base, judge, profile, reranker),
                market_stats_tool(store),
            ],
            checkpointer or InMemorySaver(),
            history_window=history_window,
            max_tool_calls=max_tool_calls,
        )

    def send(
        self, conversation_id: str, message: str, search: SearchSettings | None = None
    ) -> Iterator[AnswerEvent]:
        """Run one turn of the Conversation; never raises for expected failures.

        `search` is the embedding model and reranker choice for this turn's searches.

        Token usage of every LLM call (also those inside tools) is recorded in the store and
        yielded as a UsageEvent.
        """
        problem = input_problem(message)
        if problem is not None:
            yield self._security_event(conversation_id, "rejected_input", problem, message)
            return
        if looks_like_injection(message):
            yield self._security_event(
                conversation_id,
                "prompt_injection",
                "Your message contains phrases often used to manipulate chatbots. It was flagged "
                "and logged; the assistant still follows its rules.",
                message,
            )
        usage_events: queue.SimpleQueue[UsageEvent] = queue.SimpleQueue()
        turn_id = uuid4().hex  # also the question message's ID, which links usage to its answer
        recorder = _UsageRecorder(conversation_id, turn_id, self._store, usage_events)
        try:
            in_scope = self._scope_guard.check(message, {"callbacks": [recorder]}).on_topic
        except AppError as error:
            yield ErrorEvent(message=error.user_message)
            return
        if not in_scope:
            yield self._security_event(
                conversation_id, "off_topic", "Off-topic request: refused and logged.", message
            )
            yield TextEvent(text=OFF_TOPIC_ANSWER)
            yield from _drained(usage_events)
            return
        self._store.touch_conversation(conversation_id, _title(message))
        config = _thread(conversation_id)
        config["configurable"]["search"] = (search or self._default_search).model_dump()
        config["callbacks"] = [recorder]
        capped = False
        stream = self._agent.stream(
            {"messages": [HumanMessage(message, id=turn_id)]},
            config,
            stream_mode=["messages", "updates"],
        )
        try:
            for mode, data in stream:
                yield from _drained(usage_events)
                if mode == "messages":
                    yield from _text_events(*data)
                else:
                    for node, update in data.items():
                        for update_message in (update or {}).get("messages", []):
                            yield from _update_events(update_message)
                            reached_cap = (
                                isinstance(update_message, ToolMessage)
                                and update_message.text == TOOL_CALL_CAP_RESULT
                            )
                            if reached_cap and not capped:
                                capped = True
                                yield self._security_event(
                                    conversation_id,
                                    "tool_call_cap",
                                    "This question needed too many tool calls; the assistant "
                                    "was stopped to save time and cost.",
                                    message,
                                )
                            # Answers not written by the model node (e.g. the tool-call cap's)
                            # are not streamed as chunks.
                            if node != "model" and isinstance(update_message, AIMessage):
                                yield TextEvent(text=update_message.text)
        except AppError as error:
            yield ErrorEvent(message=error.user_message)
        except openai.APIError:
            yield ErrorEvent(message=ExternalAPIError.user_message)
        except GraphRecursionError:
            yield ErrorEvent(
                message="I could not finish this request. Please rephrase it or split it up."
            )
        yield from _drained(usage_events)

    def _security_event(
        self,
        conversation_id: str,
        event_type: SecurityEventType,
        explanation: str,
        message: str,
    ) -> SecurityEvent:
        """Log the event with an excerpt of the message, and return it for the candidate."""
        excerpt = redacted(message, self._profile)[:EXCERPT_LENGTH]
        self._store.log_security_event(
            SecurityLogEntry(
                conversation_id=conversation_id, event_type=event_type, excerpt=excerpt
            )
        )
        return SecurityEvent(event_type=event_type, message=explanation)

    def conversations(self) -> list[ConversationSummary]:
        """Every Conversation, most recently used first."""
        return self._store.conversations()

    def security_log(self) -> list[SecurityLogEntry]:
        """Every security event of every Conversation, oldest first."""
        return self._store.security_log()

    def conversation(self, conversation_id: str) -> Conversation:
        """The stored Conversation (empty if unknown): its turns and its latest Search Results.

        Restored answers have no usage events; each turn's token usage is in `Turn.usage`.
        """
        state = self._agent.get_state(_thread(conversation_id)).values
        usage = self._store.token_usage(conversation_id)
        turns: list[Turn] = []
        for message in state.get("messages", []):
            if isinstance(message, HumanMessage):
                turn_usage = [u for u in usage if u.turn_id is not None and u.turn_id == message.id]
                turns.append(Turn(question=message.text, events=[], usage=turn_usage))
            elif turns:
                turns[-1].events += _update_events(message)
                if isinstance(message, AIMessage) and message.text:
                    turns[-1].events.append(TextEvent(text=message.text))
        return Conversation(
            turns=turns,
            search_results=state.get("search_results", []),
            current_job_id=state.get("current_job_id"),
            usage=usage,
        )


def open_chat_service(settings: Settings) -> ChatService:
    """The Chat service on the real store, Knowledge Base, Conversations and agent model."""

    @cache
    def knowledge_base(name: str) -> KnowledgeBase:
        # First the embedder: an unknown name must not create a collection.
        embedder = load_embedder(settings, name)
        return KnowledgeBase(open_collection(settings.chroma_dir, name), embedder)

    knowledge_base(settings.embedding_model)  # fail at start, not on the first search
    llm = chat_model(settings, settings.agent_model)
    return ChatService(
        llm,
        JobStore(settings.database_path),
        knowledge_base,
        open_judge(settings),
        load_profile(settings.candidate_dir),
        scope_guard=LLMScopeGuard(llm),
        reranker=CrossEncoderReranker(),
        default_search=SearchSettings(embedding_model=settings.embedding_model),
        checkpointer=open_checkpointer(settings.conversations_path),
        history_window=settings.history_window,
        max_tool_calls=settings.max_tool_calls,
    )


def _title(question: str) -> str:
    question = " ".join(question.split())
    if len(question) <= TITLE_LENGTH:
        return question
    return question[: TITLE_LENGTH - 1].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"


def _drained(events: queue.SimpleQueue[UsageEvent]) -> Iterator[UsageEvent]:
    while not events.empty():
        yield events.get()


def _thread(conversation_id: str) -> dict[str, Any]:
    """The agent config that selects the Conversation's checkpointer thread."""
    return {"configurable": {"thread_id": conversation_id}}


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
            yield CitationsEvent(results=outcome.results, trace=outcome.trace)
        elif message.name == "get_job_description":
            yield JobDescriptionEvent(description=JobDescription.model_validate(message.artifact))
        elif message.name == "create_job_info":
            yield JobInfoEvent(info=JobInfo.model_validate(message.artifact))
        elif message.name == "write_cover_letter":
            yield CoverLetterEvent(letter=CoverLetter.model_validate(message.artifact))
        elif message.name == "job_market_stats":
            yield MarketStatsEvent(stats=MarketStats.model_validate(message.artifact))


class _UsageRecorder(BaseCallbackHandler):
    """Records the token usage and cost of every LLM call of one turn."""

    def __init__(
        self,
        conversation_id: str,
        turn_id: str,
        store: JobStore,
        events: queue.SimpleQueue[UsageEvent],
    ) -> None:
        self._conversation_id = conversation_id
        self._turn_id = turn_id
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
            turn_id=self._turn_id,
        )
        self._store.record_token_usage(usage)
        self._events.put(UsageEvent(usage=usage))
