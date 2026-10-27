"""Conversation memory: the agent state behind "#n" and the sliding window sent to the LLM.

The checkpointer stores the full message history; the LLM only sees the recent window plus the
latest numbered Search Results and the current job ID, so "#n" resolves to the exact job ID even
after the search itself has left the window.
"""

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from turing_projects_v2.rag.indexer import NOT_STATED
from turing_projects_v2.rag.retriever import SearchResult


def _latest[T](old: T, new: T) -> T:
    # Parallel tool calls may update the same key in one step; the last update wins.
    return new


class ConversationState(AgentState):
    search_results: NotRequired[Annotated[list[dict[str, Any]], _latest]]
    """The numbered Search Results of the latest search (JSON dicts of SearchResult)."""
    current_job_id: NotRequired[Annotated[int | None, _latest]]
    """The job the Conversation is currently about."""


class ConversationMemoryMiddleware(AgentMiddleware[ConversationState]):
    """Keeps Search Results and the current job ID in state and sends only a window to the LLM."""

    state_schema = ConversationState

    def __init__(self, window: int) -> None:
        super().__init__()
        self._window = window

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        system = request.system_prompt or ""
        memory = memory_prompt(request.state)
        return handler(
            request.override(
                messages=recent_window(request.messages, self._window),
                system_message=SystemMessage(f"{system}\n\n{memory}".strip()),
            )
        )

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        result = handler(request)
        if not isinstance(result, ToolMessage) or result.status == "error":
            return result
        name = request.tool_call["name"]
        if name == "search_jobs" and result.artifact is not None:
            update = {"search_results": result.artifact["results"]}
        elif name in (
            "get_job_description",
            "create_job_info",
            "write_cover_letter",
            "change_job_facts",
        ):
            update = {"current_job_id": request.tool_call["args"]["job_id"]}
        else:
            return result
        return Command(update={"messages": [result], **update})


def recent_window(messages: list[AnyMessage], size: int) -> list[AnyMessage]:
    """The last `size` messages, starting at a question so tool calls keep their results.

    A single turn longer than the window is kept whole.
    """
    if len(messages) <= size:
        return messages
    questions = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if not questions:
        return messages
    start = next((i for i in questions if i >= len(messages) - size), questions[-1])
    return messages[start:]


def memory_prompt(state: dict[str, Any]) -> str:
    """The latest Search Results and current job ID, for the end of the system prompt."""
    results = [SearchResult.model_validate(result) for result in state.get("search_results", [])]
    lines = ['Latest Search Results ("#n" refers to these):']
    if results:
        lines.append("<job_posting_data>")
        lines += [
            f"#{result.number} = job ID {result.job_id}: {result.title} at "
            f"{result.company or NOT_STATED}"
            for result in results
        ]
        lines.append("</job_posting_data>")
    else:
        lines.append("none yet")
    current = state.get("current_job_id")
    lines.append(f"Current job: job ID {current}" if current is not None else "Current job: none")
    return "\n".join(lines)


def open_checkpointer(path: Path) -> SqliteSaver:
    """The SQLite checkpointer that stores every Conversation's messages and state in `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Streamlit reruns the page in different threads, all sharing the one cached service.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
