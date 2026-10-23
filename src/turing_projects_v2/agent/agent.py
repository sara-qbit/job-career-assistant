"""The chat agent: LangChain `create_agent` with the domain system prompt and the job tools."""

from typing import Any

import openai
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRetryMiddleware,
    ToolErrorMiddleware,
    hook_config,
)
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from turing_projects_v2.agent.memory import ConversationMemoryMiddleware
from turing_projects_v2.errors import AppError
from turing_projects_v2.prompts import load_prompt

# Failures worth retrying; anything else (e.g. a bad request) fails at once.
TRANSIENT_API_ERRORS = (
    openai.APIConnectionError,  # includes timeouts
    openai.RateLimitError,
    openai.InternalServerError,
)

# The tool result for a call over the cap, and the answer if the LLM keeps calling tools anyway.
TOOL_CALL_CAP_RESULT = (
    "Tool-call limit reached for this question. Do not call any more tools; "
    "answer the candidate with what you have so far."
)
TOOL_CALL_CAP_ANSWER = (
    "I stopped because answering this needed too many steps. Please ask again more "
    "specifically, for example one search or one job at a time."
)


def build_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver,
    *,
    history_window: int,
    max_tool_calls: int,
) -> CompiledStateGraph:
    """The agent; each Conversation is one checkpointer thread.

    Only the last `history_window` messages are sent to the LLM; at most `max_tool_calls` tool
    calls run per question.
    """
    return create_agent(
        llm,
        tools,
        system_prompt=load_prompt("system"),
        middleware=[
            ConversationMemoryMiddleware(history_window),
            ToolCallCapMiddleware(max_tool_calls),
            ToolErrorMiddleware(_tool_error_for_llm),
            ModelRetryMiddleware(max_retries=3, retry_on=TRANSIENT_API_ERRORS, on_failure="error"),
        ],
        checkpointer=checkpointer,
    )


def _tool_error_for_llm(error: Exception, request: ToolCallRequest) -> str | None:
    """The tool result the LLM sees when a tool fails as expected; None lets the error propagate."""
    if isinstance(error, AppError):
        return f"Error: {error.user_message}"
    return None


class ToolCallCapMiddleware(AgentMiddleware):
    """Stops a looping agent after `cap` tool calls per question.

    Calls over the cap do not run; the LLM gets TOOL_CALL_CAP_RESULT instead and one more chance
    to answer. If it calls a tool again, the turn ends with TOOL_CALL_CAP_ANSWER.
    """

    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = cap

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        calls = messages[-1].tool_calls if isinstance(messages[-1], AIMessage) else []
        if not calls:
            return None
        question = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
        results = [m for m in messages[question:-1] if isinstance(m, ToolMessage)]
        stopped = [result for result in results if result.text == TOOL_CALL_CAP_RESULT]
        executed = len(results) - len(stopped)
        blocked = calls[max(self._cap - executed, 0) :]
        if not blocked:
            return None
        if not stopped:
            return {"messages": [_cap_result(call) for call in blocked]}
        return {
            "messages": [*(_cap_result(call) for call in calls), AIMessage(TOOL_CALL_CAP_ANSWER)],
            "jump_to": "end",
        }


def _cap_result(call: ToolCall) -> ToolMessage:
    return ToolMessage(
        TOOL_CALL_CAP_RESULT, tool_call_id=call["id"], name=call["name"], status="error"
    )
