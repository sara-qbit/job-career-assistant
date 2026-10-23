"""The chat agent: LangChain `create_agent` with the domain system prompt and the job tools."""

import openai
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolErrorMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from turing_projects_v2.errors import AppError
from turing_projects_v2.prompts import load_prompt

# Failures worth retrying; anything else (e.g. a bad request) fails at once.
TRANSIENT_API_ERRORS = (
    openai.APIConnectionError,  # includes timeouts
    openai.RateLimitError,
    openai.InternalServerError,
)


def build_agent(
    llm: BaseChatModel, tools: list[BaseTool], checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    """The agent; each Conversation is one checkpointer thread."""
    return create_agent(
        llm,
        tools,
        system_prompt=load_prompt("system"),
        middleware=[
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
