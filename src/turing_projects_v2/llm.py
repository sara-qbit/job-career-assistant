"""Chat models served by OpenRouter through LangChain's OpenAI-compatible client."""

from collections.abc import Sequence
from typing import Any

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from turing_projects_v2.config import Settings
from turing_projects_v2.errors import DataValidationError, ExternalAPIError


class OpenRouterChat(ChatOpenAI):
    """ChatOpenAI that keeps the cost OpenRouter reports for a call (in USD).

    OpenRouter adds `cost` to the usage of every response; LangChain drops it, so it is copied
    to the answer message's `response_metadata["cost"]`, streamed or not.
    """

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        cost = ((result.llm_output or {}).get("token_usage") or {}).get("cost")
        if cost is not None:
            for generation in result.generations:
                generation.message.response_metadata["cost"] = cost
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        cost = (chunk.get("usage") or {}).get("cost")
        if generation is not None and cost is not None:
            generation.message.response_metadata["cost"] = cost
        return generation


def chat_model(settings: Settings, model: str) -> BaseChatModel:
    """The OpenRouter chat model called `model` (e.g. `settings.agent_model`)."""
    return OpenRouterChat(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        timeout=60,
        max_retries=2,
        stream_usage=True,
    )


def ask_structured[T: BaseModel](
    llm: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage],
    config: RunnableConfig | None = None,
) -> T:
    """The LLM's answer to `messages` as a `schema` object (`config` e.g. adds callbacks).

    API failures (after the client's own retries) raise ExternalAPIError; an answer that does
    not fit the schema raises DataValidationError.
    """
    try:
        answer = llm.with_structured_output(schema).invoke(list(messages), config)
    except openai.APIError as error:
        raise ExternalAPIError(f"{schema.__name__} request failed: {error}") from error
    except (OutputParserException, PydanticValidationError) as error:
        raise DataValidationError(f"LLM answer is not a valid {schema.__name__}") from error
    if not isinstance(answer, schema):
        raise DataValidationError(f"LLM answer is not a valid {schema.__name__}: {answer!r}")
    return answer
