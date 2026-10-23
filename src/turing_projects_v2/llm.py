"""Chat models served by OpenRouter through LangChain's OpenAI-compatible client."""

from collections.abc import Sequence

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from turing_projects_v2.config import Settings
from turing_projects_v2.errors import DataValidationError, ExternalAPIError


def chat_model(settings: Settings, model: str) -> BaseChatModel:
    """The OpenRouter chat model called `model` (e.g. `settings.agent_model`)."""
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def ask_structured[T: BaseModel](
    llm: BaseChatModel, schema: type[T], messages: Sequence[BaseMessage]
) -> T:
    """The LLM's answer to `messages` as a `schema` object.

    API failures (after the client's own retries) raise ExternalAPIError; an answer that does
    not fit the schema raises DataValidationError.
    """
    try:
        answer = llm.with_structured_output(schema).invoke(list(messages))
    except openai.APIError as error:
        raise ExternalAPIError(f"{schema.__name__} request failed: {error}") from error
    except (OutputParserException, PydanticValidationError) as error:
        raise DataValidationError(f"LLM answer is not a valid {schema.__name__}") from error
    if not isinstance(answer, schema):
        raise DataValidationError(f"LLM answer is not a valid {schema.__name__}: {answer!r}")
    return answer
