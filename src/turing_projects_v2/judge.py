"""LLM-as-a-judge: decides whether a generated statement is supported by its source."""

import os
from typing import Any, Literal, Protocol

import openai
from pydantic import BaseModel

from turing_projects_v2.config import Settings
from turing_projects_v2.errors import DataValidationError, ExternalAPIError
from turing_projects_v2.prompts import load_prompt


class Verdict(BaseModel):
    value: Literal["supported", "fabricated"]
    reason: str


class Judge(Protocol):
    def check(self, statement: str, source: str) -> Verdict: ...


class RagasJudge:
    """The judge as a RAGAs `DiscreteMetric` (supported / fabricated, with a reason).

    `llm` is a RAGAs LLM (see `open_judge`); scripted RAGAs LLMs work too.
    """

    def __init__(self, llm: Any) -> None:
        # RAGAs takes about 13 s to import, so only the real judge loads it; it sends usage
        # telemetry unless told not to.
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        from ragas.metrics import DiscreteMetric

        self._llm = llm
        self._metric = DiscreteMetric(
            name="supported_by_source",
            allowed_values=["supported", "fabricated"],
            prompt=load_prompt("judge"),
        )

    def check(self, statement: str, source: str) -> Verdict:
        """The verdict on `statement`; API failures raise ExternalAPIError, an answer that is
        not a valid verdict raises DataValidationError."""
        from instructor.core.exceptions import InstructorError

        try:
            result = self._metric.score(llm=self._llm, statement=statement, source=source)
        except (openai.APIError, InstructorError) as error:
            # Instructor wraps API failures, too, in its own error after its retries.
            if _caused_by_api(error):
                raise ExternalAPIError(f"Judge request failed: {error}") from error
            raise DataValidationError("Judge answer is not a valid verdict") from error
        return Verdict(value=result.value, reason=result.reason or "")


def _caused_by_api(error: BaseException | None) -> bool:
    """Whether `error` is, or was raised from, an OpenAI-client API failure."""
    while error is not None:
        if isinstance(error, openai.APIError):
            return True
        error = error.__cause__ or error.__context__
    return False


def open_judge(settings: Settings) -> RagasJudge:
    """The judge on OpenRouter's `settings.judge_model`."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    client = openai.OpenAI(
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        timeout=60,
        max_retries=2,
    )
    from ragas.llms import llm_factory

    return RagasJudge(llm_factory(settings.judge_model, client=client, temperature=0))
