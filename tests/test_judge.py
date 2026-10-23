import httpx
import openai
import pytest
from instructor.core.exceptions import InstructorRetryException

from turing_projects_v2.errors import DataValidationError, ExternalAPIError
from turing_projects_v2.judge import RagasJudge, Verdict


class ScriptedRagasLLM:
    """A RAGAs LLM that answers every prompt with one verdict and records the prompts."""

    def __init__(self, value="supported", reason="", error=None):
        self.value, self.reason, self.error = value, reason, error
        self.prompts = []

    def generate(self, prompt, response_model):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return response_model(value=self.value, reason=self.reason)


def test_ragas_judge_returns_the_verdict_on_the_statement_and_its_source():
    llm = ScriptedRagasLLM("fabricated", "The source does not mention Mars.")
    judge = RagasJudge(llm)

    verdict = judge.check("Acme flies to Mars.", "We build {rockets}.")

    assert verdict == Verdict(value="fabricated", reason="The source does not mention Mars.")
    (prompt,) = llm.prompts
    assert "<statement_data>\nAcme flies to Mars.\n</statement_data>" in prompt
    assert "<source_data>\nWe build {rockets}.\n</source_data>" in prompt


def test_judge_api_failure_is_an_external_api_error():
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://openrouter.ai"))
    judge = RagasJudge(ScriptedRagasLLM(error=timeout))

    with pytest.raises(ExternalAPIError):
        judge.check("Acme builds rockets.", "We build rockets.")


def test_judge_api_failure_after_instructor_retries_is_an_external_api_error():
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://openrouter.ai"))
    wrapped = InstructorRetryException(str(timeout), n_attempts=3, total_usage=0)
    wrapped.__cause__ = timeout
    judge = RagasJudge(ScriptedRagasLLM(error=wrapped))

    with pytest.raises(ExternalAPIError):
        judge.check("Acme builds rockets.", "We build rockets.")


def test_invalid_judge_answer_is_a_data_validation_error():
    invalid = InstructorRetryException("no valid verdict", n_attempts=3, total_usage=0)
    judge = RagasJudge(ScriptedRagasLLM(error=invalid))

    with pytest.raises(DataValidationError):
        judge.check("Acme builds rockets.", "We build rockets.")
