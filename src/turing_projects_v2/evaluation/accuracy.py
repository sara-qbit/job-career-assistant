"""Deterministic accuracy measures: tool calling on hand-written agent cases, and the exact
fields of saved Cover Letters.

Tool Selection Accuracy: share of cases where the agent's first tool call used the expected tool.
Where it did not, a judge decides whether the call was a reasonable alternative; that gives a
second, more lenient accuracy. Parameter Accuracy: for the cases with the expected tool, the share
of expected arguments the call got right, averaged over those cases.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from turing_projects_v2.store import CoverLetter, JobPosting


class AgentCase(BaseModel):
    """A hand-written question with the tool call a correct agent makes first."""

    question: str
    expected_tool: str
    expected_args: dict[str, Any]


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]


class ToolVerdict(BaseModel):
    value: Literal["reasonable", "unreasonable"]
    reason: str


class ToolJudge(Protocol):
    def judge(self, case: AgentCase, call: ToolCall) -> ToolVerdict:
        """Whether `call` is a reasonable way to answer `case.question`."""
        ...


class CaseResult(BaseModel):
    case: AgentCase
    call: ToolCall | None
    """The agent's first tool call; None if it answered without a tool."""
    tool_correct: bool
    alternative: ToolVerdict | None
    """The judge's verdict on a different tool; None if the expected one (or none) was called."""
    parameters_correct: float | None
    """Share of expected arguments matched; None unless the expected tool was called."""


class AgentReport(BaseModel):
    results: list[CaseResult]
    tool_selection_accuracy: float
    tool_selection_accuracy_with_alternatives: float
    parameter_accuracy: float | None


def score_agent_cases(
    cases: Iterable[AgentCase], calls: Mapping[str, ToolCall | None], judge: ToolJudge
) -> AgentReport:
    """Score each case against `calls` (the agent's first tool call per question)."""
    results = []
    for case in cases:
        call = calls.get(case.question)
        correct = call is not None and call.name == case.expected_tool
        results.append(
            CaseResult(
                case=case,
                call=call,
                tool_correct=correct,
                alternative=judge.judge(case, call) if call is not None and not correct else None,
                parameters_correct=_parameters_correct(case, call) if correct else None,
            )
        )
    parameter_scores = [r.parameters_correct for r in results if r.parameters_correct is not None]
    reasonable = [r for r in results if r.alternative and r.alternative.value == "reasonable"]
    total = len(results) or 1
    return AgentReport(
        results=results,
        tool_selection_accuracy=sum(r.tool_correct for r in results) / total,
        tool_selection_accuracy_with_alternatives=(
            sum(r.tool_correct for r in results) + len(reasonable)
        )
        / total,
        parameter_accuracy=(
            sum(parameter_scores) / len(parameter_scores) if parameter_scores else None
        ),
    )


def _parameters_correct(case: AgentCase, call: ToolCall | None) -> float:
    if not case.expected_args:
        return 1.0
    args = call.args if call else {}
    matched = sum(_same(args.get(name), expected) for name, expected in case.expected_args.items())
    return matched / len(case.expected_args)


def _same(actual: Any, expected: Any) -> bool:
    """Equal; strings ignoring case and surrounding or repeated spaces."""
    if isinstance(actual, str) and isinstance(expected, str):
        return " ".join(actual.lower().split()) == " ".join(expected.lower().split())
    return actual == expected


FIELDS: dict[str, Callable[[JobPosting], str | None]] = {
    "job name": lambda posting: posting.title,
    "company": lambda posting: posting.company_name,
    "work type": lambda posting: posting.formatted_work_type,
}


class FieldReport(BaseModel):
    letters: int
    per_letter: list[dict[str, Any]]
    """Per letter: its job ID and, per field, whether it contains the exact value (None if the
    posting does not state the field)."""
    accuracy: dict[str, float | None]
    """Per field, the share of letters containing the posting's exact value; None if no
    letter's posting states the field."""


def field_accuracy(
    letters: Iterable[CoverLetter], posting: Callable[[int], JobPosting | None]
) -> FieldReport:
    """How often saved letters contain each exact field of their Job Posting."""
    letters = list(letters)
    found: dict[str, list[bool]] = {name: [] for name in FIELDS}
    per_letter: list[dict[str, Any]] = []
    for letter in letters:
        source = posting(letter.job_id)
        if source is None:
            continue
        matches: dict[str, Any] = {"job_id": letter.job_id}
        for name, field in FIELDS.items():
            value = field(source)
            matches[name] = (value in letter.text) if value else None
            if value:
                found[name].append(matches[name])
        per_letter.append(matches)
    return FieldReport(
        letters=len(letters),
        per_letter=per_letter,
        accuracy={name: (sum(hits) / len(hits) if hits else None) for name, hits in found.items()},
    )
