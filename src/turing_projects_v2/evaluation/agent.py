"""Agent evaluation: the hand-written cases through the agent model, scored for tool calling.

Only the agent's first tool call is needed, so the model is asked once with the real system
prompt and tool schemas, and the tool is not run: no searches, letters or other side effects.
"""

import tomllib
from collections.abc import Sequence
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from turing_projects_v2.evaluation.accuracy import AgentCase, ToolCall, ToolJudge, score_agent_cases
from turing_projects_v2.prompts import load_prompt
from turing_projects_v2.store import EvaluationRun, JobStore

AGENT_CASES = Path(__file__).parent / "agent_cases.toml"


def load_agent_cases(path: Path = AGENT_CASES) -> list[AgentCase]:
    """The hand-written agent cases (`[[case]]` tables with question, expected_tool and
    expected_args)."""
    with path.open("rb") as file:
        return [AgentCase.model_validate(case) for case in tomllib.load(file)["case"]]


def first_tool_call(
    llm: BaseChatModel, tools: Sequence[BaseTool], question: str
) -> ToolCall | None:
    """The tool call the agent model makes first for `question`; None if it answers directly."""
    message = llm.bind_tools(list(tools)).invoke(
        [SystemMessage(load_prompt("system")), HumanMessage(question)]
    )
    if isinstance(message, AIMessage) and message.tool_calls:
        call = message.tool_calls[0]
        return ToolCall(name=call["name"], args=call["args"])
    return None


def run_agent_evaluation(
    store: JobStore,
    cases: Sequence[AgentCase],
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    judge: ToolJudge,
) -> EvaluationRun:
    """Score the agent's first tool call on each case and store the run."""
    calls = {case.question: first_tool_call(llm, tools, case.question) for case in cases}
    report = score_agent_cases(cases, calls, judge)
    run = EvaluationRun(
        kind="agent",
        config={"cases": len(cases)},
        metrics={
            "tool_selection_accuracy": report.tool_selection_accuracy,
            "tool_selection_accuracy_with_alternatives": (
                report.tool_selection_accuracy_with_alternatives
            ),
            "parameter_accuracy": report.parameter_accuracy,
        },
        cases=[
            result.model_dump(mode="json") | {"question": result.case.question}
            for result in report.results
        ],
    )
    store.save_evaluation_run(run)
    return run
