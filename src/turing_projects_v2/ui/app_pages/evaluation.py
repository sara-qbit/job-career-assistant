"""Evaluation: everything behind the answers of the open Conversation, read-only.

History and export, the RAG process and sources of each search, token usage and cost, tool
calls, the security measures and log, and the system prompts.
"""

import json

import pandas as pd
import streamlit as st

from turing_projects_v2.agent.chat import CitationsEvent, TextEvent, ToolCallEvent
from turing_projects_v2.config import load_settings
from turing_projects_v2.errors import AppError
from turing_projects_v2.export import conversation_json, conversation_markdown
from turing_projects_v2.prompts import load_prompt, prompt_names
from turing_projects_v2.security import EXCERPT_LENGTH, INJECTION_PATTERNS, MAX_MESSAGE_LENGTH
from turing_projects_v2.store import EvaluationRun
from turing_projects_v2.ui.answer import AnswerView, escape_markdown
from turing_projects_v2.ui.session import EMBEDDING_MODEL_SHORT_NAMES, chat_service

state = st.session_state
st.title("Evaluation")
try:
    service = chat_service()
    conversation = service.conversation(state.conversation_id)
    titles = {summary.id: summary.title for summary in service.conversations()}
except AppError as error:
    st.error(str(error))
    st.stop()
title = titles.get(state.conversation_id, "New Conversation")
st.caption(f"Conversation: {escape_markdown(title)}")
turns = list(enumerate(conversation.turns, start=1))
searches = [
    (number, event)
    for number, turn in turns
    for event in turn.events
    if isinstance(event, CitationsEvent) and event.trace is not None
]

st.subheader("Conversation history")
if not turns:
    st.info("No questions in this Conversation yet. Ask one on MAIN.")
for number, turn in turns:
    with st.expander(f"{number}. {turn.question}"):
        answer = "".join(event.text for event in turn.events if isinstance(event, TextEvent))
        st.markdown(escape_markdown(answer) or "_(no answer text)_")
with st.container(horizontal=True):
    file_name = f"conversation-{state.conversation_id}"
    st.download_button(
        "Markdown",
        conversation_markdown(conversation, title),
        file_name=f"{file_name}.md",
        mime="text/markdown",
        icon=":material/download:",
    )
    st.download_button(
        "JSON",
        conversation_json(conversation),
        file_name=f"{file_name}.json",
        mime="application/json",
        icon=":material/download:",
    )

st.subheader("RAG process")
if not searches:
    st.info("No search in this Conversation yet.")
for number, event in searches:
    trace = event.trace
    with st.expander(f"Question {number}: {trace.question}", expanded=True):
        filters = trace.filters.model_dump(exclude_none=True)
        st.markdown(
            f"**Search text:** {escape_markdown(trace.search_text) or '-'}  \n"
            f"**Filters:** {escape_markdown(json.dumps(filters)) if filters else 'none'}"
        )
        st.markdown(
            "**Queries** (the search text first, then the variants):\n"
            + "\n".join(
                f"- {escape_markdown(stage.query)}: {len(stage.job_ids)} postings"
                for stage in trace.queries
            )
        )
        per_query = pd.DataFrame(
            [
                {"query": stage.query, "rank": rank, "job ID": job_id, "distance": distance}
                for stage in trace.queries
                for rank, (job_id, distance) in enumerate(
                    zip(stage.job_ids, stage.distances, strict=True), start=1
                )
            ]
        )
        st.dataframe(per_query, hide_index=True, alt="Postings found per query")
        reranked = dict(trace.reranked or [])
        fusion = pd.DataFrame(
            [
                {
                    "fusion rank": rank,
                    "job ID": job_id,
                    "RRF score": score,
                    "reranker score": reranked.get(job_id),
                }
                for rank, (job_id, score) in enumerate(trace.fusion, start=1)
            ]
        )
        st.markdown(
            "**Fusion** (Reciprocal Rank Fusion)"
            + (" **and reranker scores**" if trace.reranked is not None else ", reranker off")
        )
        st.dataframe(fusion, hide_index=True, alt="Fused ranking with reranker scores")

st.subheader("Sources")
if not searches:
    st.info("No search in this Conversation yet.")
for number, event in searches:
    st.markdown(f"**Question {number}:** {escape_markdown(event.trace.question)}")
    score_name = "reranker score" if event.trace.reranked is not None else "RRF score"
    if not event.trace.sources:
        st.caption("Matched chunks were not recorded for searches before the Evaluation tab.")
    results = {result.job_id: result for result in event.results}
    for source in event.trace.sources:
        result = results[source.job_id]
        with st.container(border=True):
            name = escape_markdown(result.title)
            link = f"[{name}]({result.url})" if result.url else name
            st.markdown(
                f"**[{result.number}] {link}** · job ID {result.job_id} · "
                f"distance {source.distance:.3f} · {score_name} {result.score:.4f}"
            )
            st.code(source.text, language=None, wrap_lines=True)

st.subheader("Token usage and cost")
per_answer = pd.DataFrame(
    [
        {
            "#": number,
            "question": turn.question,
            "LLM calls": len(turn.usage),
            "tokens": turn.total.tokens,
            "cost (USD)": turn.total.cost,
            "all costs reported": turn.total.cost_known,
        }
        for number, turn in turns
    ]
)
if turns:
    st.dataframe(per_answer, hide_index=True, alt="Token usage and cost per answer")
total = conversation.total
with st.container(horizontal=True):
    st.metric("Total tokens", f"{total.tokens:,}", border=True)
    st.metric(
        "Total cost",
        f"${total.cost:.4f}" + ("" if total.cost_known else " + not reported"),
        border=True,
        help="Includes LLM calls of refused messages, which are not in the history. "
        "'+ not reported': a provider did not report the cost of some calls.",
    )

st.subheader("Tool calls")
calls = [
    (number, event, turn)
    for number, turn in turns
    for event in turn.events
    if isinstance(event, ToolCallEvent)
]
if not calls:
    st.info("No tool calls in this Conversation yet.")
for number, call, turn in calls:
    result, card = turn.tool_result(call.call_id)
    failed = result is not None and result.is_error
    icon = ":material/error:" if failed else ":material/build:"
    with st.expander(f"Question {number}: {call.name}", icon=icon):
        st.markdown("**Arguments**")
        st.json(call.args)
        st.markdown("**Result**" + (" (error)" if failed else ""))
        st.code(result.content if result else "(no result)", language=None, wrap_lines=True)
        if isinstance(card, CitationsEvent):
            st.markdown(
                "\n".join(
                    f"{r.number}. [{escape_markdown(r.title)}]({r.url})"
                    if r.url
                    else f"{r.number}. {escape_markdown(r.title)}"
                    for r in card.results
                )
            )
        elif card is not None:
            AnswerView({}).show(card)  # the same card as on MAIN: Job Info, letter, chart, …

st.subheader("Security")
settings = load_settings()
st.markdown(
    f"- **Input validation:** empty messages, messages over {MAX_MESSAGE_LENGTH:,} characters "
    "and control characters are rejected.\n"
    f"- **Prompt-injection check:** {len(INJECTION_PATTERNS)} patterns; matches are flagged and "
    "logged, and the system prompt's rules still apply.\n"
    "- **Scope guard:** an LLM check refuses requests that are not about job search.\n"
    f"- **Tool-call cap:** at most {settings.max_tool_calls} tool calls per question.\n"
    f"- **Memory window:** only the last {settings.history_window} messages reach the LLM.\n"
    "- **Personal data:** the test_question's details reach only the Cover Letter call and are "
    f"redacted from log excerpts (at most {EXCERPT_LENGTH} characters)."
)
log = service.security_log()
if log:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "time (UTC)": entry.created_at,
                    "this Conversation": entry.conversation_id == state.conversation_id,
                    "event": entry.event_type,
                    "excerpt": entry.excerpt,
                }
                for entry in reversed(log)
            ]
        ),
        hide_index=True,
        alt="Security log, newest first",
    )
else:
    st.info("The security log is empty.")

st.subheader("System prompts")
for name in prompt_names():
    with st.expander(name):
        st.code(load_prompt(name), language="markdown", wrap_lines=True)

st.subheader("RAGAs test set")
store = service.store
test_questions = store.test_questions()


def set_approval(question_id: int, key: str) -> None:
    store.set_test_approval(question_id, st.session_state[key])


if not test_questions:
    st.info("No test set yet. Generate one with `uv run turing-projects-v2 evaluate generate`.")
else:
    approved = sum(test_question.approved is True for test_question in test_questions)
    st.caption(
        f"{approved} of {len(test_questions)} generated questions approved; only approved "
        "questions are evaluated. Hover a question for its reference answer."
    )
    for test_question in test_questions:
        key = f"approve-{test_question.id}"
        st.checkbox(
            test_question.question,
            value=test_question.approved is True,
            key=key,
            help=test_question.reference[:500],
            on_change=set_approval,
            args=(test_question.id, key),
        )

st.subheader("Evaluation results")
runs = store.evaluation_runs()
latest: dict[str, EvaluationRun] = {}
for run in runs:  # oldest first, so the latest run of each configuration wins
    latest[run.kind + json.dumps(run.config, sort_keys=True)] = run
if not runs:
    st.info(
        "No evaluation runs yet. Run `uv run turing-projects-v2 evaluate rag` (and `agent`, "
        "`fields`)."
    )

rag_runs = [run for run in latest.values() if run.kind == "rag"]
if rag_runs:
    st.markdown(
        "**RAG pipeline** (RAGAs, mean over the approved questions; 1 is best): embedding "
        "models with and without the reranker"
    )
    rag = pd.DataFrame(
        [
            {
                # Short labels: the chart's axis cuts long ones off.
                "configuration": EMBEDDING_MODEL_SHORT_NAMES.get(
                    run.config["embedding_model"], run.config["embedding_model"]
                )
                + (" + reranker" if run.config["rerank"] else ""),
                **{name.replace("_", " "): value for name, value in run.metrics.items()},
            }
            for run in sorted(
                rag_runs, key=lambda run: (run.config["embedding_model"], not run.config["rerank"])
            )
        ]
    ).set_index("configuration")
    # Categorical slots 1-3 of the validated chart palette (blue, orange, aqua), in fixed order;
    # aqua is below 3:1 contrast on white, so the table below carries the exact values.
    st.bar_chart(
        rag,
        alt="RAGAs faithfulness, context precision and context recall per configuration",
        stack=False,
        horizontal=True,
        x_label="",
        y_label="score (0 to 1)",
        color=["#2a78d6", "#eb6834", "#1baf7a"][: len(rag.columns)],
    )
    st.dataframe(rag, alt="RAGAs metrics per configuration")
    st.caption(
        f"Latest run per configuration, {len(rag_runs[0].cases)} questions each. Query "
        "rewriting and judging use LLMs, so repeated runs differ by up to about 0.1-0.2: "
        "treat smaller differences as noise. All configurations search with the same "
        "rewritten queries per question. Faithfulness is measured on short answers written from "
        "the retrieved chunks only (prompt `rag_answer`), not on the chat agent's answers."
    )

agent_run = next((run for run in reversed(runs) if run.kind == "agent"), None)
if agent_run:
    st.markdown(
        f"**Agent: Tool Selection Accuracy and Parameter Accuracy** "
        f"({agent_run.config.get('cases', len(agent_run.cases))} hand-written cases)"
    )
    with st.container(horizontal=True):
        for name, label in [
            ("tool_selection_accuracy", "Tool Selection Accuracy"),
            ("tool_selection_accuracy_with_alternatives", "incl. reasonable alternatives"),
            ("parameter_accuracy", "Parameter Accuracy"),
        ]:
            value = agent_run.metrics.get(name)
            st.metric(label, "-" if value is None else f"{value:.0%}", border=True)
    if agent_run.cases:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "question": case["question"],
                        "expected tool": case["case"]["expected_tool"],
                        "called tool": (case["call"] or {}).get("name", "(none)"),
                        "parameters correct": case["parameters_correct"],
                        "judge": (case["alternative"] or {}).get("value"),
                    }
                    for case in agent_run.cases
                ]
            ),
            hide_index=True,
            alt="Agent cases with expected and called tools",
        )

    st.caption(
        "Only the agent's first tool call is scored; the tool is not run. Arguments must match "
        "exactly (strings ignoring case and spacing), so a reworded search request counts as wrong."
    )

fields_run = next((run for run in reversed(runs) if run.kind == "fields"), None)
if fields_run:
    st.markdown(
        f"**Cover Letters: field accuracy** (exact job name, company and work type in "
        f"{fields_run.config.get('letters', 0)} saved letters)"
    )
    with st.container(horizontal=True):
        for name, value in fields_run.metrics.items():
            st.metric(name.capitalize(), "-" if value is None else f"{value:.0%}", border=True)
    if fields_run.cases and "job name" in fields_run.cases[0]:
        st.dataframe(
            pd.DataFrame(fields_run.cases),
            hide_index=True,
            alt="Which exact fields each saved Cover Letter contains",
        )
