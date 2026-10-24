"""Evaluation: everything behind the answers of the open Conversation, read-only.

History and export, the RAG process and sources of each search, token usage and cost, tool
calls, the security measures and log, and the system prompts.
"""

import json

import pandas as pd
import streamlit as st

from turing_projects_v2.agent.chat import CitationsEvent, TextEvent, ToolCallEvent, ToolResultEvent
from turing_projects_v2.config import load_settings
from turing_projects_v2.errors import AppError
from turing_projects_v2.export import conversation_json, conversation_markdown
from turing_projects_v2.prompts import load_prompt, prompt_names
from turing_projects_v2.security import EXCERPT_LENGTH, INJECTION_PATTERNS, MAX_MESSAGE_LENGTH
from turing_projects_v2.ui.answer import escape_markdown
from turing_projects_v2.ui.session import chat_service

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
    results = {result.job_id: result for result in event.results}
    for source in event.trace.sources:
        result = results[source.job_id]
        with st.container(border=True):
            name = escape_markdown(result.title)
            link = f"[{name}]({result.url})" if result.url else name
            st.markdown(
                f"**[{result.number}] {link}** · job ID {result.job_id} · "
                f"distance {source.distance:.3f} · score {result.score:.4f}"
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
            "cost (USD)": turn.total.cost if turn.total.cost_known else None,
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
        f"${total.cost:.4f}" if total.cost_known else "not reported",
        border=True,
        help="Includes LLM calls of refused messages, which are not in the history.",
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
    result = next(
        (
            event
            for event in turn.events
            if isinstance(event, ToolResultEvent) and event.call_id == call.call_id
        ),
        None,
    )
    failed = result is not None and result.is_error
    icon = ":material/error:" if failed else ":material/build:"
    with st.expander(f"Question {number}: {call.name}", icon=icon):
        st.markdown("**Arguments**")
        st.json(call.args)
        st.markdown("**Result**" + (" (error)" if failed else ""))
        st.code(result.content if result else "(no result)", language=None, wrap_lines=True)

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
    "- **Personal data:** the candidate's details reach only the Cover Letter call and are "
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
