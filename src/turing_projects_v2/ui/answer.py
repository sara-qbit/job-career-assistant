"""Shows one chatbot answer in the MAIN page, event by event, as the Chat service yields them.

Replaying the stored events of an earlier answer renders it the same way.
"""

import re

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from turing_projects_v2.agent.chat import (
    AnswerEvent,
    CitationsEvent,
    ErrorEvent,
    JobDescriptionEvent,
    JobInfoEvent,
    StatusEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from turing_projects_v2.job_info import NOT_STATED
from turing_projects_v2.judge import Verdict
from turing_projects_v2.rag.retriever import SearchResult
from turing_projects_v2.store import JobInfo

CITATION = re.compile(r"\[(\d+)\](?!\()")


def link_citations(text: str, citations: dict[int, SearchResult]) -> str:
    """Markdown for `text` with each [n] marker linked to Search Result n's original posting."""

    def link(match: re.Match[str]) -> str:
        result = citations.get(int(match.group(1)))
        if result is None or not result.url:
            return match.group(0)
        return f'[\\[{result.number}\\]]({result.url} "{result.title}")'

    return CITATION.sub(link, escape_markdown(text))


def escape_markdown(text: str) -> str:
    """`text` for Streamlit Markdown: a "$" would start a LaTeX formula (salaries contain many)."""
    return text.replace("$", r"\$")


def show_job_info(info: JobInfo) -> None:
    """The Job Info card: exact fields, then each summary with the judge's verdict."""
    with st.container(border=True):
        st.markdown(f"**Job Info: {escape_markdown(info.job_name)}**")
        st.markdown(
            f"**Company:** {escape_markdown(info.company_name or NOT_STATED)}  \n"
            f"**Work type:** {escape_markdown(info.work_type or NOT_STATED)}"
        )
        st.caption("Job name, company and work type are copied exactly from the posting.")
        summaries = [
            ("Role summary", info.role_summary, info.role_summary_verdict),
            ("Company vision", info.company_vision, info.company_vision_verdict),
        ]
        for name, summary, verdict in summaries:
            st.markdown(f"**{name}:** {escape_markdown(summary)}")
            show_verdict(verdict)


def show_verdict(verdict: Verdict | None) -> None:
    if verdict is None:
        st.caption("Not checked: the source states nothing to summarise.")
    elif verdict.value == "supported":
        st.success(
            f"Judge: supported. {escape_markdown(verdict.reason)}", icon=":material/check_circle:"
        )
    else:
        st.warning(
            f"Judge: fabricated. {escape_markdown(verdict.reason)}", icon=":material/warning:"
        )


class AnswerView:
    """Renders an answer into the current container: text, status lines, boxes, usage."""

    def __init__(self, citations: dict[int, SearchResult]) -> None:
        self.citations = citations
        self._text = ""
        self._text_area: DeltaGenerator | None = None
        self._pending_status: DeltaGenerator | None = None
        self._statuses: dict[str, DeltaGenerator] = {}
        self._tokens = 0
        self._cost = 0.0
        self._cost_known = True

    def show(self, event: AnswerEvent) -> None:
        match event:
            case TextEvent(text=text):
                if self._text_area is None:
                    self._text, self._text_area = "", st.empty()
                self._text += text
                self._text_area.markdown(link_citations(self._text, self.citations))
            case StatusEvent(text=text):
                self._text_area = None
                self._pending_status = st.status(text)
            case ToolCallEvent(call_id=call_id, args=args):
                status = self._pending_status or st.status(event.name)
                status.json(args)
                self._statuses[call_id] = status
                self._pending_status = None
            case ToolResultEvent(call_id=call_id, content=content, is_error=is_error):
                status = self._statuses.get(call_id) or st.status(event.name)
                status.text(content)
                status.update(state="error" if is_error else "complete")
            case CitationsEvent(results=results):
                self.citations = {result.number: result for result in results}
            case JobDescriptionEvent(description=description):
                self._text_area = None
                with st.container(border=True):
                    company = f" at {description.company}" if description.company else ""
                    st.markdown(f"**Job Description: {description.title}{company}**")
                    if description.url:
                        st.markdown(f"[Original posting]({description.url})")
                    # st.text would dedent and strip the text; st.code keeps it as stored
                    # (apart from one leading and one trailing newline).
                    st.code(description.text, language=None, wrap_lines=True)
            case JobInfoEvent(info=info):
                self._text_area = None
                show_job_info(info)
            case UsageEvent(usage=usage):
                self._tokens += usage.input_tokens + usage.output_tokens
                if usage.cost is None:
                    self._cost_known = False
                else:
                    self._cost += usage.cost
            case ErrorEvent(message=message):
                self._text_area = None
                st.error(message)

    def finish(self) -> None:
        """Show the answer's token usage and cost below it."""
        if self._tokens:
            cost = f"${self._cost:.4f}" if self._cost_known else "cost not reported"
            st.caption(f"{self._tokens:,} tokens · {cost}")
