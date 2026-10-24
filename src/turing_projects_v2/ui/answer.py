"""Shows one chatbot answer in the MAIN page, event by event, as the Chat service yields them.

Replaying the stored events of an earlier answer renders it the same way.
"""

import re

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from turing_projects_v2.agent.chat import (
    AnswerEvent,
    CitationsEvent,
    CoverLetterEvent,
    ErrorEvent,
    JobDescriptionEvent,
    JobInfoEvent,
    MarketStatsEvent,
    SecurityEvent,
    StatusEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from turing_projects_v2.documents import letter_docx, letter_pdf
from turing_projects_v2.job_info import NOT_STATED
from turing_projects_v2.judge import Verdict
from turing_projects_v2.rag.retriever import SearchResult
from turing_projects_v2.store import CoverLetter, JobInfo, TokenUsage, UsageTotal
from turing_projects_v2.tools.stats import MarketStats

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


def show_cover_letter(letter: CoverLetter) -> None:
    """The Cover Letter box: the letter, its check results and the downloads."""
    with st.container(border=True):
        st.markdown("**Cover Letter**")
        if letter.failed_checks or letter.fabricated:
            st.warning(
                "Check this letter by hand before sending it: "
                + "; ".join(
                    [f"{name} check failed" for name in letter.failed_checks]
                    + [f"{name} judged fabricated" for name in letter.fabricated]
                ),
                icon=":material/warning:",
            )
        # st.code keeps the letter's line breaks exactly as written.
        st.code(letter.text, language=None, wrap_lines=True)
        st.markdown("**Exact checks** (against the Job Info and the Letter Template)")
        for check in letter.checks:
            mark = ":material/check_circle:" if check.passed else ":material/cancel:"
            st.markdown(f"{mark} {check.name}")
        for name, verdict in letter.verdicts.items():
            st.markdown(f"**{name}** (written by the AI)")
            show_verdict(verdict)
        if letter.rewritten:
            st.caption("A gap judged fabricated was written a second time.")
        stamp = letter.created_at.strftime("%Y%m%d-%H%M%S")
        name = f"cover-letter-{letter.job_id}-{stamp}"
        left, right = st.columns(2)
        left.download_button(
            "Download .docx",
            letter_docx(letter.text),
            file_name=f"{name}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"{name}-docx",
            icon=":material/description:",
        )
        right.download_button(
            "Download PDF",
            letter_pdf(letter.text),
            file_name=f"{name}.pdf",
            mime="application/pdf",
            key=f"{name}-pdf",
            icon=":material/picture_as_pdf:",
        )


def show_market_stats(stats: MarketStats) -> None:
    """A statistic as a chart, with the postings it covers."""
    with st.container(border=True):
        st.markdown(f"**{escape_markdown(stats.title)}**")
        st.caption(f"Computed exactly over {stats.population}.")
        if not stats.rows:
            st.info("No postings match.")
            return
        table = pd.DataFrame([row.model_dump() for row in stats.rows])
        if stats.statistic != "salary_by_target_role":
            st.bar_chart(
                table,
                x="label",
                y="count",
                horizontal=True,
                sort="-count",
                x_label="Postings",
                y_label="",
            )
            return
        quartiles = {"p25": "25th percentile", "median": "Median", "p75": "75th percentile"}
        named = table.rename(columns={"label": "Target Role", "count": "Postings with salary"})
        named = named.rename(columns=quartiles)
        # Grouped bars show the middle half of salaries around the median for each role.
        st.bar_chart(
            named,
            x="Target Role",
            y=list(quartiles.values()),
            horizontal=True,
            stack=False,
            x_label="Yearly salary (USD)",
            y_label="",
        )
        st.dataframe(
            named,
            hide_index=True,
            column_config={
                name: st.column_config.NumberColumn(format="dollar") for name in quartiles.values()
            },
        )


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
        self._usage: list[TokenUsage] = []

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
            case CoverLetterEvent(letter=letter):
                self._text_area = None
                show_cover_letter(letter)
            case SecurityEvent(event_type="rejected_input", message=message):
                self._text_area = None
                st.error(message, icon=":material/block:")
            case SecurityEvent(message=message):
                self._text_area = None
                st.caption(f":material/shield: {escape_markdown(message)}")
            case MarketStatsEvent(stats=stats):
                self._text_area = None
                show_market_stats(stats)
            case UsageEvent(usage=usage):
                self._usage.append(usage)
            case ErrorEvent(message=message):
                self._text_area = None
                st.error(message)

    def finish(self) -> None:
        """Show the answer's token usage and cost below it."""
        total = UsageTotal.of(self._usage)
        if total.tokens:
            st.caption(total.describe())
