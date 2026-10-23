"""Job Info: the structured, verifiable summary of one Job Posting behind `create_job_info`.

Job name, company name and work type are copied exactly from the data; only the role summary and
the company vision are written by the LLM.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from turing_projects_v2.errors import UserInputError
from turing_projects_v2.judge import Judge, Verdict
from turing_projects_v2.llm import ask_structured
from turing_projects_v2.prompts import load_prompt
from turing_projects_v2.store import JobInfo, JobStore

NOT_STATED = "not stated in the posting"


class JobSummaries(BaseModel):
    role_summary: str = Field(
        description=f'what the job is about, in 2-3 sentences, or "{NOT_STATED}"'
    )
    company_vision: str = Field(
        description=f'what the company does and aims for, in 1-2 sentences, or "{NOT_STATED}"'
    )


def create_job_info(job_id: int, store: JobStore, llm: BaseChatModel, judge: Judge) -> JobInfo:
    """The Job Info of posting `job_id`: the saved one, or a new one that is then saved."""
    saved = store.get_job_info(job_id)
    if saved is not None:
        return saved
    posting = store.get_posting(job_id)
    if posting is None:
        raise UserInputError(f"No Job Posting has job ID {job_id}.")
    company = store.get_company(posting.company_id) if posting.company_id is not None else None
    role_source = _text(posting.description)
    vision_source = _text(company.description if company else None)
    summaries = JobSummaries(role_summary=NOT_STATED, company_vision=NOT_STATED)
    if role_source or vision_source:
        blocks = [
            f"<job_posting_data>\n{role_source or NOT_STATED}\n</job_posting_data>",
            f"<company_data>\n{vision_source or NOT_STATED}\n</company_data>",
        ]
        answer = ask_structured(
            llm,
            JobSummaries,
            [SystemMessage(load_prompt("job_info")), HumanMessage("\n\n".join(blocks))],
        )
        # A summary of a missing source can only be invented, whatever the LLM wrote.
        if role_source:
            summaries.role_summary = answer.role_summary
        if vision_source:
            summaries.company_vision = answer.company_vision
    info = JobInfo(
        job_id=job_id,
        job_name=posting.title,
        company_name=posting.company_name,
        work_type=posting.formatted_work_type,
        role_summary=summaries.role_summary,
        company_vision=summaries.company_vision,
        role_summary_verdict=_judged(judge, summaries.role_summary, role_source, "the role"),
        company_vision_verdict=_judged(
            judge, summaries.company_vision, vision_source, "the company's vision"
        ),
    )
    store.save_job_info(info)
    return info


def _judged(judge: Judge, summary: str, source: str | None, topic: str) -> Verdict | None:
    """The judge's verdict on `summary` (about `topic`); None when there is no source to check.

    A summary that says the source does not state the topic is checked as exactly that claim.
    """
    if source is None:
        return None
    if summary.strip() == NOT_STATED:
        summary = f"The source says nothing about {topic}."
    return judge.check(summary, source)


def _text(source: str | None) -> str | None:
    """`source` as given, or None if it has no text."""
    return source if source and source.strip() else None
