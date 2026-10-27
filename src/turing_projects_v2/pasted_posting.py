"""Pasted Job Postings: a job ad the candidate pastes is checked to be a job ad and stored.

The ad text is stored unchanged. Position and employer are taken from the LLM's answer only
when they appear word for word in the ad.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from turing_projects_v2.errors import UserInputError
from turing_projects_v2.llm import ask_structured
from turing_projects_v2.prompts import load_prompt
from turing_projects_v2.store import JobPosting, JobStore

UNKNOWN_POSITION = "Pasted job ad"
"""The title of a pasted Job Posting whose position could not be copied from the ad."""


class JobAdCheck(BaseModel):
    is_job_ad: bool = Field(description="whether the text is an advertisement for a job vacancy")
    reason: str = Field(description="one short sentence")
    position: str = Field(description='the job title, copied word for word from the ad, or ""')
    employer: str = Field(
        description='the hiring company or university, copied word for word from the ad, or ""'
    )


class NotAJobAdError(UserInputError):
    """The pasted text is not a job ad; nothing was stored."""


def store_job_ad(
    text: str, store: JobStore, llm: BaseChatModel, config: RunnableConfig | None = None
) -> tuple[JobPosting, bool]:
    """The pasted Job Posting for the ad `text`, and whether it had been pasted before.

    Raises NotAJobAdError if the text is not a job ad.
    """
    existing = store.pasted_posting(text)
    if existing is not None:
        return existing, True
    check = ask_structured(
        llm,
        JobAdCheck,
        [
            SystemMessage(load_prompt("job_ad_check")),
            HumanMessage(f"<job_posting_data>\n{text}\n</job_posting_data>"),
        ],
        config,
    )
    if not check.is_job_ad:
        raise NotAJobAdError(
            f"This text does not look like a job ad ({check.reason.rstrip('.')}), so I did not "
            "store it. Please paste the full text of a job ad."
        )
    title = _copied(check.position, text) or UNKNOWN_POSITION
    return store.add_pasted_posting(text, title, _copied(check.employer, text)), False


def _copied(value: str, text: str) -> str | None:
    """`value` if it appears word for word in `text`, else None."""
    value = value.strip()
    return value if value and value in text else None
