"""Job Facts: the facts of a pasted Job Posting, read from the ad as soon as it is pasted.

An LLM proposes each fact; deterministic checks then keep only what the ad backs up. Employer,
position, city, address and contact person must appear word for word in the ad. Employment
(Vollzeit/Teilzeit) and work mode (Präsenz/Online/Hybrid) are fixed German values whose backing
quote must appear in the ad. A fact failing its check becomes "nicht angegeben".
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from turing_projects_v2.errors import UserInputError
from turing_projects_v2.llm import ask_structured
from turing_projects_v2.pasted_posting import copied
from turing_projects_v2.profile import TargetRole
from turing_projects_v2.prompts import load_prompt
from turing_projects_v2.store import (
    NOT_STATED_DE,
    Employment,
    JobFacts,
    JobPosting,
    JobStore,
    Language,
    WorkMode,
)


class JobFactsProposal(BaseModel):
    employer: str = Field(
        description='the hiring company or university, copied word for word from the ad, or ""'
    )
    position: str = Field(description='the job title, copied word for word from the ad, or ""')
    city: str = Field(description='the city of the workplace, copied word for word, or ""')
    employment: Employment
    employment_quote: str = Field(
        description='the words of the ad that state full or part time, copied exactly, or ""'
    )
    work_mode: WorkMode
    work_mode_quote: str = Field(
        description="the words of the ad that state on-site, remote or hybrid work, copied "
        'exactly, or ""'
    )
    address: str = Field(
        description='the postal address applications go to, copied word for word, or ""'
    )
    contact_person: str = Field(
        description='the contact person for applications, copied word for word, or ""'
    )
    target_role: TargetRole
    language: Language


def create_job_facts(
    posting: JobPosting, store: JobStore, llm: BaseChatModel, config: RunnableConfig | None = None
) -> JobFacts:
    """The Job Facts of the pasted `posting`: the saved ones, or new ones that are then saved."""
    saved = store.get_job_facts(posting.job_id)
    if saved is not None:
        return saved
    ad = posting.description or ""
    proposal = ask_structured(
        llm,
        JobFactsProposal,
        [
            SystemMessage(load_prompt("job_facts")),
            HumanMessage(f"<job_posting_data>\n{ad}\n</job_posting_data>"),
        ],
        config,
    )
    employment, employment_quote = _backed(proposal.employment, proposal.employment_quote, ad)
    work_mode, work_mode_quote = _backed(proposal.work_mode, proposal.work_mode_quote, ad)
    facts = JobFacts(
        job_id=posting.job_id,
        employer=copied(proposal.employer, ad) or NOT_STATED_DE,
        position=copied(proposal.position, ad) or NOT_STATED_DE,
        city=copied(proposal.city, ad) or NOT_STATED_DE,
        employment=employment,
        employment_quote=employment_quote,
        work_mode=work_mode,
        work_mode_quote=work_mode_quote,
        address=copied(proposal.address, ad),
        contact_person=copied(proposal.contact_person, ad),
        target_role=proposal.target_role,
        language=proposal.language,
    )
    store.save_job_facts(facts)
    return facts


def change_job_facts(
    job_id: int,
    store: JobStore,
    llm: BaseChatModel,
    *,
    target_role: TargetRole | None = None,
    language: Language | None = None,
    address: str | None = None,
) -> JobFacts:
    """Set the Target Role, the language and/or the recipient address of the Job Facts of the
    pasted posting `job_id`. The address is the candidate's own, so it is stored as given.

    A posting pasted before Job Facts existed gets them created first.
    """
    posting = store.get_posting(job_id)
    if posting is None:
        raise UserInputError(f"No Job Posting has job ID {job_id}.")
    if posting.source != "pasted":
        raise UserInputError(f"Job ID {job_id} has no Job Facts: only pasted job ads have them.")
    facts = create_job_facts(posting, store, llm)
    changes = {"target_role": target_role, "language": language, "address": address}
    facts = facts.model_copy(update={k: v for k, v in changes.items() if v is not None})
    store.save_job_facts(facts)
    return facts


def _backed(value: str, quote: str, text: str) -> tuple[str, str | None]:
    """`value` and its quote as `text` writes it, if the quote appears there; otherwise
    "nicht angegeben" and no quote."""
    found = copied(quote, text)
    if value == NOT_STATED_DE or found is None:
        return NOT_STATED_DE, None
    return value, found


def facts_text(facts: JobFacts) -> str:
    """The Job Facts as lines for the LLM."""
    return "\n".join(
        [
            f"Employer: {facts.employer}",
            f"Position: {facts.position}",
            f"City: {facts.city}",
            f"Vollzeit/Teilzeit: {facts.employment}",
            f"Präsenz/Online/Hybrid: {facts.work_mode}",
            f"Target Role: {facts.target_role}",
            f"Language: {facts.language}",
            f"Address: {facts.address or NOT_STATED_DE}",
            f"Contact person: {facts.contact_person or NOT_STATED_DE}",
        ]
    )
