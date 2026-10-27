"""The job tools: `search_jobs` (the Retriever), `get_job_description` (verbatim, no LLM),
`create_job_info` (summaries checked by the judge), `write_cover_letter`, `change_job_facts`
(the candidate's Target Role and language for a pasted Job Posting), and `record_application` and
`remove_application` (the pasted Job Postings the candidate applied for), and `application_table`
(the Application Table for the Berater).

Each tool returns a short text for the LLM, most of them plus an artifact (a JSON-ready dict) that
the Chat service turns into answer events; recording and removing an Application return text
only. Expected failures raise AppError subclasses; the agent hands their user message back to the
LLM as the tool result.
"""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from turing_projects_v2 import applications, cover_letter, job_facts, job_info
from turing_projects_v2.errors import UserInputError
from turing_projects_v2.judge import Judge
from turing_projects_v2.profile import CandidateProfile, TargetRole
from turing_projects_v2.rag import retriever
from turing_projects_v2.rag.indexer import NOT_STATED, KnowledgeBase
from turing_projects_v2.rag.reranker import Reranker
from turing_projects_v2.rag.retriever import SearchSettings
from turing_projects_v2.store import JobStore, Language


class JobDescription(BaseModel):
    """The verbatim Job Description of one posting, with what is needed to show it."""

    job_id: int
    title: str
    company: str | None
    url: str | None
    text: str


def job_tools(
    store: JobStore,
    llm: BaseChatModel,
    knowledge_base: Callable[[str], KnowledgeBase],
    judge: Judge,
    profile: CandidateProfile,
    reranker: Reranker | None = None,
) -> list[BaseTool]:
    """The job tools, bound to the given services.

    `search_jobs` searches the Knowledge Base of the embedding model named in the run config's
    `configurable["search"]` (a SearchSettings dict), reranking if it says so.
    """

    @tool(response_format="content_and_artifact")
    def search_jobs(question: str, config: RunnableConfig) -> tuple[str, dict]:
        """Search the Knowledge Base for Job Postings matching the candidate's request.

        Args:
            question: the candidate's request in their own words, including any hard conditions
                (remote, work type, location, Target Role, experience level).
        """
        search = SearchSettings.model_validate(config.get("configurable", {}).get("search", {}))
        collection, embedder = knowledge_base(search.embedding_model)
        outcome = retriever.search_jobs(
            question,
            llm,
            collection,
            embedder,
            reranker=reranker if search.rerank else None,
        )
        lines = [outcome.message]
        lines += [
            f"#{result.number} [job ID {result.job_id}] {result.title} at "
            f"{result.company or NOT_STATED} ({result.location or NOT_STATED}, "
            f"{result.work_type or NOT_STATED})"
            for result in outcome.results
        ]
        return "\n".join(lines), outcome.model_dump(mode="json")

    @tool(response_format="content_and_artifact")
    def get_job_description(job_id: int) -> tuple[str, dict | None]:
        """Show the candidate the full, verbatim Job Description of one Job Posting.

        Args:
            job_id: the job ID of the posting (as listed in the search results).
        """
        posting = store.get_posting(job_id)
        if posting is None:
            raise UserInputError(f"No Job Posting has job ID {job_id}.")
        if not posting.description:
            return f"Job Posting {job_id} has no Job Description in the data.", None
        description = JobDescription(
            job_id=job_id,
            title=posting.title,
            company=posting.company_name,
            url=posting.job_posting_url,
            text=posting.description,
        )
        content = (
            f"The Job Description of job ID {job_id} ({posting.title} at "
            f"{posting.company_name or NOT_STATED}) is now shown to the candidate verbatim in its "
            "own box; do not repeat it.\n"
            f"<job_posting_data>\n{posting.description}\n</job_posting_data>"
        )
        return content, description.model_dump(mode="json")

    @tool(response_format="content_and_artifact")
    def create_job_info(job_id: int) -> tuple[str, dict]:
        """Show the candidate the Job Info of one Job Posting: job name, company, work type, a
        role summary and the company vision, each summary checked by a judge.

        Args:
            job_id: the job ID of the posting (as listed in the search results).
        """
        info = job_info.create_job_info(job_id, store, llm, judge)
        verdicts = {
            "Role summary": info.role_summary_verdict,
            "Company vision": info.company_vision_verdict,
        }
        checks = [
            f"{name} judged {verdict.value}: {verdict.reason}"
            for name, verdict in verdicts.items()
            if verdict is not None
        ]
        lines = [
            f"The Job Info of job ID {job_id} is now shown to the candidate in its own card; do "
            "not repeat it. Mention any summary judged fabricated.",
            "<job_info_data>",
            f"Job name: {info.job_name}",
            f"Company: {info.company_name or job_info.NOT_STATED}",
            f"Work type: {info.work_type or job_info.NOT_STATED}",
            f"Role summary: {info.role_summary}",
            f"Company vision: {info.company_vision}",
            *checks,
            "</job_info_data>",
        ]
        return "\n".join(lines), info.model_dump(mode="json")

    @tool(response_format="content_and_artifact")
    def write_cover_letter(job_id: int) -> tuple[str, dict]:
        """Write the candidate's Cover Letter for one Job Posting from their Letter Template and
        show it with its check results and download buttons.

        Args:
            job_id: the job ID of the posting (as listed in the search results).
        """
        letter = cover_letter.write_cover_letter(job_id, store, profile, llm, judge)
        failed = letter.failed_checks
        fabricated = [f"{name}: {reason}" for name, reason in letter.fabricated.items()]
        # The letter itself holds the candidate's personal details, so it stays out of the reply.
        lines = [
            f"The Cover Letter for job ID {job_id} is now shown to the candidate in its own box "
            "with its checks and download buttons; do not repeat it.",
            f"Failed exact checks: {', '.join(failed)}" if failed else "All exact checks passed.",
            *(
                ["Still judged fabricated after one rewrite (warn the candidate):", *fabricated]
                if fabricated
                else ["The judge found no fabricated text."]
            ),
        ]
        return "\n".join(lines), letter.model_dump(mode="json")

    @tool(response_format="content_and_artifact")
    def change_job_facts(
        job_id: int,
        target_role: TargetRole | None = None,
        language: Language | None = None,
        address: str | None = None,
    ) -> tuple[str, dict]:
        """Change the Target Role, the letter language or the recipient address in the Job
        Facts of a pasted Job Posting, when the candidate asks (e.g. "use Grant Management",
        "write it in English", "the address is ..."), and show the changed Job Facts card.

        Args:
            job_id: the job ID of the pasted posting.
            target_role: the Target Role the candidate chose, if they changed it.
            language: the language the candidate chose for the letter, if they changed it.
            address: the postal address the candidate gave, copied exactly with one line per
                address line, if they gave one.
        """
        facts = job_facts.change_job_facts(
            job_id, store, llm, target_role=target_role, language=language, address=address
        )
        content = (
            f"The changed Job Facts of job ID {job_id} are now shown to the candidate in their "
            "own card; do not repeat them.\n"
            f"<job_facts_data>\n{job_facts.facts_text(facts)}\n</job_facts_data>"
        )
        return content, facts.model_dump(mode="json")

    @tool
    def record_application(job_id: int) -> str:
        """Record an Application for a pasted Job Posting, dated today. Use it only when the
        candidate says they applied or asks to save the job ("save it", "I applied").

        Args:
            job_id: the job ID of the pasted posting.
        """
        application, new = applications.record_application(job_id, store)
        applied_on = applications.german_date(application.applied_on)
        if not new:
            return f"Already applied on {applied_on}. Nothing was added; tell the candidate."
        return f"Application for job ID {job_id} recorded: applied on {applied_on}."

    @tool
    def remove_application(job_id: int) -> str:
        """Remove a wrong Application, when the candidate asks ("remove the application", "I
        didn't apply"). The Job Posting stays stored.

        Args:
            job_id: the job ID of the posting.
        """
        application = applications.remove_application(job_id, store)
        return (
            f"Application for job ID {job_id} (applied on "
            f"{applications.german_date(application.applied_on)}) removed; the Job Posting stays "
            "stored."
        )

    @tool(response_format="content_and_artifact")
    def application_table() -> tuple[str, dict | None]:
        """Show the candidate the Application Table: every Application so far as a German table,
        with a preview and a download button for Bewerbungen.pdf. Use it when the candidate asks
        for the Applications PDF or their list of applications."""
        table = applications.application_table(profile.name, store)
        if not table.rows:
            return (
                "The candidate has no Applications yet, so no PDF was made. Tell them to say "
                '"save it" or "I applied" after pasting a job ad.'
            ), None
        # The table carries the candidate's name, so only its rows reach the LLM.
        lines = [
            f"The Application Table with {len(table.rows)} Applications is now shown to the "
            "candidate with a preview and a download button for Bewerbungen.pdf; do not repeat it.",
            "<applications_data>",
            *(
                f"{applications.german_date(row.applied_on)}: {row.position} at {row.employer}"
                for row in table.rows
            ),
            "</applications_data>",
        ]
        return "\n".join(lines), table.model_dump(mode="json")

    return [
        search_jobs,
        get_job_description,
        create_job_info,
        write_cover_letter,
        change_job_facts,
        record_application,
        remove_application,
        application_table,
    ]
