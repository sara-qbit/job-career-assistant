"""The job tools: `search_jobs` (the Retriever) and `get_job_description` (verbatim, no LLM).

Each tool returns a short text for the LLM plus an artifact (a JSON-ready dict) that the Chat
service turns into answer events. Expected failures raise AppError subclasses; the agent hands
their user message back to the LLM as the tool result.
"""

from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from turing_projects_v2.errors import UserInputError
from turing_projects_v2.rag import retriever
from turing_projects_v2.rag.indexer import NOT_STATED
from turing_projects_v2.store import JobStore


class JobDescription(BaseModel):
    """The verbatim Job Description of one posting, with what is needed to show it."""

    job_id: int
    title: str
    company: str | None
    url: str | None
    text: str


def job_tools(
    store: JobStore, llm: BaseChatModel, collection: Collection, embedder: Embeddings
) -> list[BaseTool]:
    """The tools `search_jobs` and `get_job_description`, bound to the given services."""

    @tool(response_format="content_and_artifact")
    def search_jobs(question: str) -> tuple[str, dict]:
        """Search the Knowledge Base for Job Postings matching the candidate's request.

        Args:
            question: the candidate's request in their own words, including any hard conditions
                (remote, work type, location, Target Role, experience level).
        """
        outcome = retriever.search_jobs(question, llm, collection, embedder)
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

    return [search_jobs, get_job_description]
