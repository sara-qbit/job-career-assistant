"""Retriever: the advanced RAG pipeline behind the `search_jobs` tool.

1. Self-query: the LLM turns the question into search text plus metadata filters.
2. Multi-query: the LLM rewrites the search text as 3-4 query variants.
3. Filtered similarity search in the Knowledge Base for the search text and each variant.
4. Reciprocal Rank Fusion of the per-query rankings, de-duplicated by job ID.
5. Optionally, a Reranker re-scores the best fused postings against the search text.
"""

import re
from typing import Any, Literal, cast

from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from turing_projects_v2.llm import ask_structured
from turing_projects_v2.prompts import load_prompt
from turing_projects_v2.rag.reranker import Reranker

# Constant from the original RRF paper (Cormack et al., 2009); damps the weight of top ranks.
RRF_K = 60

# Fused postings the Reranker re-scores.
RERANK_TOP = 20

# Chunks fetched per wanted posting; tagged postings have about 7 chunks on average.
_CHUNKS_PER_POSTING = 10


WorkType = Literal["Full-time", "Part-time", "Contract", "Temporary", "Internship", "Volunteer"]
TargetRoleName = Literal[
    "AI Engineer",
    "AI Innovation Manager",
    "Quantum Software Developer",
    "Grant Management Specialist",
]


class SearchFilters(BaseModel):
    """Hard conditions from the question; None means the question does not state one."""

    remote: bool | None = Field(
        default=None, description="true if remote work is asked for, false if on-site is asked for"
    )
    location: str | None = Field(
        default=None,
        description='a city ("Austin") or a two-letter US state code ("TX", never "Texas")',
    )
    work_type: WorkType | None = None
    experience_level: (
        Literal[
            "Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"
        ]
        | None
    ) = None
    target_role: TargetRoleName | None = None


class SearchQuery(BaseModel):
    search_text: str = Field(description="what the job is about, without the hard conditions")
    filters: SearchFilters = SearchFilters()


class QueryVariants(BaseModel):
    variants: list[str] = Field(
        min_length=3, max_length=4, description="3 to 4 rewordings of the search text"
    )


class SearchSettings(BaseModel):
    """The candidate's search choices for one turn (the sidebar starts with these defaults)."""

    embedding_model: str = "gemini"
    rerank: bool = True


class SearchResult(BaseModel):
    number: int
    job_id: int
    title: str
    company: str | None
    location: str | None
    work_type: str | None
    url: str | None
    score: float


class QueryStage(BaseModel):
    """One query's ranking: job IDs best first, with the distance of each job's closest chunk."""

    query: str
    job_ids: list[int]
    distances: list[float]


class MatchedChunk(BaseModel):
    """The chunk of a returned posting closest to any query: the source the answer is based on."""

    job_id: int
    text: str
    distance: float


class SearchTrace(BaseModel):
    """Every stage of one search, for display in the Evaluation tab."""

    question: str
    search_text: str
    filters: SearchFilters
    queries: list[QueryStage]
    fusion: list[tuple[int, float]]  # (job ID, RRF score), best first
    reranked: list[tuple[int, float]] | None = None  # (job ID, Reranker score), best first
    sources: list[MatchedChunk] = []  # one per returned result, in result order


class SearchOutcome(BaseModel):
    results: list[SearchResult]
    message: str
    trace: SearchTrace


def search_jobs(
    question: str,
    llm: BaseChatModel,
    collection: Collection,
    embedder: Embeddings,
    top_k: int = 5,
    per_query: int = 20,
    reranker: Reranker | None = None,
) -> SearchOutcome:
    """The `top_k` Job Postings that best match `question`, numbered from 1.

    With a `reranker`, the best RERANK_TOP fused postings are re-scored (each by its chunk
    closest to any query) and ordered by that score.
    """
    if collection.count() == 0:
        return SearchOutcome(
            results=[],
            message=(
                "The Knowledge Base for this embedding model is not indexed yet. "
                "Run `uv run turing-projects-v2 index`, or choose another embedding model."
            ),
            trace=SearchTrace(
                question=question, search_text="", filters=SearchFilters(), queries=[], fusion=[]
            ),
        )
    query = ask_structured(
        llm, SearchQuery, [SystemMessage(load_prompt("self_query")), HumanMessage(question)]
    )
    variants = ask_structured(
        llm,
        QueryVariants,
        [SystemMessage(load_prompt("multi_query")), HumanMessage(query.search_text)],
    )
    where = _where(query.filters, collection)

    stages: list[QueryStage] = []
    chunk_metadata: dict[int, Metadata] = {}
    closest_chunk: dict[int, tuple[float, str]] = {}  # job ID -> (distance, chunk text)
    for text in [query.search_text, *variants.variants]:
        stage = QueryStage(query=text, job_ids=[], distances=[])
        if where is not None:
            # ChromaDB ranks chunks, not postings: fetch enough chunks for `per_query` postings.
            found = collection.query(
                query_embeddings=[embedder.embed_query(text)],
                n_results=per_query * _CHUNKS_PER_POSTING,
                where=where or None,
                include=["metadatas", "distances", "documents"],
            )
            metadatas = (found["metadatas"] or [[]])[0]
            distances = (found["distances"] or [[]])[0]
            documents = (found["documents"] or [[]])[0]
            for metadata, distance, document in zip(metadatas, distances, documents, strict=True):
                job_id = cast(int, metadata["job_id"])
                chunk_metadata.setdefault(job_id, metadata)
                if job_id not in closest_chunk or distance < closest_chunk[job_id][0]:
                    closest_chunk[job_id] = (distance, document)
                if job_id not in stage.job_ids and len(stage.job_ids) < per_query:
                    stage.job_ids.append(job_id)
                    stage.distances.append(distance)
        stages.append(stage)

    scores: dict[int, float] = {}
    for stage in stages:
        for rank, job_id in enumerate(stage.job_ids, start=1):
            scores[job_id] = scores.get(job_id, 0.0) + 1 / (RRF_K + rank)
    fusion = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranking = fusion
    reranked = None
    if reranker is not None:
        candidates = [job_id for job_id, _ in fusion[:RERANK_TOP]]
        rerank_scores = reranker.score(
            query.search_text, [closest_chunk[job_id][1] for job_id in candidates]
        )
        reranked = sorted(
            zip(candidates, rerank_scores, strict=True), key=lambda item: item[1], reverse=True
        )
        ranking = reranked

    results = [
        SearchResult(
            number=number,
            job_id=job_id,
            title=str(chunk_metadata[job_id]["title"]),
            company=_text(chunk_metadata[job_id], "company"),
            location=_text(chunk_metadata[job_id], "location"),
            work_type=_text(chunk_metadata[job_id], "work_type"),
            url=_text(chunk_metadata[job_id], "url"),
            score=score,
        )
        for number, (job_id, score) in enumerate(ranking[:top_k], start=1)
    ]
    trace = SearchTrace(
        question=question,
        search_text=query.search_text,
        filters=query.filters,
        queries=stages,
        fusion=fusion,
        reranked=reranked,
        sources=[
            MatchedChunk(
                job_id=result.job_id,
                text=closest_chunk[result.job_id][1],
                distance=closest_chunk[result.job_id][0],
            )
            for result in results
        ],
    )
    return SearchOutcome(results=results, message=_message(results, query.filters), trace=trace)


def _message(results: list[SearchResult], filters: SearchFilters) -> str:
    if results:
        noun = "Job Posting" if len(results) == 1 else "Job Postings"
        return f"Found {len(results)} matching {noun}."
    applied = ", ".join(
        f"{name.replace('_', ' ')} = {value}"
        for name, value in filters.model_dump(exclude_none=True).items()
    )
    conditions = f" with {applied}" if applied else ""
    return (
        f"No tagged Job Posting matches this search{conditions}. "
        "Try fewer conditions or other words for the job."
    )


def _text(metadata: Metadata, name: str) -> str | None:
    value = metadata.get(name)
    return None if value is None else str(value)


def _where(filters: SearchFilters, collection: Collection) -> dict[str, Any] | None:
    """The ChromaDB `where` clause for `filters` ({} if none); None if no posting can match.

    The location matches as whole words, case-insensitively ("TX" matches "Austin, TX"); ChromaDB
    cannot do that, so it becomes a filter on the job IDs whose location matches.
    """
    conditions: list[dict[str, Any]] = [
        {name: value}
        for name, value in filters.model_dump(exclude={"location"}).items()
        if value is not None
    ]
    if filters.location:
        pattern = re.compile(rf"(?<!\w){re.escape(filters.location)}(?!\w)", re.IGNORECASE)
        metadatas = collection.get(include=["metadatas"])["metadatas"] or []
        job_ids = sorted(
            {
                cast(int, metadata["job_id"])
                for metadata in metadatas
                if pattern.search(str(metadata.get("location", "")))
            }
        )
        if not job_ids:
            return None
        conditions.append({"job_id": {"$in": job_ids}})
    if len(conditions) > 1:
        return {"$and": conditions}
    return conditions[0] if conditions else {}
