"""Indexer: chunks tagged Job Postings and writes them to one ChromaDB collection per model.

Each chunk is a piece of the Job Description prefixed with a context header, so that it is
understandable on its own.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple, cast

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from turing_projects_v2.store import JobPosting, JobStore

NOT_STATED = "not stated"

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


class KnowledgeBase(NamedTuple):
    """One embedding model's collection, with the embedder that queries it."""

    collection: Collection
    embedder: Embeddings


def open_collection(chroma_dir: Path, model_name: str) -> Collection:
    """The Knowledge Base collection for embedding model `model_name`, created if missing."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(
        f"knowledge_base_{model_name}", configuration={"hnsw": {"space": "cosine"}}
    )


class IndexReport(BaseModel):
    indexed: int
    already_indexed: int
    removed: int
    chunks: int


def index_knowledge_base(
    store: JobStore,
    collection: Collection,
    embedder: Embeddings,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
    batch_size: int = 32,
) -> IndexReport:
    """Bring `collection` in line with the tagged postings; embeds only postings not yet in it.

    Postings that lost their tag are removed; changed metadata (such as a new Target Role) is
    updated without re-embedding. New postings are embedded `batch_size` at a time, calling
    `on_progress(done, total)` before the first batch and after each one.
    """
    existing = _indexed_metadata(collection)
    tagged = {posting.job_id: posting for posting in store.tagged_postings()}

    untagged = [job_id for job_id in existing if job_id not in tagged]
    if untagged:
        collection.delete(ids=[chunk_id for job_id in untagged for chunk_id in existing[job_id]])

    for job_id, chunks in existing.items():
        posting = tagged.get(job_id)
        if posting and any(metadata != _metadata(posting) for metadata in chunks.values()):
            collection.update(ids=list(chunks), metadatas=[_metadata(posting)] * len(chunks))

    new_postings = [posting for job_id, posting in tagged.items() if job_id not in existing]
    on_progress(0, len(new_postings))
    chunk_count = 0
    for start in range(0, len(new_postings), batch_size):
        batch = new_postings[start : start + batch_size]
        ids, documents, metadatas = [], [], []
        for posting in batch:
            chunks = chunk_texts(posting)
            ids += [f"{posting.job_id}-{n}" for n in range(len(chunks))]
            documents += chunks
            metadatas += [_metadata(posting)] * len(chunks)
        embeddings: list[Sequence[float]] = list(embedder.embed_documents(documents))
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        chunk_count += len(documents)
        on_progress(start + len(batch), len(new_postings))
    return IndexReport(
        indexed=len(new_postings),
        already_indexed=len(tagged) - len(new_postings),
        removed=len(untagged),
        chunks=chunk_count,
    )


def _indexed_metadata(collection: Collection) -> dict[int, dict[str, Metadata]]:
    """Per job ID in `collection`, the metadata of each of its chunks by chunk ID."""
    stored = collection.get(include=["metadatas"])
    chunks: dict[int, dict[str, Metadata]] = {}
    for chunk_id, metadata in zip(stored["ids"], stored["metadatas"] or [], strict=True):
        chunks.setdefault(cast(int, metadata["job_id"]), {})[chunk_id] = metadata
    return chunks


def chunk_texts(posting: JobPosting) -> list[str]:
    """The Job Description split into pieces, each prefixed with the context header.

    A posting without a description still gets one chunk (the header alone), so that it can be
    found by its title.
    """
    header = (
        f"Title: {posting.title}\n"
        f"Company: {posting.company_name or NOT_STATED}\n"
        f"Work type: {posting.formatted_work_type or NOT_STATED}\n"
        f"Location: {posting.location or NOT_STATED}\n\n"
    )
    pieces = _SPLITTER.split_text(posting.description or "") or [""]
    return [header + piece for piece in pieces]


def _metadata(posting: JobPosting) -> Metadata:
    """Filterable fields of `posting`; ChromaDB rejects None, so missing fields are left out."""
    fields: dict[str, str | int | float | bool | None] = {
        "job_id": posting.job_id,
        "target_role": posting.target_role,
        "title": posting.title,
        "company": posting.company_name,
        "work_type": posting.formatted_work_type,
        # Kaggle marks remote postings with 1 and leaves the rest empty.
        "remote": bool(posting.remote_allowed),
        "location": posting.location,
        "experience_level": posting.formatted_experience_level,
        "min_salary": posting.min_salary,
        "med_salary": posting.med_salary,
        "max_salary": posting.max_salary,
        "pay_period": posting.pay_period,
        "currency": posting.currency,
        "url": posting.job_posting_url,
    }
    return {name: value for name, value in fields.items() if value is not None}
