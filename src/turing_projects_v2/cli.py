"""Command-line entry point: `uv run turing-projects-v2 <command>`."""

import argparse
import subprocess
import sys
import time
from collections.abc import Callable

from tqdm import tqdm

from turing_projects_v2.config import PROJECT_ROOT, Settings, load_settings
from turing_projects_v2.errors import AppError
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.llm import chat_model
from turing_projects_v2.profile import load_profile
from turing_projects_v2.rag.embeddings import EMBEDDING_MODELS, load_embedder
from turing_projects_v2.rag.indexer import NOT_STATED, index_knowledge_base, open_collection
from turing_projects_v2.rag.retriever import search_jobs
from turing_projects_v2.store import JobStore
from turing_projects_v2.tagging import load_title_variants, tag_postings


def check(settings: Settings) -> None:
    """check settings and the Candidate Profile (default)"""
    profile = load_profile(settings.candidate_dir)
    print(f"Settings OK (agent model: {settings.agent_model}, judge: {settings.judge_model}).")
    print(f"Candidate Profile OK for {profile.name}.")
    for role, template in profile.target_roles.items():
        missing = " is MISSING" if role in profile.missing_templates else ""
        print(f"  {role}: Letter Template '{template}'{missing}")


def import_data(settings: Settings) -> None:
    """import the Kaggle dataset into the database, then tag postings"""
    store = JobStore(settings.database_path)
    started = time.perf_counter()
    report = import_kaggle(settings.kaggle_dir, store)
    print(
        f"Imported {report.postings:,} postings and {report.companies:,} companies "
        f"in {time.perf_counter() - started:.0f}s."
    )
    _tag(settings, store)


def tag(settings: Settings) -> None:
    """re-tag postings after editing the title variant list"""
    _tag(settings, JobStore(settings.database_path))


def review_titles(settings: Settings) -> None:
    """list the titles tagged per Target Role"""
    store = JobStore(settings.database_path)
    for role, titles in store.tagged_title_counts().items():
        print(f"\n## {role}: {sum(n for _, n in titles)} postings, {len(titles)} titles")
        for title, count in titles:
            print(f"{count:>4}  {title}")


def index(settings: Settings) -> None:
    """embed tagged postings not yet in the Knowledge Base, for every embedding model"""
    store = JobStore(settings.database_path)
    for model_name in EMBEDDING_MODELS:
        print(f"\n## {model_name}")
        embedder = load_embedder(settings, model_name)
        with tqdm(unit="posting") as bar:

            def show(done: int, total: int, bar: tqdm = bar) -> None:
                bar.total = total
                bar.update(done - bar.n)

            report = index_knowledge_base(
                store, open_collection(settings.chroma_dir, model_name), embedder, show
            )
        print(
            f"Indexed {report.indexed:,} postings ({report.chunks:,} chunks); "
            f"{report.already_indexed:,} already indexed, {report.removed:,} removed."
        )


def search(settings: Settings, question: str, embedding_model: str | None) -> None:
    """search the Knowledge Base the way the chatbot does"""
    embedding_model = embedding_model or settings.embedding_model
    # First: an unknown name must not create a collection.
    embedder = load_embedder(settings, embedding_model)
    outcome = search_jobs(
        question,
        chat_model(settings, settings.agent_model),
        open_collection(settings.chroma_dir, embedding_model),
        embedder,
    )
    trace = outcome.trace
    print(f"Search text: {trace.search_text}")
    print(f"Filters: {trace.filters.model_dump(exclude_none=True) or 'none'}")
    for stage in trace.queries:
        print(f"  query '{stage.query}': {len(stage.job_ids)} postings")
    print(f"\n{outcome.message}")
    for result in outcome.results:
        print(
            f"#{result.number} {result.title} at {result.company or NOT_STATED} "
            f"({result.location or NOT_STATED}, {result.work_type or NOT_STATED}) {result.url}"
        )


def chat(settings: Settings) -> None:
    """open the chatbot in the browser (Streamlit)"""
    app = PROJECT_ROOT / "src" / "turing_projects_v2" / "ui" / "app.py"
    # Run from the project root so that Streamlit reads .streamlit/config.toml.
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], cwd=PROJECT_ROOT)


COMMANDS: dict[str, Callable[..., None]] = {
    "check": check,
    "import-data": import_data,
    "tag": tag,
    "review-titles": review_titles,
    "index": index,
    "search": search,
    "chat": chat,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="turing-projects-v2")
    subparsers = parser.add_subparsers(dest="command")
    parsers = {
        name: subparsers.add_parser(name, help=command.__doc__)
        for name, command in COMMANDS.items()
    }
    parsers["search"].add_argument("question", help="what kind of job you are looking for")
    parsers["search"].add_argument(
        "--model",
        dest="embedding_model",
        choices=EMBEDDING_MODELS,
        help="embedding model; default from settings",
    )
    options = vars(parser.parse_args())
    name = options.pop("command") or "check"
    try:
        COMMANDS[name](load_settings(), **options)
    except AppError as error:
        sys.exit(f"{name} failed: {error}")


def _tag(settings: Settings, store: JobStore) -> None:
    started = time.perf_counter()
    report = tag_postings(store, load_title_variants(settings.title_variants_path))
    print(f"Tagged in {time.perf_counter() - started:.1f}s.")
    print(f"{report.total:,} postings: {sum(report.tagged.values()):,} tagged")
    for role, count in report.tagged.items():
        print(f"  {role}: {count:,}")
    print(f"  (untagged: {report.untagged:,})")
