import httpx
import openai
import pytest
from langchain_core.embeddings import Embeddings

from tests.kaggle_fixture import posting, write_kaggle_dataset
from turing_projects_v2.errors import DataValidationError, ExternalAPIError
from turing_projects_v2.fakes import FakeChatModel
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.profile import TARGET_ROLES
from turing_projects_v2.rag.indexer import index_knowledge_base, open_collection
from turing_projects_v2.rag.retriever import (
    QueryVariants,
    SearchFilters,
    SearchQuery,
    search_jobs,
)
from turing_projects_v2.store import JobStore

VOCABULARY = ["python", "llm", "grants", "quantum", "innovation"]


class KeywordEmbedder(Embeddings):
    """Embeds text as counts of a few keywords, so similarity is predictable in tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(text.lower().count(word)) for word in VOCABULARY] + [0.1]


@pytest.fixture
def embedder():
    return KeywordEmbedder()


def knowledge_base(tmp_path, embedder, postings, roles):
    store = JobStore(":memory:")
    import_kaggle(write_kaggle_dataset(tmp_path / "raw", postings), store)
    store.set_target_roles(roles.items())
    collection = open_collection(tmp_path / "chroma", "keywords")
    index_knowledge_base(store, collection, embedder)
    return collection


def scripted_llm(search_text, variants, **filters):
    return FakeChatModel(
        responses=[
            SearchQuery(search_text=search_text, filters=SearchFilters(**filters)),
            QueryVariants(variants=variants),
        ]
    )


def test_results_are_numbered_with_the_posting_details(tmp_path, embedder):
    collection = knowledge_base(
        tmp_path,
        embedder,
        [
            posting(1, "Python LLM Engineer", description="Python and LLM work."),
            posting(2, "Grants Officer", description="Grants."),
        ],
        {1: "AI Engineer", 2: "Grant Management Specialist"},
    )
    llm = scripted_llm("python llm", ["python", "llm", "python llm engineer"])

    outcome = search_jobs("I want to build LLM apps in Python", llm, collection, embedder)

    first = outcome.results[0]
    assert first.number == 1
    assert first.job_id == 1
    assert first.title == "Python LLM Engineer"
    assert first.company == "Acme Corp"
    assert first.location == "Austin, TX"
    assert first.work_type == "Full-time"
    assert first.url == "https://www.linkedin.com/jobs/view/1"
    assert [result.number for result in outcome.results] == list(range(1, len(outcome.results) + 1))


FILTER_POSTINGS = [
    posting(1, "Python Engineer", remote_allowed="1.0", location="Austin, TX"),
    posting(2, "Python Engineer", formatted_work_type="Contract", location="Seattle, WA"),
    posting(3, "Python Grants Officer", formatted_experience_level="Director"),
    posting(4, "Python Engineer", location="Dallas, TX", formatted_experience_level="Entry level"),
]
FILTER_ROLES = {
    1: "AI Engineer",
    2: "AI Engineer",
    3: "Grant Management Specialist",
    4: "AI Engineer",
}


@pytest.mark.parametrize(
    ("filters", "expected_job_ids"),
    [
        ({"remote": True}, {1}),
        ({"remote": False}, {2, 3, 4}),
        ({"work_type": "Contract"}, {2}),
        ({"experience_level": "Director"}, {3}),
        ({"target_role": "Grant Management Specialist"}, {3}),
        ({"location": "seattle"}, {2}),
        ({"location": "TX"}, {1, 3, 4}),
        ({"location": "TX", "experience_level": "Entry level"}, {4}),
    ],
)
def test_filters_are_honoured(tmp_path, embedder, filters, expected_job_ids):
    collection = knowledge_base(tmp_path, embedder, FILTER_POSTINGS, FILTER_ROLES)
    llm = scripted_llm("python", ["python", "python engineer", "python developer"], **filters)

    outcome = search_jobs("python jobs", llm, collection, embedder)

    assert {result.job_id for result in outcome.results} == expected_job_ids


def test_location_matches_whole_words_only(tmp_path, embedder):
    collection = knowledge_base(
        tmp_path,
        embedder,
        [posting(1, "Python Engineer", location="Austin, TX"), posting(2, "Python Engineer")],
        {1: "AI Engineer", 2: "AI Engineer"},
    )
    llm = scripted_llm("python", ["python", "python engineer", "python dev"], location="Aus")

    assert search_jobs("python jobs in Aus", llm, collection, embedder).results == []


def test_job_ranked_well_by_most_queries_comes_first(tmp_path, embedder):
    collection = knowledge_base(
        tmp_path,
        embedder,
        [
            posting(1, "Quantum Developer", description="Quantum circuits."),
            posting(2, "Python Engineer", description="Python services."),
            posting(3, "Innovation Lead", description="Innovation programmes."),
        ],
        dict.fromkeys([1, 2, 3], "AI Engineer"),
    )
    # The quantum posting wins one query; the Python posting wins the other three.
    llm = scripted_llm("python", ["quantum", "python developer", "python code"])

    outcome = search_jobs("python jobs", llm, collection, embedder)

    assert [result.job_id for result in outcome.results][:2] == [2, 1]
    assert outcome.results[0].score > outcome.results[1].score


def test_each_job_appears_once_even_if_several_chunks_match(tmp_path, embedder):
    long_python = "\n\n".join(f"Python task {n}. " * 20 for n in range(10))
    collection = knowledge_base(
        tmp_path,
        embedder,
        [posting(1, "Python Engineer", description=long_python), posting(2, "Python Analyst")],
        {1: "AI Engineer", 2: "AI Engineer"},
    )
    llm = scripted_llm("python", ["python", "python engineer", "python developer"])

    outcome = search_jobs("python jobs", llm, collection, embedder)

    assert sorted(result.job_id for result in outcome.results) == [1, 2]


def test_at_most_five_results_are_returned(tmp_path, embedder):
    collection = knowledge_base(
        tmp_path,
        embedder,
        [posting(job_id, f"Python Engineer {job_id}") for job_id in range(1, 9)],
        dict.fromkeys(range(1, 9), "AI Engineer"),
    )
    llm = scripted_llm("python", ["python", "python engineer", "python developer"])

    outcome = search_jobs("python jobs", llm, collection, embedder)

    assert [result.number for result in outcome.results] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "filters",
    [{"location": "Boston"}, {"work_type": "Volunteer"}, {"remote": True, "work_type": "Contract"}],
)
def test_no_match_says_so_and_suggests_broadening(tmp_path, embedder, filters):
    collection = knowledge_base(tmp_path, embedder, FILTER_POSTINGS, FILTER_ROLES)
    llm = scripted_llm("python", ["python", "python engineer", "python developer"], **filters)

    outcome = search_jobs("python jobs", llm, collection, embedder)

    assert outcome.results == []
    assert "No tagged Job Posting matches" in outcome.message
    assert "fewer conditions" in outcome.message


def test_found_results_have_no_broadening_hint(tmp_path, embedder):
    collection = knowledge_base(tmp_path, embedder, FILTER_POSTINGS, FILTER_ROLES)
    llm = scripted_llm("python", ["python", "python engineer", "python developer"])

    outcome = search_jobs("python jobs", llm, collection, embedder)

    assert outcome.message == "Found 4 matching Job Postings."


def test_trace_records_every_stage(tmp_path, embedder):
    collection = knowledge_base(
        tmp_path,
        embedder,
        [
            posting(1, "Quantum Developer", description="Quantum circuits."),
            posting(2, "Python Engineer", description="Python services."),
        ],
        {1: "Quantum Software Developer", 2: "AI Engineer"},
    )
    llm = scripted_llm("python", ["quantum", "python developer", "python code"], remote=False)

    trace = search_jobs("python jobs, not remote", llm, collection, embedder).trace

    assert trace.question == "python jobs, not remote"
    assert trace.filters == SearchFilters(remote=False)
    assert [stage.query for stage in trace.queries] == [
        "python",
        "quantum",
        "python developer",
        "python code",
    ]
    assert trace.queries[1].job_ids[0] == 1  # "quantum" ranks the quantum posting first
    assert trace.queries[0].job_ids[0] == 2
    assert [job_id for job_id, _ in trace.fusion] == [2, 1]
    assert trace.fusion[0][1] == pytest.approx(3 / 61 + 1 / 62)


def test_a_posting_with_many_chunks_does_not_crowd_out_other_jobs(tmp_path, embedder):
    long_python = "\n\n".join(f"Python python python task {n}. " * 12 for n in range(30))
    collection = knowledge_base(
        tmp_path,
        embedder,
        [posting(1, "Python Lead", description=long_python)]
        # Less similar to "python" than every chunk of posting 1, so those chunks rank first.
        + [
            posting(job_id, f"Python Engineer {job_id}", description="Grants and innovation.")
            for job_id in range(2, 7)
        ],
        dict.fromkeys(range(1, 7), "AI Engineer"),
    )
    llm = scripted_llm("python", ["python", "python engineer", "python developer"])

    outcome = search_jobs("python jobs", llm, collection, embedder, per_query=6)

    assert all(len(stage.job_ids) == 6 for stage in outcome.trace.queries)
    assert len(outcome.results) == 5


def test_llm_failure_is_an_external_api_error(tmp_path, embedder):
    collection = knowledge_base(tmp_path, embedder, FILTER_POSTINGS, FILTER_ROLES)
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://openrouter.ai"))

    with pytest.raises(ExternalAPIError):
        search_jobs("python jobs", FakeChatModel(responses=[timeout]), collection, embedder)


def test_malformed_llm_answer_is_a_data_validation_error(tmp_path, embedder):
    collection = knowledge_base(tmp_path, embedder, FILTER_POSTINGS, FILTER_ROLES)
    llm = FakeChatModel(responses=["not a search query"])

    with pytest.raises(DataValidationError):
        search_jobs("python jobs", llm, collection, embedder)


def test_target_role_filter_offers_exactly_the_target_roles():
    offered = SearchFilters.model_json_schema()["properties"]["target_role"]["anyOf"][0]["enum"]

    assert set(offered) == TARGET_ROLES
