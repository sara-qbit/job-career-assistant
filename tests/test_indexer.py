import pytest

from tests.kaggle_fixture import posting, write_kaggle_dataset
from turing_projects_v2.fakes import FakeEmbedder
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.rag.indexer import index_knowledge_base, open_collection
from turing_projects_v2.store import JobStore

LONG_DESCRIPTION = "\n\n".join(
    f"Paragraph {n}: we are looking for someone who enjoys building things. " * 4
    for n in range(1, 8)
)


@pytest.fixture
def store():
    return JobStore(":memory:")


@pytest.fixture
def collection(tmp_path):
    return open_collection(tmp_path / "chroma", "fake")


def load(tmp_path, store, postings, roles):
    import_kaggle(write_kaggle_dataset(tmp_path / "raw", postings), store)
    store.set_target_roles(roles.items())


def chunks_of(collection, job_id):
    return collection.get(where={"job_id": job_id}, include=["documents", "metadatas"])


def test_every_chunk_starts_with_the_context_header(tmp_path, store, collection):
    load(
        tmp_path,
        store,
        [posting(1, "AI Engineer", description=LONG_DESCRIPTION, formatted_work_type="Contract")],
        {1: "AI Engineer"},
    )

    index_knowledge_base(store, collection, FakeEmbedder())
    documents = chunks_of(collection, 1)["documents"]

    assert len(documents) > 1
    header = "Title: AI Engineer\nCompany: Acme Corp\nWork type: Contract\nLocation: Austin, TX\n\n"
    assert all(document.startswith(header) for document in documents)
    assert all(len(document) <= len(header) + 1000 for document in documents)


def test_every_chunk_carries_the_posting_metadata(tmp_path, store, collection):
    load(
        tmp_path,
        store,
        [
            posting(
                7,
                "Grant Writer",
                description=LONG_DESCRIPTION,
                remote_allowed="1.0",
                formatted_experience_level="Mid-Senior level",
                min_salary="60000.0",
                max_salary="80000.0",
                pay_period="YEARLY",
                currency="USD",
            )
        ],
        {7: "Grant Management Specialist"},
    )

    index_knowledge_base(store, collection, FakeEmbedder())
    metadatas = chunks_of(collection, 7)["metadatas"]

    assert len(metadatas) > 1
    for metadata in metadatas:
        assert metadata["job_id"] == 7
        assert metadata["target_role"] == "Grant Management Specialist"
        assert metadata["work_type"] == "Full-time"
        assert metadata["remote"] is True
        assert metadata["location"] == "Austin, TX"
        assert metadata["experience_level"] == "Mid-Senior level"
        assert metadata["min_salary"] == 60000.0
        assert metadata["max_salary"] == 80000.0
        assert metadata["pay_period"] == "YEARLY"
        assert metadata["currency"] == "USD"


def test_missing_fields_are_stated_as_such(tmp_path, store, collection):
    load(
        tmp_path,
        store,
        [posting(8, "AI Engineer", company_name="", location="", formatted_work_type="")],
        {8: "AI Engineer"},
    )

    index_knowledge_base(store, collection, FakeEmbedder())
    chunks = chunks_of(collection, 8)

    assert chunks["documents"][0].startswith(
        "Title: AI Engineer\nCompany: not stated\nWork type: not stated\nLocation: not stated\n"
    )
    assert "location" not in chunks["metadatas"][0]


def test_posting_without_remote_flag_is_not_remote(tmp_path, store, collection):
    # Kaggle's remote_allowed is either 1 or empty, so empty means not remote.
    load(tmp_path, store, [posting(9, "AI Engineer", remote_allowed="")], {9: "AI Engineer"})

    index_knowledge_base(store, collection, FakeEmbedder())

    assert chunks_of(collection, 9)["metadatas"][0]["remote"] is False


def indexed_job_ids(collection):
    return {metadata["job_id"] for metadata in collection.get(include=["metadatas"])["metadatas"]}


def test_only_tagged_postings_are_indexed(tmp_path, store, collection):
    load(
        tmp_path,
        store,
        [posting(1, "AI Engineer"), posting(2, "Accountant"), posting(3, "Grant Writer")],
        {1: "AI Engineer", 2: None, 3: "Grant Management Specialist"},
    )

    index_knowledge_base(store, collection, FakeEmbedder())

    assert indexed_job_ids(collection) == {1, 3}


def test_each_embedding_model_has_its_own_collection(tmp_path, store):
    load(tmp_path, store, [posting(1, "AI Engineer")], {1: "AI Engineer"})
    minilm = open_collection(tmp_path / "chroma", "minilm")

    index_knowledge_base(store, minilm, FakeEmbedder())

    assert indexed_job_ids(minilm) == {1}
    assert indexed_job_ids(open_collection(tmp_path / "chroma", "gemini")) == set()


def test_reindex_embeds_only_postings_not_yet_in_the_collection(tmp_path, store, collection):
    load(
        tmp_path,
        store,
        [posting(1, "AI Engineer"), posting(2, "Data Scientist")],
        {1: "AI Engineer", 2: None},
    )
    index_knowledge_base(store, collection, FakeEmbedder())
    store.set_target_roles([(2, "AI Engineer")])
    embedder = FakeEmbedder()

    report = index_knowledge_base(store, collection, embedder)

    assert indexed_job_ids(collection) == {1, 2}
    assert all(text.startswith("Title: Data Scientist\n") for text in embedder.embedded)
    assert report.indexed == 1
    assert report.already_indexed == 1


def test_reindex_without_changes_embeds_nothing(tmp_path, store, collection):
    load(tmp_path, store, [posting(1, "AI Engineer")], {1: "AI Engineer"})
    index_knowledge_base(store, collection, FakeEmbedder())
    embedder = FakeEmbedder()

    index_knowledge_base(store, collection, embedder)

    assert embedder.embedded == []
    assert len(chunks_of(collection, 1)["documents"]) == 1


def test_reindex_removes_postings_that_lost_their_tag(tmp_path, store, collection):
    load(
        tmp_path,
        store,
        [posting(1, "AI Engineer"), posting(2, "Data Scientist")],
        {1: "AI Engineer", 2: "AI Engineer"},
    )
    index_knowledge_base(store, collection, FakeEmbedder())
    store.set_target_roles([(2, None)])

    report = index_knowledge_base(store, collection, FakeEmbedder())

    assert indexed_job_ids(collection) == {1}
    assert report.removed == 1


def test_reindex_updates_the_target_role_of_retagged_postings(tmp_path, store, collection):
    load(tmp_path, store, [posting(1, "Innovation Manager")], {1: "AI Engineer"})
    index_knowledge_base(store, collection, FakeEmbedder())
    store.set_target_roles([(1, "AI Innovation Manager")])

    index_knowledge_base(store, collection, FakeEmbedder())

    assert {m["target_role"] for m in chunks_of(collection, 1)["metadatas"]} == {
        "AI Innovation Manager"
    }


def test_progress_is_reported_until_every_new_posting_is_indexed(tmp_path, store, collection):
    load(
        tmp_path,
        store,
        [posting(job_id, "AI Engineer") for job_id in range(1, 6)],
        dict.fromkeys(range(1, 6), "AI Engineer"),
    )
    progress = []

    index_knowledge_base(
        store,
        collection,
        FakeEmbedder(),
        on_progress=lambda done, total: progress.append((done, total)),
        batch_size=2,
    )

    assert progress == [(0, 5), (2, 5), (4, 5), (5, 5)]
