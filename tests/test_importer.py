import pytest

from tests.kaggle_fixture import TITLES, posting, write_kaggle_dataset
from turing_projects_v2.errors import ConfigError, DataValidationError
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.store import JobStore


@pytest.fixture
def store():
    return JobStore(":memory:")


def test_all_postings_and_companies_are_stored(tmp_path, store):
    report = import_kaggle(write_kaggle_dataset(tmp_path), store)

    assert report.postings == len(TITLES) == store.count_postings()
    assert report.companies == 1 == store.count_companies()


def test_posting_is_stored_with_its_verbatim_description_and_typed_fields(tmp_path, store):
    raw_dir = write_kaggle_dataset(
        tmp_path, [posting(42, "AI Engineer", remote_allowed="1.0", max_salary="150000.0")]
    )

    import_kaggle(raw_dir, store)
    stored = store.get_posting(42)

    assert stored.title == "AI Engineer"
    assert stored.description == "Description of job 42.\nSecond line, with a comma."
    assert stored.company_id == 2774458
    assert stored.remote_allowed is True
    assert stored.max_salary == 150000.0
    assert stored.formatted_experience_level is None
    assert stored.target_role is None


def test_reimport_creates_no_duplicates_and_keeps_updates(tmp_path, store):
    raw_dir = write_kaggle_dataset(tmp_path)
    import_kaggle(raw_dir, store)

    write_kaggle_dataset(tmp_path, [posting(1001, "Machine Learning Engineer II")])
    import_kaggle(raw_dir, store)

    assert store.count_postings() == len(TITLES)
    assert store.get_posting(1001).title == "Machine Learning Engineer II"


def test_unknown_job_id_returns_none(store):
    assert store.get_posting(999) is None


def test_missing_dataset_is_a_config_error(tmp_path, store):
    with pytest.raises(ConfigError, match=r"postings\.csv"):
        import_kaggle(tmp_path, store)


def test_csv_without_expected_columns_is_rejected(tmp_path, store):
    (tmp_path / "companies").mkdir()
    (tmp_path / "postings.csv").write_text("job_id,title\n1,AI Engineer\n", encoding="utf-8")
    (tmp_path / "companies" / "companies.csv").write_text("company_id\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="description"):
        import_kaggle(tmp_path, store)
