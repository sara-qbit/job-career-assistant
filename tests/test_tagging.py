import pytest

from tests.kaggle_fixture import TITLES, write_kaggle_dataset
from turing_projects_v2.config import PROJECT_ROOT
from turing_projects_v2.errors import DataValidationError
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.profile import TARGET_ROLES
from turing_projects_v2.store import JobStore
from turing_projects_v2.tagging import load_title_variants, tag_postings

VARIANTS_TOML = """\
[[role]]
name = "Quantum Software Developer"
include = ["quantum"]

[[role]]
name = "Grant Management Specialist"
include = ["grant", "grants"]
exclude = ["grants pass"]

[[role]]
name = "AI Innovation Manager"
include = ["innovation manager", "head of innovation"]

[[role]]
name = "AI Engineer"
include = ["machine learning engineer", "ai engineer", "ml engineer", "generative ai"]
exclude = ["ai content writer"]
"""


@pytest.fixture
def store(tmp_path):
    store = JobStore(":memory:")
    import_kaggle(write_kaggle_dataset(tmp_path / "raw"), store)
    return store


def variants_file(tmp_path, text=VARIANTS_TOML):
    path = tmp_path / "title_variants.toml"
    path.write_text(text, encoding="utf-8")
    return path


def roles_of(store):
    return {job_id: store.get_posting(job_id).target_role for job_id in TITLES}


def test_postings_are_tagged_by_title_variants(tmp_path, store):
    tag_postings(store, load_title_variants(variants_file(tmp_path)))

    roles = roles_of(store)
    assert roles[1001] == "AI Engineer"  # Machine Learning Engineer
    assert roles[1002] == "AI Engineer"  # Senior AI Engineer - Remote
    assert roles[1003] == "AI Engineer"  # AI/ML Engineer (Contract)
    assert roles[1020] == "AI Engineer"  # Generative AI Engineer
    assert roles[1006] == "Grant Management Specialist"
    assert roles[1007] == "Grant Management Specialist"
    assert roles[1009] == "AI Innovation Manager"
    assert roles[1010] == "AI Innovation Manager"
    assert roles[1011] == "Quantum Software Developer"


def test_non_matching_and_excluded_titles_stay_untagged(tmp_path, store):
    tag_postings(store, load_title_variants(variants_file(tmp_path)))

    roles = roles_of(store)
    assert roles[1004] is None  # Data Scientist: not in this variant list
    assert roles[1005] is None  # AI Content Writer: excluded
    assert roles[1008] is None  # store manager - Grants Pass, OR: excluded
    assert roles[1012] is None  # Search Engine Optimization Specialist
    assert roles[1013] is None  # Marketing Coordinator
    assert roles[1015] is None  # Software Engineer


def test_variants_match_whole_words_only(tmp_path, store):
    tag_postings(store, load_title_variants(variants_file(tmp_path)))

    assert roles_of(store)[1019] is None  # "Maintenance" contains "ai" but not as a word


def test_report_counts_tagged_postings_per_target_role(tmp_path, store):
    report = tag_postings(store, load_title_variants(variants_file(tmp_path)))

    assert report.total == len(TITLES)
    assert report.tagged == {
        "Quantum Software Developer": 1,
        "Grant Management Specialist": 2,
        "AI Innovation Manager": 2,
        "AI Engineer": 4,
    }
    assert report.untagged == len(TITLES) - 9


def test_retag_after_editing_the_list_needs_no_reimport(tmp_path, store):
    tag_postings(store, load_title_variants(variants_file(tmp_path)))
    edited = VARIANTS_TOML.replace('"generative ai"]', '"generative ai", "data scientist"]')

    tag_postings(store, load_title_variants(variants_file(tmp_path, edited)))

    assert store.get_posting(1004).target_role == "AI Engineer"


def test_retag_removes_tags_that_no_longer_match(tmp_path, store):
    tag_postings(store, load_title_variants(variants_file(tmp_path)))
    edited = VARIANTS_TOML.replace('include = ["quantum"]', 'include = ["qubit"]')

    tag_postings(store, load_title_variants(variants_file(tmp_path, edited)))

    assert store.get_posting(1011).target_role is None


def test_reimport_keeps_the_tags(tmp_path, store):
    tag_postings(store, load_title_variants(variants_file(tmp_path)))

    import_kaggle(tmp_path / "raw", store)

    assert store.get_posting(1001).target_role == "AI Engineer"


def test_variants_for_an_unknown_target_role_are_rejected(tmp_path):
    text = VARIANTS_TOML.replace("AI Innovation Manager", "Chef")

    with pytest.raises(DataValidationError, match="Chef"):
        load_title_variants(variants_file(tmp_path, text))


def test_candidates_variant_list_covers_every_target_role():
    variants = load_title_variants(PROJECT_ROOT / "candidate" / "title_variants.toml")

    assert {role.name for role in variants.roles} == TARGET_ROLES
