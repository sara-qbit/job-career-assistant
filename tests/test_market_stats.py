import pytest

from tests.kaggle_fixture import posting, write_kaggle_dataset
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.store import JobStore
from turing_projects_v2.tools.stats import market_stats_tool

# job ID: (title, Target Role, work type, remote, location, yearly salary, currency)
POSTINGS = {
    1: ("AI Engineer", "AI Engineer", "Full-time", "1", "Austin, TX", "100000", "USD"),
    2: ("ML Engineer", "AI Engineer", "Contract", "", "Austin, TX", "140000", "USD"),
    3: ("LLM Engineer", "AI Engineer", "Full-time", "", "New York, NY", "120000", "USD"),
    4: (
        "Grants Manager",
        "Grant Management Specialist",
        "Full-time",
        "1",
        "New York, NY",
        "60000",
        "USD",
    ),
    5: ("Grant Writer", "Grant Management Specialist", "Part-time", "", "Boston, MA", "", ""),
    6: ("Nurse", None, "Full-time", "", "Austin, TX", "50000", "USD"),
    7: ("Accountant", None, "Contract", "1", "Denver, CO", "90000", "EUR"),
}


@pytest.fixture
def stats(tmp_path):
    """The `job_market_stats` tool on a store with five Target-Role and two other postings."""
    rows = [
        posting(
            job_id,
            title,
            formatted_work_type=work_type,
            remote_allowed=remote,
            location=location,
            normalized_salary=salary,
            currency=currency,
        )
        for job_id, (title, _, work_type, remote, location, salary, currency) in POSTINGS.items()
    ]
    store = JobStore(":memory:")
    import_kaggle(write_kaggle_dataset(tmp_path / "raw", rows), store)
    store.set_target_roles((job_id, values[1]) for job_id, values in POSTINGS.items())
    return market_stats_tool(store)


def ask(tool, **args):
    """The tool's reply to a call with `args`, as the agent would make it."""
    return tool.invoke({"type": "tool_call", "name": tool.name, "args": args, "id": "c1"})


def counts(reply):
    return {row["label"]: row["count"] for row in reply.artifact["rows"]}


def test_postings_per_target_role_are_counted_over_target_role_postings(stats):
    reply = ask(stats, statistic="postings_by_target_role")

    assert counts(reply) == {"AI Engineer": 3, "Grant Management Specialist": 2}
    assert reply.artifact["population"] == "the 5 Target-Role postings"
    assert "the 5 Target-Role postings" in reply.text


def test_postings_per_work_type_over_all_postings(stats):
    reply = ask(stats, statistic="postings_by_work_type", population="all_postings")

    assert counts(reply) == {"Full-time": 4, "Contract": 2, "Part-time": 1}
    assert reply.artifact["population"] == "all 7 postings"


def test_remote_postings_are_counted(stats):
    reply = ask(stats, statistic="postings_by_remote")

    assert counts(reply) == {"On-site or not stated": 3, "Remote": 2}


def test_top_locations_keep_the_whole_population_in_the_label(stats):
    reply = ask(stats, statistic="postings_by_location", population="all_postings", limit=2)

    assert counts(reply) == {"Austin, TX": 3, "New York, NY": 2}
    assert reply.artifact["population"] == "all 7 postings"


def test_filters_narrow_the_population_and_are_named_in_it(stats):
    reply = ask(stats, statistic="postings_by_work_type", target_role="AI Engineer")

    assert counts(reply) == {"Full-time": 2, "Contract": 1}
    assert reply.artifact["population"] == "the 3 Target-Role postings for AI Engineer"

    reply = ask(
        stats, statistic="postings_by_location", population="all_postings", work_type="Contract"
    )

    assert counts(reply) == {"Austin, TX": 1, "Denver, CO": 1}
    assert reply.artifact["population"] == "all 2 postings with work type Contract"


@pytest.mark.parametrize(
    "args",
    [
        {"statistic": "DROP TABLE postings"},
        {"statistic": "postings_by_location", "limit": 0},
        {"statistic": "postings_by_location", "limit": 26},
        {"statistic": "postings_by_work_type", "target_role": "Chef"},
        {"statistic": "postings_by_work_type", "population": "everything"},
    ],
)
def test_invalid_parameters_are_rejected(stats, args):
    with pytest.raises(ValueError):
        ask(stats, **args)


def test_salary_ranges_per_target_role_use_stated_usd_salaries_only(stats):
    reply = ask(stats, statistic="salary_by_target_role")

    rows = {row["label"]: row for row in reply.artifact["rows"]}
    assert rows["AI Engineer"] == {
        "label": "AI Engineer",
        "count": 3,
        "p25": 110000.0,
        "median": 120000.0,
        "p75": 130000.0,
    }
    assert (
        rows["Grant Management Specialist"]["count"],
        rows["Grant Management Specialist"]["median"],
    ) == (1, 60000.0)
    assert (
        reply.artifact["population"] == "the 4 Target-Role postings that state a yearly USD salary"
    )
    assert "median $120,000" in reply.text


def test_salary_over_all_postings_leaves_out_other_currencies(stats):
    reply = ask(stats, statistic="salary_by_target_role", population="all_postings")

    assert counts(reply) == {
        "AI Engineer": 3,
        "Grant Management Specialist": 1,
        "No Target Role": 1,
    }
