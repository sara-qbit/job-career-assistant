import pytest

from tests.kaggle_fixture import posting, write_kaggle_dataset
from turing_projects_v2.errors import UserInputError
from turing_projects_v2.fakes import FakeChatModel, FakeJudge
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.job_info import NOT_STATED, JobSummaries, create_job_info
from turing_projects_v2.judge import Verdict
from turing_projects_v2.store import JobStore

TITLE = "Sr. AI/ML Engineer – (Remote)  "  # noqa: RUF001 (an exact copy keeps the en dash)
DESCRIPTION = "You will build RAG pipelines in Python for our clients."


def job_store(tmp_path, path=":memory:", **overrides):
    """A store with job 1 at Acme Corp, whose company description is "We build rockets."."""
    store = JobStore(path)
    row = posting(
        1,
        TITLE,
        company_name="ACME corp, Inc.",
        formatted_work_type="Contract",
        description=DESCRIPTION,
    )
    import_kaggle(write_kaggle_dataset(tmp_path / "raw", [row | overrides]), store)
    return store


def summaries(role="Builds RAG pipelines in Python.", vision="Acme builds rockets."):
    return JobSummaries(role_summary=role, company_vision=vision)


def test_job_name_company_and_work_type_are_copied_exactly(tmp_path):
    llm = FakeChatModel(responses=[summaries()])

    info = create_job_info(1, job_store(tmp_path), llm, FakeJudge())

    assert info.job_id == 1
    assert info.job_name == TITLE
    assert info.company_name == "ACME corp, Inc."
    assert info.work_type == "Contract"
    assert info.role_summary == "Builds RAG pipelines in Python."
    assert info.company_vision == "Acme builds rockets."


def test_summaries_are_made_from_the_posting_and_company_description_only(tmp_path):
    llm = FakeChatModel(responses=[summaries()])

    create_job_info(1, job_store(tmp_path), llm, FakeJudge())

    (sent,) = llm.calls
    prompt = "\n".join(message.text for message in sent)
    assert f"<job_posting_data>\n{DESCRIPTION}\n</job_posting_data>" in prompt
    assert "<company_data>\nWe build rockets.\n</company_data>" in prompt


@pytest.mark.parametrize("company_id", ["", "999"])
def test_company_vision_is_not_stated_without_a_company_description(tmp_path, company_id):
    llm = FakeChatModel(responses=[summaries(vision="Acme is a leader in space travel.")])
    store = job_store(tmp_path, company_id=company_id)

    info = create_job_info(1, store, llm, FakeJudge())

    assert info.company_vision == NOT_STATED
    assert info.role_summary == "Builds RAG pipelines in Python."


def test_without_any_source_text_nothing_is_asked(tmp_path):
    llm = FakeChatModel(responses=[])
    store = job_store(tmp_path, company_id="", description="")

    info = create_job_info(1, store, llm, FakeJudge())

    assert (info.role_summary, info.company_vision) == (NOT_STATED, NOT_STATED)
    assert info.job_name == TITLE


def test_each_summary_is_judged_against_its_own_source(tmp_path):
    llm = FakeChatModel(responses=[summaries()])
    judge = FakeJudge(
        [
            Verdict(value="supported", reason="The posting mentions RAG and Python."),
            Verdict(value="fabricated", reason="Rockets are mentioned, travel is not."),
        ]
    )

    info = create_job_info(1, job_store(tmp_path), llm, judge)

    assert judge.checked == [
        ("Builds RAG pipelines in Python.", DESCRIPTION),
        ("Acme builds rockets.", "We build rockets."),
    ]
    assert info.role_summary_verdict == Verdict(
        value="supported", reason="The posting mentions RAG and Python."
    )
    assert info.company_vision_verdict == Verdict(
        value="fabricated", reason="Rockets are mentioned, travel is not."
    )


def test_a_not_stated_summary_of_an_existing_source_is_judged_as_that_claim(tmp_path):
    llm = FakeChatModel(responses=[summaries(vision=NOT_STATED)])
    judge = FakeJudge(
        [
            Verdict(value="supported", reason="RAG and Python are named."),
            Verdict(value="fabricated", reason="The company says it builds rockets."),
        ]
    )

    info = create_job_info(1, job_store(tmp_path), llm, judge)

    assert judge.checked[1] == (
        "The source says nothing about the company's vision.",
        "We build rockets.",
    )
    assert info.company_vision == NOT_STATED
    assert info.company_vision_verdict.value == "fabricated"


def test_a_summary_without_source_is_not_judged(tmp_path):
    llm = FakeChatModel(responses=[summaries()])
    judge = FakeJudge()

    info = create_job_info(1, job_store(tmp_path, company_id=""), llm, judge)

    assert judge.checked == [("Builds RAG pipelines in Python.", DESCRIPTION)]
    assert info.company_vision_verdict is None


def test_job_info_is_saved_and_reused_not_regenerated(tmp_path):
    database = tmp_path / "jobs.sqlite"
    judge = FakeJudge([Verdict(value="fabricated", reason="Not in the posting.")])
    first = create_job_info(
        1, job_store(tmp_path, database), FakeChatModel(responses=[summaries()]), judge
    )

    reopened = JobStore(database)
    again = create_job_info(1, reopened, FakeChatModel(responses=[]), FakeJudge())

    assert again == first
    assert again.role_summary_verdict == Verdict(value="fabricated", reason="Not in the posting.")


def test_unknown_job_id_is_a_user_input_error(tmp_path):
    with pytest.raises(UserInputError, match="No Job Posting has job ID 999"):
        create_job_info(999, job_store(tmp_path), FakeChatModel(responses=[]), FakeJudge())
