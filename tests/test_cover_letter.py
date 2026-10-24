from datetime import UTC, datetime

import pytest

from tests.kaggle_fixture import posting, write_kaggle_dataset
from turing_projects_v2.cover_letter import LetterGaps, check_letter, write_cover_letter
from turing_projects_v2.errors import UserInputError
from turing_projects_v2.fakes import FakeChatModel, FakeEmbedder, FakeJudge
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.job_info import NOT_STATED
from turing_projects_v2.judge import Verdict
from turing_projects_v2.profile import CandidateProfile, LetterTemplate
from turing_projects_v2.rag.indexer import open_collection
from turing_projects_v2.store import JobInfo, JobStore
from turing_projects_v2.tools.jobs import job_tools

TEMPLATE = """\
{{candidate_name}}, {{candidate_street}}, {{candidate_city}}, {{candidate_postal_code}}
{{candidate_country}}, Tel.: {{candidate_phone}}

{{company_name}}
{{company_address}}

I am writing to apply for {{job_name}} {{work_type}} at {{company_name}}. I support achieving \
the {{company_vision}}
My projects are at {{github}}. I am highly motivated to {{role_summary}}
I would like to contribute to {{company_name}}.
Yours sincerely,
{{candidate_name}}
"""


GRANT_TEMPLATE = """\
{{candidate_name}}
Dear Hiring Manager,
I apply for the {{work_type}} position of {{job_name}} at {{company_name}}. I want to support \
committees in {{role_summary}}
I am eager to extend my experience to {{company_vision}}
"""


def profile(template=TEMPLATE, **options):
    return CandidateProfile(
        name="Ada Example",
        street="Main Street 1",
        postal_code="12345",
        city="Sample Town",
        country="Germany",
        phone="0123 456",
        email="ada@example.com",
        github="https://github.com/ada",
        target_roles={
            "AI Engineer": "ai",
            "AI Innovation Manager": "ai",
            "Quantum Software Developer": "ai",
            "Grant Management Specialist": "grant",
        },
        letter_templates={
            "ai": LetterTemplate(name="ai", text=template),
            "grant": LetterTemplate(name="grant", text=GRANT_TEMPLATE),
        },
        **options,
    )


ADDRESS = {"address": "1 Rocket Way", "city": "Austin", "state": "TX", "zip_code": "78701"}


def letter_store(tmp_path, company=ADDRESS, path=":memory:", **job_info):
    """A store with tagged AI Engineer job 1 at Acme Corp in Austin, and its saved Job Info."""
    store = JobStore(path)
    raw = write_kaggle_dataset(tmp_path / "raw", [posting(1, "AI Engineer")], company)
    import_kaggle(raw, store)
    store.set_target_roles([(1, "AI Engineer")])
    info = {
        "job_id": 1,
        "job_name": "AI Engineer",
        "company_name": "Acme Corp",
        "work_type": "Full-time",
        "role_summary": "The engineer builds RAG pipelines in Python for clients.",
        "company_vision": "Acme builds rockets to make space travel affordable.",
        "role_summary_verdict": None,
        "company_vision_verdict": None,
    }
    store.save_job_info(JobInfo(**info | job_info))
    return store


GAPS = LetterGaps(
    company_vision="company's goal of making space travel affordable with its rockets",
    role_summary="build RAG pipelines in Python for your clients",
)


def test_only_the_gaps_of_the_letter_template_are_filled(tmp_path):
    llm = FakeChatModel(responses=[GAPS])

    letter = write_cover_letter(1, letter_store(tmp_path), profile(), llm, FakeJudge())

    assert letter.text == (
        "Ada Example, Main Street 1, Sample Town, 12345\n"
        "Germany, Tel.: 0123 456\n"
        "\n"
        "Acme Corp\n"
        "1 Rocket Way, Austin, TX 78701\n"
        "\n"
        "I am writing to apply for AI Engineer Full-time at Acme Corp. I support achieving the "
        "company's goal of making space travel affordable with its rockets.\n"
        "My projects are at https://github.com/ada. I am highly motivated to build RAG pipelines "
        "in Python for your clients.\n"
        "I would like to contribute to Acme Corp.\n"
        "Yours sincerely,\n"
        "Ada Example\n"
    )
    assert (letter.job_id, letter.target_role, letter.template_name) == (1, "AI Engineer", "ai")


def test_a_correct_letter_passes_every_check(tmp_path):
    llm = FakeChatModel(responses=[GAPS])

    letter = write_cover_letter(1, letter_store(tmp_path), profile(), llm, FakeJudge())

    assert [(check.name, check.passed) for check in letter.checks] == [
        ("Job name", True),
        ("Company", True),
        ("Work type", True),
        ("Letter Template text", True),
    ]


@pytest.mark.parametrize(
    ("wrong", "right", "failing"),
    [
        ("for AI Engineer Full-time", "for ML Engineer Full-time", "Job name"),
        ("at Acme Corp.", "at Acme Inc.", "Company"),
        ("Engineer Full-time at", "Engineer Contract at", "Work type"),
        ("I would like to contribute", "I would love to contribute", "Letter Template text"),
    ],
)
def test_checks_fail_when_the_letter_differs_from_the_job_info_or_template(
    tmp_path, wrong, right, failing
):
    store = letter_store(tmp_path)
    letter = write_cover_letter(1, store, profile(), FakeChatModel(responses=[GAPS]), FakeJudge())
    tampered = letter.text.replace(wrong, right)
    assert tampered != letter.text

    checks = check_letter(tampered, TEMPLATE, store.get_job_info(1))

    failed = {check.name for check in checks if not check.passed}
    assert failing in failed
    if failing != "Letter Template text":  # a changed fixed text fails every check
        assert failed <= {failing, "Letter Template text"}


def test_missing_work_type_and_address_are_left_out_and_still_pass_the_checks(tmp_path):
    store = letter_store(tmp_path, company={}, work_type=None)

    letter = write_cover_letter(1, store, profile(), FakeChatModel(responses=[GAPS]), FakeJudge())

    assert "Acme Corp\n\nI am writing to apply for AI Engineer at Acme Corp. I" in letter.text
    assert all(check.passed for check in letter.checks)


def test_summaries_not_stated_in_the_posting_get_neutral_fixed_wording(tmp_path):
    store = letter_store(tmp_path, role_summary=NOT_STATED, company_vision=NOT_STATED)

    letter = write_cover_letter(1, store, profile(), FakeChatModel(responses=[]), FakeJudge())

    assert "I support achieving the company's goals.\n" in letter.text
    assert "I am highly motivated to contribute to this role.\n" in letter.text


def test_only_the_stated_summary_is_written_by_the_llm(tmp_path):
    store = letter_store(tmp_path, company_vision=NOT_STATED)

    letter = write_cover_letter(1, store, profile(), FakeChatModel(responses=[GAPS]), FakeJudge())

    assert "I support achieving the company's goals.\n" in letter.text
    assert (
        "I am highly motivated to build RAG pipelines in Python for your clients.\n" in letter.text
    )


def test_a_posting_without_company_name_gets_no_letter(tmp_path):
    store = letter_store(tmp_path, company_name=None)

    with pytest.raises(UserInputError, match="no company name"):
        write_cover_letter(1, store, profile(), FakeChatModel(responses=[]), FakeJudge())


SUPPORTED = Verdict(value="supported", reason="Stated in the Job Info.")
FABRICATED = Verdict(value="fabricated", reason="Affordable is not in the Job Info.")
REWRITTEN = LetterGaps(
    company_vision="company's goal of building rockets for affordable space travel",
    role_summary="build RAG pipelines in Python for your clients",
)


def test_written_gaps_are_judged_against_the_posting_and_company_description(tmp_path):
    judge = FakeJudge([SUPPORTED, SUPPORTED])
    llm = FakeChatModel(responses=[GAPS])

    letter = write_cover_letter(1, letter_store(tmp_path), profile(), llm, judge)

    assert judge.checked == [
        ("company's goal of making space travel affordable with its rockets.", "We build rockets."),
        (
            "build RAG pipelines in Python for your clients.",
            "Description of job 1.\nSecond line, with a comma.",
        ),
    ]
    assert (letter.company_vision_verdict, letter.role_summary_verdict) == (SUPPORTED, SUPPORTED)
    assert not letter.rewritten


def test_a_fabricated_gap_is_rewritten_once_with_the_judges_reason(tmp_path):
    judge = FakeJudge([FABRICATED, SUPPORTED, SUPPORTED])
    llm = FakeChatModel(responses=[GAPS, REWRITTEN])

    letter = write_cover_letter(1, letter_store(tmp_path), profile(), llm, judge)

    assert "Affordable is not in the Job Info." in llm.calls[1][-1].text
    assert "achieving the company's goal of building rockets for affordable space travel." in (
        letter.text
    )
    assert judge.checked[2][0] == "company's goal of building rockets for affordable space travel."
    assert letter.rewritten
    assert letter.company_vision_verdict == SUPPORTED
    assert all(check.passed for check in letter.checks)


def test_a_gap_still_fabricated_after_the_rewrite_is_kept_with_the_verdict(tmp_path):
    judge = FakeJudge([FABRICATED, SUPPORTED, FABRICATED])
    llm = FakeChatModel(responses=[GAPS, REWRITTEN])

    letter = write_cover_letter(1, letter_store(tmp_path), profile(), llm, judge)

    assert letter.rewritten
    assert letter.company_vision_verdict == FABRICATED
    assert letter.role_summary_verdict == SUPPORTED
    assert "company's goal of building rockets for affordable space travel." in letter.text


def test_fixed_wording_is_not_judged(tmp_path):
    store = letter_store(tmp_path, company_vision=NOT_STATED)
    judge = FakeJudge()

    letter = write_cover_letter(1, store, profile(), FakeChatModel(responses=[GAPS]), judge)

    assert [statement for statement, _ in judge.checked] == [
        "build RAG pipelines in Python for your clients."
    ]
    assert letter.company_vision_verdict is None


def test_each_letter_is_saved_with_its_checks_verdicts_and_date(tmp_path):
    database = tmp_path / "jobs.sqlite"
    store = letter_store(tmp_path, path=database)
    judge = FakeJudge([FABRICATED, SUPPORTED, FABRICATED])
    first = write_cover_letter(
        1, store, profile(), FakeChatModel(responses=[GAPS, REWRITTEN]), judge
    )
    second = write_cover_letter(1, store, profile(), FakeChatModel(responses=[GAPS]), FakeJudge())

    saved = JobStore(database).cover_letters(1)

    assert saved == [first, second]
    assert saved[0].company_vision_verdict == FABRICATED
    assert saved[0].created_at.date() == datetime.now(UTC).date()


def test_a_target_role_without_letter_template_gets_a_clear_message(tmp_path):
    store = letter_store(tmp_path)
    store.set_target_roles([(1, "Grant Management Specialist")])
    without_grant = profile().model_copy(
        update={"letter_templates": {"ai": LetterTemplate(name="ai", text=TEMPLATE)}}
    )

    with pytest.raises(UserInputError, match=r"No Letter Template for .*Grant Management"):
        write_cover_letter(1, store, without_grant, FakeChatModel(responses=[]), FakeJudge())


def test_kaggle_placeholders_in_the_company_address_are_left_out(tmp_path):
    company = {"address": "0", "city": "Sao Paulo", "state": "-", "zip_code": "0", "country": "BR"}
    store = letter_store(tmp_path, company=company)

    letter = write_cover_letter(1, store, profile(), FakeChatModel(responses=[GAPS]), FakeJudge())

    assert "\nAcme Corp\nSao Paulo, BR\n" in letter.text


def test_the_tool_reply_warns_about_fabricated_text_without_the_letter(tmp_path):
    judge = FakeJudge([FABRICATED, SUPPORTED, FABRICATED])
    llm = FakeChatModel(responses=[GAPS, REWRITTEN])
    collection = open_collection(tmp_path / "chroma", "fake")
    tools = job_tools(letter_store(tmp_path), llm, collection, FakeEmbedder(), judge, profile())
    (write,) = [tool for tool in tools if tool.name == "write_cover_letter"]

    reply = write.invoke(
        {"type": "tool_call", "name": "write_cover_letter", "args": {"job_id": 1}, "id": "c1"}
    )

    assert "All exact checks passed." in reply.text
    assert "Still judged fabricated" in reply.text
    assert "Company vision: Affordable is not in the Job Info." in reply.text
    assert "Ada Example" not in reply.text and "Main Street" not in reply.text
    assert reply.artifact["text"].startswith("Ada Example, Main Street 1")


GRANT_GAPS = LetterGaps(
    company_vision="the field of affordable space travel",
    role_summary="building RAG pipelines in Python for clients",
)


def test_grant_jobs_get_the_grant_template_written_to_its_own_sentences(tmp_path):
    store = letter_store(tmp_path)
    store.set_target_roles([(1, "Grant Management Specialist")])
    llm = FakeChatModel(responses=[GRANT_GAPS])

    letter = write_cover_letter(1, store, profile(), llm, FakeJudge())

    assert letter.template_name == "grant"
    assert letter.text == (
        "Ada Example\n"
        "Dear Hiring Manager,\n"
        "I apply for the Full-time position of AI Engineer at Acme Corp. I want to support "
        "committees in building RAG pipelines in Python for clients.\n"
        "I am eager to extend my experience to the field of affordable space travel.\n"
    )
    assert all(check.passed for check in letter.checks)
    prompt = llm.calls[0][-1].text
    assert '- role_summary completes: "I want to support committees in"' in prompt
    assert '- company_vision completes: "I am eager to extend my experience to"' in prompt


def test_summaries_not_stated_get_the_templates_own_wording(tmp_path):
    store = letter_store(tmp_path, role_summary=NOT_STATED, company_vision=NOT_STATED)
    store.set_target_roles([(1, "Grant Management Specialist")])
    wording = {"grant": {"role_summary": "the duties of this role", "company_vision": "your field"}}

    letter = write_cover_letter(
        1, store, profile(not_stated_wording=wording), FakeChatModel(responses=[]), FakeJudge()
    )

    assert "support committees in the duties of this role.\n" in letter.text
    assert "extend my experience to your field.\n" in letter.text
