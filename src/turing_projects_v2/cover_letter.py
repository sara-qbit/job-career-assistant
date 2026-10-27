"""Cover Letter: the Letter Template of the posting's Target Role, filled in for one posting.

A pasted Job Posting takes its Target Role, language, employer and position from its Job Facts, so
a German ad gets the German Letter Template (never a translation). A Kaggle Job Posting gets the
English Letter Template of its tagged Target Role.

Only the gaps are filled. Candidate details and the job name, company and work type are copied
exactly by code; the LLM writes only the two summary gaps, each completing its template sentence
from the Job Info. The LLM never sees the candidate's personal details.
"""

import re
from datetime import UTC, date, datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from turing_projects_v2.errors import ConfigError, UserInputError
from turing_projects_v2.job_facts import create_job_facts
from turing_projects_v2.job_info import NOT_STATED, create_job_info
from turing_projects_v2.judge import Judge, Verdict
from turing_projects_v2.llm import ask_structured
from turing_projects_v2.profile import GAP_PATTERN, CandidateProfile, Language, NotStatedWording
from turing_projects_v2.prompts import load_prompt
from turing_projects_v2.store import (
    NOT_STATED_DE,
    Company,
    CoverLetter,
    JobFacts,
    JobInfo,
    JobStore,
    LetterCheck,
)

GERMAN_MONTHS = (
    "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
    "November", "Dezember",
)  # fmt: skip


class LetterGaps(BaseModel):
    company_vision: str = Field(
        description="the words that complete the template sentence ending in this gap, "
        "written from the company vision"
    )
    role_summary: str = Field(
        description="the words that complete the template sentence ending in this gap, "
        "written from the role summary"
    )


# The gaps the LLM writes; each has the name of the Job Info summary it is written from.
WRITTEN_GAPS = ("company_vision", "role_summary")


def write_cover_letter(
    job_id: int, store: JobStore, profile: CandidateProfile, llm: BaseChatModel, judge: Judge
) -> CoverLetter:
    """The Cover Letter for posting `job_id`, from its Job Info (created if not saved yet).

    The letter is saved with its checks, verdicts and date.
    """
    posting = store.get_posting(job_id)
    if posting is None:
        raise UserInputError(f"No Job Posting has job ID {job_id}.")
    facts = create_job_facts(posting, store, llm) if posting.source == "pasted" else None
    target_role = facts.target_role if facts else posting.target_role
    language: Language = facts.language if facts else "English"
    if target_role is None:
        raise UserInputError(
            f"Job Posting {job_id} has no Target Role, so it has no Letter Template."
        )
    try:
        template = profile.template_for(target_role, language)
    except ConfigError as error:
        # The candidate can fix this (add the template), so they get the exact message.
        raise UserInputError(str(error)) from error
    info = create_job_info(job_id, store, llm, judge)
    if facts:
        info = _with_job_facts(info, facts)
    if not info.company_name:
        raise UserInputError(
            f"Job Posting {job_id} has no company name, so no Cover Letter can be written for it."
        )
    company = store.get_company(posting.company_id) if posting.company_id is not None else None
    wording = profile.wording_for(template)
    gaps = _written_gaps(template.text, info, wording, llm)
    # Judged against the posting itself, so a fabricated Job Info summary cannot pass on.
    sources = {
        "company_vision": company.description if company else None,
        "role_summary": posting.description,
    }
    verdicts = {gap: _judged(judge, gap, gaps, info, sources) for gap in WRITTEN_GAPS}
    fabricated = {
        gap: verdict.reason
        for gap, verdict in verdicts.items()
        if verdict is not None and verdict.value == "fabricated"
    }
    if fabricated:
        rewrite = _written_gaps(template.text, info, wording, llm, fabricated, gaps)
        gaps = gaps.model_copy(update={gap: getattr(rewrite, gap) for gap in fabricated})
        verdicts |= {gap: _judged(judge, gap, gaps, info, sources) for gap in fabricated}
    written_at = datetime.now(UTC)
    local_date = written_at.astimezone()
    values = {
        "candidate_name": profile.name,
        "candidate_street": profile.street,
        "candidate_postal_code": profile.postal_code,
        "candidate_city": profile.city,
        "candidate_country": profile.country,
        "candidate_phone": profile.phone,
        "candidate_email": profile.email,
        "date": letter_date(local_date.date(), language),
        "github": profile.github,
        "company_name": info.company_name,
        "company_address": _address(company),
        "job_name": info.job_name,
        "work_type": info.work_type,
        "company_vision": gaps.company_vision,
        "role_summary": gaps.role_summary,
    }
    text = fill(template.text, values)
    letter = CoverLetter(
        job_id=job_id,
        target_role=target_role,
        template_name=template.name,
        text=text,
        checks=check_letter(text, template.text, info),
        company_vision_verdict=verdicts["company_vision"],
        role_summary_verdict=verdicts["role_summary"],
        rewritten=bool(fabricated),
        created_at=written_at,
    )
    store.save_cover_letter(letter)
    return letter


def _with_job_facts(info: JobInfo, facts: JobFacts) -> JobInfo:
    """`info` with the employer and position of the Job Facts, as far as the ad states them."""
    stated = {"company_name": facts.employer, "job_name": facts.position}
    return info.model_copy(
        update={field: value for field, value in stated.items() if value != NOT_STATED_DE}
    )


def letter_date(day: date, language: Language) -> str:
    """`day` written in the letter's language, e.g. "30 November 2026" or "30. November 2026"."""
    if language == "German":
        return f"{day.day}. {GERMAN_MONTHS[day.month - 1]} {day.year}"
    return f"{day.day} {day:%B %Y}"


def _written_gaps(
    template: str,
    info: JobInfo,
    wording: NotStatedWording,
    llm: BaseChatModel,
    fabricated: dict[str, str] | None = None,
    previous: LetterGaps | None = None,
) -> LetterGaps:
    """The summary gaps written from the Job Info, each ending its sentence with a full stop.

    A summary not stated in the posting gets the template's fixed `wording`. For a rewrite,
    `fabricated` holds the judge's reason per gap of the `previous` wording.
    """
    stated = {gap: getattr(info, gap) != NOT_STATED for gap in WRITTEN_GAPS}
    written: LetterGaps | NotStatedWording = wording
    if any(stated.values()):
        beginnings = "\n".join(
            f'- {gap} completes: "{_sentence_beginning(template, gap)}"' for gap in WRITTEN_GAPS
        )
        messages = [
            SystemMessage(load_prompt("cover_letter")),
            HumanMessage(
                f"<letter_template>\n{template}\n</letter_template>\n\n"
                f"<job_info_data>\nRole summary: {info.role_summary}\n"
                f"Company vision: {info.company_vision}\n</job_info_data>\n\n"
                f"Gaps to write:\n{beginnings}"
            ),
        ]
        if fabricated and previous:
            problems = "\n".join(
                f'- {gap}: "{getattr(previous, gap)}" ({reason})'
                for gap, reason in fabricated.items()
            )
            messages.append(
                HumanMessage(
                    "A judge found facts that are not in the Job Info in your wording:\n"
                    f"{problems}\nWrite these gaps again using only the Job Info."
                )
            )
        written = ask_structured(llm, LetterGaps, messages)
    return LetterGaps(
        **{
            gap: _sentence_end(getattr(written if is_stated else wording, gap))
            for gap, is_stated in stated.items()
        }
    )


def _sentence_beginning(template: str, gap: str) -> str:
    """The words of the template sentence before `{{gap}}`, e.g. "I am highly motivated to"."""
    before = template.split(f"{{{{{gap}}}}}", 1)[0]
    sentence = re.split(r"(?<=[.!?:])\s+|\n", before)[-1]
    return sentence.strip()


def _judged(
    judge: Judge, gap: str, gaps: LetterGaps, info: JobInfo, sources: dict[str, str | None]
) -> Verdict | None:
    """The judge's verdict on a written gap against the source text its Job Info summary was
    made from; None for fixed wording."""
    if getattr(info, gap) == NOT_STATED:
        return None
    return judge.check(getattr(gaps, gap), sources[gap] or getattr(info, gap))


def fill(text: str, values: dict[str, str | None]) -> str:
    """`text` with each {{gap}} replaced by its value.

    A gap without a value is left out: with its whole line if it stands alone on it, else with
    the space before it.
    """
    for gap, value in values.items():
        if value is None:
            text = re.sub(rf"^[ \t]*\{{\{{{gap}\}}\}}[ \t]*\n", "", text, flags=re.MULTILINE)
            text = re.sub(rf" ?\{{\{{{gap}\}}\}}", "", text)
    return GAP_PATTERN.sub(lambda match: values[match.group(1)] or "", text)


# The gaps copied exactly from the Job Info (gap and JobInfo field share the name), by check name.
EXACT_FIELDS = {"Job name": "job_name", "Company": "company_name", "Work type": "work_type"}


def check_letter(text: str, template: str, info: JobInfo) -> list[LetterCheck]:
    """Checks that the letter is the template with the Job Info's exact fields in their gaps.

    Each field is checked on its own, with the other gaps allowed to hold anything; the last
    check, "Letter Template text", needs all of them in place and the fixed text unchanged. A
    changed fixed text fails every check.
    """
    expected = {gap: getattr(info, gap) for gap in EXACT_FIELDS.values()}
    checks = [
        LetterCheck(
            name=name,
            passed=_letter_pattern(template, {gap: expected[gap]}).fullmatch(text) is not None,
        )
        for name, gap in EXACT_FIELDS.items()
    ]
    whole = _letter_pattern(template, expected).fullmatch(text) is not None
    return [*checks, LetterCheck(name="Letter Template text", passed=whole)]


def _letter_pattern(template: str, pinned: dict[str, str | None]) -> re.Pattern[str]:
    """A pattern for letters filled from `template` (see `fill`).

    Pinned gaps must hold exactly their value (None: left out); any other gap may hold any one
    line of text or be left out.
    """
    parts: list[str] = []
    position = 0
    for match in GAP_PATTERN.finditer(template):
        gap = match.group(1)
        fixed = template[position : match.start()]
        position = match.end()
        alone = (not fixed or fixed.endswith("\n")) and template.startswith("\n", position)
        if alone:
            position += 1  # the gap's line break goes with the gap
        spaced = not alone and fixed.endswith(" ")
        if spaced:
            fixed = fixed[:-1]  # the space before the gap goes with the gap
        parts.append(re.escape(fixed))
        if gap in pinned and pinned[gap] is None:
            continue
        value = re.escape(pinned[gap]) if gap in pinned else r"[^\n]*?"
        if alone:
            parts.append(rf"{value}\n" if gap in pinned else rf"(?:{value}\n)?")
        elif spaced:
            parts.append(f" {value}" if gap in pinned else f"(?: {value})?")
        else:
            parts.append(value)
    parts.append(re.escape(template[position:]))
    return re.compile("".join(parts))


def _address(company: Company | None) -> str | None:
    """The company's address on one line, from the fields the data knows; None without any."""
    if company is None:
        return None

    def known(part: str | None) -> str | None:
        # The Kaggle data writes "0" or "-" for an unknown address field.
        return part if part and part.strip() not in ("0", "-") else None

    region = " ".join(part for part in (known(company.state), known(company.zip_code)) if part)
    parts = [known(company.address), known(company.city), region, known(company.country)]
    return ", ".join(part for part in parts if part) or None


def _sentence_end(phrase: str) -> str:
    """`phrase` ending with exactly one full stop (it ends a template sentence)."""
    return phrase.strip().rstrip(".") + "."
