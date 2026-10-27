"""Candidate Profile and Letter Templates, loaded from the candidate directory.

Layout of the candidate directory:
    profile.toml          candidate details and the Target Role -> Letter Template mapping
    templates/<name>.txt  Letter Templates; gaps are marked as {{gap_name}}
"""

import re
import tomllib
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from turing_projects_v2.errors import ConfigError, DataValidationError

type TargetRole = Literal[
    "AI Engineer",
    "AI Innovation Manager",
    "Quantum Software Developer",
    "Grant Management Specialist",
]

TARGET_ROLES: frozenset[str] = frozenset(get_args(TargetRole.__value__))

GAP_PATTERN = re.compile(r"\{\{(\w+)\}\}")

PROFILE_GAPS = frozenset(
    {
        "candidate_name",
        "candidate_street",
        "candidate_postal_code",
        "candidate_city",
        "candidate_country",
        "candidate_phone",
        "candidate_email",
        "github",
    }
)
JOB_GAPS = frozenset(
    {"company_name", "company_address", "job_name", "work_type", "company_vision", "role_summary"}
)
LETTER_GAPS = frozenset({"date"})
"""Filled when the letter is written: the day, e.g. "25 October 2026"."""
KNOWN_GAPS = PROFILE_GAPS | JOB_GAPS | LETTER_GAPS


class LetterTemplate(BaseModel):
    """Fixed cover-letter text; only the marked gaps are ever filled in."""

    model_config = ConfigDict(frozen=True)

    name: str
    text: str

    @property
    def gaps(self) -> set[str]:
        return set(GAP_PATTERN.findall(self.text))


class NotStatedWording(BaseModel):
    """A Letter Template's wording for a summary gap whose Job Info summary is not stated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    company_vision: str = "company's goals"
    role_summary: str = "contribute to this role"


class CandidateProfile(BaseModel):
    """The facts about the candidate, supplied by the candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    street: str
    postal_code: str
    city: str
    country: str
    phone: str
    email: str
    github: str
    target_roles: dict[str, str]
    letter_templates: dict[str, LetterTemplate] = {}
    not_stated_wording: dict[str, NotStatedWording] = {}
    """Per Letter Template name; templates without an entry get the default wording."""

    @field_validator("target_roles")
    @classmethod
    def every_target_role_is_mapped(cls, target_roles: dict[str, str]) -> dict[str, str]:
        unmapped = TARGET_ROLES - target_roles.keys()
        if unmapped:
            raise ValueError(f"no Letter Template mapped for {', '.join(sorted(unmapped))}")
        return target_roles

    @model_validator(mode="after")
    def wording_is_for_known_templates(self) -> CandidateProfile:
        unknown = self.not_stated_wording.keys() - set(self.target_roles.values())
        if unknown:
            raise ValueError(
                f"not_stated_wording for unknown templates: {', '.join(sorted(unknown))}"
            )
        return self

    @property
    def missing_templates(self) -> dict[str, str]:
        """Target Roles whose Letter Template file does not exist yet (role -> template name)."""
        return {
            role: template
            for role, template in self.target_roles.items()
            if template not in self.letter_templates
        }

    def wording_for(self, template: LetterTemplate) -> NotStatedWording:
        """The template's wording for summary gaps whose Job Info summary is not stated."""
        return self.not_stated_wording.get(template.name, NotStatedWording())

    def template_for(self, target_role: str) -> LetterTemplate:
        if target_role not in self.target_roles:
            known = ", ".join(self.target_roles)
            raise ConfigError(f"Unknown Target Role '{target_role}'. Known roles: {known}.")
        template_name = self.target_roles[target_role]
        if template_name not in self.letter_templates:
            raise ConfigError(
                f"No Letter Template for Target Role '{target_role}': "
                f"add templates/{template_name}.txt to the candidate directory."
            )
        return self.letter_templates[template_name]


def load_profile(candidate_dir: Path) -> CandidateProfile:
    """Load and validate the Candidate Profile and every Letter Template it refers to."""
    profile_path = candidate_dir / "profile.toml"
    try:
        raw = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Candidate Profile not found: {profile_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise DataValidationError(f"{profile_path} is not valid TOML: {error}") from error

    templates = {}
    for template_name in set(raw.get("target_roles", {}).values()):
        template_path = candidate_dir / "templates" / f"{template_name}.txt"
        if template_path.exists():
            templates[template_name] = _load_template(template_name, template_path)

    try:
        return CandidateProfile(**raw, letter_templates=templates)
    except PydanticValidationError as error:
        raise DataValidationError(
            f"Invalid Candidate Profile in {profile_path}: {error}"
        ) from error


def _load_template(name: str, path: Path) -> LetterTemplate:
    template = LetterTemplate(name=name, text=path.read_text(encoding="utf-8"))
    unknown = template.gaps - KNOWN_GAPS
    if unknown:
        raise DataValidationError(
            f"Letter Template {path} has unknown gaps: {', '.join(sorted(unknown))}. "
            f"Known gaps: {', '.join(sorted(KNOWN_GAPS))}."
        )
    return template
