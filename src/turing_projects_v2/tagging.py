"""Target Role tagging: matches posting titles against the reviewed list of title variants.

A posting gets the first Target Role (in file order) that has an `include` phrase in its title
and no `exclude` phrase. Phrases match case-insensitively and as whole words only.
"""

import re
import tomllib
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import ValidationError as PydanticValidationError

from turing_projects_v2.errors import ConfigError, DataValidationError
from turing_projects_v2.profile import TARGET_ROLES
from turing_projects_v2.store import JobStore


class RoleVariants(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    include: list[str]
    exclude: list[str] = []

    @field_validator("name")
    @classmethod
    def is_a_target_role(cls, name: str) -> str:
        if name not in TARGET_ROLES:
            raise ValueError(f"'{name}' is not a Target Role ({', '.join(sorted(TARGET_ROLES))})")
        return name

    def matches(self, title: str) -> bool:
        return bool(_phrases(tuple(self.include)).search(title)) and not (
            self.exclude and _phrases(tuple(self.exclude)).search(title)
        )


class TitleVariants(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    roles: list[RoleVariants]

    def target_role_for(self, title: str) -> str | None:
        return next((role.name for role in self.roles if role.matches(title)), None)


class TagReport(BaseModel):
    total: int
    tagged: dict[str, int]
    untagged: int


def load_title_variants(path: Path) -> TitleVariants:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"Title variant list not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise DataValidationError(f"{path} is not valid TOML: {error}") from error
    try:
        return TitleVariants(roles=raw.get("role", []))
    except PydanticValidationError as error:
        raise DataValidationError(f"Invalid title variant list in {path}: {error}") from error


def tag_postings(store: JobStore, variants: TitleVariants) -> TagReport:
    """Set every posting's Target Role (or none) from its title; replaces earlier tags."""
    store.set_target_roles(
        (job_id, variants.target_role_for(title)) for job_id, title in store.all_titles()
    )
    counts = store.count_by_target_role()
    tagged = {role.name: counts.get(role.name, 0) for role in variants.roles}
    return TagReport(total=sum(counts.values()), tagged=tagged, untagged=counts.get(None, 0))


@cache
def _phrases(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """One case-insensitive whole-word pattern matching any of `phrases`."""
    alternatives = "|".join(re.escape(phrase) for phrase in phrases)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)
