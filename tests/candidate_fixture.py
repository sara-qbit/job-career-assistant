"""Writes the repo's example candidate directory: the example profile and Letter Templates.

Locally, candidate/profile.toml may be the candidate's private profile (git skip-worktree), so
tests that need the shipped example build their own copy: the example profile from the text
below, and the example Letter Templates copied from candidate/templates/, where they are tracked.
"""

import shutil
from pathlib import Path

from turing_projects_v2.config import PROJECT_ROOT

# The example candidate/profile.toml committed in the repo, without its comments; keep them alike.
EXAMPLE_PROFILE_TOML = """\
name = "Ada Example"
street = "Example Street 1"
postal_code = "12345"
city = "Sample Town"
country = "Germany"
phone = "0123 4567890"
email = "ada@example.com"
github = "https://github.com/sara-qbit"

[target_roles]
"AI Engineer" = { English = "ai", German = "ai_de" }
"AI Innovation Manager" = { English = "ai", German = "ai_de" }
"Quantum Software Developer" = { English = "ai", German = "ai_de" }
"Grant Management Specialist" = { English = "grant", German = "grant_de" }

[not_stated_wording.ai]
company_vision = "company's goals"
role_summary = "contribute to this role"

[not_stated_wording.grant]
company_vision = "your organisation's field of work"
role_summary = "fulfilling the responsibilities of this role"

[not_stated_wording.ai_de]
company_vision = "der Verwirklichung Ihrer Unternehmensziele"
role_summary = "zu dieser Stelle beizutragen"

[not_stated_wording.grant_de]
company_vision = "der Verwirklichung Ihrer Ziele"
role_summary = "die Aufgaben dieser Stelle zu übernehmen"
"""

EXAMPLE_TEMPLATES = ("ai", "grant", "ai_de", "grant_de")


def write_example_candidate_dir(candidate_dir: Path) -> Path:
    """Write the example profile and copy the example Letter Templates into `candidate_dir`."""
    (candidate_dir / "templates").mkdir(parents=True, exist_ok=True)
    (candidate_dir / "profile.toml").write_text(EXAMPLE_PROFILE_TOML, encoding="utf-8")
    for name in EXAMPLE_TEMPLATES:
        shutil.copy(
            PROJECT_ROOT / "candidate" / "templates" / f"{name}.txt",
            candidate_dir / "templates" / f"{name}.txt",
        )
    return candidate_dir
