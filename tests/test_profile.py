from pathlib import Path

import pytest

from turing_projects_v2.config import PROJECT_ROOT
from turing_projects_v2.errors import ConfigError, DataValidationError
from turing_projects_v2.profile import load_profile

CANDIDATE_DIR = PROJECT_ROOT / "candidate"

PROFILE_TOML = """\
name = "Ada Example"
street = "Main Street 1"
postal_code = "12345"
city = "Sample Town"
country = "Germany"
phone = "0123 456"
email = "ada@example.com"
github = "https://github.com/ada"

[target_roles]
"AI Engineer" = "ai"
"AI Innovation Manager" = "ai"
"Quantum Software Developer" = "ai"
"Grant Management Specialist" = "grant"
"""


def write_candidate_dir(tmp_path: Path, profile: str, templates: dict[str, str]) -> Path:
    (tmp_path / "templates").mkdir()
    (tmp_path / "profile.toml").write_text(profile, encoding="utf-8")
    for name, text in templates.items():
        (tmp_path / "templates" / f"{name}.txt").write_text(text, encoding="utf-8")
    return tmp_path


def test_candidate_profile_holds_the_candidates_details(tmp_path):
    # Its own profile file: the real one in candidate/ may hold the candidate's private details.
    templates = {
        name: (CANDIDATE_DIR / "templates" / f"{name}.txt").read_text(encoding="utf-8")
        for name in ("ai", "grant")
    }
    profile = load_profile(write_candidate_dir(tmp_path, PROFILE_TOML, templates))

    assert profile.name == "Ada Example"
    assert profile.email == "ada@example.com"
    assert profile.github == "https://github.com/ada"
    assert profile.phone == "0123 456"


def test_ai_roles_use_the_ai_letter_template():
    profile = load_profile(CANDIDATE_DIR)

    for role in ["AI Engineer", "AI Innovation Manager", "Quantum Software Developer"]:
        template = profile.template_for(role)
        assert template.name == "ai"
        assert "I am currently transitioning my career towards AI engineering" in template.text


def test_ai_letter_template_marks_its_gaps():
    template = load_profile(CANDIDATE_DIR).template_for("AI Engineer")

    assert template.gaps == {
        "candidate_name",
        "candidate_street",
        "candidate_postal_code",
        "candidate_city",
        "candidate_country",
        "candidate_phone",
        "candidate_email",
        "github",
        "date",
        "company_name",
        "company_address",
        "job_name",
        "work_type",
        "company_vision",
        "role_summary",
    }


def test_grant_management_uses_the_grant_letter_template_with_the_same_gaps():
    profile = load_profile(CANDIDATE_DIR)

    template = profile.template_for("Grant Management Specialist")

    assert template.name == "grant"
    assert "support top management committees in {{role_summary}}" in template.text
    assert template.gaps == load_profile(CANDIDATE_DIR).template_for("AI Engineer").gaps - {
        "github"
    }
    assert profile.missing_templates == {}


def test_each_letter_template_has_its_not_stated_wording():
    profile = load_profile(CANDIDATE_DIR)

    for role in ["AI Engineer", "Grant Management Specialist"]:
        template = profile.template_for(role)
        assert template.name in profile.not_stated_wording


def test_missing_template_is_reported_not_fatal(tmp_path):
    profile = load_profile(write_candidate_dir(tmp_path, PROFILE_TOML, {"ai": "Hi {{job_name}}"}))

    assert profile.missing_templates == {"Grant Management Specialist": "grant"}
    with pytest.raises(ConfigError, match=r"Grant Management Specialist.*grant\.txt"):
        profile.template_for("Grant Management Specialist")


def test_unknown_target_role_is_rejected(tmp_path):
    profile = load_profile(write_candidate_dir(tmp_path, PROFILE_TOML, {"ai": "Hi {{job_name}}"}))

    with pytest.raises(ConfigError, match="Chef"):
        profile.template_for("Chef")


def test_template_with_an_unknown_gap_is_rejected(tmp_path):
    candidate_dir = write_candidate_dir(tmp_path, PROFILE_TOML, {"ai": "Hi {{favourite_colour}}"})

    with pytest.raises(DataValidationError, match="favourite_colour"):
        load_profile(candidate_dir)


def test_profile_missing_a_field_is_rejected(tmp_path):
    profile_without_email = PROFILE_TOML.replace('email = "ada@example.com"\n', "")
    candidate_dir = write_candidate_dir(tmp_path, profile_without_email, {"ai": "Hi"})

    with pytest.raises(DataValidationError, match="email"):
        load_profile(candidate_dir)


def test_missing_profile_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"profile\.toml"):
        load_profile(tmp_path)


def test_profile_must_map_every_target_role(tmp_path):
    profile_without_grant = PROFILE_TOML.replace('"Grant Management Specialist" = "grant"\n', "")
    candidate_dir = write_candidate_dir(tmp_path, profile_without_grant, {"ai": "Hi"})

    with pytest.raises(DataValidationError, match="Grant Management Specialist"):
        load_profile(candidate_dir)


def test_not_stated_wording_for_an_unknown_template_is_rejected(tmp_path):
    profile = PROFILE_TOML + '\n[not_stated_wording.grnat]\nrole_summary = "this role"\n'
    candidate_dir = write_candidate_dir(tmp_path, profile, {"ai": "Hi"})

    with pytest.raises(DataValidationError, match="grnat"):
        load_profile(candidate_dir)
