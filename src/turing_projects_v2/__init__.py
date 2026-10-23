"""Job Letter Assistant: finds matching Job Postings and writes Cover Letters."""

import sys

from turing_projects_v2.config import load_settings
from turing_projects_v2.errors import AppError
from turing_projects_v2.profile import load_profile


def main() -> None:
    """Check that settings and the Candidate Profile load, and report what is missing."""
    try:
        settings = load_settings()
        profile = load_profile(settings.candidate_dir)
    except AppError as error:
        sys.exit(f"Startup failed: {error}")

    print(f"Settings OK (agent model: {settings.agent_model}, judge: {settings.judge_model}).")
    print(f"Candidate Profile OK for {profile.name}.")
    for role in profile.target_roles:
        if role in profile.missing_templates:
            print(f"  {role}: Letter Template '{profile.target_roles[role]}' is MISSING")
        else:
            print(f"  {role}: Letter Template '{profile.target_roles[role]}'")
