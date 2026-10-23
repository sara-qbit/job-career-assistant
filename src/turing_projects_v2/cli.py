"""Command-line entry point: `uv run turing-projects-v2 <command>`."""

import argparse
import sys
import time
from collections.abc import Callable

from turing_projects_v2.config import Settings, load_settings
from turing_projects_v2.errors import AppError
from turing_projects_v2.importer import import_kaggle
from turing_projects_v2.profile import load_profile
from turing_projects_v2.store import JobStore
from turing_projects_v2.tagging import load_title_variants, tag_postings


def check(settings: Settings) -> None:
    """check settings and the Candidate Profile (default)"""
    profile = load_profile(settings.candidate_dir)
    print(f"Settings OK (agent model: {settings.agent_model}, judge: {settings.judge_model}).")
    print(f"Candidate Profile OK for {profile.name}.")
    for role, template in profile.target_roles.items():
        missing = " is MISSING" if role in profile.missing_templates else ""
        print(f"  {role}: Letter Template '{template}'{missing}")


def import_data(settings: Settings) -> None:
    """import the Kaggle dataset into the database, then tag postings"""
    store = JobStore(settings.database_path)
    started = time.perf_counter()
    report = import_kaggle(settings.kaggle_dir, store)
    print(
        f"Imported {report.postings:,} postings and {report.companies:,} companies "
        f"in {time.perf_counter() - started:.0f}s."
    )
    _tag(settings, store)


def tag(settings: Settings) -> None:
    """re-tag postings after editing the title variant list"""
    _tag(settings, JobStore(settings.database_path))


def review_titles(settings: Settings) -> None:
    """list the titles tagged per Target Role"""
    store = JobStore(settings.database_path)
    for role, titles in store.tagged_title_counts().items():
        print(f"\n## {role}: {sum(n for _, n in titles)} postings, {len(titles)} titles")
        for title, count in titles:
            print(f"{count:>4}  {title}")


COMMANDS: dict[str, Callable[[Settings], None]] = {
    "check": check,
    "import-data": import_data,
    "tag": tag,
    "review-titles": review_titles,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="turing-projects-v2")
    subparsers = parser.add_subparsers(dest="command")
    for name, command in COMMANDS.items():
        subparsers.add_parser(name, help=command.__doc__)
    name = parser.parse_args().command or "check"
    try:
        COMMANDS[name](load_settings())
    except AppError as error:
        sys.exit(f"{name} failed: {error}")


def _tag(settings: Settings, store: JobStore) -> None:
    started = time.perf_counter()
    report = tag_postings(store, load_title_variants(settings.title_variants_path))
    print(f"Tagged in {time.perf_counter() - started:.1f}s.")
    print(f"{report.total:,} postings: {sum(report.tagged.values()):,} tagged")
    for role, count in report.tagged.items():
        print(f"  {role}: {count:,}")
    print(f"  (untagged: {report.untagged:,})")
