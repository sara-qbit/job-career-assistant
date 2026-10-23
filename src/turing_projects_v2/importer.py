"""One-time import of the Kaggle LinkedIn job postings dataset into the data store.

Expected layout of the raw directory (as downloaded from Kaggle):
    postings.csv
    companies/companies.csv
"""

import csv
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

from pydantic import BaseModel

from turing_projects_v2.errors import ConfigError, DataValidationError
from turing_projects_v2.store import COMPANY_COLUMNS, POSTING_COLUMNS, JobStore

# Job Descriptions can be longer than the csv module's default field limit.
csv.field_size_limit(sys.maxsize)


class ImportReport(BaseModel):
    postings: int
    companies: int


def import_kaggle(raw_dir: Path, store: JobStore) -> ImportReport:
    """Store every posting and company from `raw_dir`; safe to re-run (upsert by ID)."""
    postings = store.upsert_postings(_read_csv(raw_dir / "postings.csv", POSTING_COLUMNS))
    companies = store.upsert_companies(
        _read_csv(raw_dir / "companies" / "companies.csv", COMPANY_COLUMNS)
    )
    return ImportReport(postings=postings, companies=companies)


def _read_csv(path: Path, columns: Mapping[str, str]) -> Iterator[dict[str, str | None]]:
    """Yield the rows of `path` with empty fields turned into None."""
    try:
        file = path.open(encoding="utf-8", newline="")
    except FileNotFoundError as error:
        raise ConfigError(f"Kaggle file not found: {path}. See README section 'Data'.") from error
    with file:
        reader = csv.DictReader(file)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(f"{path} lacks columns: {', '.join(sorted(missing))}")
        for row in reader:
            yield {name: value or None for name, value in row.items()}
