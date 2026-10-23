"""Data store: SQLite repository for Job Postings, companies, Job Info, Cover Letters and logs.

Callers use domain operations only; the SQL stays inside this module.
"""

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from pydantic import BaseModel

# Kaggle postings.csv columns with their SQLite types. Type affinity turns "2774458.0" into 2774458.
POSTING_COLUMNS = {
    "job_id": "INTEGER PRIMARY KEY",
    "company_name": "TEXT",
    "title": "TEXT",
    "description": "TEXT",
    "max_salary": "REAL",
    "pay_period": "TEXT",
    "location": "TEXT",
    "company_id": "INTEGER",
    "views": "INTEGER",
    "med_salary": "REAL",
    "min_salary": "REAL",
    "formatted_work_type": "TEXT",
    "applies": "INTEGER",
    "original_listed_time": "INTEGER",
    "remote_allowed": "INTEGER",
    "job_posting_url": "TEXT",
    "application_url": "TEXT",
    "application_type": "TEXT",
    "expiry": "INTEGER",
    "closed_time": "INTEGER",
    "formatted_experience_level": "TEXT",
    "skills_desc": "TEXT",
    "listed_time": "INTEGER",
    "posting_domain": "TEXT",
    "sponsored": "INTEGER",
    "work_type": "TEXT",
    "currency": "TEXT",
    "compensation_type": "TEXT",
    "normalized_salary": "REAL",
    "zip_code": "TEXT",
    "fips": "TEXT",
}

# Kaggle companies/companies.csv columns with their SQLite types.
COMPANY_COLUMNS = {
    "company_id": "INTEGER PRIMARY KEY",
    "name": "TEXT",
    "description": "TEXT",
    "company_size": "INTEGER",
    "state": "TEXT",
    "country": "TEXT",
    "city": "TEXT",
    "zip_code": "TEXT",
    "address": "TEXT",
    "url": "TEXT",
}


class JobPosting(BaseModel):
    """One Job Posting as imported, plus its Target Role tag."""

    job_id: int
    title: str
    company_name: str | None
    company_id: int | None
    description: str | None
    location: str | None
    formatted_work_type: str | None
    remote_allowed: bool | None
    formatted_experience_level: str | None
    min_salary: float | None
    med_salary: float | None
    max_salary: float | None
    pay_period: str | None
    currency: str | None
    job_posting_url: str | None
    target_role: str | None


class JobStore:
    """SQLite repository. Pass ":memory:" for a throwaway store in tests."""

    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        posting_columns = ", ".join(f"{name} {kind}" for name, kind in POSTING_COLUMNS.items())
        company_columns = ", ".join(f"{name} {kind}" for name, kind in COMPANY_COLUMNS.items())
        with self._db:
            self._db.execute(
                f"CREATE TABLE IF NOT EXISTS postings ({posting_columns}, target_role TEXT)"
            )
            self._db.execute(f"CREATE TABLE IF NOT EXISTS companies ({company_columns})")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS postings_target_role ON postings (target_role)"
            )

    def upsert_postings(self, rows: Iterable[Mapping[str, str | None]]) -> int:
        """Insert or update postings by job ID, keeping their Target Role tag; return the count."""
        return self._upsert("postings", POSTING_COLUMNS, rows)

    def upsert_companies(self, rows: Iterable[Mapping[str, str | None]]) -> int:
        """Insert or update companies by company ID; return the count."""
        return self._upsert("companies", COMPANY_COLUMNS, rows)

    def _upsert(
        self, table: str, columns: Mapping[str, str], rows: Iterable[Mapping[str, str | None]]
    ) -> int:
        names = list(columns)
        key, *others = names
        sql = (
            f"INSERT INTO {table} ({', '.join(names)}) "
            f"VALUES ({', '.join('?' for _ in names)}) "
            f"ON CONFLICT({key}) DO UPDATE SET "
            + ", ".join(f"{name} = excluded.{name}" for name in others)
        )
        count = 0

        def values() -> Iterator[list[str | None]]:
            nonlocal count
            for row in rows:
                count += 1
                yield [row[name] for name in names]

        with self._db:
            self._db.executemany(sql, values())
        return count

    def count_postings(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM postings").fetchone()[0]

    def count_companies(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM companies").fetchone()[0]

    def all_titles(self) -> list[tuple[int, str]]:
        """(job ID, title) of every posting."""
        return self._db.execute("SELECT job_id, title FROM postings").fetchall()

    def set_target_roles(self, tags: Iterable[tuple[int, str | None]]) -> None:
        """Set the Target Role (or None) of each given job ID in one transaction."""
        with self._db:
            self._db.executemany(
                "UPDATE postings SET target_role = ? WHERE job_id = ?",
                ((role, job_id) for job_id, role in tags),
            )

    def count_by_target_role(self) -> dict[str | None, int]:
        """Number of postings per Target Role; the None key counts untagged postings."""
        rows = self._db.execute(
            "SELECT target_role, COUNT(*) FROM postings GROUP BY target_role"
        ).fetchall()
        return {role: count for role, count in rows}

    def tagged_title_counts(self) -> dict[str, list[tuple[str, int]]]:
        """Per Target Role, its distinct tagged titles with counts, most frequent first."""
        rows = self._db.execute(
            "SELECT target_role, title, COUNT(*) AS n FROM postings "
            "WHERE target_role IS NOT NULL GROUP BY target_role, title "
            "ORDER BY target_role, n DESC, title"
        ).fetchall()
        result: dict[str, list[tuple[str, int]]] = {}
        for role, title, count in rows:
            result.setdefault(role, []).append((title, count))
        return result

    def tagged_postings(self) -> list[JobPosting]:
        """Every posting with a Target Role, in job ID order."""
        rows = self._db.execute(
            "SELECT * FROM postings WHERE target_role IS NOT NULL ORDER BY job_id"
        ).fetchall()
        return [JobPosting.model_validate(dict(row)) for row in rows]

    def get_posting(self, job_id: int) -> JobPosting | None:
        row = self._db.execute("SELECT * FROM postings WHERE job_id = ?", (job_id,)).fetchone()
        return JobPosting.model_validate(dict(row)) if row else None
