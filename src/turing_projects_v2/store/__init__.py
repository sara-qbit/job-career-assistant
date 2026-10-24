"""Data store: SQLite repository for Job Postings, companies, Job Info, Cover Letters and logs.

Callers use domain operations only; the SQL stays inside this module.
"""

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from turing_projects_v2.judge import Verdict

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


class Company(BaseModel):
    """One company as imported."""

    company_id: int
    description: str | None
    address: str | None
    city: str | None
    state: str | None
    zip_code: str | None
    country: str | None


class JobInfo(BaseModel):
    """The structured summary of one Job Posting, with the judge's verdicts on its summaries."""

    job_id: int
    job_name: str
    company_name: str | None
    work_type: str | None
    role_summary: str
    company_vision: str
    role_summary_verdict: Verdict | None
    """The judge's verdict on the role summary; None when there was nothing to check."""
    company_vision_verdict: Verdict | None


class LetterCheck(BaseModel):
    """One deterministic check of a Cover Letter."""

    name: str
    passed: bool


class CoverLetter(BaseModel):
    """A Letter Template filled in for one Job Posting, with its check results."""

    job_id: int
    target_role: str
    template_name: str
    text: str
    checks: list[LetterCheck]
    company_vision_verdict: Verdict | None
    """The judge's verdict on the written company vision gap; None for fixed wording."""
    role_summary_verdict: Verdict | None
    rewritten: bool
    """Whether a gap judged fabricated was written a second time."""
    created_at: datetime

    @property
    def verdicts(self) -> dict[str, Verdict | None]:
        """The judge's verdict per written gap, by display name."""
        return {
            "Company vision": self.company_vision_verdict,
            "Role summary": self.role_summary_verdict,
        }

    @property
    def failed_checks(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]

    @property
    def fabricated(self) -> dict[str, str]:
        """The judge's reason per written gap still judged fabricated, by display name."""
        return {
            name: verdict.reason
            for name, verdict in self.verdicts.items()
            if verdict is not None and verdict.value == "fabricated"
        }


class TokenUsage(BaseModel):
    """Tokens used by one LLM call; cost in USD, None when the provider did not report it."""

    conversation_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float | None


class JobStore:
    """SQLite repository. Pass ":memory:" for a throwaway store in tests."""

    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        # The agent runs tools in worker threads and Streamlit reruns the page in new threads.
        self._db = sqlite3.connect(path, check_same_thread=False)
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
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS job_info ("
                "job_id INTEGER PRIMARY KEY REFERENCES postings (job_id), job_name TEXT NOT NULL, "
                "company_name TEXT, work_type TEXT, role_summary TEXT NOT NULL, "
                "company_vision TEXT NOT NULL, role_summary_verdict TEXT, "
                "role_summary_reason TEXT, company_vision_verdict TEXT, "
                "company_vision_reason TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            # The whole letter as JSON, plus the columns needed to find it.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cover_letters ("
                "id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL REFERENCES postings (job_id), "
                "target_role TEXT NOT NULL, template_name TEXT NOT NULL, letter TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS token_usage ("
                "id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, model TEXT NOT NULL, "
                "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost REAL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
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

    def get_company(self, company_id: int) -> Company | None:
        row = self._db.execute(
            "SELECT company_id, description, address, city, state, zip_code, country "
            "FROM companies WHERE company_id = ?",
            (company_id,),
        ).fetchone()
        return Company.model_validate(dict(row)) if row else None

    def save_job_info(self, info: JobInfo) -> None:
        """Save the Job Info of its posting, replacing an earlier one."""
        verdicts = (info.role_summary_verdict, info.company_vision_verdict)
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO job_info (job_id, job_name, company_name, work_type, "
                "role_summary, company_vision, role_summary_verdict, role_summary_reason, "
                "company_vision_verdict, company_vision_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    info.job_id,
                    info.job_name,
                    info.company_name,
                    info.work_type,
                    info.role_summary,
                    info.company_vision,
                    *(
                        value
                        for verdict in verdicts
                        for value in ((verdict.value, verdict.reason) if verdict else (None, None))
                    ),
                ),
            )

    def get_job_info(self, job_id: int) -> JobInfo | None:
        row = self._db.execute("SELECT * FROM job_info WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        fields = dict(row)
        for name in ("role_summary", "company_vision"):
            value, reason = fields.pop(f"{name}_verdict"), fields.pop(f"{name}_reason")
            fields[f"{name}_verdict"] = (
                Verdict(value=value, reason=reason) if value is not None else None
            )
        return JobInfo.model_validate(fields)

    def save_cover_letter(self, letter: CoverLetter) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO cover_letters "
                "(job_id, target_role, template_name, letter, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    letter.job_id,
                    letter.target_role,
                    letter.template_name,
                    letter.model_dump_json(),
                    letter.created_at.isoformat(),
                ),
            )

    def cover_letters(self, job_id: int) -> list[CoverLetter]:
        """Every saved Cover Letter for the posting, oldest first."""
        rows = self._db.execute(
            "SELECT letter FROM cover_letters WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
        return [CoverLetter.model_validate_json(row["letter"]) for row in rows]

    def record_token_usage(self, usage: TokenUsage) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO token_usage "
                "(conversation_id, model, input_tokens, output_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    usage.conversation_id,
                    usage.model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cost,
                ),
            )

    def token_usage(self, conversation_id: str) -> list[TokenUsage]:
        """Every recorded LLM call of the Conversation, oldest first."""
        rows = self._db.execute(
            "SELECT conversation_id, model, input_tokens, output_tokens, cost FROM token_usage "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [TokenUsage.model_validate(dict(row)) for row in rows]
