"""Data store: SQLite repository for Job Postings, companies, Job Facts, Job Info, Cover Letters
and logs.

Callers use domain operations only; the SQL stays inside this module.
"""

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from turing_projects_v2.judge import Verdict
from turing_projects_v2.profile import TargetRole

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

PASTED_JOB_ID_START = 100_000_000_000
"""The first job ID of a pasted Job Posting; Kaggle (LinkedIn) job IDs stay far below it."""

# Only Kaggle postings make up the job market: the Knowledge Base, tagging and statistics.
_KAGGLE = "source = 'kaggle'"

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
    source: Literal["kaggle", "pasted"] = "kaggle"
    pasted_on: date | None = None
    """The day a pasted Job Posting was first pasted; None for Kaggle postings."""


class Company(BaseModel):
    """One company as imported."""

    company_id: int
    description: str | None
    address: str | None
    city: str | None
    state: str | None
    zip_code: str | None
    country: str | None


NOT_STATED_DE = "nicht angegeben"
"""A Job Fact the ad does not state (or whose backing quote is not in the ad)."""

type Language = Literal["German", "English"]
type Employment = Literal["Vollzeit", "Teilzeit", "nicht angegeben"]
type WorkMode = Literal["Präsenz", "Online", "Hybrid", "nicht angegeben"]


class JobFacts(BaseModel):
    """The facts of one pasted Job Posting, each checked against its Job Description.

    Employer, position and city are copied word for word from the ad; employment and work mode
    are fixed German values backed by a quote from the ad. A fact not found is NOT_STATED_DE;
    address and contact person (word for word too) are None then.
    """

    job_id: int
    employer: str
    position: str
    city: str
    employment: Employment
    employment_quote: str | None
    work_mode: WorkMode
    work_mode_quote: str | None
    address: str | None
    contact_person: str | None
    target_role: TargetRole
    """The suggested Target Role, or the one the candidate chose."""
    language: Language
    """The ad's language, or the one the candidate chose for the letter."""


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


# What postings can be counted by, as SQL expressions; nothing else ever reaches the SQL.
_COUNT_BY = {
    "target_role": "COALESCE(target_role, 'No Target Role')",
    "work_type": "COALESCE(formatted_work_type, 'Not stated')",
    "remote": "CASE WHEN remote_allowed = 1 THEN 'Remote' ELSE 'On-site or not stated' END",
    "location": "COALESCE(location, 'Not stated')",
}


class ConversationSummary(BaseModel):
    """A Conversation as the sidebar lists it; messages live in the agent checkpointer."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class EvalQuestion(BaseModel):
    """A RAGAs test-set question with its reference answer and the postings text it came from.

    `approved` is None until the candidate decides; only approved questions are evaluated.
    """

    id: int | None = None
    question: str
    reference: str
    reference_contexts: list[str]
    approved: bool | None = None


class EvaluationRun(BaseModel):
    """One evaluation run: what was measured (`kind`), with which configuration, the metric
    values, and the per-case results."""

    id: int | None = None
    created_at: datetime | None = None
    kind: str
    config: dict[str, Any]
    metrics: dict[str, float | None]
    cases: list[dict[str, Any]]


class SecurityLogEntry(BaseModel):
    """One blocked or flagged input, or another security event; never holds personal data."""

    conversation_id: str
    event_type: str
    excerpt: str
    created_at: datetime | None = None


class TokenUsage(BaseModel):
    """Tokens used by one LLM call; cost in USD, None when the provider did not report it.

    `turn_id` is the ID of the question message whose answer made the call.
    """

    conversation_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float | None
    turn_id: str | None = None


class UsageTotal(BaseModel):
    """Tokens and cost of several LLM calls; `cost_known` is False if a cost was not reported."""

    tokens: int = 0
    cost: float = 0.0
    cost_known: bool = True

    @classmethod
    def of(cls, usages: Iterable[TokenUsage]) -> UsageTotal:
        total = cls()
        for usage in usages:
            total.tokens += usage.input_tokens + usage.output_tokens
            if usage.cost is None:
                total.cost_known = False
            else:
                total.cost += usage.cost
        total.cost = round(total.cost, 10)  # sums of float costs pick up rounding noise
        return total

    def describe(self) -> str:
        """E.g. "3,192 tokens · $0.0005", or "… · $0.0005 + not reported" if a cost is missing."""
        if self.cost_known:
            cost = f"${self.cost:.4f}"
        elif self.cost:
            cost = f"${self.cost:.4f} + not reported"
        else:
            cost = "cost not reported"
        return f"{self.tokens:,} tokens · {cost}"


def _posting_filter(
    target_roles_only: bool, target_role: str | None, work_type: str | None
) -> tuple[str, tuple[str, ...]]:
    """The WHERE clause and its parameters for the postings a statistic covers."""
    conditions, parameters = [_KAGGLE], []
    if target_roles_only:
        conditions.append("target_role IS NOT NULL")
    if target_role is not None:
        conditions.append("target_role = ?")
        parameters.append(target_role)
    if work_type is not None:
        conditions.append("formatted_work_type = ?")
        parameters.append(work_type)
    return "WHERE " + " AND ".join(conditions), tuple(parameters)


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
            # Databases created before pasted Job Postings lack these columns. `ad_key` is the
            # hash of a pasted ad's whitespace-normalised text, which finds a repeated paste.
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(postings)")}
            for name, kind in (
                ("source", "TEXT NOT NULL DEFAULT 'kaggle'"),
                ("pasted_on", "TEXT"),
                ("ad_key", "TEXT"),
            ):
                if name not in columns:
                    self._db.execute(f"ALTER TABLE postings ADD COLUMN {name} {kind}")
            self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS postings_ad_key ON postings (ad_key)"
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
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS job_facts ("
                "job_id INTEGER PRIMARY KEY REFERENCES postings (job_id), facts TEXT NOT NULL)"
            )
            # The whole letter as JSON, plus the columns needed to find it.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cover_letters ("
                "id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL REFERENCES postings (job_id), "
                "target_role TEXT NOT NULL, template_name TEXT NOT NULL, letter TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS security_log ("
                "id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, event_type TEXT NOT NULL, "
                "excerpt TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS eval_questions ("
                "id INTEGER PRIMARY KEY, question TEXT NOT NULL, reference TEXT NOT NULL, "
                "reference_contexts TEXT NOT NULL, approved INTEGER)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS eval_runs ("
                "id INTEGER PRIMARY KEY, kind TEXT NOT NULL, config TEXT NOT NULL, "
                "metrics TEXT NOT NULL, cases TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS token_usage ("
                "id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, model TEXT NOT NULL, "
                "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost REAL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, turn_id TEXT)"
            )
            # Databases created before answers had turn IDs lack the column.
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(token_usage)")}
            if "turn_id" not in columns:
                self._db.execute("ALTER TABLE token_usage ADD COLUMN turn_id TEXT")

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
        """The number of Kaggle postings."""
        return self._db.execute(f"SELECT COUNT(*) FROM postings WHERE {_KAGGLE}").fetchone()[0]

    def count_companies(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM companies").fetchone()[0]

    def all_titles(self) -> list[tuple[int, str]]:
        """(job ID, title) of every Kaggle posting."""
        return self._db.execute(f"SELECT job_id, title FROM postings WHERE {_KAGGLE}").fetchall()

    def set_target_roles(self, tags: Iterable[tuple[int, str | None]]) -> None:
        """Set the Target Role (or None) of each given job ID in one transaction."""
        with self._db:
            self._db.executemany(
                "UPDATE postings SET target_role = ? WHERE job_id = ?",
                ((role, job_id) for job_id, role in tags),
            )

    def count_by_target_role(self) -> dict[str | None, int]:
        """Number of Kaggle postings per Target Role; the None key counts untagged postings."""
        rows = self._db.execute(
            f"SELECT target_role, COUNT(*) FROM postings WHERE {_KAGGLE} GROUP BY target_role"
        ).fetchall()
        return {role: count for role, count in rows}

    def tagged_title_counts(self) -> dict[str, list[tuple[str, int]]]:
        """Per Target Role, its distinct tagged titles with counts, most frequent first."""
        rows = self._db.execute(
            "SELECT target_role, title, COUNT(*) AS n FROM postings "
            f"WHERE target_role IS NOT NULL AND {_KAGGLE} GROUP BY target_role, title "
            "ORDER BY target_role, n DESC, title"
        ).fetchall()
        result: dict[str, list[tuple[str, int]]] = {}
        for role, title, count in rows:
            result.setdefault(role, []).append((title, count))
        return result

    def tagged_postings(self) -> list[JobPosting]:
        """Every Kaggle posting with a Target Role (the Knowledge Base), in job ID order."""
        rows = self._db.execute(
            f"SELECT * FROM postings WHERE target_role IS NOT NULL AND {_KAGGLE} ORDER BY job_id"
        ).fetchall()
        return [JobPosting.model_validate(dict(row)) for row in rows]

    def get_posting(self, job_id: int) -> JobPosting | None:
        row = self._db.execute("SELECT * FROM postings WHERE job_id = ?", (job_id,)).fetchone()
        return JobPosting.model_validate(dict(row)) if row else None

    def pasted_posting(self, text: str) -> JobPosting | None:
        """The pasted Job Posting with this ad text, ignoring differences in whitespace."""
        row = self._db.execute(
            "SELECT * FROM postings WHERE ad_key = ?", (_ad_key(text),)
        ).fetchone()
        return JobPosting.model_validate(dict(row)) if row else None

    def add_pasted_posting(self, text: str, title: str, company_name: str | None) -> JobPosting:
        """Store the ad `text` unchanged as a new pasted Job Posting, pasted today."""
        with self._db:
            (last,) = self._db.execute(
                "SELECT MAX(job_id) FROM postings WHERE source = 'pasted'"
            ).fetchone()
            job_id = max(last or 0, PASTED_JOB_ID_START - 1) + 1
            self._db.execute(
                "INSERT INTO postings "
                "(job_id, title, company_name, description, source, pasted_on, ad_key) "
                "VALUES (?, ?, ?, ?, 'pasted', ?, ?)",
                (job_id, title, company_name, text, date.today().isoformat(), _ad_key(text)),
            )
            row = self._db.execute("SELECT * FROM postings WHERE job_id = ?", (job_id,)).fetchone()
        return JobPosting.model_validate(dict(row))

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

    def save_job_facts(self, facts: JobFacts) -> None:
        """Save the Job Facts of their posting, replacing earlier ones."""
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO job_facts (job_id, facts) VALUES (?, ?)",
                (facts.job_id, facts.model_dump_json()),
            )

    def get_job_facts(self, job_id: int) -> JobFacts | None:
        row = self._db.execute("SELECT facts FROM job_facts WHERE job_id = ?", (job_id,)).fetchone()
        return JobFacts.model_validate_json(row["facts"]) if row else None

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

    def all_cover_letters(self) -> list[CoverLetter]:
        """Every saved Cover Letter, oldest first."""
        rows = self._db.execute("SELECT letter FROM cover_letters ORDER BY id").fetchall()
        return [CoverLetter.model_validate_json(row["letter"]) for row in rows]

    def posting_counts(
        self,
        by: str,
        *,
        target_roles_only: bool,
        target_role: str | None = None,
        work_type: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[tuple[str, int]], int]:
        """Postings per value of `by` (a _COUNT_BY key), most first, and the number counted.

        `target_roles_only`, `target_role` and `work_type` narrow the postings counted; `limit`
        keeps only the top values.
        """
        where, parameters = _posting_filter(target_roles_only, target_role, work_type)
        rows = self._db.execute(
            f"SELECT {_COUNT_BY[by]} AS label, COUNT(*) AS n FROM postings {where} "
            "GROUP BY label ORDER BY n DESC, label LIMIT ?",
            (*parameters, -1 if limit is None else limit),
        ).fetchall()
        total = self._db.execute(f"SELECT COUNT(*) FROM postings {where}", parameters).fetchone()
        return [(row["label"], row["n"]) for row in rows], total[0]

    def salaries(
        self,
        *,
        target_roles_only: bool,
        target_role: str | None = None,
        work_type: str | None = None,
    ) -> dict[str, list[float]]:
        """The stated yearly USD salaries of the postings, per Target Role ('No Target Role').

        Kaggle's normalized salary is the posting's salary converted to a yearly amount.
        """
        where, parameters = _posting_filter(target_roles_only, target_role, work_type)
        where = f"{where} AND currency = 'USD' AND normalized_salary > 0"
        rows = self._db.execute(
            f"SELECT {_COUNT_BY['target_role']} AS label, normalized_salary FROM postings {where} "
            "ORDER BY label",
            parameters,
        ).fetchall()
        result: dict[str, list[float]] = {}
        for row in rows:
            result.setdefault(row["label"], []).append(row["normalized_salary"])
        return result

    def touch_conversation(self, conversation_id: str, title: str) -> None:
        """Mark the Conversation as just used; `title` is kept only if it is new."""
        now = datetime.now(UTC).isoformat()
        with self._db:
            self._db.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET updated_at = excluded.updated_at",
                (conversation_id, title, now, now),
            )

    def conversations(self) -> list[ConversationSummary]:
        """Every Conversation, most recently used first."""
        rows = self._db.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [ConversationSummary.model_validate(dict(row)) for row in rows]

    def save_test_questions(self, questions: Iterable[EvalQuestion]) -> None:
        """Add generated test-set questions, undecided."""
        with self._db:
            self._db.executemany(
                "INSERT INTO eval_questions (question, reference, reference_contexts) "
                "VALUES (?, ?, ?)",
                [(q.question, q.reference, json.dumps(q.reference_contexts)) for q in questions],
            )

    def test_questions(self) -> list[EvalQuestion]:
        """Every test-set question, approved or not, in the order generated."""
        return self._eval_questions("")

    def approved_test_questions(self) -> list[EvalQuestion]:
        return self._eval_questions("WHERE approved = 1")

    def set_test_approval(self, question_id: int, approved: bool) -> None:
        with self._db:
            self._db.execute(
                "UPDATE eval_questions SET approved = ? WHERE id = ?", (approved, question_id)
            )

    def _eval_questions(self, where: str) -> list[EvalQuestion]:
        rows = self._db.execute(
            "SELECT id, question, reference, reference_contexts, approved "
            f"FROM eval_questions {where} ORDER BY id"
        ).fetchall()
        return [
            EvalQuestion(
                id=row["id"],
                question=row["question"],
                reference=row["reference"],
                reference_contexts=json.loads(row["reference_contexts"]),
                approved=None if row["approved"] is None else bool(row["approved"]),
            )
            for row in rows
        ]

    def save_evaluation_run(self, run: EvaluationRun) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO eval_runs (kind, config, metrics, cases) VALUES (?, ?, ?, ?)",
                (run.kind, json.dumps(run.config), json.dumps(run.metrics), json.dumps(run.cases)),
            )

    def evaluation_runs(self) -> list[EvaluationRun]:
        """Every evaluation run, oldest first."""
        rows = self._db.execute(
            "SELECT id, created_at, kind, config, metrics, cases FROM eval_runs ORDER BY id"
        ).fetchall()
        return [
            EvaluationRun(
                id=row["id"],
                created_at=row["created_at"],
                kind=row["kind"],
                config=json.loads(row["config"]),
                metrics=json.loads(row["metrics"]),
                cases=json.loads(row["cases"]),
            )
            for row in rows
        ]

    def log_security_event(self, entry: SecurityLogEntry) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO security_log (conversation_id, event_type, excerpt) VALUES (?, ?, ?)",
                (entry.conversation_id, entry.event_type, entry.excerpt),
            )

    def security_log(self) -> list[SecurityLogEntry]:
        """Every security event, oldest first."""
        rows = self._db.execute(
            "SELECT conversation_id, event_type, excerpt, created_at FROM security_log ORDER BY id"
        ).fetchall()
        return [SecurityLogEntry.model_validate(dict(row)) for row in rows]

    def record_token_usage(self, usage: TokenUsage) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO token_usage "
                "(conversation_id, model, input_tokens, output_tokens, cost, turn_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    usage.conversation_id,
                    usage.model,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cost,
                    usage.turn_id,
                ),
            )

    def token_usage(self, conversation_id: str) -> list[TokenUsage]:
        """Every recorded LLM call of the Conversation, oldest first."""
        rows = self._db.execute(
            "SELECT conversation_id, model, input_tokens, output_tokens, cost, turn_id "
            "FROM token_usage "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [TokenUsage.model_validate(dict(row)) for row in rows]


def _ad_key(text: str) -> str:
    """The key that finds a pasted ad again: a hash of its text with whitespace normalised."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
