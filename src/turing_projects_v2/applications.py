"""Applications: the pasted Job Postings the candidate applied for, each dated the local day they
said so ("save it", "I applied"). Kaggle Job Postings cannot have one: only pasted Job Postings go
on the Application Table for the Berater.
"""

from datetime import date, datetime

from pydantic import BaseModel

from turing_projects_v2.errors import UserInputError
from turing_projects_v2.store import NOT_STATED_DE, Application, Employment, JobStore, WorkMode

TABLE_COLUMNS = ["Datum", "Arbeitgeber", "Stelle", "Ort", "Vollzeit/Teilzeit", "Präsenz/Online"]
"""The Application Table's columns, in the order of `ApplicationRow.cells`."""


class ApplicationRow(BaseModel):
    """One row of the Application Table: the application date and the Job Facts."""

    applied_on: date
    employer: str = NOT_STATED_DE
    position: str = NOT_STATED_DE
    city: str = NOT_STATED_DE
    employment: Employment = NOT_STATED_DE
    work_mode: WorkMode = NOT_STATED_DE

    def cells(self) -> list[str]:
        """The row as the table shows it, one value per column of TABLE_COLUMNS."""
        return [
            german_date(self.applied_on),
            self.employer,
            self.position,
            self.city,
            self.employment,
            self.work_mode,
        ]


class ApplicationTable(BaseModel):
    """The Application Table for the Berater: every Application, oldest first."""

    candidate_name: str
    made_at: datetime
    """When it was made (local time); "Stand" shows its date."""
    rows: list[ApplicationRow]


def german_date(day: date) -> str:
    """E.g. "30.11.2026"."""
    return f"{day:%d.%m.%Y}"


def application_table(candidate_name: str, store: JobStore) -> ApplicationTable:
    """Every Application with its Job Facts, oldest first, made today. Facts that were never read
    (the Job Facts step failed) are NOT_STATED_DE."""
    rows = []
    for application in store.applications():
        facts = store.get_job_facts(application.job_id)
        known = facts.model_dump(include=set(ApplicationRow.model_fields)) if facts else {}
        rows.append(ApplicationRow(applied_on=application.applied_on, **known))
    return ApplicationTable(candidate_name=candidate_name, made_at=datetime.now(), rows=rows)


def record_application(job_id: int, store: JobStore) -> tuple[Application, bool]:
    """Record an Application for the pasted posting `job_id`, dated today; return it and whether
    it is new. A posting applied for before keeps its Application and date."""
    posting = store.get_posting(job_id)
    if posting is None:
        raise UserInputError(f"No Job Posting has job ID {job_id}.")
    if posting.source != "pasted":
        raise UserInputError(
            f"Job ID {job_id} is a Kaggle Job Posting and cannot be saved as an Application: only "
            "pasted job ads go on the Application Table for the Berater. Paste the job ad from the "
            "employer's website to save it."
        )
    earlier = store.get_application(job_id)
    if earlier is not None:
        return earlier, False
    application = Application(job_id=job_id, applied_on=date.today())
    store.add_application(application)
    return application, True


def remove_application(job_id: int, store: JobStore) -> Application:
    """Remove the Application for `job_id` and return it; the Job Posting stays stored."""
    application = store.get_application(job_id)
    if application is None:
        raise UserInputError(f"Job ID {job_id} has no Application to remove.")
    store.remove_application(job_id)
    return application
