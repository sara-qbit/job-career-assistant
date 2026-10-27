"""Applications: the pasted Job Postings the candidate applied for, each dated the local day they
said so ("save it", "I applied"). Kaggle Job Postings cannot have one: only pasted Job Postings go
on the Application Table for the Berater.
"""

from datetime import date

from turing_projects_v2.errors import UserInputError
from turing_projects_v2.store import Application, JobStore


def german_date(day: date) -> str:
    """E.g. "30.11.2026"."""
    return f"{day:%d.%m.%Y}"


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
