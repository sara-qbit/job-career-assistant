"""Writes a small dataset shaped like the Kaggle LinkedIn job postings download."""

import csv
from pathlib import Path

POSTING_HEADER = [
    "job_id",
    "company_name",
    "title",
    "description",
    "max_salary",
    "pay_period",
    "location",
    "company_id",
    "views",
    "med_salary",
    "min_salary",
    "formatted_work_type",
    "applies",
    "original_listed_time",
    "remote_allowed",
    "job_posting_url",
    "application_url",
    "application_type",
    "expiry",
    "closed_time",
    "formatted_experience_level",
    "skills_desc",
    "listed_time",
    "posting_domain",
    "sponsored",
    "work_type",
    "currency",
    "compensation_type",
    "normalized_salary",
    "zip_code",
    "fips",
]
COMPANY_HEADER = "company_id,name,description,company_size,state,country,city,zip_code,address,url"
COMPANY_HEADER = COMPANY_HEADER.split(",")

TITLES = {
    1001: "Machine Learning Engineer",
    1002: "Senior AI Engineer - Remote",
    1003: "AI/ML Engineer (Contract)",
    1004: "Data Scientist",
    1005: "AI Content Writer",
    1006: "Grants Management Specialist",
    1007: "Grant Writer",
    1008: "store manager - Grants Pass, OR",
    1009: "Innovation Manager",
    1010: "Head of Innovation",
    1011: "Post Doctoral Fellow, Quantum Computing",
    1012: "Search Engine Optimization Specialist",
    1013: "Marketing Coordinator",
    1014: "Registered Nurse",
    1015: "Software Engineer",
    1016: "Retail Sales Associate",
    1017: "Accountant",
    1018: "Process Optimization Engineer",
    1019: "Maintenance Technician",
    1020: "Generative AI Engineer",
}


def posting(job_id: int, title: str, **overrides: str) -> dict[str, str]:
    row = dict.fromkeys(POSTING_HEADER, "")
    row.update(
        job_id=str(job_id),
        title=title,
        company_name="Acme Corp",
        company_id="2774458.0",
        description=f"Description of job {job_id}.\nSecond line, with a comma.",
        formatted_work_type="Full-time",
        location="Austin, TX",
        job_posting_url=f"https://www.linkedin.com/jobs/view/{job_id}",
    )
    row.update(overrides)
    return row


def write_kaggle_dataset(
    raw_dir: Path,
    postings: list[dict[str, str]] | None = None,
    company: dict[str, str] | None = None,
) -> Path:
    """Write postings.csv and companies/companies.csv under `raw_dir`; return `raw_dir`.

    The one company is Acme Corp (ID 2774458); `company` overrides its columns.
    """
    if postings is None:
        postings = [posting(job_id, title) for job_id, title in TITLES.items()]
    (raw_dir / "companies").mkdir(parents=True, exist_ok=True)
    with (raw_dir / "postings.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, POSTING_HEADER)
        writer.writeheader()
        writer.writerows(postings)
    with (raw_dir / "companies" / "companies.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, COMPANY_HEADER)
        writer.writeheader()
        writer.writerow(
            dict.fromkeys(COMPANY_HEADER, "")
            | {"company_id": "2774458", "name": "Acme Corp", "description": "We build rockets."}
            | (company or {})
        )
    return raw_dir
