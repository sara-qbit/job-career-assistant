"""The `job_market_stats` tool: a fixed set of statistics computed exactly from the database.

The LLM chooses a statistic and its parameters through the tool schema; it never writes SQL.
"""

import statistics
from typing import Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from turing_projects_v2.rag.retriever import TargetRoleName, WorkType
from turing_projects_v2.store import JobStore

# Statistic -> (what the store counts postings by, chart title).
STATISTICS = {
    "postings_by_target_role": ("target_role", "Postings per Target Role"),
    "postings_by_work_type": ("work_type", "Postings per work type"),
    "postings_by_remote": ("remote", "Remote and on-site postings"),
    "postings_by_location": ("location", "Postings per location (top locations)"),
    "salary_by_target_role": ("salary", "Yearly salary per Target Role (USD)"),
}


class StatsQuery(BaseModel):
    statistic: Literal[
        "postings_by_target_role",
        "postings_by_work_type",
        "postings_by_remote",
        "postings_by_location",
        "salary_by_target_role",
    ] = Field(
        description="which statistic to compute; salary_by_target_role gives the middle half "
        "(25th to 75th percentile) and the median of stated yearly USD salaries"
    )
    population: Literal["target_role_postings", "all_postings"] = Field(
        default="target_role_postings",
        description="count only the postings tagged with a Target Role (the default), or all "
        "postings in the database",
    )
    target_role: TargetRoleName | None = Field(
        default=None, description="only postings of this Target Role"
    )
    work_type: WorkType | None = Field(default=None, description="only postings of this work type")
    limit: int = Field(
        default=10, ge=1, le=25, description="how many top locations to show (locations only)"
    )


class StatRow(BaseModel):
    label: str
    count: int
    """Postings counted; for salaries, the postings that state one."""
    p25: float | None = None
    median: float | None = None
    p75: float | None = None


class MarketStats(BaseModel):
    """One computed statistic, ready for a chart."""

    statistic: str
    title: str
    population: str
    """Which postings the numbers cover, e.g. "the 1,108 Target-Role postings"."""
    rows: list[StatRow]


def market_stats_tool(store: JobStore) -> BaseTool:
    """The `job_market_stats` tool on the given store."""

    @tool(args_schema=StatsQuery, response_format="content_and_artifact")
    def job_market_stats(
        statistic: str,
        population: str = "target_role_postings",
        target_role: str | None = None,
        work_type: str | None = None,
        limit: int = 10,
    ) -> tuple[str, dict]:
        """Compute an exact job market statistic from the database of Job Postings, shown to the
        candidate as a chart. Use it for questions such as "how many contract jobs per Target
        Role" or "where are most AI Engineer jobs"."""
        by, title = STATISTICS[statistic]
        target_roles_only = population == "target_role_postings"
        if by == "salary":
            salaries = store.salaries(
                target_roles_only=target_roles_only, target_role=target_role, work_type=work_type
            )
            rows = [_salary_row(label, values) for label, values in salaries.items()]
            total = sum(row.count for row in rows)
            covered = " that state a yearly USD salary"
        else:
            counted, total = store.posting_counts(
                by,
                target_roles_only=target_roles_only,
                target_role=target_role,
                work_type=work_type,
                limit=limit if by == "location" else None,
            )
            rows = [StatRow(label=label, count=count) for label, count in counted]
            covered = ""
        stats = MarketStats(
            statistic=statistic,
            title=title,
            population=_population(total, target_roles_only, target_role, work_type) + covered,
            rows=rows,
        )
        lines = [
            f"{stats.title}, computed exactly over {stats.population}; shown to the candidate as "
            "a chart. State the population in your answer.",
            *(_row_text(row) for row in stats.rows),
        ]
        return "\n".join(lines), stats.model_dump(mode="json")

    return job_market_stats


def _salary_row(label: str, salaries: list[float]) -> StatRow:
    """The quartiles of the salaries; the middle half is robust to Kaggle's odd outliers."""
    if len(salaries) == 1:
        p25 = median = p75 = salaries[0]
    else:
        p25, median, p75 = statistics.quantiles(salaries, n=4, method="inclusive")
    return StatRow(label=label, count=len(salaries), p25=p25, median=median, p75=p75)


def _row_text(row: StatRow) -> str:
    if row.median is None:
        return f"- {row.label}: {row.count:,}"
    return (
        f"- {row.label}: median ${row.median:,.0f}, middle half ${row.p25:,.0f} to "
        f"${row.p75:,.0f} ({row.count:,} postings)"
    )


def _population(
    total: int, target_roles_only: bool, target_role: str | None, work_type: str | None
) -> str:
    """Which postings a statistic covers, e.g. "the 3 Target-Role postings for AI Engineer"."""
    label = (
        f"the {total:,} Target-Role postings" if target_roles_only else f"all {total:,} postings"
    )
    if target_role is not None:
        label += f" for {target_role}"
    if work_type is not None:
        label += f" with work type {work_type}"
    return label
