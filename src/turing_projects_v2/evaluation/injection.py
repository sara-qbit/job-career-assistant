"""How well the Injection Guard tells attacks from honest input, stored as an evaluation run."""

import tomllib
from pathlib import Path

from pydantic import BaseModel

from turing_projects_v2.security import InjectionGuard, InjectionSource
from turing_projects_v2.store import EvaluationRun, JobStore

INJECTION_CASES = Path(__file__).parent / "injection_cases.toml"


class InjectionCase(BaseModel):
    """One labelled input: a typed message or a pasted job ad, and whether it is an attack."""

    source: InjectionSource
    attack: bool
    text: str


def load_injection_cases(path: Path = INJECTION_CASES) -> list[InjectionCase]:
    with path.open("rb") as file:
        return [InjectionCase.model_validate(case) for case in tomllib.load(file)["case"]]


def run_injection_evaluation(
    store: JobStore, cases: list[InjectionCase], guard: InjectionGuard
) -> EvaluationRun:
    """Check every case with the guard; the share of attacks blocked and of honest inputs
    blocked by mistake, with one case per input."""
    outcomes = []
    for case in cases:
        verdict = guard.check(case.text, case.source)
        outcomes.append(
            {
                "source": case.source,
                "attack": case.attack,
                "blocked": verdict.injection,
                "reason": verdict.reason,
                "text": case.text,
            }
        )
    attacks = [outcome for outcome in outcomes if outcome["attack"]]
    honest = [outcome for outcome in outcomes if not outcome["attack"]]
    run = EvaluationRun(
        kind="injection",
        config={"attacks": len(attacks), "honest": len(honest)},
        metrics={
            "attacks caught": _blocked_share(attacks),
            "honest wrongly blocked": _blocked_share(honest),
        },
        cases=outcomes,
    )
    store.save_evaluation_run(run)
    return run


def injection_report(run: EvaluationRun) -> list[str]:
    """The run's scores as counts, then every input the guard judged wrongly."""
    attacks = [case for case in run.cases if case["attack"]]
    honest = [case for case in run.cases if not case["attack"]]
    lines = [
        f"Attacks caught: {sum(case['blocked'] for case in attacks)}/{len(attacks)}",
        f"Honest inputs wrongly blocked: {sum(case['blocked'] for case in honest)}/{len(honest)}",
    ]
    for case in run.cases:
        if case["blocked"] != case["attack"]:
            mistake = "missed attack" if case["attack"] else "wrongly blocked"
            lines.append(f"{mistake} ({case['source']}): {case['text'][:60]!r} - {case['reason']}")
    return lines


def _blocked_share(outcomes: list[dict]) -> float | None:
    return sum(outcome["blocked"] for outcome in outcomes) / len(outcomes) if outcomes else None
