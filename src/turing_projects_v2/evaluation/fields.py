"""Field accuracy over every saved Cover Letter, stored as an evaluation run."""

from turing_projects_v2.evaluation.accuracy import field_accuracy
from turing_projects_v2.store import EvaluationRun, JobStore


def run_field_evaluation(store: JobStore) -> EvaluationRun:
    """How often the saved letters contain their posting's exact job name, company and work
    type; one case per letter, with which fields matched."""
    letters = store.all_cover_letters()
    report = field_accuracy(letters, store.get_posting)
    run = EvaluationRun(
        kind="fields",
        config={"letters": report.letters},
        metrics=report.accuracy,
        cases=report.per_letter,
    )
    store.save_evaluation_run(run)
    return run
