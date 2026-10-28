# 01: Injection Guard (v3 security)

**What to build:** an LLM Injection Guard checks every chat message and every pasted or attached job ad before the agent runs. A detected prompt injection is blocked and logged as `prompt_injection`; the agent never sees it and a blocked ad is not saved. A failed guard call stops the turn. The regex check stays as a flag-only layer, the scope guard and input limits stay as they are. A labelled set of 20 inputs measures the guard.

**Blocked by:** none

**Status:** done

- [x] `InjectionGuard` (prompt `prompts/injection_guard.md`, same small model as the scope guard) checks the message and the job ad
- [x] Detected injection: blocked, logged as `prompt_injection`, ad not saved, agent not run
- [x] Guard call fails: the turn stops with an error message (security is never skipped)
- [x] `src/turing_projects_v2/evaluation/injection_cases.toml` (data/ is gitignored): 6 attack messages, 4 attack ads, 6 honest messages (some tricky), 4 honest ads
- [x] `uv run turing-projects-v2 evaluate injection` prints attacks caught x/10 and honest inputs wrongly blocked y/10 and saves an `EvaluationRun` (kind `injection`) shown on the Evaluation page
- [x] Test: no setting or toggle turns security off
- [x] README v3 section with the numbers; version 3.0.0; tag v3.0

## Comments

2026-10-28: Built test-first. Real run of `evaluate injection` with the agent model: attacks caught 10/10, honest inputs wrongly blocked 0/10. 300 tests and ruff green.
