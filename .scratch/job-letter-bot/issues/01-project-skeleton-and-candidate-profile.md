# 01: Project skeleton + Candidate Profile

**What to build:** A runnable, well-organised Python project (v0.0) that loads the Candidate Profile and the AI Letter Template and validates them, so every later ticket has config, error types, prompt loading and test fakes to build on. See the spec (Implementation Decisions: Stack, Modules, Errors, Prompts).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Project managed with uv (pyproject + lockfile); ruff configured and clean; pytest runs
- [ ] Module layout keeps config, data store, RAG, tools, agent, prompts, UI and evaluation apart
- [ ] Typed settings from an environment file (OpenRouter key, agent model `openai/gpt-6-luna`, judge model `anthropic/claude-haiku-5.5`, default embedding model, paths, tool-call cap, window size); clear startup error when the key is missing; example env file without secrets; env file git-ignored
- [ ] Distinct error types for external API failures, validation errors and user-input problems
- [ ] Prompts are loaded by name from separate prompt files
- [ ] Candidate Profile holds name, address, phone, email (ada@example.com) and GitHub (https://github.com/sara-qbit); the AI Letter Template is stored as fixed text with marked gaps, taken from idea.md
- [ ] Target Role to Letter Template mapping validated (AI template for AI Engineer, AI Innovation Manager, Quantum Software Developer; Grant Management Specialist mapped but allowed to be missing until ticket 10, reported clearly)
- [ ] Deterministic fakes for LLM, embedder and judge available to tests
- [ ] README skeleton: purpose, install, configure, run, Kaggle CC BY-SA credit
