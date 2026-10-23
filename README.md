# Job Letter Assistant

A chatbot that finds job postings matching the candidate's Target Roles and writes a personalised cover letter for a chosen job, without inventing facts about the candidate. Built for the Turing College Sprint 2 project (LangChain, advanced RAG, tool calling, Streamlit).

> **Status: v0.0.** Project skeleton: settings, error types, prompt loading, Candidate Profile and AI Letter Template. Search, chat and letters follow in later versions.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync
```

## Configure

1. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`. `.env` is git-ignored; the app refuses to start without the key.
2. Optional settings (models, tool-call cap, history window) are listed in `.env.example`.
3. The Candidate Profile lives in `candidate/profile.toml`; Letter Templates live in `candidate/templates/<name>.txt` with gaps marked as `{{gap_name}}`. The `[target_roles]` table maps each Target Role to its Letter Template.

## Run

```bash
uv run turing-projects-v2   # checks settings and the Candidate Profile
```

## Develop

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Project layout

| Path | Purpose |
| --- | --- |
| `src/turing_projects_v2/config.py` | Typed settings from `.env` |
| `src/turing_projects_v2/errors.py` | Error types (external API, validation, user input, config) |
| `src/turing_projects_v2/profile.py` | Candidate Profile and Letter Templates |
| `src/turing_projects_v2/prompts/` | One file per prompt, loaded by name |
| `src/turing_projects_v2/fakes.py` | Deterministic LLM, embedder and judge for tests |
| `src/turing_projects_v2/store/`, `rag/`, `tools/`, `agent/`, `ui/`, `evaluation/` | Later versions |

## Data credit

Job postings come from [LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by Arsh Koneru on Kaggle, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
