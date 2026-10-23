# Spec: Job Letter Assistant (Sprint 2, v1)

Status: ready-for-agent

## Problem Statement

The candidate is moving from an academic/engineering background into AI-related roles and needs to find suitable vacancies and send tailored cover letters. Searching job boards by keyword misses relevant jobs whose titles differ from the wording the candidate uses, and writing a personalised letter for each job is slow. Letting a generic LLM write the letter is risky: it invents qualifications, misnames companies or misstates the work type, which is embarrassing in an application. The candidate also wants to verify every claim themselves against the original posting.

The project is also the Sprint 2 course deliverable and must demonstrate advanced RAG (query translation and structured retrieval), at least three domain tools, LangChain with OpenRouter, a Streamlit UI showing sources, tool results and progress, security measures, and the optional tasks (conversation history and export, RAG visualisation, citations, token and cost display, tool-result visualisation, and a RAGAs evaluation with reranking shown on the website).

## Solution

A ChatGPT-style Streamlit chatbot over a Knowledge Base built from the Kaggle LinkedIn job postings dataset. The candidate describes the jobs they want in their own words. The chatbot translates the question into metadata filters and several query variants, searches the Knowledge Base, and lists numbered Search Results with citations. The candidate can open the verbatim Job Description of any posting, ask for the Job Info (a structured, verifiable summary of the job and company), and ask for a Cover Letter. The Cover Letter fills the gaps in the candidate's fixed Letter Template for the posting's Target Role, using only the Job Info and the Candidate Profile. Every filled-in gap is checked: exact fields deterministically, and summarised fields by an LLM-as-a-judge. The letter downloads as .docx and PDF.

A second tab, "Evaluation", shows everything behind the answers: conversation history and export, the RAG process, full sources, token usage and cost, tool-call results, the security measures and their log, the system prompts, and RAGAs evaluation results, including an embedding-model comparison and a reranker on/off comparison.

## User Stories

### Candidate Profile and Letter Templates

1. As the candidate, I want my personal details (name, address, phone, email, GitHub link) stored once in a Candidate Profile, so that every Cover Letter uses them consistently.
2. As the candidate, I want my Letter Template stored as fixed text with clearly marked gaps, so that only the gaps are ever generated.
3. As the candidate, I want a separate Letter Template per Target Role (AI template; Grant Management template), so that a grant job does not receive a letter about moving into AI engineering.
4. As the candidate, I want AI Engineer, AI Innovation Manager and Quantum Software Developer jobs to use the AI Letter Template, so that I only maintain two templates.
5. As the candidate, I want the Candidate Profile to be the only source of facts about me, so that the chatbot can never invent qualifications.
6. As the candidate, I want the app to tell me clearly if the Letter Template for a Target Role is missing, so that I am not surprised by a failed letter.

### Data import and Target Roles

7. As the candidate, I want all Kaggle Job Postings and companies imported into the database once, so that I never re-read the large CSV for later changes.
8. As the candidate, I want each Job Posting tagged with a Target Role (or none) from a reviewed list of title variants, so that I can see and control exactly which jobs count as relevant.
9. As the candidate, I want to review the generated title variants for each Target Role before they are used, so that obviously wrong titles are excluded.
10. As the candidate, I want to add new job titles or Target Roles later and re-tag in seconds, so that extending the project does not require a re-import.
11. As the candidate, I want only tagged Job Postings to be searchable, so that search results stay focused on my Target Roles.
12. As the candidate, I want the import to report how many postings were imported and how many were tagged per Target Role, so that I can sanity-check the data.
13. As the candidate, I want re-running the import to be safe (no duplicates), so that I can repeat it without corrupting data.

### Knowledge Base and embeddings

14. As the candidate, I want the Job Descriptions of tagged postings split into overlapping chunks, each starting with a context header (title, company, work type, location), so that every chunk is understandable on its own.
15. As the candidate, I want every chunk to carry metadata (job ID, Target Role, work type, remote flag, location, experience level, salary), so that searches can be filtered by hard conditions.
16. As the candidate, I want the Knowledge Base embedded with three models (local MiniLM, Gemini Embedding 2, Qwen3-Embedding-8B), so that I can compare them.
17. As the candidate, I want a progress indicator while indexing runs, so that I know it is working during long operations.
18. As the candidate, I want indexing to batch API calls and retry on rate limits, so that a temporary API problem does not abort the whole build.
19. As the candidate, I want newly tagged postings to be embedded without re-embedding everything, so that adding a Target Role stays cheap.

### Search (advanced RAG)

20. As the candidate, I want to describe jobs in my own words, so that I do not need to know the exact job titles companies use.
21. As the candidate, I want hard conditions in my question (remote, contract/full-time/part-time, location, Target Role, experience level) applied as exact filters, so that results never violate them.
22. As the candidate, I want my question rephrased into several variants that are each searched, so that jobs with unusual titles are still found.
23. As the candidate, I want results from all variants merged fairly into one ranking, so that jobs found by several variants rank higher.
24. As the candidate, I want each job to appear only once in the results even if several of its chunks matched, so that the list is not cluttered.
25. As the candidate, I want the top results numbered (#1, #2, ...), so that I can refer to them in follow-up messages.
26. As the candidate, I want each result to show title, company, location, work type and a link to the original posting, so that I can judge it quickly.
27. As the candidate, I want a clear message when no job matches, with a suggestion to broaden the search, so that I am not left with an empty answer.
28. As the candidate, I want to switch the embedding model in the sidebar (Gemini preselected), so that I can see how results differ.
29. As the candidate, I want to switch the reranker on or off in the sidebar (on by default), so that I can compare result quality.

### Job Description and Job Info

30. As the candidate, I want to ask for the full Job Description of any job and receive it verbatim, exactly as imported, so that I can verify the chatbot's claims myself.
31. As the candidate, I want the verbatim Job Description shown in its own box with a link to the original posting, so that it is clearly separate from generated text.
32. As the candidate, I want to ask "tell me about #2" and get the Job Info for that Search Result, so that I understand the job without reading the whole posting.
33. As the candidate, I want the job name, company name and work type in the Job Info copied exactly from the data, so that they are never paraphrased or wrong.
34. As the candidate, I want the role summary and the company vision summarised only from the posting and company description, so that they reflect the source.
35. As the candidate, I want a field to say "not stated in the posting" when the source has no information, so that nothing is guessed.
36. As the candidate, I want each summarised field checked by a judge with a visible verdict and reason, so that I can trust or question it.
37. As the candidate, I want the Job Info saved, so that it is not regenerated and does not change between views.

### Cover Letter

38. As the candidate, I want to ask "write the cover letter for #2" and get a complete Cover Letter, so that I can apply quickly.
39. As the candidate, I want only the gaps of my Letter Template filled (job name, work type, company, company vision, role summary and the motivation sentence), so that the rest of my letter stays exactly as I wrote it.
40. As the candidate, I want the motivation sentence built only from the role summary and facts already in my Letter Template, so that nothing new is claimed about me.
41. As the candidate, I want the job name, company and work type in the letter checked exactly against the Job Info, so that the most embarrassing errors are impossible.
42. As the candidate, I want the generated sentences checked by an LLM judge, with one automatic rewrite if a sentence is judged fabricated, so that most problems fix themselves.
43. As the candidate, I want a letter that still fails the check shown with a warning and the judge's reason, so that I know what to fix by hand.
44. As the candidate, I want the check results shown next to the letter, so that I see at a glance that it is safe to send.
45. As the candidate, I want to download the Cover Letter as .docx and as PDF, so that I can edit it or send it directly.
46. As the candidate, I want each generated Cover Letter saved with its check results and date, so that I can find it again.

### Market statistics

47. As the candidate, I want to ask questions such as "how many contract jobs per Target Role" or "salary range for AI Engineer", so that I understand the market.
48. As the candidate, I want statistics computed exactly from the database, not estimated by the LLM, so that the numbers are correct.
49. As the candidate, I want statistics answers to state whether they cover all postings or only Target-Role postings, so that I interpret them correctly.
50. As the candidate, I want statistics shown as charts, so that they are easy to read.

### Conversation and memory

51. As the candidate, I want a ChatGPT-like page with my messages, the chatbot's answers, and the input box at the bottom, so that it feels familiar.
52. As the candidate, I want answers streamed word by word with a status line while tools run, so that I see progress during long operations.
53. As the candidate, I want small [n] citation markers in answers linking to the original postings, so that I can see where each claim comes from.
54. As the candidate, I want "#2" to always mean the second result of my latest search, even late in a long Conversation, so that follow-ups are reliable.
55. As the candidate, I want the chatbot to remember the recent Conversation, so that I do not repeat myself.
56. As the candidate, I want a "New chat" button and a list of past Conversations in the left sidebar, so that I can switch between job searches.
57. As the candidate, I want to reopen a past Conversation with its full history and its Search Results restored, so that I can continue where I left off.
58. As the candidate, I want long Conversations to stay fast and cheap, so that cost does not grow with every message.

### Security and robustness

59. As the candidate, I want empty, overlong or malformed messages rejected with a clear explanation, so that I know what to change.
60. As the candidate, I want invalid or unknown job IDs reported clearly, so that typos do not crash the app.
61. As the candidate, I want instructions hidden inside Job Descriptions ignored, so that a malicious posting cannot take over the chatbot.
62. As the candidate, I want suspicious prompt-injection phrases in my own messages flagged, so that misuse is visible.
63. As the candidate, I want off-topic requests politely refused, so that the chatbot stays a job assistant.
64. As the candidate, I want my personal details sent to the LLM only when a Cover Letter is written, and never logged, so that my data is protected.
65. As the candidate, I want the API key kept only in an environment file excluded from git, and a clear startup error if it is missing, so that secrets never leak.
66. As the candidate, I want the agent stopped after a fixed number of tool calls per question, so that a confused agent cannot loop and waste time or money.
67. As the candidate, I want a failing tool or a slow API to produce a friendly answer instead of a stack trace, so that the app never looks broken.
68. As the candidate, I want temporary API failures retried automatically, so that short outages do not interrupt me.

### Evaluation tab

69. As the candidate, I want an "Evaluation" tab next to "MAIN", so that the chat stays clean while the details remain available.
70. As the candidate, I want to see and export the Conversation history as Markdown and JSON, so that I can keep or share it.
71. As the candidate, I want to see the RAG process for each search (filters, query variants, per-variant results, fusion and reranker scores), so that I understand why each job was found.
72. As the candidate, I want the full sources (matched chunks with scores and job links) for each answer, so that I can audit citations.
73. As the candidate, I want token usage and cost per answer and in total, so that I know what the chatbot costs.
74. As the candidate, I want every tool call with its arguments and result visualised, so that I can see what the agent did.
75. As the candidate, I want to see which security measures are active and a log of blocked or flagged inputs, so that I can demonstrate them.
76. As the candidate, I want to read the system prompts in the app, so that I can explain the chatbot's behaviour.
77. As the candidate, I want RAGAs metrics (faithfulness, context precision, context recall) for the search pipeline shown on the website, so that RAG quality is measured, not guessed.
78. As the candidate, I want the three embedding models compared on the same tagged postings, so that I can choose the best one with evidence.
79. As the candidate, I want results with and without the reranker compared, so that I can show what reranking adds.
80. As the candidate, I want Tool Selection Accuracy and Parameter Accuracy measured on hand-written agent cases, so that tool calling is evaluated deterministically.
81. As the candidate, I want an LLM-as-a-judge verdict for agent cases where a different tool choice may still be reasonable, so that strict checks do not mislead.
82. As the candidate, I want field accuracy (job name, company, work type) aggregated over generated letters, so that I can show the letters are reliable.
83. As the candidate, I want the RAGAs test set generated from the Knowledge Base, filtered, and approved by me, so that it is realistic but trustworthy.

### Course reviewer and developer

84. As a course reviewer, I want a README that explains what the project does, how to install, configure and run it, the architecture, and the Kaggle data credit, so that I can assess and run it.
85. As a course reviewer, I want the README to state that v1 uses Kaggle and that the Bundesagentur Jobsuche API is planned for v2, so that the deviation from the brief is explicit.
86. As a course reviewer, I want a reflection section listing known problems and improvement ideas, so that the learner's understanding is visible.
87. As a course reviewer, I want a visible version history (v0.0 to v1.0), so that iterative scoping is evident.
88. As a developer, I want prompts kept in separate files, so that they can be reviewed and changed without touching code.
89. As a developer, I want all structured LLM output enforced by schemas, so that malformed JSON cannot slip through.
90. As a developer, I want distinct error types for API failures, validation errors and user-input problems, so that each is handled appropriately.
91. As a developer, I want the LLM, embedder and judge injectable, so that tests run free, offline and deterministically.
92. As a developer, I want dependencies managed with uv and a lockfile, and code formatted and linted with ruff, so that the project is reproducible and PEP 8 compliant.
93. As a developer, I want the agent model, judge model and default embedding model configurable via settings, so that switching models needs no code change.

## Implementation Decisions

### Stack
- Python with uv (pyproject plus lockfile), ruff, pytest. LangChain for the agent and RAG, OpenRouter via LangChain's OpenAI-compatible chat model, ChromaDB for vectors, SQLite for everything else, Streamlit for the UI, RAGAs for evaluation, python-docx and a PDF writer for downloads.
- Agent model: `openai/gpt-6-luna`. Judge model: `anthropic/claude-haiku-5.5` (different vendor to reduce self-grading bias). Both are set in configuration; any OpenRouter model with tool calling and structured output may be substituted (e.g. a free model during development).
- Settings come from an environment file via a typed settings object; startup fails clearly if the OpenRouter key is missing.

### Modules (deep modules with small interfaces)
- **Config**: typed settings (keys, model names, paths, limits such as the tool-call cap and window size).
- **Profile**: loads the Candidate Profile and Letter Templates and validates that every Target Role has a template mapping (AI template for AI Engineer, AI Innovation Manager and Quantum Software Developer; Grant template for Grant Management Specialist).
- **Data store** (SQLite repository): Job Postings, companies, Target Role tags, Job Info, Cover Letters, Conversations metadata, security log, token-usage records. Interface in domain terms (get posting by job ID, list tagged postings, save Job Info, save Cover Letter, run statistics query), hiding SQL.
- **Importer**: one-time import of all Kaggle postings and companies (idempotent upsert keyed by job ID); a tagging step that applies the reviewed Target Role title list; a re-tag command.
- **Indexer**: chunks tagged postings (recursive splitting, about 1,000 characters with 150 overlap, prefixed with a context header and carrying metadata) and writes one ChromaDB collection per embedding model; batches requests, retries on rate limits, reports progress, and embeds only postings not yet indexed.
- **Embedding providers**: local sentence-transformers MiniLM; Gemini Embedding 2 and Qwen3-Embedding-8B via OpenRouter embeddings. One interface; selected by name.
- **Retriever** (the advanced RAG pipeline): self-query (LLM produces search text plus metadata filters via a schema), multi-query (3 to 4 variants via a schema), filtered similarity search per variant, Reciprocal Rank Fusion with de-duplication by job ID, optional cross-encoder reranking (local `cross-encoder/ms-marco-MiniLM-L-6-v2`) of the top ~20, top 5 returned. Returns Search Results plus a trace of every stage for the Evaluation tab.
- **Tools** (LangChain tools):
  - `search_jobs`: the Retriever; stores the numbered Search Results in agent state.
  - `get_job_description`: verbatim Job Description by job ID from the data store; no LLM.
  - `create_job_info`: retrieves the posting and company by job ID; copies job name, company name and work type exactly; LLM summarises role and company vision via a schema ("not stated in the posting" when absent); judge checks both summaries; saves Job Info.
  - `write_cover_letter`: picks the Letter Template for the posting's Target Role; fills the gaps from Job Info; generates the motivation sentence from the role summary and Letter Template facts only; runs deterministic checks (job name, company, work type) and the judge; one automatic rewrite on failure; saves and returns the letter with check results.
  - `job_market_stats`: a fixed set of parameterised SQL statistics over all postings or tagged postings (counts by Target Role, work type, remote, location; salary ranges); returns data suitable for charts. The LLM chooses a statistic and its parameters via the tool schema; it never writes SQL.
- **Judge**: RAGAs `DiscreteMetric` with values `supported` / `fabricated` and a reason; used live (Job Info summaries, Cover Letter sentences) and in evaluation.
- **Agent**: LangChain `create_agent` with the five tools, a domain system prompt, and the security rules. State includes the message history, the latest numbered Search Results, and the current job ID. Persistence via the LangGraph SQLite checkpointer, one thread per Conversation. Only a sliding window of recent messages (about 20) is sent to the LLM; the full history stays stored. Tool-call cap of 6 per question; tool errors are returned to the LLM as tool results.
- **Chat service** (test seam 3): one turn = send a message to a Conversation, receive a stream of answer events (text chunks, status updates, tool calls with arguments and results, citations, Job Description, Job Info, Cover Letter, token usage). Applies input validation, the injection pattern check and the scope guard before calling the agent; records token usage and security events.
- **Security**: input validation (non-empty, at most 2,000 characters, numeric and existing job IDs); prompt-injection defence (posting text wrapped in clearly delimited data blocks plus a system-prompt rule; pattern-based flagging of user input); scope guard; personal data only passed in `write_cover_letter` and excluded from logs; secrets only in the environment file; tool-call cap.
- **Errors**: distinct types for external API failures (with retry and backoff), validation errors, and user-input problems; each mapped to a user-facing message. No bare catch-all.
- **Prompts**: one file per prompt (system, self-query, multi-query, Job Info, cover letter, motivation sentence, judge), loaded by name and also displayed read-only on the Evaluation tab.
- **Evaluation**: RAGAs test-set generation from tagged postings (about 25 generated, filtered to about 20 approved by the candidate) plus 10 hand-written agent cases; computes RAGAs metrics per embedding model and with or without the reranker, Tool Selection Accuracy, Parameter Accuracy, the judge verdicts, and field accuracy over saved Cover Letters; stores results for the UI.
- **UI** (Streamlit): two tabs, "MAIN" and "Evaluation". MAIN is ChatGPT-style (chat messages, bottom input, streaming, status lines, [n] citations, boxes for Job Description, Job Info and Cover Letter with .docx and PDF download, charts for statistics). The left sidebar has New chat, past Conversations, the embedding dropdown (Gemini preselected) and the reranker toggle (on by default). Evaluation shows history and export, RAG process, full sources, tokens and cost, tool calls, security measures and log, system prompts, and RAGAs results.

### Schema (SQLite, conceptual)
- Job Postings: all Kaggle posting columns keyed by job ID, plus the Target Role tag (nullable).
- Companies: Kaggle company columns keyed by company ID.
- Job Info: job ID (link to the posting), job name, company name, work type, company vision, role summary, judge verdicts and reasons, created at.
- Cover Letters: job ID, Target Role, Letter Template used, full text, deterministic check results, judge verdicts, rewrite flag, created at.
- Conversations: ID (also the agent thread ID), title, created and updated at. Messages and agent state live in the checkpointer tables.
- Token usage: per LLM call (Conversation, model, input and output tokens, cost).
- Security log: timestamp, Conversation, event type, excerpt (personal data never logged).
- Evaluation runs: run ID, configuration (embedding model, reranker), metric values, per-case results.

### Versions (each is committed and working before the next)
- v0.0 setup, module layout, config, errors, Candidate Profile and AI Letter Template, README skeleton.
- v0.1 import of all postings and companies; Target Role title list and tagging; re-tag.
- v0.2 chunking and the three ChromaDB indexes.
- v0.3 Retriever (self-query, multi-query, fusion).
- v0.4 Agent with `search_jobs` and `get_job_description`, memory, Chat service, ChatGPT-style MAIN with citations and streaming; token recording.
- v0.5 Job Info with judge.
- v0.6 Cover Letter with templates, checks, rewrite, downloads (Grant Letter Template needed here).
- v0.7 `job_market_stats` with charts; all security measures.
- v0.8 sidebar: Conversations, embedding dropdown, reranker toggle.
- v0.9 Evaluation tab sections.
- v1.0 RAGAs evaluation and comparisons; README completion with reflection.

## Testing Decisions

- A good test exercises external behaviour through one of the three agreed seams and asserts on observable outcomes (stored rows, returned Search Results, answer events), never on internal calls or private helpers. LLM, embedder and judge are replaced by deterministic fakes (scripted structured outputs, fixed vectors, fixed verdicts), so tests are free, offline and repeatable.
- **Seam 1: data pipeline** (importer, tagging, re-tag, indexer) on a fixture of about 20 Kaggle-shaped rows: all postings stored; idempotent re-import; correct Target Role tags including title variants and non-matches; re-tag after list change; chunks have context header and metadata; only tagged postings indexed; only new postings embedded on re-index.
- **Seam 2: the five tools** called directly: filters are honoured; Search Results are de-duplicated and numbered; reranker on/off changes order as scripted; verbatim Job Description is byte-identical; exact Job Info fields; "not stated in the posting" on missing company description; deterministic check fails on wrong company, title or work type; rewrite path and final warning; missing Letter Template error; statistics are exact on fixture data and state their population.
- **Seam 3: Chat service** with a scripted fake LLM and in-memory storage: "#2" resolves to the right job ID after window trimming; reopening a Conversation restores history and Search Results; the tool-call cap stops a looping agent; a failing tool yields a friendly answer; validation, injection flagging and scope guard; personal data absent from logs and only present in the letter call; token usage recorded.
- Evaluation computations (Tool Selection Accuracy, Parameter Accuracy, field accuracy) are tested on fixed case data with a fake judge.
- UI: a single smoke test that the Streamlit app starts and renders both tabs.
- Prior art: none (fresh repo). These tests set the patterns for the project.

## Out of Scope

- Bundesagentur Jobsuche API or any scraping (v2).
- Target Roles beyond the four v1 roles (AI Engineer, Grant Management Specialist, AI Innovation Manager, Quantum Software Developer) (v2).
- German-language Cover Letters and a dedicated Quantum Letter Template (v2).
- CV upload or CV retrieval; a motivation paragraph supplied by the candidate.
- Multiple candidates or user accounts; deployment beyond running locally.
- Sending applications (email, Gmail), calendar integration, approval workflows.
- Summary memory; a chat-model selector in the UI.
- Embedding all untagged postings.

## Further Notes

- Data: Kaggle "LinkedIn Job Postings (2023-2024)" by Arsh Koneru, CC BY-SA 4.0 (attribution required in the README). About 124k mostly US postings; already downloaded into the git-ignored data folder. Roughly 1,081 postings match the four Target Roles; only about 20 are quantum-related.
- Candidate Profile values supplied so far: GitHub https://github.com/sara-qbit, email ada@example.com; the remaining details come from the existing Letter Template text. The Grant Management Letter Template will be supplied by the candidate before v0.6.
- Estimated one-time embedding cost: about $0.35. Running cost per search-to-letter session: under $0.01.
- Deadline: Sunday 2026-10-25. Core requirements are complete after v0.7; v0.8 to v1.0 deliver the optional tasks.
- See the glossary for terms and ADR 0001 for the decision to store all postings while searching only tagged ones.
