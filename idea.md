Overview

RAG chatbot flow: vector retrieval, LLM reasoning and tool calling
This is the project for Sprint 2, where you bring together LangChain, RAG, function calling, and vector databases to build a specialised, domain-focused chatbot. Unlike the Sprint 1 project, this one requires advanced RAG with query translation, tool calling for practical tasks, and a polished user interface.
---
Task description
This project represents a significant step up from basic chatbot implementation. You will create a specialised chatbot that leverages advanced RAG techniques and tool calling to provide domain-specific assistance. The goal is to build something that could be valuable in a real-world context.
We will be using LangChain, your framework (Streamlit) of choice, and implementing advanced RAG techniques.
The intended code editor for this project is VS Code.
Your chatbot will:
Implement advanced RAG with query translation and structured retrieval
Include tool calling capabilities for practical tasks
Focus on a specific domain or use case
Provide detailed, context-aware responses

Chatbot that sparse the Job website search engine named: https://www.arbeitsagentur.de/jobsuche/. The specialised chatbot should leverages advanced RAG techniques and tool calling to provide domain-specific assistance. the Chat 

---
Task requirements
Core requirements
RAG Implementation:
Create a knowledge base relevant to your domain
Implement standard document retrieval with embeddings
Use chunking strategies and similarity search
Tool Calling:
Implement at least 3 different tool calls
Functions should be relevant to your domain
Examples: data analysis, calculations, API integrations
Domain Specialisation:
Choose a specific domain or use case
Create a focused knowledge base
Implement domain-specific prompts and responses
Add relevant security measures for your domain
Technical Implementation:
Use LangChain with OpenRouter (OpenAI-compatible SDK) for LLM integration
Implement proper error handling
Include user input validation
User Interface:
Create an intuitive interface using Streamlit 
Show relevant context and sources
Display tool call results
Include progress indicators for long operations
---
Optional tasks to get 100 points
After the main functionality is implemented and your code works correctly, the project should be upgraded. 
Caution: Some of the tasks in medium or hard categories may involve concepts or libraries that are introduced in later sections or even require outside knowledge/time to research outside of the course.

Easy

Add conversation history and export functionality
Add visualisation of RAG process
Include source citations in responses

Medium

Calculate and display token usage and costs
Add visualisation of tool call results

Hard

Implement an evaluation of your RAG system, using RAGAs please use reranking and show display in website
---
Evaluation criteria
Understanding Core Concepts:
The learner understands the basic principles of how RAG works
The learner can explain tool calling implementation clearly
The learner demonstrates good code organisation practices
The learner can identify potential error scenarios and edge cases
Technical Implementation:
The learner knows how to use a front-end library using their knowledge and/or external resources
The learner's project works as intended; you can chat with a chatbot and get answers
The learner has created a relevant knowledge base for their domain
The learner has implemented appropriate security considerations
Reflection and Improvement:
The learner understands the potential problems with the application
The learner can offer suggestions on improving the code and the project
Bonus Points:
For maximum points, the learner should implement at least 2 medium and 1 hard optional task from the Optional tasks list.

---


Please avoid the most common mistakes Most common mistakes in Python AI Engineering projects

Structured output via prompt instructions instead of a schema. The expected JSON format is described in the prompt text rather than enforced with the API's structured-output support (e.g. a Pydantic model or JSON schema passed to the model).
Weak error handling. Errors are either not handled at all or caught with a single catch-all except Exception, with no distinction between API failures, validation errors, and user input problems.
Prompts hardcoded in the code. Prompts are embedded as strings inside functions instead of being kept in separate files or a dedicated module, which makes them hard to review, version, and change.
No uv for project and dependency management. Dependencies are installed ad hoc or listed in an unpinned requirements.txt instead of being managed through uv with a pyproject.toml and lockfile.
Unclear or incomplete README. The README does not explain what the project does, how to install and run it, or how it is configured.
No scoping or iterative versions. The project is not scoped to a clear MVP and then extended in versions; everything is attempted at once, with no visible progression.
No tests. There are no unit tests for core logic, and no tests that mock LLM calls.
Code does not follow PEP 8. Inconsistent naming, formatting, and line lengths; no linter or formatter (e.g. ruff) is used.
No separation of concerns. The code is a single monolithic script instead of a module structure where LLM calls, prompts, business logic, and I/O are kept apart.
---


USE CASE
Phase 1 — Candidate Profile Store
Goal: A place to store the "source of truth" data described in spec — CV, Motivation Letter and target roles — that nothing is allowed to fabricate.



Phase 2 — Job Discovery 
Goal: Pull real vacancies from authorized sources only: Kaggle

Phase 3 — Semantic search 
Here I think, system prompt engineering + RAG (keyword search or vector search) 
{"Name of job": ------,    
"Name of company": ------,   
"Part/full ---": ------,   
"Vision and goal of the company": ------,
    "Summary of the role": ------,
     "clever motivation why the transition to this specific job domain": ------,
  }
Store this as job_info, linked to jobs_raw

This is Level 1 ("Recommendation") from your spec — fully working before you touch generation or approval logic.
Skills gained: structured LLM output, prompt design for consistent JSON, basic evaluation of LLM output quality (does the reasoning actually make sense?).



Phase 4 — Application Generation ("Personalisation without Fabrication")
Goal: use the phase 3 to fill the part which should be added to paragraph 1 and paragraph 3.
Build:
Skills gained: prompt constraints/guardrails, output verification patterns (a core AI-agent-reliability skill).

---
use ChromaDB and SQLlite

---


Cover Letter

Ada Example
Example Street 1
Sample Town, 12345
Germany
Tel.: 0123 4567890


[Company name]
[Address of the Company]

Dear Mr/Ms Hiring Manager,
I am writing to apply for [Job Name] ["Part/full"] at [Company]. If possible, I would like to start on 1 January 2027. I am currently transitioning my career towards AI engineering and would greatly appreciate the opportunity to apply my AI engineering knowledge to the design and management of AI systems. I am highly motivated to take on a responsible and dedicated role and to support the top high management in achieving the [the vision and goal of the Company]
My foundation in AI dates back to my doctoral studies, which I completed in 2018, before the current generation of generative AI existed. Rather than viewing this as a gap, I have been working purposefully to close it. I am currently completing a government-funded training programme in AI Engineering through the German Federal Employment Agency (Agentur für Arbeit), which ends on 21 December 2026 and covers the following topics:
- Fundamentals of LLM application development
- Application development using LangChain, RAG, and Streamlit
- AI agents
- Evaluation benchmarks
- Final project in AI engineering
The current Projects that I made are in my Github link [Github Link]. I am highly motivated to ["Summary of the role"] 
During my work at Budapest University of Technology and Economics and the 6G Health Institute, I developed strong skills in analysing and selecting suitable strategies to solve complex engineering problems in data science and suppoting in developing and pushing forward new use cases for EU Funding projects. I am fluent in German, English, French, and Arabic.
I would welcome the opportunity to discuss in an interview how I could contribute to achieving the objectives of [Company name]. I look forward to hearing from you and to the possibility of a positive response.
Yours sincerely,
Ada Example

