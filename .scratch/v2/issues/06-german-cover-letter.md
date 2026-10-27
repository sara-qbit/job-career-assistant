# 06: German Cover Letter

**What to build:** Each Target Role has an English and a German Letter Template. A Cover Letter for a pasted Job Posting uses its Target Role and language from the Job Facts; a German ad gets a German letter from the German template, never a translation. Kaggle Job Postings stay English. See spec: Cover Letter.

**Blocked by:** 02

**Status:** ready-for-agent

- [x] The Candidate Profile maps each Target Role to a Letter Template per language (English, German); the example profile in the repo has example German templates with no real personal details
- [x] A Cover Letter for a pasted Job Posting uses the Job Facts' Target Role and language, and its employer and position
- [x] Kaggle Job Postings always get the English Letter Template, as in v1
- [x] The date gap is written in the letter's language (e.g. "30. November 2026")
- [x] A missing Letter Template for a role and language gives a clear message naming the file to add
- [x] v1's exact checks and judge run on German letters; Word and PDF downloads work with the DIN 5008 layout
- [x] The system prompt says never to translate a Letter Template
- [x] Tests through the chat service for German and English pasted jobs and a Kaggle job; full suite and ruff green
