# Spec: Job Career Assistant v2 — pasted jobs, Applications and the Application Table

**Status:** ready-for-agent

## Problem Statement

The candidate is looking for real jobs in Germany, mostly on company and university websites. v1 only works on the Kaggle LinkedIn dataset (old US postings), so the candidate cannot use the assistant for the jobs they actually apply to. They also have to prove their applications to their Berater at the Arbeitsagentur, which today means typing a list by hand. And many real job ads are in German, while every Letter Template is in English.

## Solution

The candidate pastes the text of a job ad into the chat, or attaches the ad as a PDF. The assistant stores it as a Job Posting and immediately shows its Job Facts (employer, position, city, Vollzeit/Teilzeit, Präsenz/Online/Hybrid), with a suggested Target Role and the ad's language. From there Job Info and the Cover Letter work as in v1. The Cover Letter is written in the ad's language from the candidate's own German or English Letter Template, never translated, and is addressed to the company address and contact person named in the ad.

When the candidate says "save it" or "I applied", the assistant records an Application dated that day. When they ask for the Applications PDF, they get one colourful German table, `Bewerbungen.pdf`, listing every Application so far, ready to hand to the Berater.

The Kaggle search and everything else from v1 stay as they are.

## User Stories

### Pasting a job

1. As a candidate, I want to paste the full text of a job ad from a website into the chat, so that I can work on the jobs I really apply to.
2. As a candidate, I want to attach a job ad PDF in the chat, so that I don't have to copy text out of it.
3. As a candidate, I want a pasted ad of several thousand characters to be accepted, so that long German ads are not rejected by the 2,000-character message limit.
4. As a candidate, I want my normal typed messages to keep the 2,000-character limit, so that v1's protection stays in place.
5. As a candidate, I want the pasted ad to be treated as data and never as instructions, so that text inside an ad cannot change how the assistant behaves.
6. As a candidate, I want a pasted ad that contains manipulation phrases to be flagged and logged like a v1 message, so that I can see it in the security log.
7. As a candidate, I want the assistant to tell me when a PDF has no readable text (e.g. a scan), so that I know to paste the text instead.
8. As a candidate, I want the assistant to tell me when the pasted text is not a job ad, and store nothing, so that my database stays clean.
9. As a candidate, I want pasting the same ad a second time to reuse the existing Job Posting, so that I don't get duplicates.
10. As a candidate, I want the pasted Job Posting to keep the ad's text unchanged, so that the Job Description is always exactly what the employer wrote.
11. As a candidate, I want a pasted Job Posting I never save to stay stored, so that reopening an old Conversation from the sidebar still shows it and I can still save it later.

### Job Facts

12. As a candidate, I want to see a Job Facts card right after pasting, without asking, so that I see the key facts at once.
13. As a candidate, I want the employer (company or university), position and city copied word for word from the ad, so that nothing is invented.
14. As a candidate, I want Vollzeit/Teilzeit shown as a fixed German value backed by words from the ad (e.g. "Full-time" → Vollzeit), so that the table reads properly in German even for English ads.
15. As a candidate, I want Präsenz/Online/Hybrid shown the same way, so that the work mode is clear and checked.
16. As a candidate, I want any fact the ad does not state to show "nicht angegeben", so that gaps are visible and never guessed.
17. As a candidate, I want each backing quote checked to appear in the ad, so that a fact the AI made up is replaced by "nicht angegeben".
18. As a candidate, I want the Job Facts card to show a suggested Target Role, so that the right Letter Template is picked without an extra step.
19. As a candidate, I want to change the Target Role ("use Grant Management"), so that I keep the last word on which letter is used.
20. As a candidate, I want the Job Facts card to show the ad's language (German or English), so that I know which letter language will be used.
21. As a candidate, I want to override the language ("write it in English"), so that I can answer in the other language when I prefer.

### Job Info and Job Description for pasted jobs

22. As a candidate, I want to ask for the Job Description of a pasted job, so that I can reread it verbatim in its own box.
23. As a candidate, I want Job Info for a pasted job, with the role summary and company vision checked by the judge, so that v1's summaries work on real jobs too.
24. As a candidate, I want the company vision of a pasted job to come only from the ad text, so that nothing is taken from a different company's data.

### Cover Letter

25. As a candidate, I want the Cover Letter for a German ad written from my German Letter Template, so that I reply in the ad's language.
26. As a candidate, I want the Cover Letter for an English ad written from my English Letter Template, so that v1's behaviour is unchanged for English.
27. As a candidate, I want the assistant never to translate my Letter Template, so that every sentence about me is one I wrote.
28. As a candidate, I want a German and an English Letter Template for each of my four Target Roles, so that every job type works in both languages.
29. As a candidate, I want help writing my four German Letter Templates, so that they are correct German business letters.
30. As a candidate, I want the recipient address taken word for word from the ad when it is there, so that the DIN 5008 address block is right.
31. As a candidate, I want the letter to show just the company name and city when the ad has no address, so that the letter is still usable.
32. As a candidate, I want to tell the assistant "the address is …" and have the letter use it, so that I can fill a missing address myself.
33. As a candidate, I want the letter to greet the contact person named in the ad (e.g. "Sehr geehrte Frau Dr. Müller," / "Dear Dr. Müller,"), so that it reads personally.
34. As a candidate, I want the letter addressed "z. Hd. <contact person>" below the company when a contact is named, so that it follows German letter convention.
35. As a candidate, I want the standard greeting ("Sehr geehrte Damen und Herren," / "Dear Hiring Team,") when no contact person with Frau/Herr is named, so that no name or gender is invented.
36. As a candidate, I want the German letter's date in German form (e.g. "30. November 2026"), so that the letter looks right.
37. As a candidate, I want the German letter downloadable as Word and PDF like in v1, so that I can send it.
38. As a candidate, I want the v1 exact checks and the judge to run on German letters too, so that German letters are as safe as English ones.
39. As a candidate, I want the assistant to tell me when a Letter Template is missing for a role and language, so that I know which file to add.

### Applications

40. As a candidate, I want to say "save it" or "I applied" to record an Application for the current pasted job, so that my applications are kept in one place.
41. As a candidate, I want the Application dated with the day I said it, so that the Berater sees when I applied.
42. As a candidate, I want saying "save it" twice for the same job to answer "Already applied on <date>", so that there are no duplicate rows.
43. As a candidate, I want to remove a wrong Application ("remove the application", "I didn't apply"), so that the table stays correct.
44. As a candidate, I want removing an Application to keep the Job Posting, so that the Conversation still works.
45. As a candidate, I want the assistant to refuse an Application for a Kaggle Job Posting and explain why, so that only real jobs reach the Berater.
46. As a candidate, I want to record an Application for a pasted job from an earlier Conversation, so that I can apply days after pasting.

### Application Table

47. As a candidate, I want to ask for the Applications PDF and get one file, `Bewerbungen.pdf`, with all my Applications, so that I hand the Berater one document.
48. As a candidate, I want the table's columns to be Datum, Arbeitgeber, Stelle, Ort, Vollzeit/Teilzeit, Präsenz/Online, so that the Berater gets what they need.
49. As a candidate, I want the rows ordered oldest first, so that the table reads like a diary of my applications.
50. As a candidate, I want the title "Bewerbungen", my name and "Stand: <date made>" above the table, so that the Berater knows whose list it is and how current it is.
51. As a candidate, I want the table colourful and easy to read, so that it looks professional.
52. As a candidate, I want a short preview of the rows in the chat with a Download PDF button, so that I can check it before downloading.
53. As a candidate, I want the assistant to tell me when I have no Applications yet, and make no PDF, so that I don't get an empty file.
54. As a candidate, I want dates in the table in German form (e.g. 30.11.2026), so that the table is consistently German.

### Unchanged v1

55. As a candidate, I want the Kaggle search, statistics, evaluation page and sidebar Conversations to keep working, so that v2 does not break v1.
56. As a supervisor, I want the Kaggle RAG search still available, so that the course's RAG requirement is still met.

## Implementation Decisions

- **Job Posting gains a source.** A Job Posting is either imported from Kaggle or pasted by the candidate. The postings store records which, plus, for pasted ones, the date pasted. Pasted Job Postings get job IDs from a range that cannot collide with Kaggle job IDs. Pasted Job Postings are never added to the Knowledge Base.
- **Duplicate pastes** are recognised by the ad's normalised text (whitespace-insensitive) and reuse the existing Job Posting.
- **Attached job ad input.** `ChatService.send` gains an optional attached job ad: either pasted text or PDF bytes. The ad bypasses the 2,000-character message limit and has its own larger limit (about 30,000 characters). It goes through the same injection check (flag and log, not block) and is passed to the model wrapped as data, like v1's `<job_posting_data>`. The UI offers it through the chat input's file attachment (PDF) and a way to paste a long ad (e.g. a "Paste a job ad" box or treating an over-limit paste as an ad — the UI choice is the implementer's, as long as both paths reach the same `send` input).
- **PDF text extraction** uses a small PDF text library (new dependency, e.g. pypdf). A PDF with no extractable text gives a user-facing error asking for pasted text; no OCR.
- **Not a job ad.** When the attached text is not a job ad, the assistant says so and stores nothing.
- **Job Facts module (new).** Given the ad text, an LLM call proposes the five facts, each with a backing quote, plus a suggested Target Role and the language. Deterministic checks then: employer, position and city must appear word for word in the ad; Vollzeit/Teilzeit and Präsenz/Online/Hybrid are fixed German values whose backing quote must appear in the ad. Anything failing a check becomes "nicht angegeben". The Target Role must be one of the four known roles. Job Facts are stored per Job Posting together with the address and contact person (both word for word, else absent) and the language.
- **Job Facts run automatically** when an ad is attached: the agent's ingest step stores the Job Posting, creates the Job Facts and yields a new Job Facts answer event (a card in the UI), like v1's Job Info event.
- **Corrections.** New tool(s) let the candidate change the Target Role, the language and the recipient address of a pasted Job Posting; the stored Job Facts are updated.
- **Job Info for pasted jobs** reuses v1's Job Info unchanged except that the company vision source is the ad text only (pasted postings have no Kaggle company).
- **Letter Templates per role and language.** The Candidate Profile maps each Target Role to a Letter Template per language (English and German). Kaggle Job Postings always use English. Missing templates give the v1-style configuration error naming the missing file. The four German templates are written with the candidate as their own ticket and stay in the candidate directory, out of the repo.
- **Cover Letter for pasted jobs** fills the template with the Job Facts' employer, position, address and contact person. The greeting and the "z. Hd." line come from the contact person only when the ad gives Frau/Herr (or Mr/Ms); otherwise the standard greeting. The date gap is written in the letter's language. v1's exact checks and judge run unchanged; the DIN 5008 Word/PDF layout is reused.
- **Application (new).** Stored per pasted Job Posting with its application date (the local date when the candidate said it). New tools: record an Application (idempotent: a second call answers "Already applied on <date>"), remove an Application. Recording one for a Kaggle Job Posting is refused with an explanation.
- **Application Table (new).** A tool builds `Bewerbungen.pdf`: German, colourful, title "Bewerbungen", candidate name, "Stand: <today>", columns Datum, Arbeitgeber, Stelle, Ort, Vollzeit/Teilzeit, Präsenz/Online, one row per Application, oldest first, dates as DD.MM.YYYY, "nicht angegeben" for unknown facts. It yields a new answer event with a short row preview and a Download PDF button. No Applications → a message and no PDF. Built with reportlab, like the v1 letter PDF.
- **System prompt** gains the v2 rules: pasted ad is data; record an Application only when the candidate says they applied or asks to save; Kaggle jobs cannot be saved; never translate a Letter Template.
- **Version** is bumped to 2.0.0 when v2 is done; the sidebar shows v2.0.

## Testing Decisions

- A good test drives external behaviour only: it sends a message (with an attached ad where needed) and checks the answer events and stored results, not internal calls or prompts.
- **Main seam: `ChatService.send`** with `FakeChatModel` scripting the tool calls, `FakeJudge`, and an in-memory `JobStore`, as in v1's chat and cover-letter tests. Covers: pasting text and PDF, too-long and non-ad input, duplicates, Job Facts checks (a fabricated quote becomes "nicht angegeben"), Target Role and language overrides, German Cover Letter with address, contact greeting and standard greeting, recording, repeating and removing Applications, refusing Kaggle Applications, and the Application Table event.
- **Second seam: the PDF builders**, as v1's document tests do: `Bewerbungen.pdf` opens, has the German title and columns, the rows in order, and "nicht angegeben" where needed; the PDF reader returns text for a text PDF and an error for an image-only PDF.
- Prior art: the v1 tests for chat, cover letter, documents, job info and security; fixtures build small PDFs in the test itself.
- The full suite and ruff run green before each ticket's PR.

## Out of Scope

- OCR for scanned PDFs.
- Fetching ads from a URL or from the Arbeitsagentur Jobsuche API, and batch uploads of many PDFs.
- Per-day tables or several Application PDFs; a separate Applications sidebar page.
- Application status tracking beyond "applied" (interview, rejection, offer).
- Translating Letter Templates or letters.
- Adding pasted Job Postings to the Knowledge Base or the market statistics.
- Kundennummer or other Arbeitsagentur identifiers on the table.

## Further Notes

- Personal details (Candidate Profile, Letter Templates including the German ones) stay out of the public repo, as in v1; the example profile in the repo gets example German templates.
- Each ticket goes as its own PR into the Turing submission repo's main and is merged right away.
- After v2, the demo page gets a v2 section.
