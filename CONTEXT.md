# Job Career Assistant

A chatbot that finds job postings matching the candidate's target domain and writes a personalised cover letter for a chosen job, without inventing facts about the candidate.

## Language

**Job Posting**:
One vacancy as imported from the Kaggle LinkedIn job postings dataset, unchanged.
_Avoid_: Job, vacancy, listing (when meaning the raw record)

**Job Description**:
The original, verbatim text of a Job Posting; always shown to the candidate exactly as imported, never rewritten.
_Avoid_: Summary, job details

**Job Info**:
The structured summary of one Job Posting (job name, company, work type, company vision, role summary), every field verifiable against the posting; feeds the Cover Letter.
_Avoid_: Job summary, job details, extraction

**Knowledge Base**:
The Job Postings tagged with a Target Role, which the chatbot can search by meaning; untagged postings are stored but not searchable.
_Avoid_: Corpus, dataset (when meaning what the bot searches)

**Candidate Profile**:
The facts about the candidate supplied by the candidate themselves; the only source the chatbot may draw on when writing about the candidate.
_Avoid_: CV, resume, user data

**Target Role**:
A job title or domain the candidate wants to apply for; part of the Candidate Profile and the filter for what enters the Knowledge Base.
_Avoid_: Profile (when meaning a job title), position type

**Letter Template**:
The candidate's fixed cover-letter text with placeholders to be filled per Job Posting.
_Avoid_: Motivation Letter, template letter

**Search Result**:
A Job Posting returned by the latest search in a Conversation, numbered so the candidate can refer to it as "#n".
_Avoid_: Hit, match, result item

**Conversation**:
One saved chat between the candidate and the chatbot, reopenable from the sidebar.
_Avoid_: Session, thread, chat history (when meaning one chat)

**Cover Letter**:
The Letter Template filled in for one specific Job Posting.
_Avoid_: Motivation Letter, recommendation letter, application letter
