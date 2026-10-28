# Job Career Assistant

A chatbot that finds job postings matching the candidate's target domain and writes a personalised cover letter for a chosen job, without inventing facts about the candidate. It also takes job ads the candidate pastes, records the ones they applied to, and lists them in the Application Table, a German table for their Berater.

## Language

**Job Posting**:
One vacancy, either imported from the Kaggle LinkedIn job postings dataset or pasted (as text or a PDF) by the candidate; kept unchanged either way.
_Avoid_: Job, vacancy, listing (when meaning the raw record), Pasted Job

**Job Description**:
The original, verbatim text of a Job Posting; always shown to the candidate exactly as imported or pasted, never rewritten.
_Avoid_: Summary, job details

**Job Info**:
The structured summary of one Job Posting (job name, company, work type, company vision, role summary), every field verifiable against the posting; feeds the Cover Letter.
_Avoid_: Job summary, job details, extraction

**Job Facts**:
The five facts read from a pasted Job Posting as soon as it is pasted: employer, position, city, full or part time, on-site or online; employer, position and city are copied word for word from the Job Description; full or part time and on-site or online are fixed German values (Vollzeit/Teilzeit, Präsenz/Online/Hybrid), each backed by words that appear in the Job Description; any fact not found is "nicht angegeben". Stored with them: a suggested Target Role and the ad's language (German or English), both changeable by the candidate, and the address and contact person if the ad states them word for word (the candidate can give the address themselves).
_Avoid_: Job Info (which adds summaries), extraction, read job

**Knowledge Base**:
The Job Postings tagged with a Target Role, which the chatbot can search by meaning; untagged and pasted postings are stored but not searchable.
_Avoid_: Corpus, dataset (when meaning what the bot searches)

**Candidate Profile**:
The facts about the candidate supplied by the candidate themselves; the only source the chatbot may draw on when writing about the candidate.
_Avoid_: CV, resume, user data

**Target Role**:
A job title or domain the candidate wants to apply for; part of the Candidate Profile and the filter for what enters the Knowledge Base. A pasted Job Posting gets a suggested Target Role, which the candidate can change; it picks the Letter Template.
_Avoid_: Profile (when meaning a job title), position type

**Letter Template**:
The candidate's fixed cover-letter text with placeholders to be filled per Job Posting; one per Target Role in English and one in German, never translated by the chatbot.
_Avoid_: Motivation Letter, template letter

**Search Result**:
A Job Posting returned by the latest search in a Conversation, numbered so the candidate can refer to it as "#n".
_Avoid_: Hit, match, result item

**Conversation**:
One saved chat between the candidate and the chatbot, reopenable from the sidebar.
_Avoid_: Session, thread, chat history (when meaning one chat)

**Application**:
A pasted Job Posting the candidate has applied for, recorded when they say so ("save it", "I applied"); the day they said it is its application date. A pasted Job Posting without an Application stays stored. Kaggle Job Postings cannot have an Application.
_Avoid_: Saved Job, bookmarked job, Bewerbung (except in the Application Table itself)

**Application Table**:
The one German PDF table of all the candidate's Applications so far, each row dated with its application date, for the candidate's Berater at the Arbeitsagentur.
_Avoid_: Report, list, Bewerbungsliste

**Injection Guard**:
The LLM check that blocks a typed message or a pasted job ad trying to change the assistant's (or another AI's) instructions, before the agent sees it. Unlike the regex injection flag, it blocks; unlike the scope guard, it judges intent to manipulate, not topic.
_Avoid_: classifier, filter, prompt-injection check (that is the regex flag)

**Cover Letter**:
The Letter Template filled in for one specific Job Posting.
_Avoid_: Motivation Letter, recommendation letter, application letter
