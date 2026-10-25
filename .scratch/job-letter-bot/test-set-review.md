# Test-set review (ticket 15)

RAGAs generated 24 candidate questions (`turing-projects-v2 evaluate generate`, 40 tagged postings
spread over the Target Roles). The candidate asked Claude to make the selection overnight; the
decisions below are saved in `eval_questions.approved` and can be changed on the Evaluation tab.

**Criteria.** Approve a question that a job seeker could plausibly ask this chatbot (roles,
requirements, duties, pay, comparisons), that the postings can answer, and that does not duplicate
another question. RAGAs' informal personas ("wat", "so like if im…") stay: real users type that way.
Reject company trivia, personal data, contrived questions and near-duplicates.

**Result: 18 approved, 6 rejected.**

| # | Decision | Reason |
|---|---|---|
| 1 | rejected | Company trivia (ZEISS business segments), not a job question |
| 2 | approved | Licensing and drug-test requirements of a casino director role |
| 3 | approved | Requirements of the ARETUM Proposal Writer role |
| 4 | approved | Functions of JPMorgan Chase's Global Philanthropy team (context of a grant role) |
| 5 | approved | Contact for a Director of Development search (a recruiter named in the posting) |
| 6 | approved | Responsibilities and qualifications of a Director of Development role |
| 7 | rejected | Company trivia (InSite Group founding year) |
| 8 | approved | Tasks of a Grants Administrator role (contrived persona, real question) |
| 9 | approved | What Oracle needs from a Senior Director of Proposal Operations |
| 10 | approved | Comparison of two AI roles' skills |
| 11 | approved | Comparison of two telecom data roles (keyword style) |
| 12 | rejected | Contrived: one question chaining three unrelated postings |
| 13 | approved | NLP Scientist vs Healthcare Statistician responsibilities |
| 14 | approved | AI and innovation strategy in two leadership roles |
| 15 | approved | Proposal Manager vs Proposal Writer responsibilities |
| 16 | rejected | Near-duplicate of 17 (same two Quantum SME postings) |
| 17 | approved | Duties of two Quantum SME contract roles |
| 18 | approved | SaaS experience and benefits of the Vaco AI Product Manager role |
| 19 | approved | Microsoft Office and other requirements of a Proposal Writer role |
| 20 | approved | Requirements of the PriceSenz Chief AI Officer role (keyword style) |
| 21 | rejected | Asks about a named person's LinkedIn profile (personal data) |
| 22 | approved | Importance of SAS in two healthcare data roles |
| 23 | approved | Director of Development responsibilities (keyword style) |
| 24 | rejected | Contrived persona unrelated to the question (packaging engineer vs Agile) |

**Finding for the title tagging.** Questions 16 and 17 are about "Quantum SME" contract roles for
*Quantum* treasury-management software, not quantum computing. They are tagged Quantum Software
Developer, so the title variant list matches them by the word "Quantum" alone. Worth tightening
`candidate/title_variants.toml` later; the evaluation keeps 17 because it is still a fair question
about postings in the Knowledge Base.
