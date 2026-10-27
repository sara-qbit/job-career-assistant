# 02: Job Facts card

**What to build:** Right after a job ad is pasted, the assistant shows a Job Facts card without being asked: employer, position, city, Vollzeit/Teilzeit, Präsenz/Online/Hybrid, plus a suggested Target Role and the ad's language. Every fact is checked against the ad. The candidate can change the Target Role and the language. See spec: Job Facts.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] Pasting an ad yields a Job Facts answer event, shown as a card in the UI
- [ ] Employer, position and city are copied word for word from the ad; a value not found in the ad becomes "nicht angegeben"
- [ ] Vollzeit/Teilzeit and Präsenz/Online/Hybrid are fixed German values, each backed by a quote that must appear in the ad; a quote not found makes the value "nicht angegeben"
- [ ] The suggested Target Role is one of the four known Target Roles
- [ ] The language is German or English
- [ ] The address and contact person are also read word for word if the ad states them (stored for ticket 07, not shown yet if not needed)
- [ ] Job Facts are stored per pasted Job Posting and survive reopening the Conversation
- [ ] "Use Grant Management" (or another role) changes the stored Target Role; "write it in English/German" changes the stored language; the card shows the change
- [ ] Glossary terms (Job Facts, Target Role) used in code and UI text
- [ ] Tests through the chat service, including a fabricated quote turning into "nicht angegeben"; full suite and ruff green
