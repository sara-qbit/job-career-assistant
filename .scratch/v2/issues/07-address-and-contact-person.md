# 07: Address and contact person in the letter

**What to build:** A Cover Letter for a pasted Job Posting is addressed to the company address and contact person named in the ad, with a matching greeting; otherwise it uses the company name and city and the standard greeting. The candidate can give the address themselves. See spec: Cover Letter.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] The recipient address block uses the address read word for word from the ad; without one, just the employer and city
- [ ] "The address is …" stores the candidate's address for that Job Posting and the next Cover Letter uses it
- [ ] With a contact person given as Frau/Herr (or Mr/Ms), the letter has "z. Hd. <name>" below the company and the greeting "Sehr geehrte Frau …," / "Sehr geehrter Herr …," / "Dear …,"
- [ ] Without such a contact person: "Sehr geehrte Damen und Herren," / "Dear Hiring Team,"
- [ ] The name is copied from the ad, never invented, and the exact checks cover it
- [ ] Tests through the chat service for each case; full suite and ruff green
