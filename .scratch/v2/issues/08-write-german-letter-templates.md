# 08: Write the candidate's four German Letter Templates

**What to build:** Together with the candidate, write a German Letter Template for each of the four Target Roles (AI Engineer, AI Innovation Manager, Quantum Software Developer, Grant Management Specialist), based on their English templates and in their own words. They go in the candidate's private directory, never in the repo.

**Blocked by:** 06

**Status:** done

- [x] Four German Letter Templates exist in the private candidate directory, using only known gaps
- [x] The candidate has read and approved each one
- [x] The app loads the profile without errors and writes a German Cover Letter for a German test ad in each role
- [x] No personal details are committed (pre-commit hook passes)

## Notes

- The four German templates are `ai_engineer_de`, `innovation_de`, `quantum_de` and `grant_management_de` in the private `candidate/templates/`, kept out of git by the local `.git/info/exclude`; the private `profile.toml` maps each role to its own. The repo's example `ai_de` / `grant_de` stay as they are.
- At the candidate's request, Quantum Software Developer also got its own English template, `quantum` (from the candidate's qFLOW letter), private in the same way.
- Tried through `ChatService.send` on copies of the data: a German ad per role and an English Quantum ad each got a letter from the right template, all exact checks passed and nothing was judged fabricated.
- Found on the way, in ticket 07's address code (not in the templates), now ticket 10: the company name repeated when the ad's address starts with it; the greeting keeps the first name ("Sehr geehrte Frau Dr. Anna Weber,"); "z. Hd. Herr" instead of "z. Hd. Herrn"; a double full stop after a name ending in "." ("e. V..").
- With a private profile in `candidate/`, tests that read it broke. They now build the repo's example candidate directory in a temporary folder (`tests/candidate_fixture.py`), so the candidate's private files never affect the suite.
