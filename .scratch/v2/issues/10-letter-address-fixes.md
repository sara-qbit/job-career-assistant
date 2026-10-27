# 10: Letter address and greeting fixes

**What to build:** Four small defects in the German letter's address block and greeting, found while trying the candidate's German Letter Templates (ticket 08). They come from ticket 07's address code, not from the templates. See spec: Cover Letter, user stories 30-35.

**Blocked by:** 07

**Status:** ready-for-agent

- [ ] The company name is not repeated when the ad's address already starts with it (e.g. "Nordlicht Analytics GmbH" above "Nordlicht Analytics GmbH, Karl-Liebknecht-Straße 12, 04107 Leipzig")
- [ ] The greeting leaves out the first name: "Sehr geehrte Frau Dr. Weber," not "Sehr geehrte Frau Dr. Anna Weber," (English likewise: "Dear Dr. Weber,")
- [ ] The German "z. Hd." line uses "Herrn": "z. Hd. Herrn Prof. Dr. Thomas Krüger"
- [ ] A company name ending in "." gets no second full stop ("bei Quantum Lab Dresden e. V.", not "e. V..")
- [ ] Tests through the chat service for each case; full suite and ruff green
