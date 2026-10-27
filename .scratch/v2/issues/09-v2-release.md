# 09: v2.0 release

**What to build:** v2 is finished and documented: the README describes pasting jobs, Job Facts, German Cover Letters, Applications and `Bewerbungen.pdf`; the version is 2.0.0 and the sidebar shows v2.0.

**Blocked by:** 01, 02, 03, 04, 05, 06, 07, 08, 10

**Status:** done

- [x] README updated for v2 (features, how to paste, how to get the Applications PDF, out of scope)
- [x] Version 2.0.0; sidebar shows "Job Career Assistant v2.0"
- [x] Glossary (CONTEXT.md) matches the final terms
- [x] Full test suite and ruff green; the app starts and a manual run-through (paste, Job Facts, German letter, save, Applications PDF) works
- [x] Tag v2.0

## Comments

2026-10-27: Released. 294 tests and ruff green; the app starts. Manual run-through with the real models on copies of the databases (German ad for "AI Engineer (m/w/d)" at a made-up Nordlicht Analytics GmbH, Leipzig): Job Facts card with address and "Frau Dr. Anna Weber"; German letter from the German template with "z. Hd. Frau Dr. Anna Weber" and "Sehr geehrte Frau Dr. Weber,", all checks passed; "save it" recorded 27.10.2026, a second "save it" answered "Already applied on 27.10.2026"; Bewerbungen.pdf with the row. Cost $0.0012. Seen on the way, not fixed: the agent answered in German to English messages after a German ad, and one written letter gap read clumsily ("unterstütze ... bei der Unterstützung von Unternehmen").
