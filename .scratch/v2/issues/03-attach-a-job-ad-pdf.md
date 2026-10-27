# 03: Attach a job ad PDF

**What to build:** The candidate attaches a job ad as a PDF in the chat input, and it works exactly like pasted text: a pasted Job Posting is stored and the Job Facts card appears. See spec: Pasting a job.

**Blocked by:** 01 (and 02 for the card, which is done by then)

**Status:** ready-for-agent

- [ ] The chat input accepts a PDF attachment
- [ ] The PDF's text is extracted (new PDF text dependency) and handled as an attached job ad, with the same limits, checks and duplicate detection as pasted text
- [ ] A PDF without extractable text (e.g. a scan) gives a clear message asking the candidate to paste the text; nothing is stored
- [ ] A file that is not a PDF is refused with a clear message
- [ ] Tests build small PDFs in the test (text and image-only); full suite and ruff green
