# 05: Application Table (Bewerbungen.pdf)

**What to build:** When the candidate asks for the Applications PDF, the assistant shows a short preview of the rows and a Download PDF button for `Bewerbungen.pdf`: one colourful German table of all Applications so far, for the Berater. See spec: Application Table.

**Blocked by:** 02, 04

**Status:** ready-for-agent

- [x] The PDF is titled "Bewerbungen", shows the candidate's name and "Stand: <today as DD.MM.YYYY>"
- [x] Columns: Datum, Arbeitgeber, Stelle, Ort, Vollzeit/Teilzeit, Präsenz/Online, filled from the Application date and the Job Facts
- [x] One row per Application, oldest first; dates as DD.MM.YYYY; unknown facts as "nicht angegeben"
- [x] The table is colourful and readable (header colour, alternating rows); long values wrap
- [x] The download file is named `Bewerbungen.pdf`
- [x] With no Applications, the assistant says so and makes no PDF
- [x] Removed Applications do not appear
- [x] Tests: through the chat service for the event, and on the PDF builder for title, columns, row order and German text; full suite and ruff green
