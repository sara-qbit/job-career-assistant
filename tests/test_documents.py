import io

import docx

from turing_projects_v2.documents import letter_docx, letter_pdf

LETTER = (
    "Ada Example\n"
    "Sample Town, 12345\n"
    "\n"
    "Dear Mr/Ms Hiring Manager,\n"
    "I apply for Sr. AI/ML Engineer – (Remote) at R&D <Labs> 李.\n"  # noqa: RUF001 (the en dash is on purpose)
    "Yours sincerely,\n"
    "Ada Example\n"
)


def test_docx_holds_the_letter_line_by_line():
    document = docx.Document(io.BytesIO(letter_docx(LETTER)))

    assert [paragraph.text for paragraph in document.paragraphs] == LETTER.splitlines()


def test_pdf_is_a_pdf_document_even_with_unusual_characters():
    pdf = letter_pdf(LETTER)

    assert pdf.startswith(b"%PDF-")
    assert b"/Type /Page" in pdf
