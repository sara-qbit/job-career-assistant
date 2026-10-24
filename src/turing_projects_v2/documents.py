"""Cover Letter downloads: the letter text as a .docx and as a PDF, one paragraph per line."""

import io
from xml.sax.saxutils import escape

import docx
from docx.shared import Pt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate

FONT_SIZE = 11


def letter_docx(text: str) -> bytes:
    """The letter as a Word document."""
    document = docx.Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(FONT_SIZE)
    style.paragraph_format.space_after = Pt(0)
    for line in text.splitlines():
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def letter_pdf(text: str) -> bytes:
    """The letter as an A4 PDF in Helvetica.

    Helvetica covers Western European text (umlauts, ß, dashes, curly quotes); other characters
    show as blanks.
    """
    style = ParagraphStyle("letter", fontName="Helvetica", fontSize=FONT_SIZE, leading=14)
    # An empty paragraph has no height, so a blank line is kept as a non-breaking space.
    paragraphs = [Paragraph(escape(line) or "&nbsp;", style) for line in text.splitlines()]
    buffer = io.BytesIO()
    margin = 2.5 * cm
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="Cover Letter",
    )
    document.build(paragraphs)
    return buffer.getvalue()
