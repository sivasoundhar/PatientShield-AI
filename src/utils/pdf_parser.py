"""Text extraction from PDF, DOCX, and TXT medical documents.

Imports for PyMuPDF/python-docx are deferred into each extractor function
rather than at module level — most of this project's test runs never touch a
PDF or DOCX, and lazy imports keep `import src.utils.pdf_parser` cheap.

OCR for scanned (image-only) PDFs is intentionally not implemented yet: it
needs PaddleOCR, which is commented out in requirements.txt until a later day
actually exercises it (see CLAUDE.md section 2, "do not build ahead"). A
scanned PDF fails loudly below rather than silently returning empty text.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedFileTypeError(ValueError):
    """Raised when a file's extension isn't one PatientShield AI can parse."""


def extract_text(file_path: str) -> str:
    """Extract plain text from a PDF, DOCX, or TXT file.

    Args:
        file_path: Path to the uploaded document on disk.

    Returns:
        The document's full text content.

    Use when: called once per upload, before any PHI/clinical processing —
    every downstream agent operates on plain text, never on file bytes.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    if suffix == ".txt":
        return _extract_txt(file_path)

    # Edge: bad file type — fail loudly per rule 8, don't guess a fallback parser.
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def _extract_pdf(file_path: str) -> str:
    import fitz  # PyMuPDF

    with fitz.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    if not text.strip():
        # No text layer usually means a scanned/image-only PDF. Raising here
        # instead of returning "" prevents silently processing a blank
        # document — de-identification of empty text is trivially "safe"
        # but useless, and would hide the real problem from the caller.
        raise ValueError(
            f"No extractable text in '{file_path}' — likely a scanned PDF. "
            "OCR fallback (PaddleOCR) is not yet implemented."
        )
    return text


def _extract_docx(file_path: str) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def _extract_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    import tempfile

    # Self-test: TXT round-trip and unsupported-extension rejection. PDF/DOCX
    # extraction is covered by tests/test_health.py fixtures instead, since
    # building a valid PDF/DOCX in a one-liner here would obscure the point.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Patient reports mild headache.")
        txt_path = f.name

    assert extract_text(txt_path) == "Patient reports mild headache."

    try:
        extract_text("document.xyz")
        raise AssertionError("expected UnsupportedFileTypeError")
    except UnsupportedFileTypeError:
        pass

    Path(txt_path).unlink()
    print("pdf_parser.py self-test passed: TXT extraction and unsupported-type rejection both work.")
