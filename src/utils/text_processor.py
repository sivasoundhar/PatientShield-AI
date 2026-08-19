"""Text cleaning and normalization applied to every document right after extraction.

Runs before the PHI Agent ever sees the text. Kept deliberately minimal —
this is NOT where de-identification happens (that's phi_agent.py, Day 4).
This module only removes noise that would otherwise confuse entity detection
(null bytes, inconsistent line endings, stray control characters) without
altering clinically meaningful content.
"""

import re
import unicodedata

# Matches 2+ blank lines so paragraph breaks collapse to exactly one blank
# line, and 2+ spaces/tabs so mid-line spacing collapses to one space.
# Kept as module-level constants (not magic numbers per rule 4) since these
# are structural text-cleaning rules, not tunable thresholds.
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Normalize a raw extracted document into consistent, parseable text.

    Args:
        text: Raw text straight out of pdf_parser.extract_text().

    Returns:
        Cleaned text: consistent line endings, no null bytes, no excess
        whitespace, Unicode-normalized (NFC) so accented characters compare
        equal regardless of how the source encoded them.

    Use when: called once per document, immediately after extraction and
    before it reaches any agent.
    """
    if not text:
        return ""

    text = text.replace("\x00", "")  # PDF extraction occasionally leaves null bytes
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normalize Windows/old-Mac line endings
    text = unicodedata.normalize("NFC", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def truncate_for_preview(text: str, max_chars: int = 200) -> str:
    """Shorten text for logging/display without ever logging full PHI-laden content.

    Args:
        text: Any document or chunk text.
        max_chars: Maximum characters to keep before truncating.

    Returns:
        `text` unchanged if short enough, otherwise the first `max_chars`
        characters plus an ellipsis marker.

    Use when: building a log message or audit `details` field that
    references document content — never interpolate raw document text
    directly into logs (rule 8: fail loud at edges, but don't leak PHI while
    doing it).
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


if __name__ == "__main__":
    messy = "Patient  reports\r\nheadache.\x00\n\n\n\nFollow up in  2 weeks."
    cleaned = clean_text(messy)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "  " not in cleaned

    assert truncate_for_preview("short") == "short"
    assert truncate_for_preview("x" * 300, max_chars=10) == "x" * 10 + "…"

    print("text_processor.py self-test passed: cleaning and preview truncation both work.")
