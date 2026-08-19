# Python 3.11-slim: matches the pinned interpreter version (CLAUDE.md section 2 —
# 3.12 is "too new for 10-day sprint" compatibility with the AI library stack).
FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF/spacy build steps; kept minimal on purpose.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy's model isn't on PyPI under a version pin, so it's not in
# requirements.txt — install via direct wheel URL rather than
# `python -m spacy download`, which shells out to its own requests session
# to fetch a compatibility manifest first and is one more thing that can
# fail in a restricted build environment (see PROGRESS.md Day 4 notes).
RUN pip install --no-cache-dir "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
