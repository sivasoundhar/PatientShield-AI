# PatientShield AI

**A multi-agent clinical document intelligence platform.** Upload a medical document, and a
pipeline of five coordinated agents de-identifies protected health information (PHI), extracts
structured clinical findings, indexes the de-identified text for retrieval, and answers questions
about it with cited, grounded answers — with every decision logged to an auditable trail.

![PatientShield AI demo: upload a document, run the 5-agent pipeline, review de-identified PHI side-by-side, ask a grounded question, and inspect the audit trail](assets/demo.gif)

---

## Aim & Goal

Clinical notes are dense, unstructured, and full of PHI that makes them hard to search, share, or
build tooling around. PatientShield AI exists to answer one question: **can a coordinated set of
small, single-responsibility agents make a medical document safe to search and reason over,
without ever exposing the patient behind it?**

Concretely, the system aims to:

- **Detect and mask PHI** with high precision, distinguishing genuine identifiers ("Patient Smith")
  from clinical noise that merely looks like one ("Dr. Smith", a title, not an identity).
- **Extract clinical signal** — diagnoses, medications, findings, follow-ups — from the
  de-identified text, prioritized by clinical relevance, not just mention frequency.
- **Enable grounded Q&A** over the document with real citations, and a hard refusal instead of a
  guess when the answer genuinely isn't in the text.
- **Make every agent decision auditable** — timestamped, attributed, and reviewable — because a
  healthcare-shaped system that can't explain itself isn't trustworthy, regardless of accuracy.

This is a portfolio-grade build: the engineering patterns (multi-agent orchestration, confidence
gating, graceful degradation, audit logging) are production-shaped, but the project has not been
through a compliance review and is not a certified HIPAA product. See
[Known Limitations](#known-limitations).

---

## Architecture

```
                    Medical Document (PDF / DOCX / TXT)
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Planner Agent   │  Builds the execution plan
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │  PHI Reasoner    │  Presidio + LLM context verification
                         │     Agent        │  → de-identified text
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Clinical Agent   │  Diagnoses, meds, findings, follow-ups
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │ Knowledge Agent  │  Indexes de-identified text for
                         │                  │  hybrid (BM25 + dense) retrieval
                         └────────┬─────────┘
                                  ▼
                         ┌─────────────────┐
                         │  Audit Agent     │  Consolidates the full event trail,
                         │                  │  flags any agent that silently no-ops
                         └────────┬─────────┘
                                  ▼
        De-identified text • Clinical summary • Searchable, citable index
```

The pipeline is a **LangGraph `StateGraph`**: a single `PipelineState` object flows through all
five nodes in sequence, each node reading what it needs and appending to a shared audit trail. No
node has side effects (database writes happen once, after the graph completes) and no node's
failure takes down the ones before it — the Knowledge Agent, for example, degrades to an honest
"unavailable" state rather than crashing the whole run if its retrieval backend isn't reachable.

---

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| **Backend** | FastAPI + Uvicorn | Async, stateless, typed request/response models on every route |
| **Orchestration** | LangGraph + LangChain | `StateGraph`-based 5-node pipeline with typed shared state |
| **LLM Gateway** | Groq (primary + fallback model) → Ollama (local fallback) → Claude (demo mode) | Chained fallback, not a single point of failure; every response tags which provider served it |
| **PHI Detection** | Microsoft Presidio + spaCy (`en_core_web_sm`) | Regex/NER candidate detection, gated by LLM context verification before anything is trusted as PHI |
| **Clinical Extraction** | Structured-JSON LLM extraction | Diagnosis / Medication / Finding / Follow-up categories, priority-scored, negation-aware |
| **Retrieval** | Hybrid BM25 (keyword) + dense embeddings | Keyword catches exact clinical terms; dense catches semantic paraphrase |
| **Vector Store / Embeddings** | ChromaDB + `sentence-transformers` (BAAI/bge-small-en-v1.5) | Local, persistent, no external vector DB dependency |
| **Document Parsing** | PyMuPDF, `python-docx` | Structured PDF and DOCX text extraction |
| **Database** | SQLAlchemy + SQLite | 5 tables: documents, PHI entities, clinical findings, audit logs, test metrics |
| **Frontend** | React 18 + TypeScript + Vite + Tailwind CSS v4 | 5 routed pages, typed API client, shared chat/audit components |
| **Testing** | pytest + pytest-asyncio | ~65 tests across unit, integration, and system-level suites — no mocked LLM calls in the real test suites |
| **Config** | Pydantic Settings + `.env` | Every threshold, model name, and path is inspectable, none hardcoded |

**Python 3.11** (pinned — see `pyproject.toml`). **Node 18+** for the frontend.

---

## Key Features

- **Context-aware PHI detection** — a bare regex match isn't enough; a professional title
  immediately preceding a name (`Dr. Smith`) is deterministically excluded, while genuinely
  ambiguous cases go to LLM judgment.
- **Confidence-gated everything** — PHI entities, clinical findings, and Q&A answers all carry a
  confidence score and are only surfaced above a configured threshold (`src/config.py`), not
  hardcoded magic numbers.
- **Hallucination-resistant Q&A** — the LLM is instructed to answer `NOT_FOUND` rather than guess,
  and a low-confidence "answer" is treated identically to an explicit refusal.
- **Per-document isolation** — retrieval is scoped so one document's Q&A session can never surface
  another document's content.
- **Full audit trail** — every agent decision (detection, extraction, retrieval, refusal) is
  logged with a timestamp, confidence score, and reasoning string, queryable per document.

---

## API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | `GET` | System health check |
| `/upload` | `POST` | Upload a medical document (PDF/DOCX/TXT) |
| `/process` | `POST` | Run the full 5-agent pipeline on an uploaded document |
| `/chat` | `POST` | Ask a grounded question against a processed, indexed document |
| `/documents/{document_id}` | `GET` | Document metadata and processing status |
| `/audit/{document_id}` | `GET` | Full audit trail for a document |
| `/pipeline-status/{document_id}` | `GET` | Live pipeline execution status |

Interactive docs are available at `/docs` (Swagger UI) once the backend is running.

---

## Project Structure

```
PatientShield-AI/
├── src/
│   ├── agents/            # Planner, PHI, Clinical, Knowledge, Audit agents
│   ├── orchestrator/       # LLM fallback manager, LangGraph pipeline, retrieval connector
│   ├── utils/              # Document parsing, text processing, metrics
│   ├── validation/         # PHI detection scoring against ground truth
│   ├── main.py              # FastAPI app
│   ├── config.py           # Pydantic Settings (all thresholds, model names, paths)
│   ├── database.py         # SQLAlchemy models
│   └── models.py           # Pydantic request/response schemas
├── medintel-ui/            # React + TypeScript frontend
│   └── src/
│       ├── pages/          # Dashboard, Processing, Workspace, Chat, Audit Trail
│       ├── services/       # Typed Axios API client
│       └── components/     # Shared UI + chat/audit components
├── tests/                  # pytest suites — unit, integration, and system-level
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python **3.11** (`python --version` — if it resolves to something else, use a version manager
  or an explicit `py -3.11` / `python3.11` invocation)
- Node.js 18+
- A [Groq API key](https://console.groq.com) (free tier) — or a locally running
  [Ollama](https://ollama.com) instance as an offline fallback

### Backend setup

```bash
# From the repository root
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# spaCy's English model isn't pip-installable by version pin alone:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

cp .env.example .env
# Edit .env — at minimum, set GROQ_API_KEY (or configure Ollama locally)

uvicorn src.main:app --reload --port 8000
```

Visit `http://localhost:8000/health` — you should see `{"status": "healthy", ...}`.

### Frontend setup

```bash
cd medintel-ui
npm install
cp .env.example .env   # VITE_API_URL should point at the backend above
npm run dev
```

Visit `http://localhost:5173`.

### Optional: enable Chat / knowledge retrieval

The Knowledge Agent indexes de-identified text to a separate hybrid-retrieval (BM25 + dense) RAG
service and calls it for grounded Q&A. Point `UNIRAG_BASE_URL` in `.env` at a running instance of
a compatible retrieval service. **Without it, PHI detection and clinical extraction still run
fully** — only document indexing and `/chat` degrade to an honest "unavailable" response instead
of failing the whole pipeline.

### Running tests

```bash
pytest tests/ -v
```

Test suites run against real Presidio, real spaCy, and a real LLM call chain — nothing is mocked
in the primary suites, by design, so a passing suite reflects real model behavior.

---

## Configuration

All runtime behavior is controlled through `.env` (see `.env.example` for the full list with
inline explanations) — no thresholds or model names are hardcoded in source. Notable ones:

| Variable | Default | Meaning |
|---|---|---|
| `PHI_CONFIDENCE_THRESHOLD` | `0.80` | Minimum confidence for a PHI entity to be masked |
| `CLINICAL_PRIORITY_THRESHOLD` | `0.70` | Minimum priority for a clinical finding to be surfaced |
| `QA_CONFIDENCE_THRESHOLD` | `0.85` | Minimum confidence for Q&A to answer instead of refuse |
| `DEMO_MODE` | `false` | Routes LLM calls through Claude instead of Groq/Ollama |

---

## Known Limitations

Being explicit about scope here matters more than sounding finished:

- **PHI detection** relies on Presidio's entity patterns plus LLM context judgment — it will miss
  identifiers that don't match any known shape (e.g., a hospital-internal ID format it's never
  seen).
- **De-identification masks, it doesn't synthesize** — PHI is replaced with `[ENTITY_TYPE]`
  placeholders, not realistic fake data, so the output isn't suited for model-training use cases.
- **Q&A performs best on structured clinical notes**; narrative, unstructured prose is harder to
  ground reliably.
- **Single-user, single-session** — no multi-user access control or persistent conversation memory
  across sessions is implemented.
- **Not a certified medical or compliance product.** The system is built with HIPAA-relevant
  patterns (audit logging, confidence gating, de-identification-before-indexing) but has not been
  through formal compliance certification. Test data used throughout development is synthetic.

---

## Disclaimer

This project processes **synthetic and de-identified test data only** during development. It is
not intended for use with real patient data without independent security and compliance review.

---

## License

Released under the [MIT License](LICENSE).
