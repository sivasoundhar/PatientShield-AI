"""FastAPI entrypoint — the 7-endpoint API surface defined in CLAUDE.md section 6.

/process runs the real LangGraph pipeline (Day 2); phi_agent (Day 4),
clinical_agent (Day 5), and knowledge_agent (Day 6) are real, audit_agent
remains an honest placeholder until Day 7 (see pipeline.py). /chat runs real
retrieval-grounded Q&A against UniRAG (Day 6) as of today. Every other
endpoint is fully functional against the database built in database.py.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.agents.knowledge_agent import KnowledgeAgent
from src.config import settings
from src.database import AuditLogRecord, Document, get_db, init_db
from src.models import (
    AuditEvent,
    AuditTrailResponse,
    ChatRequest,
    DocumentResponse,
    HealthResponse,
    PipelineStatusResponse,
    ProcessingStatus,
    ProcessRequest,
    ProcessResponse,
    QAResponse,
    UploadResponse,
)
from src.orchestrator.pipeline import PatientShieldPipeline
from src.utils.pdf_parser import UnsupportedFileTypeError, extract_text
from src.utils.text_processor import clean_text

logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.ensure_storage_dirs()
    init_db()
    yield


app = FastAPI(title="PatientShield AI", version=settings.APP_VERSION, lifespan=lifespan)

# Constructed once at import time: compiling the LangGraph graph is the
# relatively expensive part, and the compiled graph holds no per-document
# state, so one instance safely serves every /process request.
pipeline = PatientShieldPipeline()

# Separate instance from pipeline.py's own module-level KnowledgeAgent
# singleton (used only for indexing during /process) — /chat runs
# independently of any pipeline execution, potentially long after indexing
# finished, so sharing an instance would buy nothing beyond a slightly
# smaller audit-trail list. Both talk to the same UniRAG service; there's no
# shared in-memory state that actually needs to be the same object.
knowledge_agent = KnowledgeAgent()

# Local dev only: React frontend (Day 3) runs on Vite's default port.
# Tightened before any real deployment — Render env vars would set this
# explicitly rather than leaving a wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_document_or_404(document_id: str, db: Session) -> Document:
    """Fetch a Document by id or raise the 404 every read/write endpoint needs.

    Use when: any endpoint keyed by document_id — keeps the not-found
    behavior (and its message) identical everywhere instead of redefined per
    route.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return document


def _audit_trail_for(document_id: str, db: Session) -> list[AuditEvent]:
    rows = (
        db.query(AuditLogRecord)
        .filter(AuditLogRecord.document_id == document_id)
        .order_by(AuditLogRecord.timestamp)
        .all()
    )
    return [
        AuditEvent(timestamp=r.timestamp, agent_name=r.agent_name, action=r.action, status=r.status, details=r.details)
        for r in rows
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # LLMManager exists as of Day 2, but /health doesn't yet probe Groq/Ollama/
    # Claude availability — that's a live network call, not appropriate for a
    # liveness check to make on every hit. llm_status stays a static value
    # until there's a real readiness check to report (not scoped to any day).
    return HealthResponse(status="healthy", version=settings.APP_VERSION, llm_status="not_configured")


@app.post("/upload", response_model=UploadResponse)
def upload(
    file: UploadFile = File(...),
    patient_id: str | None = Form(None),
    db: Session = Depends(get_db),
) -> UploadResponse:
    document_id = str(uuid.uuid4())
    dest_path = Path(settings.UPLOADS_PATH) / f"{document_id}_{file.filename}"
    dest_path.write_bytes(file.file.read())

    try:
        text = clean_text(extract_text(str(dest_path)))
    except (UnsupportedFileTypeError, ValueError) as exc:
        # Edge: bad file type or unreadable content — fail loudly with a
        # message the caller can act on, per rule 8.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document = Document(
        id=document_id,
        filename=file.filename,
        patient_id=patient_id,
        original_text=text,
        processing_status=ProcessingStatus.UPLOADED.value,
    )
    db.add(document)
    db.commit()

    return UploadResponse(document_id=document_id, filename=file.filename, status=ProcessingStatus.UPLOADED)


@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest, db: Session = Depends(get_db)) -> ProcessResponse:
    document = _get_document_or_404(request.document_id, db)

    # Mark processing before the pipeline runs (not after) so a client
    # polling /pipeline-status mid-run sees "processing" rather than a stale
    # "uploaded" — the whole point of that endpoint (section 6).
    document.processing_status = ProcessingStatus.PROCESSING.value
    db.commit()

    try:
        result = await pipeline.run_pipeline(document.id, document.original_text or "")
    except Exception as exc:
        # Edge: the pipeline itself raised (not a node returning a placeholder
        # event, an actual exception). Fail loudly per rule 8 rather than
        # silently reporting success.
        document.processing_status = ProcessingStatus.FAILED.value
        db.commit()
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

    document.de_identified_text = result["de_identified_text"]
    document.phi_detected = result["phi_results"]
    document.clinical_analysis = result["clinical_results"]
    document.processing_time = result["processing_time"]
    document.processing_status = ProcessingStatus.COMPLETED.value
    db.commit()

    # Persist every node's audit event to the audit_logs table — this is what
    # makes /audit/{id} and /pipeline-status/{id} (both DB-backed) reflect
    # this run, not just this response.
    for event in result["audit_events"]:
        db.add(
            AuditLogRecord(
                document_id=document.id,
                agent_name=event.agent_name,
                action=event.action,
                status=event.status,
                details=event.details,
                timestamp=event.timestamp,
            )
        )
    db.commit()

    return ProcessResponse(
        document_id=document.id,
        processing_status=ProcessingStatus.COMPLETED,
        original_text=document.original_text,
        de_identified_text=result["de_identified_text"],
        phi_detected=result["phi_results"],
        clinical_analysis=result["clinical_results"],
        knowledge_indexed=result["knowledge_indexed"],
        audit_events=result["audit_events"],
        processing_time=result["processing_time"],
    )


@app.post("/chat", response_model=QAResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)) -> QAResponse:
    document = _get_document_or_404(request.document_id, db)

    events_before = len(knowledge_agent.get_audit_trail())
    result = await knowledge_agent.answer_question(request.document_id, request.question)
    new_events = knowledge_agent.get_audit_trail()[events_before:]

    # Denormalized snapshot on the document row (mirrors phi_detected/
    # clinical_analysis) plus the durable audit trail — same two-writes
    # pattern /process uses, just scoped to one Q&A turn instead of a full
    # pipeline run.
    document.qa_results = result.model_dump(mode="json")
    for event in new_events:
        db.add(
            AuditLogRecord(
                document_id=document.id,
                agent_name=event.agent_name,
                action=event.action,
                status=event.status,
                details=event.details,
                timestamp=event.timestamp,
            )
        )
    db.commit()

    return QAResponse(
        answer=result.answer,
        confidence=result.confidence,
        source_citation=result.source_citation,
        found_in_document=result.found_in_document,
    )


@app.get("/audit/{document_id}", response_model=AuditTrailResponse)
def get_audit_trail(document_id: str, db: Session = Depends(get_db)) -> AuditTrailResponse:
    _get_document_or_404(document_id, db)
    return AuditTrailResponse(document_id=document_id, audit_trail=_audit_trail_for(document_id, db))


@app.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentResponse:
    document = _get_document_or_404(document_id, db)
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        patient_id=document.patient_id,
        processing_status=ProcessingStatus(document.processing_status),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@app.get("/pipeline-status/{document_id}", response_model=PipelineStatusResponse)
def pipeline_status(document_id: str, db: Session = Depends(get_db)) -> PipelineStatusResponse:
    document = _get_document_or_404(document_id, db)
    return PipelineStatusResponse(
        document_id=document_id,
        status=ProcessingStatus(document.processing_status),
        processing_time=document.processing_time,
        events=_audit_trail_for(document_id, db),
    )
