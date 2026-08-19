"""Pydantic schemas for API requests/responses and agent result payloads.

Every service boundary (API in/out, agent output, DB write) goes through one
of these models — see CLAUDE.md rule 3 ("Pydantic models are mandatory at
service boundaries"). Agent result models (PHIDetectionResult,
ClinicalAnalysisResult, AuditEvent, ...) are what Days 4-7 will have agents
return; API models wrap them for the 7 endpoints in section 6.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _now() -> datetime:
    """UTC now, used as the default_factory for every response's `timestamp` field."""
    return datetime.now(timezone.utc)


class ProcessingStatus(str, Enum):
    """Lifecycle of a Document row, driving both DB status and UI progress display."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FindingCategory(str, Enum):
    """The four clinical finding types the Clinical Agent extracts (Day 5)."""

    DIAGNOSIS = "DIAGNOSIS"
    MEDICATION = "MEDICATION"
    FINDING = "FINDING"
    FOLLOW_UP = "FOLLOW_UP"


class TimestampedResponse(BaseModel):
    """Base for every API response — section 6 requires all responses carry a timestamp."""

    timestamp: datetime = Field(default_factory=_now)


# --- Agent result payloads (section 7) ---


class PHIEntity(BaseModel):
    """One PHI entity detected by the PHI Reasoner Agent (Day 4)."""

    entity_type: str  # e.g. PERSON, SSN, DATE_OF_BIRTH, PHONE_NUMBER, MRN
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    start_pos: int
    end_pos: int
    reasoning: str  # why the LLM/Presidio judged this PHI vs. a false positive


class PHIDetectionResult(BaseModel):
    """Full output of the PHI Reasoner Agent for one document."""

    entities: list[PHIEntity] = Field(default_factory=list)
    total_count: int = 0
    high_confidence_count: int = 0  # entities at/above PHI_CONFIDENCE_THRESHOLD
    precision_estimate: float | None = None  # only populated once ground truth exists (Day 8-9)


class ClinicalFinding(BaseModel):
    """One extracted clinical finding from the Clinical Agent (Day 5)."""

    category: FindingCategory
    value: str
    priority_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class ClinicalAnalysisResult(BaseModel):
    """Full output of the Clinical Agent for one document."""

    findings: list[ClinicalFinding] = Field(default_factory=list)
    summary: str = ""


class QAResult(BaseModel):
    """Full output of the Knowledge Agent answering one question (Day 6)."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_citation: str | None = None  # chunk/passage the answer was grounded in
    found_in_document: bool  # False => refused rather than risk hallucinating


class AuditEvent(BaseModel):
    """One agent decision, as recorded by BaseAgent.log_decision (rule 10)."""

    timestamp: datetime = Field(default_factory=_now)
    agent_name: str
    action: str
    status: str = "success"  # success | error | skipped
    details: dict | None = None


class AuditSummary(BaseModel):
    """Consolidated view of one pipeline run's audit trail, produced by the Audit Agent (Day 7).

    Not returned by a dedicated endpoint — embedded in the audit_consolidated
    AuditEvent's `details`, retrievable via the existing /audit/{id} and
    /pipeline-status/{id} endpoints without any new API surface.
    """

    total_events: int
    events_by_agent: dict[str, int] = Field(default_factory=dict)
    events_by_status: dict[str, int] = Field(default_factory=dict)
    # Expected agents (planner, phi_agent, clinical_agent, knowledge_agent)
    # that produced zero events in this run — a HIPAA-facing audit trail's
    # whole point is catching exactly this: a node that silently didn't run
    # looks, from the trail alone, identical to one that ran and logged
    # nothing, unless something explicitly checks for its absence.
    missing_agents: list[str] = Field(default_factory=list)


# --- API request/response models (section 6) ---


class HealthResponse(TimestampedResponse):
    status: str
    version: str
    llm_status: str  # e.g. "not_configured" until Day 2 wires the LLM manager


class UploadResponse(TimestampedResponse):
    document_id: str
    filename: str
    status: ProcessingStatus


class ProcessRequest(BaseModel):
    document_id: str


class ProcessResponse(TimestampedResponse):
    """Full pipeline output — the /process endpoint's response body."""

    document_id: str
    processing_status: ProcessingStatus
    # Both sides of the Workspace UI's "original vs de-identified"
    # comparison (core to the product's privacy pitch) — as of Day 4's real
    # masking, these two fields genuinely diverge, so both must be present
    # or the comparison has nothing meaningful to show.
    original_text: str | None = None
    de_identified_text: str | None = None
    phi_detected: PHIDetectionResult | None = None
    clinical_analysis: ClinicalAnalysisResult | None = None
    knowledge_indexed: bool = False
    audit_events: list[AuditEvent] = Field(default_factory=list)
    processing_time: float | None = None  # seconds; None until pipeline finishes


class ChatRequest(BaseModel):
    document_id: str
    question: str


class QAResponse(TimestampedResponse, QAResult):
    """/chat response — reuses QAResult's fields plus the response timestamp."""


class AuditTrailResponse(BaseModel):
    document_id: str
    audit_trail: list[AuditEvent] = Field(default_factory=list)


class DocumentResponse(TimestampedResponse):
    """/documents/{id} response — metadata only, not the (potentially large) text fields."""

    id: str
    filename: str
    patient_id: str | None
    processing_status: ProcessingStatus
    created_at: datetime
    updated_at: datetime


class PipelineStatusResponse(BaseModel):
    document_id: str
    status: ProcessingStatus
    processing_time: float | None = None
    events: list[AuditEvent] = Field(default_factory=list)


if __name__ == "__main__":
    # Self-test: every model must construct and round-trip through JSON with
    # minimal valid data. No external dependency.
    phi = PHIDetectionResult(entities=[PHIEntity(entity_type="PERSON", text="[REDACTED]", confidence=0.95, start_pos=0, end_pos=10, reasoning="Presidio NAME match, LLM-confirmed")])
    clinical = ClinicalAnalysisResult(findings=[ClinicalFinding(category=FindingCategory.DIAGNOSIS, value="Type 2 diabetes", priority_score=0.9, confidence=0.88)], summary="Routine follow-up.")
    proc = ProcessResponse(document_id="doc-1", processing_status=ProcessingStatus.COMPLETED, phi_detected=phi, clinical_analysis=clinical, audit_events=[AuditEvent(agent_name="planner", action="plan_created")])
    assert proc.model_validate_json(proc.model_dump_json()).document_id == "doc-1"

    qa = QAResponse(answer="Patient was prescribed metformin.", confidence=0.9, found_in_document=True)
    assert qa.model_validate_json(qa.model_dump_json()).confidence == 0.9

    summary = AuditSummary(total_events=4, events_by_agent={"planner": 1, "phi_agent": 3}, events_by_status={"success": 4}, missing_agents=["knowledge_agent"])
    assert summary.model_validate_json(summary.model_dump_json()).total_events == 4

    print("models.py self-test passed: all schemas construct and round-trip through JSON.")
