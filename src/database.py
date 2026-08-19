"""SQLAlchemy ORM models and session management.

Five tables back the whole pipeline: `documents` (the record itself, plus a
denormalized snapshot of each agent's output for fast reads), and four
detail tables (`phi_entities`, `clinical_findings`, `audit_logs`,
`test_metrics`) that keep per-entity rows for querying, filtering, and the
Day 8-9 precision/recall math.

Denormalizing agent output onto `documents` (as JSON columns) *and* storing
normalized rows in the detail tables is deliberate: the JSON snapshot is what
the API returns for a single document in one query; the detail tables are
what audit search, PHI metrics, and the AuditTrail UI page filter across many
documents. See docs/DATABASE_SCHEMA.md (user-authored) for the full rationale.
"""

import uuid
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """Timezone-aware UTC now, for DateTime column defaults — keeps every audit
    timestamp directly comparable instead of mixing naive/aware datetimes."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """UUID4 string primary key — avoids leaking row-count via sequential IDs."""
    return str(uuid.uuid4())


class Document(Base):
    """A single uploaded medical document and the latest snapshot of pipeline output."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    patient_id: Mapped[str | None] = mapped_column(String, nullable=True)

    original_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    de_identified_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Denormalized JSON snapshots of each agent's result, for single-query reads.
    phi_detected: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    clinical_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    qa_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    audit_trail: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # uploaded -> processing -> completed | failed
    processing_status: Mapped[str] = mapped_column(String, default="uploaded", nullable=False)
    # Wall-clock seconds the last /process run took — populated once, never
    # updated again by /pipeline-status, /audit/{id}, or /chat. Without a
    # persisted column here, /pipeline-status's own PipelineStatusResponse
    # (section 6: "{status, processing_time, events}") could never report
    # this after the fact — /process's own direct response was the only
    # place it was ever visible, a gap found by Day 7's integration tests.
    processing_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    phi_entities: Mapped[list["PHIEntityRecord"]] = relationship(back_populates="document")
    clinical_findings: Mapped[list["ClinicalFindingRecord"]] = relationship(back_populates="document")
    audit_logs: Mapped[list["AuditLogRecord"]] = relationship(back_populates="document")


class PHIEntityRecord(Base):
    """One detected PHI entity, normalized for cross-document precision/recall queries."""

    __tablename__ = "phi_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)

    entity_type: Mapped[str] = mapped_column(String, nullable=False)  # e.g. PERSON, SSN, DATE_OF_BIRTH
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    start_pos: Mapped[int] = mapped_column(Integer, nullable=False)
    end_pos: Mapped[int] = mapped_column(Integer, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="phi_entities")


class ClinicalFindingRecord(Base):
    """One extracted clinical finding (diagnosis, medication, lab finding, or follow-up)."""

    __tablename__ = "clinical_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)

    category: Mapped[str] = mapped_column(String, nullable=False)  # DIAGNOSIS | MEDICATION | FINDING | FOLLOW_UP
    value: Mapped[str] = mapped_column(Text, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="clinical_findings")


class AuditLogRecord(Base):
    """One event emitted by any agent. This table is the HIPAA-facing audit trail."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)

    agent_name: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    # Mirrors AuditEvent.status ("success" | "error" | "skipped") — without a
    # column for this, every persisted event silently reverted to "success"
    # on read, hiding the difference between an agent that ran and one that
    # was only a Day 2 placeholder. Traceable status is the whole point of a
    # HIPAA-facing audit trail (rule 10), so this isn't optional.
    status: Mapped[str] = mapped_column(String, default="success", nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="audit_logs")


class TestMetricRecord(Base):
    """A single precision/recall/accuracy measurement, populated during Days 8-9 system testing."""

    __tablename__ = "test_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"), nullable=True)

    agent_name: Mapped[str] = mapped_column(String, nullable=False)  # phi | clinical | knowledge
    metric_name: Mapped[str] = mapped_column(String, nullable=False)  # precision | recall | f1 | accuracy | hallucination_rate
    value: Mapped[float] = mapped_column(Float, nullable=False)

    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


# SQLite needs check_same_thread=False because FastAPI's threadpool can hand a
# request to a different thread than the one that opened the connection; every
# other engine we might swap in (section 3: SQLAlchemy + SQLite is fixed) ignores
# this arg harmlessly if ever changed.
_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they don't already exist.

    Returns:
        None.

    Use when: called once at FastAPI startup. Safe to call repeatedly —
    `create_all` no-ops on existing tables.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session, closing it after the request completes.

    Returns:
        A SQLAlchemy Session, via FastAPI's dependency-injection generator pattern.

    Use when: declared as a FastAPI route dependency, e.g. `db: Session = Depends(get_db)`.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Self-test: round-trip one row per table on an in-memory DB. No external dependency.
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    with sessionmaker(bind=test_engine)() as session:
        doc = Document(filename="sample.txt", processing_status="uploaded")
        session.add(doc)
        session.commit()
        session.refresh(doc)

        rows = [
            PHIEntityRecord(document_id=doc.id, entity_type="PERSON", text="[REDACTED]", confidence=0.95, start_pos=0, end_pos=10),
            ClinicalFindingRecord(document_id=doc.id, category="DIAGNOSIS", value="Type 2 diabetes", priority_score=0.9, confidence=0.88),
            AuditLogRecord(document_id=doc.id, agent_name="planner", action="plan_created", status="success", details={"steps": 4}),
            TestMetricRecord(document_id=doc.id, agent_name="phi", metric_name="precision", value=0.95),
        ]
        session.add_all(rows)
        session.commit()

        for model in (Document, PHIEntityRecord, ClinicalFindingRecord, AuditLogRecord, TestMetricRecord):
            assert session.query(model).count() == 1, f"{model.__name__} round-trip failed"

    print("database.py self-test passed: all 5 tables created and round-tripped a row.")
