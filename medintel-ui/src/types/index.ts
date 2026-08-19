/**
 * TypeScript mirror of src/models.py — every interface here corresponds
 * 1:1 to a Pydantic model on the backend. Keep field names, optionality,
 * and enum values in exact sync with models.py; this file is the contract
 * the whole frontend is built against (CLAUDE.md Day 3 instruction 3).
 *
 * Dates: Pydantic serializes `datetime` fields as ISO 8601 strings over
 * JSON, so every backend `datetime` becomes a TS `string` here, not `Date`
 * — parse with `new Date(...)` at the point of use (see lib/format.ts).
 */

// --- Enums (src/models.py) ---

export type ProcessingStatus = "uploaded" | "processing" | "completed" | "failed"

export type FindingCategory = "DIAGNOSIS" | "MEDICATION" | "FINDING" | "FOLLOW_UP"

export type AuditStatus = "success" | "error" | "skipped"

// --- Agent result payloads (CLAUDE.md section 7) ---

export interface PHIEntity {
  entity_type: string
  text: string
  confidence: number
  start_pos: number
  end_pos: number
  reasoning: string
}

export interface PHIDetectionResult {
  entities: PHIEntity[]
  total_count: number
  high_confidence_count: number
  precision_estimate: number | null
}

export interface ClinicalFinding {
  category: FindingCategory
  value: string
  priority_score: number
  confidence: number
}

export interface ClinicalAnalysisResult {
  findings: ClinicalFinding[]
  summary: string
}

export interface QAResult {
  answer: string
  confidence: number
  source_citation: string | null
  found_in_document: boolean
}

export interface AuditEvent {
  timestamp: string
  agent_name: string
  action: string
  status: AuditStatus
  details: Record<string, unknown> | null
}

// --- API request/response models (CLAUDE.md section 6) ---

export interface HealthResponse {
  timestamp: string
  status: string
  version: string
  llm_status: string
}

export interface UploadResponse {
  timestamp: string
  document_id: string
  filename: string
  status: ProcessingStatus
}

export interface ProcessRequest {
  document_id: string
}

export interface ProcessResponse {
  timestamp: string
  document_id: string
  processing_status: ProcessingStatus
  original_text: string | null
  de_identified_text: string | null
  phi_detected: PHIDetectionResult | null
  clinical_analysis: ClinicalAnalysisResult | null
  knowledge_indexed: boolean
  audit_events: AuditEvent[]
  processing_time: number | null
}

export interface ChatRequest {
  document_id: string
  question: string
}

export interface QAResponse extends QAResult {
  timestamp: string
}

export interface AuditTrailResponse {
  document_id: string
  audit_trail: AuditEvent[]
}

export interface DocumentResponse {
  timestamp: string
  id: string
  filename: string
  patient_id: string | null
  processing_status: ProcessingStatus
  created_at: string
  updated_at: string
}

export interface PipelineStatusResponse {
  document_id: string
  status: ProcessingStatus
  processing_time: number | null
  events: AuditEvent[]
}

/** Backend error envelope — every non-2xx FastAPI response has this shape. */
export interface ApiErrorBody {
  detail: string
}
