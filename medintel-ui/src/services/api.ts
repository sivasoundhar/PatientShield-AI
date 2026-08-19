import axios, { type AxiosInstance } from "axios"

import type {
  AuditTrailResponse,
  ChatRequest,
  DocumentResponse,
  HealthResponse,
  PipelineStatusResponse,
  ProcessRequest,
  ProcessResponse,
  QAResponse,
  UploadResponse,
} from "@/types"

/**
 * Thin typed wrapper over the 7 backend endpoints (CLAUDE.md section 6).
 * One method per endpoint, named after what it does rather than the HTTP
 * verb+path, so call sites read as intent ("processDocument(id)") instead
 * of re-deriving the route every time.
 */
const client: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
})

export const api = {
  /** GET /health */
  async getHealth(): Promise<HealthResponse> {
    const { data } = await client.get<HealthResponse>("/health")
    return data
  },

  /** POST /upload — multipart form with the file and optional patient_id. */
  async uploadDocument(file: File, patientId?: string): Promise<UploadResponse> {
    const form = new FormData()
    form.append("file", file)
    if (patientId) form.append("patient_id", patientId)
    const { data } = await client.post<UploadResponse>("/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    return data
  },

  /** POST /process — runs the 5-agent pipeline on an already-uploaded document. */
  async processDocument(documentId: string): Promise<ProcessResponse> {
    const payload: ProcessRequest = { document_id: documentId }
    const { data } = await client.post<ProcessResponse>("/process", payload)
    return data
  },

  /** POST /chat — ask a question against an indexed document. */
  async askQuestion(documentId: string, question: string): Promise<QAResponse> {
    const payload: ChatRequest = { document_id: documentId, question }
    const { data } = await client.post<QAResponse>("/chat", payload)
    return data
  },

  /** GET /audit/{document_id} */
  async getAuditTrail(documentId: string): Promise<AuditTrailResponse> {
    const { data } = await client.get<AuditTrailResponse>(`/audit/${documentId}`)
    return data
  },

  /** GET /documents/{document_id} */
  async getDocument(documentId: string): Promise<DocumentResponse> {
    const { data } = await client.get<DocumentResponse>(`/documents/${documentId}`)
    return data
  },

  /** GET /pipeline-status/{document_id} — polled by Processing.tsx. */
  async getPipelineStatus(documentId: string): Promise<PipelineStatusResponse> {
    const { data } = await client.get<PipelineStatusResponse>(`/pipeline-status/${documentId}`)
    return data
  },
}

/** Extracts a readable message from any error this client can throw — API error, network error, or otherwise. */
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = (error.response?.data as { detail?: string } | undefined)?.detail
    if (detail) return detail
    if (error.code === "ERR_NETWORK") return "Can't reach the backend — is the API server running?"
    return error.message
  }
  if (error instanceof Error) return error.message
  return "An unexpected error occurred."
}
