/**
 * Client-side "recent documents" tracking.
 *
 * The backend has no GET /documents list endpoint (CLAUDE.md section 6 only
 * defines GET /documents/{id} for one document at a time) — adding one would
 * be backend scope creep on a frontend-only day. Instead, every successful
 * /upload response is remembered here (id + filename + when), and Dashboard
 * hydrates each entry's *live* status via the real GET /documents/{id} call.
 * This is honestly "documents this browser has uploaded," not a fabricated
 * global document list — it resets if localStorage is cleared, which is an
 * acceptable tradeoff for a portfolio-scale single-user app.
 */
const STORAGE_KEY = "patientshield.recentDocuments"
const MAX_TRACKED = 20

export interface RecentDocumentEntry {
  documentId: string
  filename: string
  uploadedAt: string
}

export function getRecentDocuments(): RecentDocumentEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function trackUploadedDocument(entry: RecentDocumentEntry): void {
  const existing = getRecentDocuments().filter((d) => d.documentId !== entry.documentId)
  const updated = [entry, ...existing].slice(0, MAX_TRACKED)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
}
