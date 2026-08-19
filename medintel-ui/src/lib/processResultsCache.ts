import type { ProcessResponse } from "@/types"

/**
 * Caches the last /process result per document, in this browser only.
 *
 * Why this exists: GET /documents/{id} deliberately returns metadata only
 * (see DocumentResponse's docstring in src/models.py) — there is no backend
 * endpoint to re-fetch phi_detected/clinical_analysis/de_identified_text
 * after the one-time POST /process response. Extending that is a backend
 * decision for a later day, not something to slip in on a frontend-only
 * day. Until then, Workspace.tsx shows real results for documents processed
 * in this session and an honest "re-run processing to see this" empty state
 * otherwise — never fabricated data.
 */
const STORAGE_KEY = "patientshield.processResults"

function readAll(): Record<string, ProcessResponse> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

export function getCachedProcessResult(documentId: string): ProcessResponse | null {
  return readAll()[documentId] ?? null
}

export function cacheProcessResult(documentId: string, result: ProcessResponse): void {
  const all = readAll()
  all[documentId] = result
  localStorage.setItem(STORAGE_KEY, JSON.stringify(all))
}
